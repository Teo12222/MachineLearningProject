"""DAgger for the split arm network (PLAN.md step 21): run a split model, label the arm's frames with the expert,
retrain on everything collected so far, repeat.

Usage (from the robotnik_arm_task folder):
    pixi run python dagger.py --start split_arrived_s0 --rounds 3          # rounds 1-3, 200 training targets each
    pixi run python dagger.py --start split_arrived_s0 --rounds 3 --val-set  # + compare.py --val-set per round
    pixi run python dagger.py --start split_dart_a60x2_s0 --extra-data dart/a60 dart/a60b --name split_dartdagger
                                                   # behind a DART model, its DART data in every round (PLAN.md A2)

Round k (models split_dagger_r<k>_s<seed>, round 1 starting from --start):
1. The previous model runs headless on --n TRAINING targets (never validation or test ones), the next block of a
   fixed random order (wrapping around once all are used: a newer policy visits different states on the same
   target), from each episode's stored start pose, like compare.py (failing runs last the full 30 s).
2. Each run's arm phase is labelled with the expert's arm torques (expert_relabel.py): from the latch until --window
   seconds later, or until the expert's label stays at its torque limit (see label_run), whichever comes first.
   Saved as dagger/split_dagger_r<k>/ep_XXXX.npz in data_main's episode format (see save_episode).
3. train.py --split --dagger <--extra-data folders + all rounds' folders> trains the round's arm from scratch, with the
   base copied from --start (--base-from) and the settings of --start: sizes and gravity residual from its checkpoint,
   lr, batch and weight decay from its results file (a tuned start model keeps its tuned settings).
4. With --val-set: compare.py --val-set into compare_out/<model>/, the 50 validation episodes used for decisions
   such as which round to keep (never writes results; test episodes are only for the final numbers).

Steps that are already done are skipped (episode folder complete, checkpoint present), so an interrupted run can be
restarted with the same command. dagger/ is git-ignored. Round 1's folder records --start and --extra-data
(dagger_info.json); a run whose --name already belongs to another start or extra-data set stops before anything runs,
because skipping "done" steps would otherwise silently reuse another run's data and models.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + numpy each ship an OpenMP runtime on Windows

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from bc_data import clean_ids, split_episodes
from bc_split import SplitPolicy, load_policy, net_activation, net_output_act, net_settings
from compare import POLICY_SCRIPT, POLICY_TIMEOUT, load_npz, run_episode, stored_start
from results import read_results

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_main"
DAGGER_DIR = SCRIPT_DIR / "dagger"
TORQUE_LIMIT = 30.0     # N*m, the expert's torque clip
SATURATED_RUN = 50      # frames (ms): the expert's label at its limit this long ends the labelled part ...
SATURATION_GRACE = 500  # frames: ... but not in the first 0.5 s after the latch, where the expert's own reach starts
                        # with a short kick at the limit (J6 starts below its joint limit, PLAN.md step 21)
LOW_Z = 0.5             # m: targets below this are "low" (--low-extra; all collapses were below it, PLAN.md C18e)


def label_run(raw_path, out_path, window, posture_free=False, posture_kp=0.0):
    """Label one policy run's arm phase with the expert and save it as a DAgger episode. Returns the labelled frames.

    Labelled from the latch until window seconds later, or until the expert's label stays at its torque limit for
    SATURATED_RUN frames: by then the arm is so far off that the expert itself is at its limits (its own data has
    such frames only in the short kick at the start of the reach), and those huge labels, many standard deviations
    outside the expert's data, would dominate the training loss. Checked on 3 runs: before that point the labels are
    the size of the expert's own (|z| median 1-3), after it ~6-8 with 60-100% of frames at the limit.
    Runs in a worker process (expert_relabel builds its own robot model when imported).
    posture_free (PLAN.md C18e): each label's posture-only part (the arm's null space, which depends on the expert's
    hidden q_des) is replaced by the expert's damping on the measured joint velocity (relabel_posture.remove_posture),
    BEFORE the saturation check, so the stop judges the cleaned labels. The expert's q_des still integrates as before.
    posture_kp (with posture_free, PLAN.md C18e 4 Nov): plus a spring pulling joints 1, 3, 5 to 0 in the null space
    (remove_posture's kp_ref), against the drift on low targets.
    """
    from expert_relabel import relabel_arm
    d = load_npz(raw_path)
    n = len(d["time"])
    mask = np.zeros(n, dtype=bool)
    torque = np.full((n, 7), np.nan, dtype=np.float32)   # nan where there is no label, so misuse can't go unnoticed
    # C19 Route A: label from the arm's own flag (its trigger) instead of the base's latch, and save that flag as
    # nav_done, because training feeds nav_done to the arm network as its "reach" input. Runs without arm_on (older
    # policy files) or with the trigger off: the latch, exactly as before
    flag = d["arm_on"] if "arm_on" in d else d["nav_done"]
    if flag.any():   # a run that never latched has no arm phase to label
        k = int(np.argmax(flag))
        stop = min(n, k + round(window * 1000))
        # The expert holds the arm's start pose until it arrives, so that is its joint reference when the reach starts
        labels = relabel_arm(d, k, stop, d["q_mes"][0, 4:11])
        if posture_free:
            from relabel_posture import remove_posture
            # torque_arm holds the policy's torques here: put the expert's labels in first (only up to stop, so
            # remove_posture doesn't process the frames after it)
            t = d["torque_arm"][:stop].astype(float).copy()
            t[k:stop] = labels
            labels = remove_posture({**d, "torque_arm": t}, k, posture_kp)[k:stop]
        saturated = (np.abs(labels) >= TORQUE_LIMIT - 0.01).any(axis=1)
        runs = np.convolve(saturated, np.ones(SATURATED_RUN), "valid") == SATURATED_RUN   # run starting at each frame
        runs[:SATURATION_GRACE] = False
        if runs.any():
            stop = k + int(np.argmax(runs))
        torque[k:stop] = labels[:stop - k]
        mask[k:stop] = True
    save_episode(out_path, d, torque, mask, flag)
    raw_path.unlink()
    return int(mask.sum())


def save_episode(path, d, torque, mask, flag):
    """A DAgger episode in data_main's format, without the expert-only fields (nav_phase, q_des_arm, base_goal).

    torque_arm holds the expert's labels where arm_label_mask is True (nan elsewhere); nav_done is the policy's
    latched flag. What the policy itself commanded is kept as policy_wheel_cmds / policy_torque_arm. No wheel labels
    yet (base DAgger comes later).
    C19 Route A: nav_done is the arm's flag (flag, on from the trigger), the base's latch is kept as base_nav_done;
    without the trigger the two are the same.
    """
    np.savez(
        path,
        time=d["time"], base_pos=d["base_pos"], base_bearing=d["base_bearing"], q_mes=d["q_mes"], qd_mes=d["qd_mes"],
        ee_pos=d["ee_pos"], target=d["target"], start_pose=d["start_pose"], nav_done=flag, base_nav_done=d["nav_done"],
        torque_arm=torque, arm_label_mask=mask,
        policy_wheel_cmds=d["wheel_cmds"], policy_torque_arm=d["torque_arm"],
        final_error=d["final_error"], stop_message=d["stop_message"],
    )


def collect(model, ids, out_dir, workers, window, posture_free=False, posture_kp=0.0):
    """Run model on the training episodes ids and save their labelled DAgger episodes in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ROBOENV_POLICY"] = str(SCRIPT_DIR / "checkpoints" / f"{model}.pt")   # passed on to the policy runs
    experts = {i: load_npz(DATA_DIR / f"ep_{i:04d}.npz") for i in ids}
    print(f"  running {model} on {len(ids)} training targets, {workers} at a time ...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {i: pool.submit(run_episode, POLICY_SCRIPT, experts[i]["target"], out_dir / f"raw_{i:04d}.npz", True,
                               str(SCRIPT_DIR), POLICY_TIMEOUT, True, stored_start(experts[i])) for i in ids}
        done = [i for i in ids if jobs[i].result() is not None]
    print(f"  {len(done)} runs finished; labelling their arm phase with the expert ...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(label_run, [out_dir / f"raw_{i:04d}.npz" for i in done],
                               [out_dir / f"ep_{i:04d}.npz" for i in done], [window] * len(done),
                               [posture_free] * len(done), [posture_kp] * len(done)))
    print(f"  labelled {sum(frames)} frames ({sum(f > 0 for f in frames)} of {len(done)} runs latched)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="trained split model the first round starts from")
    parser.add_argument("--rounds", type=int, default=3, help="number of DAgger rounds")
    parser.add_argument("--n", type=int, default=200, help="training targets per round (a new set each round)")
    parser.add_argument("--name", default="split_dagger", help="models <name>_r<k>_s<seed>, data dagger/<name>_r<k>")
    parser.add_argument("--seed", type=int, default=0, help="training seed of the round models")
    parser.add_argument("--window", type=float, default=5.0, help="seconds of the arm phase labelled after the latch")
    parser.add_argument("--posture-free", action="store_true",
                        help="labels without their posture-only part (relabel_posture.remove_posture, PLAN.md C18e)")
    parser.add_argument("--posture-kp", type=float, default=0.0,
                        help="with --posture-free: spring gain (N*m/rad) pulling joints 1, 3, 5 to 0 in the null space, "
                             "against the drift on low targets (PLAN.md C18e, 4 Nov); 0 = damping only")
    parser.add_argument("--low-extra", type=int, default=0,
                        help="extra runs per round on low training targets (z < LOW_Z), on top of --n, from their own "
                             "fixed order (PLAN.md C18e, 4 Nov: the arm drifts only on low targets)")
    parser.add_argument("--arm-start", type=float, default=0.0,
                        help="C19 Route A: the policy's arm starts reaching once the target is within this distance (m) "
                             "of the base (ROBOENV_ARM_START, passed on to the policy runs and the val runs); the "
                             "rounds are labelled from that trigger. 0 = off")
    parser.add_argument("--workers", type=int, default=16, help="policy runs / labelling processes in parallel")
    parser.add_argument("--val-set", action="store_true", help="compare.py --val-set after each round")
    parser.add_argument("--extra-data", nargs="+", default=[], metavar="FOLDER",
                        help="episode folders added to every round's training next to the rounds' own, e.g. the DART "
                             "data of the --start model (run_batch.py --noise); without them a DART start model's "
                             "rounds would train without its DART data (PLAN.md A2)")
    args = parser.parse_args()

    # Checked up front, so a typo stops the run before any simulation
    extra = [str(Path(f).resolve()) for f in args.extra_data]
    for folder in extra:
        if not list(Path(folder).glob("ep_*.npz")):
            parser.error(f"--extra-data {folder}: no ep_*.npz episodes there")
    # A --name already used by another run: its "done" steps would be skipped, i.e. its data and models reused
    info_file = DAGGER_DIR / f"{args.name}_r1" / "dagger_info.json"
    if args.posture_kp and not args.posture_free:
        parser.error("--posture-kp needs --posture-free (the spring replaces the expert's posture part)")
    info = dict(start=args.start, extra_data=extra, **({"posture_free": True} if args.posture_free else {}),
                **({"posture_kp": args.posture_kp} if args.posture_kp else {}),
                **({"low_extra": args.low_extra} if args.low_extra else {}),
                **({"arm_start": args.arm_start} if args.arm_start else {}))
    if args.arm_start:
        os.environ["ROBOENV_ARM_START"] = str(args.arm_start)   # inherited by every policy run and compare.py
    if info_file.exists():
        if json.loads(info_file.read_text()) != info:
            parser.error(f"--name {args.name} belongs to another run ({info_file}: {info_file.read_text().strip()}); "
                         "pick another --name")
    elif ((DAGGER_DIR / f"{args.name}_r1").exists()
          or (SCRIPT_DIR / "checkpoints" / f"{args.name}_r1_s{args.seed}.pt").exists()):
        parser.error(f"--name {args.name}: round 1's data or model exists from a run without a record of its start "
                     "model (e.g. the r1-r5 of 30 Oct); pick another --name")

    start = load_policy(SCRIPT_DIR / "checkpoints" / f"{args.start}.pt")
    if not isinstance(start, SplitPolicy):
        parser.error(f"{args.start} is not a split model (DAgger labels only the split arm network)")
    cfg = start.config
    # Same training settings as the start model, so the rounds differ from it only in the DAgger data. The checkpoint
    # holds the sizes; lr, batch and weight decay are only in its results file (empty there: train.py's default)
    row = read_results(args.start)
    # Each network's own size and dropout (a tuned arm can differ from the base); the shared ones are the base's
    (base_h, base_n, base_p), (arm_h, arm_n, arm_p) = net_settings(cfg, "base"), net_settings(cfg, "arm")
    train_options = ["--split", "--gravity-residual" if cfg["gravity_residual"] else "--no-gravity-residual",
                     "--hidden", str(base_h), "--layers", str(base_n), "--dropout", str(base_p),
                     "--arm-hidden", str(arm_h), "--arm-layers", str(arm_n), "--arm-dropout", str(arm_p),
                     "--base-activation", net_activation(cfg, "base"), "--arm-activation", net_activation(cfg, "arm"),
                     # Output type too: train.py refuses a copied base whose output differs from the run's (PLAN.md C15)
                     "--base-output-act", net_output_act(cfg, "base"), "--arm-output-act", net_output_act(cfg, "arm"),
                     "--seed", str(args.seed), "--base-from", args.start]
    for option, column in (("--lr", "lr"), ("--batch", "batch"), ("--weight-decay", "weight_decay")):
        if row.get(column):
            train_options += [option, row[column]]
    print(f"round settings: {' '.join(train_options)}")
    if extra:
        print(f"extra data in every round: {', '.join(extra)}")

    # Training targets in a fixed random order; round k takes the k-th block of n, wrapping around to the start once
    # all have been used (696 training targets: 3 rounds of 200 before any repeats). Repeating a target is fine in
    # DAgger: it is only a starting point, and a newer policy visits different states from it
    train_ids, _, _ = split_episodes(DATA_DIR)
    train_ids = clean_ids(DATA_DIR, train_ids)
    order = np.random.default_rng(0).permutation(train_ids)
    # --low-extra: the low training targets in their own fixed order, the k-th block of low_extra in round k
    low_ids = [i for i in train_ids if np.load(DATA_DIR / f"ep_{i:04d}.npz")["target"][2] < LOW_Z]
    low_order = np.random.default_rng(1).permutation(low_ids)
    env = dict(os.environ, MPLBACKEND="Agg")   # train.py / compare.py plots are saved, not shown

    model, folders = args.start, list(extra)
    for k in range(1, args.rounds + 1):
        new_model = f"{args.name}_r{k}_s{args.seed}"
        folder = DAGGER_DIR / f"{args.name}_r{k}"
        ids = [int(i) for i in np.take(order, range((k - 1) * args.n, k * args.n), mode="wrap")]
        if args.low_extra:
            # Episode files are named by target id, so a low target already in this round's block is run once
            extra_low = np.take(low_order, range((k - 1) * args.low_extra, k * args.low_extra), mode="wrap")
            ids += [int(i) for i in extra_low if int(i) not in ids]
        print(f"\n== DAgger round {k}: {model} -> {new_model} ({len(ids)} targets)", flush=True)
        if len(list(folder.glob("ep_*.npz"))) >= len(ids):
            print(f"  {folder.name} already collected, skipping the runs")
        else:
            if k == 1:
                folder.mkdir(parents=True, exist_ok=True)
                info_file.write_text(json.dumps(info))
            collect(model, ids, folder, args.workers, args.window, args.posture_free, args.posture_kp)
        folders.append(str(folder))
        if (SCRIPT_DIR / "checkpoints" / f"{new_model}.pt").exists():
            print(f"  {new_model} already trained, skipping the training")
        else:
            subprocess.run([sys.executable, "train.py", "--model", new_model, *train_options, "--dagger", *folders],
                           cwd=SCRIPT_DIR, env=env, check=True)
        if args.val_set:
            subprocess.run([sys.executable, "compare.py", "--val-set", "--model", new_model,
                            "--out-dir", str(SCRIPT_DIR / "compare_out" / new_model)], cwd=SCRIPT_DIR, env=env, check=True)
        model = new_model


if __name__ == "__main__":
    main()
