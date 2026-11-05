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
   Saved as dagger/split_dagger_r<k>/ep_XXXX.npz in data_course_expert's episode format (see save_episode).
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
DATA_DIR = SCRIPT_DIR / "data_course_expert"
DAGGER_DIR = SCRIPT_DIR / "dagger"
TORQUE_LIMIT = 30.0     # N*m, the expert's torque clip
SATURATED_RUN = 50      # frames (ms): the expert's label at its limit this long ends the labelled part ...
SATURATION_GRACE = 500  # frames: ... but not in the first 0.5 s after the latch, where the expert's own reach starts
                        # with a short kick at the limit (J6 starts below its joint limit, PLAN.md step 21)
LOW_Z = 0.5             # m: targets below this are "low" (--low-extra; all collapses were below it, PLAN.md C18e)


def label_run(raw_path, out_path, window, posture_free=False, posture_kp=0.0, q_des0=None, drive=None,
              postpark=None):
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
    q_des0 (PLAN.md step 28, concurrent expert): the expert's joint reference when its reach starts, instead of the
    arm's start pose. The concurrent expert moves the arm to its final pose q_final while driving, and with early IK
    off its reference at the latch is exactly q_final (checked on 6 episodes: relabelling from it gives back the
    saved torques to 0.012 N*m, from the start pose 7.9 N*m off, from the measured pose 1.1 N*m).
    """
    from expert_relabel import relabel_arm
    d = load_npz(raw_path)
    if postpark is not None:
        return label_run_postpark(d, raw_path, out_path, window, posture_free, posture_kp, postpark)
    n = len(d["time"])
    mask = np.zeros(n, dtype=bool)
    torque = np.full((n, 7), np.nan, dtype=np.float32)   # nan where there is no label, so misuse can't go unnoticed
    if d["nav_done"].any():   # a run that never latched has no arm phase to label
        k = int(np.argmax(d["nav_done"]))
        stop = min(n, k + round(window * 1000))
        # The expert holds the arm's start pose until it arrives, so that is its joint reference when the reach starts
        # (the concurrent expert's is its q_final: q_des0)
        labels = relabel_arm(d, k, stop, d["q_mes"][0, 4:11] if q_des0 is None else q_des0)
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
        if drive is not None and k > 0:
            # Route B drive-phase DAgger (--label-drive, PLAN.md C19, 4 Nov): the frames before the latch too (the
            # base turning and driving, flag 0 in the saved nav_done), labelled by the concurrent expert's arm path
            # (expert_relabel.relabel_arm_drive). Kept even at the torque limit (as C19 Route A, option 1): they are
            # the expert's bounded pull back toward the path; the stop rule above still judges the reach after it
            from expert_relabel import relabel_arm_drive
            torque[:k] = relabel_arm_drive(d, k, *drive)
            mask[:k] = True
    save_episode(out_path, d, torque, mask)
    raw_path.unlink()
    return int(mask.sum())


def label_run_postpark(d, raw_path, out_path, window, posture_free, posture_kp, q_final):
    """label_run for the sequential pre-shaping expert (--label-postpark, PLAN.md step 28, 5 Nov).

    Every frame up to window seconds after the latch gets expert_relabel.relabel_arm_postpark's label: the start pose
    held before the latch (flag 0 there, from the saved nav_done), the joint path to q_final after it, then the
    Cartesian reach. The path labels are kept even at the torque limit (bounded by the lead limit, as the drive-phase
    labels); posture_free and the saturation stop act on the Cartesian part only, as they do on the plain labels.
    """
    from expert_relabel import relabel_arm_postpark
    n = len(d["time"])
    mask = np.zeros(n, dtype=bool)
    torque = np.full((n, 7), np.nan, dtype=np.float32)
    if d["nav_done"].any():
        k = int(np.argmax(d["nav_done"]))
        stop = min(n, k + round(window * 1000))
        labels, switch = relabel_arm_postpark(d, k, stop, q_final)
        if posture_free and switch < stop:
            from relabel_posture import remove_posture
            t = d["torque_arm"][:stop].astype(float).copy()
            t[:stop] = labels
            labels[switch:] = remove_posture({**d, "torque_arm": t}, switch, posture_kp)[switch:stop]
        if stop - switch >= SATURATED_RUN:
            saturated = (np.abs(labels[switch:]) >= TORQUE_LIMIT - 0.01).any(axis=1)
            runs = np.convolve(saturated, np.ones(SATURATED_RUN), "valid") == SATURATED_RUN
            runs[:SATURATION_GRACE] = False   # grace counted from the switch, where the expert's reach starts
            if runs.any():
                stop = switch + int(np.argmax(runs))
        torque[:stop] = labels[:stop]
        mask[:stop] = True
    save_episode(out_path, d, torque, mask)
    raw_path.unlink()
    return int(mask.sum())


def save_episode(path, d, torque, mask):
    """A DAgger episode in data_course_expert's format, without the expert-only fields (nav_phase, q_des_arm, base_goal).

    torque_arm holds the expert's labels where arm_label_mask is True (nan elsewhere); nav_done is the policy's
    latched flag. What the policy itself commanded is kept as policy_wheel_cmds / policy_torque_arm. No wheel labels
    yet (base DAgger comes later).
    """
    np.savez(
        path,
        time=d["time"], base_pos=d["base_pos"], base_bearing=d["base_bearing"], q_mes=d["q_mes"], qd_mes=d["qd_mes"],
        ee_pos=d["ee_pos"], target=d["target"], start_pose=d["start_pose"], nav_done=d["nav_done"],
        torque_arm=torque, arm_label_mask=mask,
        policy_wheel_cmds=d["wheel_cmds"], policy_torque_arm=d["torque_arm"],
        final_error=d["final_error"], stop_message=d["stop_message"],
    )


def collect(model, ids, out_dir, workers, window, posture_free=False, posture_kp=0.0, from_final=False,
            label_drive=False, label_postpark=False):
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
                               [posture_free] * len(done), [posture_kp] * len(done),
                               [experts[i]["q_final"] if from_final else None for i in done],
                               # --label-drive: the expert's final pose and parking spot for this target
                               [(experts[i]["q_final"].astype(float), experts[i]["base_goal"][:2].astype(float))
                                if label_drive else None for i in done],
                               # --label-postpark: the sequential pre-shaping expert's final pose for this target
                               [experts[i]["q_final"].astype(float) if label_postpark else None for i in done]))
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
    parser.add_argument("--workers", type=int, default=16, help="policy runs / labelling processes in parallel")
    parser.add_argument("--val-set", action="store_true", help="compare.py --val-set after each round")
    parser.add_argument("--extra-data", nargs="+", default=[], metavar="FOLDER",
                        help="episode folders added to every round's training next to the rounds' own, e.g. the DART "
                             "data of the --start model (run_batch.py --noise); without them a DART start model's "
                             "rounds would train without its DART data (PLAN.md A2)")
    parser.add_argument("--data-dir", default="data_course_expert",
                        help="expert episode folder (training targets, start poses, split), relative to this script; "
                             "passed on to train.py and compare.py (PLAN.md step 28: data_concurrent_expert)")
    parser.add_argument("--relabel-from-final", action="store_true",
                        help="the expert's reach starts from q_final of the target's expert episode, not the arm's "
                             "start pose (label_run's q_des0; for the concurrent expert, PLAN.md step 28)")
    parser.add_argument("--label-drive", action="store_true",
                        help="Route B: also label the frames before the latch (turning and driving) with the concurrent "
                             "expert's arm path to q_final (expert_relabel.relabel_arm_drive; needs --relabel-from-final "
                             "and a --data-dir of concurrent-expert episodes, which store q_final and base_goal)")
    parser.add_argument("--label-postpark", action="store_true",
                        help="sequential pre-shaping expert (concurrent_expert.py with ROBOENV_ARM_SEQUENTIAL=1, PLAN.md "
                             "step 28, 5 Nov): label with its own arm, start pose held before the latch (flag 0), joint "
                             "path to q_final after it, then the Cartesian reach (expert_relabel.relabel_arm_postpark); "
                             "needs a --data-dir of its episodes, which store q_final")
    args = parser.parse_args()
    if args.label_postpark and (args.label_drive or args.relabel_from_final):
        parser.error("--label-postpark replaces --relabel-from-final / --label-drive (its own labels from the start pose)")
    if args.label_drive and not args.relabel_from_final:
        parser.error("--label-drive needs --relabel-from-final (the expert's reach after the latch starts from q_final)")
    global DATA_DIR
    DATA_DIR = SCRIPT_DIR / args.data_dir

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
                **({"data_dir": args.data_dir} if args.data_dir != "data_course_expert" else {}),
                **({"relabel_from_final": True} if args.relabel_from_final else {}),
                **({"label_drive": True} if args.label_drive else {}),
                **({"label_postpark": True} if args.label_postpark else {}))
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
                     "--seed", str(args.seed), "--base-from", args.start, "--data-dir", args.data_dir]
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
            collect(model, ids, folder, args.workers, args.window, args.posture_free, args.posture_kp,
                    args.relabel_from_final, args.label_drive, args.label_postpark)
        folders.append(str(folder))
        if (SCRIPT_DIR / "checkpoints" / f"{new_model}.pt").exists():
            print(f"  {new_model} already trained, skipping the training")
        else:
            subprocess.run([sys.executable, "train.py", "--model", new_model, *train_options, "--dagger", *folders],
                           cwd=SCRIPT_DIR, env=env, check=True)
        if args.val_set:
            subprocess.run([sys.executable, "compare.py", "--val-set", "--model", new_model, "--data-dir", args.data_dir,
                            "--out-dir", str(SCRIPT_DIR / "compare_out" / new_model)], cwd=SCRIPT_DIR, env=env, check=True)
        model = new_model


if __name__ == "__main__":
    main()
