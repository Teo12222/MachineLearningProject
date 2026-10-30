"""DAgger for the split arm network (PLAN.md step 21): run a split model, label the arm's frames with the expert,
retrain on everything collected so far, repeat.

Usage (from the robotnik_arm_task folder):
    pixi run python dagger.py --start split_arrived_s0 --rounds 3          # rounds 1-3, 200 training targets each
    pixi run python dagger.py --start split_arrived_s0 --rounds 3 --val-set  # + compare.py --val-set per round

Round k (models split_dagger_r<k>_s<seed>, round 1 starting from --start):
1. The previous model runs headless on --n TRAINING targets (never validation or test ones), the next block of a
   fixed random order (wrapping around once all are used: a newer policy visits different states on the same
   target), from each episode's stored start pose, like compare.py (failing runs last the full 30 s).
2. Each run's arm phase is labelled with the expert's arm torques (expert_relabel.py): from the latch until --window
   seconds later, or until the expert's label stays at its torque limit (see label_run), whichever comes first.
   Saved as dagger/split_dagger_r<k>/ep_XXXX.npz in data_main's episode format (see save_episode).
3. train.py --split --dagger <all rounds' folders> trains the round's model from scratch (base unchanged: same data
   and seed give the same weights; settings and gravity residual taken from --start's checkpoint).
4. With --val-set: compare.py --val-set into compare_out/<model>/, the 50 validation episodes used for decisions
   such as which round to keep (never writes results; test episodes are only for the final numbers).

Steps that are already done are skipped (episode folder complete, checkpoint present), so an interrupted run can be
restarted with the same command. dagger/ is git-ignored.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + numpy each ship an OpenMP runtime on Windows

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from bc_data import clean_ids, split_episodes
from bc_split import SplitPolicy, load_policy
from compare import POLICY_SCRIPT, POLICY_TIMEOUT, load_npz, run_episode, stored_start

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_main"
DAGGER_DIR = SCRIPT_DIR / "dagger"
TORQUE_LIMIT = 30.0     # N*m, the expert's torque clip
SATURATED_RUN = 50      # frames (ms): the expert's label at its limit this long ends the labelled part ...
SATURATION_GRACE = 500  # frames: ... but not in the first 0.5 s after the latch, where the expert's own reach starts
                        # with a short kick at the limit (J6 starts below its joint limit, PLAN.md step 21)


def label_run(raw_path, out_path, window):
    """Label one policy run's arm phase with the expert and save it as a DAgger episode. Returns the labelled frames.

    Labelled from the latch until window seconds later, or until the expert's label stays at its torque limit for
    SATURATED_RUN frames: by then the arm is so far off that the expert itself is at its limits (its own data has
    such frames only in the short kick at the start of the reach), and those huge labels, many standard deviations
    outside the expert's data, would dominate the training loss. Checked on 3 runs: before that point the labels are
    the size of the expert's own (|z| median 1-3), after it ~6-8 with 60-100% of frames at the limit.
    Runs in a worker process (expert_relabel builds its own robot model when imported).
    """
    from expert_relabel import relabel_arm
    d = load_npz(raw_path)
    n = len(d["time"])
    mask = np.zeros(n, dtype=bool)
    torque = np.full((n, 7), np.nan, dtype=np.float32)   # nan where there is no label, so misuse can't go unnoticed
    if d["nav_done"].any():   # a run that never latched has no arm phase to label
        k = int(np.argmax(d["nav_done"]))
        stop = min(n, k + round(window * 1000))
        # The expert holds the arm's start pose until it arrives, so that is its joint reference when the reach starts
        labels = relabel_arm(d, k, stop, d["q_mes"][0, 4:11])
        saturated = (np.abs(labels) >= TORQUE_LIMIT - 0.01).any(axis=1)
        runs = np.convolve(saturated, np.ones(SATURATED_RUN), "valid") == SATURATED_RUN   # run starting at each frame
        runs[:SATURATION_GRACE] = False
        if runs.any():
            stop = k + int(np.argmax(runs))
        torque[k:stop] = labels[:stop - k]
        mask[k:stop] = True
    save_episode(out_path, d, torque, mask)
    raw_path.unlink()
    return int(mask.sum())


def save_episode(path, d, torque, mask):
    """A DAgger episode in data_main's format, without the expert-only fields (nav_phase, q_des_arm, base_goal).

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


def collect(model, ids, out_dir, workers, window):
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
                               [out_dir / f"ep_{i:04d}.npz" for i in done], [window] * len(done)))
    print(f"  labelled {sum(frames)} frames ({sum(f > 0 for f in frames)} of {len(done)} runs latched)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="trained split model the first round starts from")
    parser.add_argument("--rounds", type=int, default=3, help="number of DAgger rounds")
    parser.add_argument("--n", type=int, default=200, help="training targets per round (a new set each round)")
    parser.add_argument("--name", default="split_dagger", help="models <name>_r<k>_s<seed>, data dagger/<name>_r<k>")
    parser.add_argument("--seed", type=int, default=0, help="training seed of the round models")
    parser.add_argument("--window", type=float, default=5.0, help="seconds of the arm phase labelled after the latch")
    parser.add_argument("--workers", type=int, default=16, help="policy runs / labelling processes in parallel")
    parser.add_argument("--val-set", action="store_true", help="compare.py --val-set after each round")
    args = parser.parse_args()

    start = load_policy(SCRIPT_DIR / "checkpoints" / f"{args.start}.pt")
    if not isinstance(start, SplitPolicy):
        parser.error(f"{args.start} is not a split model (DAgger labels only the split arm network)")
    cfg = start.config
    # Same training settings as the start model, so the rounds differ from it only in the DAgger data
    train_options = ["--split", "--gravity-residual" if cfg["gravity_residual"] else "--no-gravity-residual",
                     "--hidden", str(cfg["hidden"]), "--layers", str(cfg["layers"]), "--dropout", str(cfg["dropout"]),
                     "--seed", str(args.seed)]

    # Training targets in a fixed random order; round k takes the k-th block of n, wrapping around to the start once
    # all have been used (696 training targets: 3 rounds of 200 before any repeats). Repeating a target is fine in
    # DAgger: it is only a starting point, and a newer policy visits different states from it
    train_ids, _, _ = split_episodes(DATA_DIR)
    train_ids = clean_ids(DATA_DIR, train_ids)
    order = np.random.default_rng(0).permutation(train_ids)
    env = dict(os.environ, MPLBACKEND="Agg")   # train.py / compare.py plots are saved, not shown

    model, folders = args.start, []
    for k in range(1, args.rounds + 1):
        new_model = f"{args.name}_r{k}_s{args.seed}"
        folder = DAGGER_DIR / f"{args.name}_r{k}"
        ids = [int(i) for i in np.take(order, range((k - 1) * args.n, k * args.n), mode="wrap")]
        print(f"\n== DAgger round {k}: {model} -> {new_model}", flush=True)
        if len(list(folder.glob("ep_*.npz"))) >= args.n:
            print(f"  {folder.name} already collected, skipping the runs")
        else:
            collect(model, ids, folder, args.workers, args.window)
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
