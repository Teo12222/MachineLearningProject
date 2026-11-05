"""Run the expert and the learned policy on the same target and plot them together.

Usage (from the robotnik_arm_task folder, inside your roboenv environment):
    python compare.py                             # default target (0, 1.1, 0.5)
    python compare.py --target 1.5,-0.8,0.6
    python compare.py --target 1.5,-0.8,0.6 --start 0.5,0.5,1.57   # both start from another base pose
    python compare.py --episode 292               # target + stored expert run of data_main/ep_0292.npz
    python compare.py --test-set                  # policy on every test episode's target, results table
    python compare.py --test-set --model mlp_worldframe   # the policy loads checkpoints/mlp_worldframe.pt
    python compare.py --test-set --quick          # fixed 50 of the test episodes, for quick checks (no results written)
    python compare.py --val-set --model X         # fixed 50 validation episodes: the check for decisions (no results)
    python compare.py --val-full --model X        # all 150 validation episodes, into val_set_full (no results)

--target: the expert (robotnik_arm_task.py) runs headless while the policy (robotnik_arm_policy_task.py) runs in the
PyBullet GUI so you can watch it. Press 'q' in the PyBullet window to end the policy run early.
--episode: the expert is not rerun; its run is read from the saved episode file. The policy runs in the GUI.
--test-set: the policy runs headless on the target of every test episode (same split as train.py), several at once,
and is compared with the stored expert runs: per-episode table, averages and success rates, plus a plot of the
final error per episode (saved as test_set_final_error.png). Only this mode writes results: the sim_* columns of
results/<model>.csv and results/expert.csv (the stored expert runs on the same test episodes).
--quick (with --test-set): only a fixed subset of 50 test episodes (bc_data.quick_test_ids), for quick checks while
developing; prints the same table and plot but never writes results, so quick checks never mix with reported numbers.
To see how a model does, not to choose between models: choices made on test episodes would leak into the final
numbers (PLAN.md, "Recording results").
--val-set: the same check on a fixed 50 validation episodes (bc_data.quick_val_ids, the ones tune.py checks too).
Use this one for every decision (which model, setting or DAgger round to keep). Never writes results.
If the policy has no model loaded or its run fails, only the expert is plotted.
Episode files and the plots (.png) are written to --out-dir.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

from bc_data import quick_test_ids, quick_val_ids, split_episodes
from results import update_results

EXPERT_SCRIPT = "robotnik_arm_task.py"
POLICY_SCRIPT = "robotnik_arm_policy_task.py"
EXPERT_TIMEOUT = 180.0  # s, the headless expert takes a few seconds per episode
POLICY_TIMEOUT = 300.0  # s, wall time; a failing policy runs until its 30 s simulated time cap
SUCCESS_TOLERANCE = 0.005  # m, the task scripts' REACH_TOLERANCE (shown as the success line in the test-set plot)


def load_npz(path):
    """All arrays of a saved episode file, as a dict."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def outcome(data):
    """Short outcome label from an episode's stop message (same categories as run_batch.py)."""
    message = str(data["stop_message"])
    if message.startswith("Target reached"):
        return "reached"
    if message.startswith("Target not reachable"):
        return "stalled"
    if message.startswith(("Reached time cap", "Reached step cap")):   # step cap: older episode files
        return "time_cap"
    return "other"


def stored_start(data):
    """Start pose (x, y, yaw) of a saved episode; files from before the start-pose option all started at the origin."""
    return data["start_pose"] if "start_pose" in data else np.zeros(3)


def run_episode(script, target, out_file, headless, script_dir, timeout=None, quiet=False, start=(0.0, 0.0, 0.0)):
    """Run one episode script on target from base pose start (x, y, yaw) and return its saved data, or None if it
    produced no result.

    quiet hides the script's own output (used when many episodes run at once).
    """
    # A file left over from an earlier run must not be mistaken for this one
    out_file.unlink(missing_ok=True)
    env = dict(os.environ)
    env["ROBOENV_TARGET"] = ",".join(f"{v:.6f}" for v in target)
    env["ROBOENV_OUT"] = str(out_file)
    env["ROBOENV_START"] = ",".join(f"{v:.6f}" for v in start)
    if headless:
        env["ROBOENV_HEADLESS"] = "1"
    else:
        env.pop("ROBOENV_HEADLESS", None)

    if not quiet:
        print(f"\n---- {script} ({'headless' if headless else 'GUI'}) ----", flush=True)
    try:
        proc = subprocess.run([sys.executable, script], cwd=script_dir, env=env, timeout=timeout,
                              capture_output=quiet)
    except subprocess.TimeoutExpired:
        print(f"{script} ({out_file.name}): timed out after {timeout:.0f} s")
        return None
    if proc.returncode != 0 or not out_file.exists():
        print(f"{script} ({out_file.name}): no result (exit code {proc.returncode})")
        return None
    return load_npz(out_file)


def plot_runs(runs, target, out_dir):
    """Plot every run in runs (list of (label, data, color)) on shared axes and save the figures to out_dir."""
    figures = {}

    # 1. Cartesian end-effector coordinates
    fig = plt.figure(figsize=(10, 8))
    coords = ["X", "Y", "Z"]
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        for label, data, color in runs:
            plt.plot(data["time"], data["ee_pos"][:, idx], label=f"{label} EE {coords[idx]}", color=color)
        plt.axhline(y=target[idx], color="r", linestyle="--", label=f"Target EE {coords[idx]}")
        plt.ylabel(f"{coords[idx]} Position (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.xlabel("Time (s)")
    plt.suptitle("End-Effector Cartesian Position Reaching")
    plt.tight_layout()
    figures["ee_position"] = fig

    # 2. Cartesian distance error
    fig = plt.figure(figsize=(8, 4))
    for label, data, color in runs:
        plt.plot(data["time"], np.linalg.norm(target - data["ee_pos"], axis=1), label=label, color=color)
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.title("Cartesian End-Effector Reaching Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figures["ee_error"] = fig

    # 3. Arm joint positions
    fig = plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        for label, data, color in runs:
            plt.plot(data["time"], data["q_mes"][:, 4 + i], label=f"{label} J{i+1}", color=color)
        plt.ylabel(f"J{i+1} (rad)")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
    plt.xlabel("Time (s)")
    plt.suptitle("Arm Joint Position Profiles")
    plt.tight_layout()
    figures["arm_joints"] = fig

    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=120)
    plt.show()


def record_sim_results(name, runs, n):
    """sim_* columns of results/<name>.csv from the runs on the n test episodes (None = no result, counts as failed)."""
    reached = [r for r in runs if r is not None and outcome(r) == "reached"]
    errors = [float(r["final_error"]) for r in runs if r is not None]
    update_results(
        name, sim_success_rate=len(reached) / n,
        sim_final_err_mean=np.mean(errors) if errors else None,
        sim_final_err_median=np.median(errors) if errors else None,
        # Episode length: time to reach + the fixed 1 s settle, the same offset for every model and the expert
        sim_time_to_reach=np.mean([float(r["time"][-1]) for r in reached]) if reached else None,
    )


def run_test_set(data_dir, out_dir, script_dir, workers, model, quick=False, val=False, val_full=False):
    """Run the policy headless on every test episode's target and compare it with the stored expert runs.

    quick: only the fixed quick subset, and no results are written. val: the fixed 50 validation episodes instead of
    test ones (for decisions), no results written either. val_full (with val, added 4 Nov): all 150 validation
    episodes (the fixed 50 among them), for a less noisy comparison of close models; no results written.
    """
    _, val_ids, test_ids = split_episodes(data_dir)
    if val:
        test_ids = list(val_ids) if val_full else quick_val_ids(val_ids)
    elif quick:
        test_ids = quick_test_ids(test_ids)
    # Separate folders, so a quick or validation run never overwrites a full one
    name = ("val_set_full" if val_full else "val_set") if val else "test_set_quick" if quick else "test_set"
    subset = ((" (all validation)" if val_full else " (validation subset)") if val
              else " (quick subset)" if quick else "")
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Episode files of an earlier run (possibly another model or split) would otherwise sit next to this run's
    for old in run_dir.glob("ep_*.npz"):
        old.unlink()
    experts = {i: load_npz(data_dir / f"ep_{i:04d}.npz") for i in test_ids}

    print(f"Running the policy headless on {len(test_ids)} test targets, {workers} at a time ...", flush=True)
    policies = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {i: pool.submit(run_episode, POLICY_SCRIPT, experts[i]["target"], run_dir / f"ep_{i:04d}.npz",
                               True, script_dir, POLICY_TIMEOUT, True, stored_start(experts[i]))
                for i in test_ids}
        for done, i in enumerate(test_ids, start=1):
            policies[i] = jobs[i].result()
            label = outcome(policies[i]) if policies[i] is not None else "no result"
            print(f"  [{done}/{len(test_ids)}] episode {i}: {label}", flush=True)

    # Per-episode table, same layout as the test table of train.py. A missing policy result counts as a failure.
    pol_err, exp_err, pol_time, exp_time = [], [], [], []
    print(f"\n{'Validation' if val else 'Test'} set{subset}: {len(test_ids)} episodes, policy vs expert in simulation")
    print(f"{'episode':>8} {'policy err (m)':>15} {'expert err (m)':>15} {'policy time (s)':>16} "
          f"{'expert time (s)':>16}  policy outcome")
    for i in test_ids:
        e, p = experts[i], policies[i]
        exp_err.append(float(e["final_error"]))
        exp_time.append(float(e["time"][-1]))
        pol_err.append(float(p["final_error"]) if p is not None else np.nan)
        pol_time.append(float(p["time"][-1]) if p is not None else np.nan)
        print(f"{i:8d} {pol_err[-1]:15.4f} {exp_err[-1]:15.4f} {pol_time[-1]:16.2f} {exp_time[-1]:16.2f}  "
              f"{outcome(p) if p is not None else 'no result'}")
    print(f"{'average':>8} {np.nanmean(pol_err):15.4f} {np.mean(exp_err):15.4f} {np.nanmean(pol_time):16.2f} "
          f"{np.mean(exp_time):16.2f}   (mean of the per-episode values)")

    n = len(test_ids)
    pol_reached = sum(p is not None and outcome(p) == "reached" for p in policies.values())
    exp_reached = sum(outcome(e) == "reached" for e in experts.values())
    print(f"\nSuccess rate: policy {pol_reached}/{n} ({100 * pol_reached / n:.1f}%), "
          f"expert {exp_reached}/{n} ({100 * exp_reached / n:.1f}%)")
    print(f"Policy episode files: {run_dir}")
    if quick or val:
        print(f"{subset.strip(' ()').capitalize()}: no results written (run the full --test-set for that)")
    else:
        record_sim_results(model, list(policies.values()), n)
        record_sim_results("expert", list(experts.values()), n)
        print(f"Results written to results/{model}.csv and results/expert.csv")

    # Final error per test episode, policy next to expert. Log scale: expert errors are a few mm (runs go on 1 s
    # after reaching), policy errors can be m.
    pos = np.arange(n)
    width = 0.4
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(pos - width / 2, exp_err, width, label="Expert", color="b")
    ax.bar(pos + width / 2, pol_err, width, label="Policy", color="tab:orange")   # nan (no result) is left empty
    ax.axhline(SUCCESS_TOLERANCE, color="gray", linestyle="--", linewidth=1,
               label=f"Success threshold ({SUCCESS_TOLERANCE * 1000:.0f} mm)")
    ax.set_yscale("log")
    # Label the powers of ten as plain distances (1 mm, 1 cm, 10 cm, 1 m, ...) instead of 10^-3, 10^-2, ...
    # Only ticks inside the plotted range are set, so the axis is not stretched to show them all.
    tick_labels = {0.0001: "0.1 mm", 0.001: "1 mm", 0.01: "1 cm", 0.1: "10 cm", 1: "1 m", 10: "10 m", 100: "100 m"}
    low, high = ax.get_ylim()
    shown = [t for t in tick_labels if low <= t <= high]
    ax.set_yticks(shown, [tick_labels[t] for t in shown])
    ax.minorticks_off()   # hide the unlabelled ticks between the powers of ten
    ax.set_xticks(pos, [str(i) for i in test_ids], rotation=90, fontsize=7)
    ax.set_xlabel(f"{'Validation' if val else 'Test'} episode")
    ax.set_ylabel("Final EE error (log scale)")
    ax.set_title(f"Final end-effector error per {'validation' if val else 'test'} episode{subset} ({model}) - "
                 f"policy reached {pol_reached}/{n}, expert {exp_reached}/{n}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_final_error.png", dpi=120)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--target", default="0.0,1.1,0.5", help="target EE position x,y,z (m)")
    mode.add_argument("--episode", type=int, help="use the target and stored expert run of this saved episode")
    mode.add_argument("--test-set", action="store_true", help="run the policy on all test episodes (table + error plot)")
    mode.add_argument("--val-set", action="store_true",
                      help="run the policy on the fixed 50 validation episodes (for decisions), no results written")
    mode.add_argument("--val-full", action="store_true",
                      help="run the policy on all 150 validation episodes (the fixed 50 among them; into val_set_full), "
                           "no results written (added 4 Nov: less noisy than 50 for close models)")
    parser.add_argument("--quick", action="store_true",
                        help="with --test-set: only the fixed quick subset of 50 test episodes, no results written")
    parser.add_argument("--start", help="start pose of the base x,y,yaw (m, m, rad), --target only; "
                                         "--episode / --test-set use each episode's stored start pose")
    parser.add_argument("--data-dir", default="data_main", help="episode folder, relative to this script")
    parser.add_argument("--workers", type=int, default=16, help="policy runs in parallel for --test-set")
    parser.add_argument("--out-dir", default="compare_out", help="folder for the episode files and plots")
    parser.add_argument("--model", default="split_conc_drive_r1_s0",
                        help="model name: the policy loads checkpoints/<model>.pt, --test-set writes results/<model>.csv")
    args = parser.parse_args()
    if args.start and (args.test_set or args.val_set or args.val_full or args.episode is not None):
        parser.error("--start only works with --target")
    if args.quick and not args.test_set:
        parser.error("--quick only works with --test-set")

    script_dir = str(Path(__file__).resolve().parent)
    data_dir = Path(script_dir) / args.data_dir
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # The policy script loads the checkpoint named by ROBOENV_POLICY; run_episode passes the environment on
    checkpoint = Path(script_dir) / "checkpoints" / f"{args.model}.pt"
    if not checkpoint.exists():
        parser.error(f"no checkpoint {checkpoint}")
    os.environ["ROBOENV_POLICY"] = str(checkpoint)

    if args.test_set or args.val_set or args.val_full:
        run_test_set(data_dir, out_dir, script_dir, args.workers, args.model, args.quick,
                     args.val_set or args.val_full, args.val_full)
        return

    if args.episode is not None:
        # Stored expert run: no need to simulate the expert again
        expert = load_npz(data_dir / f"ep_{args.episode:04d}.npz")
        target = expert["target"]
        if args.episode not in split_episodes(data_dir)[2]:
            print(f"Note: episode {args.episode} is not a test episode, so the model may have trained on it")
        policy = run_episode(POLICY_SCRIPT, target, out_dir / "policy.npz", False, script_dir,
                             start=stored_start(expert))
    else:
        target = np.array([float(v) for v in args.target.split(",")])
        start = np.array([float(v) for v in args.start.split(",")]) if args.start else np.zeros(3)
        with ThreadPoolExecutor(max_workers=1) as pool:
            # Expert runs headless in the background while the policy runs in the GUI
            expert_job = pool.submit(run_episode, EXPERT_SCRIPT, target, out_dir / "expert.npz", True, script_dir,
                                     EXPERT_TIMEOUT, start=start)
            policy = run_episode(POLICY_SCRIPT, target, out_dir / "policy.npz", False, script_dir, start=start)
            expert = expert_job.result()

    runs = []
    if expert is not None:
        runs.append(("Expert", expert, "b"))
    if policy is not None:
        runs.append(("Policy", policy, "tab:orange"))
    else:
        print("\nPolicy produced no result: plotting the expert only")
    if not runs:
        print("Neither run produced a result, nothing to plot")
        return

    print("\n==================================================================")
    print(f"Target: {target}")
    for label, data, _ in runs:
        print(f"{label:<7}| final error {float(data['final_error']):.4f} m | sim time {data['time'][-1]:.2f} s "
              f"| {data['stop_message']}")
    print("==================================================================")

    plot_runs(runs, target, out_dir)


if __name__ == "__main__":
    main()
