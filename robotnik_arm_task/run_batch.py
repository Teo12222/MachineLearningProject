"""Driver: run the expert (robotnik_arm_task.py) on many random targets and save one .npz per episode.

Usage (from the robotnik_arm_task folder, inside your roboenv environment):
    python run_batch.py --n 5 --out-dir data_test            # tiny test
    python run_batch.py --n 50 --seed 0 --out-dir data_pilot # pilot to map the reachable region
    python run_batch.py --n 500 --seed 1 --workers 4 --out-dir data_train
    python run_batch.py --n 50 --start-pose 1.0,-0.5,1.57 --out-dir data_start_test   # base starts elsewhere
    python run_batch.py --n 100 --seed 2 --noise 5 --out-dir dart/noise5   # DART (PLAN.md step 19)

Each episode runs in its own process, so one crash or hang cannot kill the batch.
Each target is determined by --seed and the episode number, so a batch is reproducible, and episodes whose .npz
already exists are skipped, so an interrupted batch can simply be restarted.
An index.csv in the output folder lists every episode with its target and outcome.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ---- Target sampling box: first guesses, to be checked with a pilot batch --------------------
# Targets are drawn in polar coordinates around the origin (the robot's start unless --start-pose).
RADIUS_RANGE = (1.0, 3.0)    # m, horizontal distance of the target from the origin
HEIGHT_RANGE = (0.2, 1.1)    # m, inside the standoff table of robotnik_arm_task.py (0.1 .. 1.25)
# The angle is uniform over the full circle, so targets behind the robot are included.
# ------------------------------------------------------------------------------------------

SUCCESS_PREFIX = "Target reached"
INDEX_FIELDS = ["episode", "x", "y", "z", "status", "outcome", "final_error", "sim_time", "wall_time", "file"]


def sample_target(seed, idx):
    """Target for episode idx. Seeded per episode, so it never depends on how many episodes are requested."""
    rng = np.random.default_rng([seed, idx])
    angle = rng.uniform(-np.pi, np.pi)
    # sqrt of a uniform r^2 gives targets uniform over the annulus area (plain uniform r crowds the inner edge)
    radius = np.sqrt(rng.uniform(RADIUS_RANGE[0] ** 2, RADIUS_RANGE[1] ** 2))
    z = rng.uniform(*HEIGHT_RANGE)
    return np.array([radius * np.cos(angle), radius * np.sin(angle), z])


def read_episode(row, out_file):
    """Fill final_error, sim_time and outcome in row from a saved episode file."""
    with np.load(out_file) as data:
        message = str(data["stop_message"])
        row["final_error"] = f"{float(data['final_error']):.5f}"
        row["sim_time"] = f"{float(data['time'][-1]):.2f}"
    if message.startswith(SUCCESS_PREFIX):
        row["outcome"] = "reached"
    elif message.startswith("Target not reachable"):
        row["outcome"] = "stalled"
    elif message.startswith(("Reached time cap", "Reached step cap")):   # step cap: older episode files
        row["outcome"] = "time_cap"
    else:
        row["outcome"] = "other"


def run_episode(idx, target, args, script_dir):
    """Run one episode in a subprocess and return a row for the index file."""
    out_file = Path(args.out_dir) / f"ep_{idx:04d}.npz"
    row = {"episode": idx, "x": f"{target[0]:.4f}", "y": f"{target[1]:.4f}", "z": f"{target[2]:.4f}",
           "status": "", "outcome": "", "final_error": "", "sim_time": "", "wall_time": "", "file": out_file.name}
    if out_file.exists():
        # Still read the file, so episodes finished by an interrupted run can be added to the index
        row["status"] = "skipped_exists"
        try:
            read_episode(row, out_file)
        except Exception:
            row["outcome"] = "unreadable"
        return row

    env = dict(os.environ)
    env["ROBOENV_HEADLESS"] = "1"
    env["ROBOENV_TARGET"] = ",".join(f"{v:.6f}" for v in target)
    env["ROBOENV_OUT"] = str(out_file)
    env["ROBOENV_START"] = args.start_pose
    env["ROBOENV_NOISE"] = f"{args.noise},{args.wheel_noise},{args.noise_tau},{idx}"
    if args.cart_noise:
        env["ROBOENV_CART_NOISE"] = str(args.cart_noise)
    if args.noise_fade:
        env["ROBOENV_NOISE_FADE"] = args.noise_fade
    if args.posture_gain:
        env["ROBOENV_POSTURE_GAIN"] = str(args.posture_gain)
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, args.script], cwd=script_dir, env=env,
            capture_output=True, text=True, timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
        row["wall_time"] = f"{time.time() - start:.1f}"
        row["file"] = ""
        return row
    row["wall_time"] = f"{time.time() - start:.1f}"

    if proc.returncode != 0 or not out_file.exists():
        row["status"] = f"crash(rc={proc.returncode})"
        # Keep the end of stderr so you can see why it failed
        (Path(args.out_dir) / f"ep_{idx:04d}.err.txt").write_text(proc.stderr[-3000:])
        row["file"] = ""
        return row

    row["status"] = "ok"
    read_episode(row, out_file)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=300, help="number of episodes")
    parser.add_argument("--seed", type=int, default=0, help="seed for the target sampler")
    parser.add_argument("--out-dir", default="data_main", help="folder for the episode files and index.csv")
    parser.add_argument("--workers", type=int, default=16, help="episodes to run in parallel")
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds before one episode is killed")
    parser.add_argument("--script", default="robotnik_arm_task.py", help="episode script to run")
    parser.add_argument("--start", type=int, default=0, help="first episode index (use to extend a dataset)")
    parser.add_argument("--start-pose", default="0,0,0",
                        help="base start pose x,y,yaw (m, m, rad) for every episode; targets stay around the origin")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="DART noise scale on the applied arm torques (PLAN.md step 19); labels stay clean")
    parser.add_argument("--wheel-noise", type=float, default=0.0, help="the same for the wheels, while driving")
    parser.add_argument("--noise-tau", type=float, default=0.1, help="correlation time of that noise (s)")
    parser.add_argument("--cart-noise", type=float, default=0.0,
                        help="std (N) of a random force on the end effector, applied as J^T F instead of --noise "
                             "(PLAN.md C18d)")
    parser.add_argument("--posture-gain", type=float, default=0.0,
                        help="expert pulls the arm's null space toward its hold pose at this gain (1/s, PLAN.md C18f); 0 = off")
    parser.add_argument("--noise-fade", default="",
                        help="r0,r1 (m): arm noise off within r0 of the target, full beyond r1, linear between (PLAN.md C18c)")
    args = parser.parse_args()

    script_dir = str(Path(__file__).resolve().parent)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Each target depends only on (seed, episode index): extending a dataset with --start never repeats one
    jobs = [(idx, sample_target(args.seed, idx)) for idx in range(args.start, args.start + args.n)]

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_episode, idx, tgt, args, script_dir) for idx, tgt in jobs]
        # Print in completion order, so one slow episode doesn't hold back the progress output
        for done, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            rows.append(row)
            print(f"[{done}/{len(jobs)}] ep {row['episode']:04d} target=({row['x']}, {row['y']}, {row['z']}) "
                  f"-> {row['status']} {row['outcome']} err={row['final_error']} wall={row['wall_time']}s", flush=True)

    # Merge with any existing index: one row per episode, newest result wins.
    # "skipped" rows only fill gaps (episodes an interrupted run finished but never indexed).
    index_path = Path(args.out_dir) / "index.csv"
    merged = {}
    if index_path.exists():
        with open(index_path, newline="") as f:
            for r in csv.DictReader(f):
                merged[int(r["episode"])] = r
    for r in rows:
        if r["status"] != "skipped_exists" or r["episode"] not in merged:
            merged[r["episode"]] = r
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(merged[k] for k in sorted(merged))

    counts = {}
    for r in rows:
        key = r["outcome"] or r["status"]
        counts[key] = counts.get(key, 0) + 1
    print(f"\nfinished {len(rows)} episodes in {time.time() - t0:.0f} s: {counts}")
    print(f"index written to {index_path}")


if __name__ == "__main__":
    main()