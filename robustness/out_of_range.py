"""Out-of-range targets: base at the default start (0, 0, 0), targets closer than 1 m or farther than 3 m
(horizontal) from it, heights inside the training range 0.2-1.1 m, so only the distance changes.

Training and the other tests draw targets 1-3 m away (run_batch.py). This runs the same episodes as run_batch.py
(run_batch.run_episode, unchanged: same subprocess, same ep_XXXX.npz files, same index.csv); only the targets differ.
Bands: near 0.6-1.0 m (under ~0.6 m the target can sit over the base itself), far 3.0-5.0 m; uniform over the
annulus area, as run_batch.py.

Usage, from robotnik_arm_task/ (robustness/robustness_runs.sh calls it):
    python ../robustness/out_of_range.py --band near --n 50 --out-dir oor/expert_near
    ROBOENV_POLICY=checkpoints/<model>.pt python ../robustness/out_of_range.py --band far --n 200 \
        --script robotnik_arm_policy_task.py --out-dir oor/<model>_far
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parent.parent / "robotnik_arm_task"
sys.path.insert(0, str(TASK_DIR))
from run_batch import HEIGHT_RANGE, INDEX_FIELDS, run_episode  # noqa: E402

BANDS = {"near": (0.6, 1.0), "far": (3.0, 5.0)}   # m, horizontal distance from the start


def sample(seed, idx, radius):
    """Target idx of a band: the same targets for every model and the expert (seeded by seed and idx)."""
    rng = np.random.default_rng([seed, idx])
    angle = rng.uniform(-np.pi, np.pi)
    r = np.sqrt(rng.uniform(radius[0] ** 2, radius[1] ** 2))
    return np.array([r * np.cos(angle), r * np.sin(angle), rng.uniform(*HEIGHT_RANGE)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--band", choices=BANDS, required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--script", default="robotnik_arm_task.py", help="episode script (the expert by default)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--timeout", type=float, default=180.0)
    a = p.parse_args()
    # The other fields run_episode reads, as run_batch.py's defaults: default start, no noise
    a.start_pose, a.noise, a.wheel_noise, a.noise_tau, a.cart_noise, a.noise_fade = "0,0,0", 0.0, 0.0, 0.1, 0.0, ""
    a.out_dir = str(Path(a.out_dir).resolve())   # the episodes run with robotnik_arm_task/ as their folder
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(i, sample(a.seed, i, BANDS[a.band])) for i in range(a.n)]
    rows, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for fut in as_completed([pool.submit(run_episode, i, t, a, str(TASK_DIR)) for i, t in jobs]):
            rows.append(fut.result())
    with open(Path(a.out_dir) / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["episode"]))
    counts = {}
    for r in rows:
        k = r["outcome"] or r["status"]
        counts[k] = counts.get(k, 0) + 1
    print(f"finished {len(rows)} episodes in {time.time() - t0:.0f} s: {counts}")


if __name__ == "__main__":
    main()
