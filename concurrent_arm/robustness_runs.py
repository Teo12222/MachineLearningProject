"""The three robustness tests of PLAN.md step 11 on one model, with Matteo's targets (5 Nov, sequential v concurrent).

- start pose: run_batch.py --seed 21 targets, poses default (0, 0, 0), offset (2.0, 1.5, 0), rotated (0, 0, 3.0),
  both (-1.5, 2.0, -2.0); for offset / both episodes 0-154 / 0-130, kept if 1-3 m (horizontal) from the start pose;
- out of range: default start, seed 31, targets uniform over the annulus area like run_batch.sample_target, near
  0.6-1.0 m (50) and far 3.0-5.0 m (200), heights 0.2-1.1 m;
- payload: the 50 default-start targets with ROBOENV_PAYLOAD = 0.5 / 1 / 2 kg (0 kg = the start-pose default run).
The target counts reproduce Matteo's tables (50 per pose; near 24 / 26, far 35 / 53 / 56 / 56 per band).
Runs go through run_batch.run_episode unchanged, into starts/<model>_<pose>, oor/<model>_<near|far>,
payload/<model>_<kg>kg (git-ignored). Usage, from robotnik_arm_task/:
    python ../concurrent_arm/robustness_runs.py run MODEL [--workers 10]
    python ../concurrent_arm/robustness_runs.py eval MODEL [MODEL ...]
"""
import argparse
import os
import sys
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "robotnik_arm_task"
sys.path.insert(0, str(ROOT))
import run_batch as rb  # noqa: E402
from analyse_distances import run_stats  # noqa: E402

POSES = {"default": "0,0,0", "offset": "2.0,1.5,0", "rotated": "0,0,3.0", "both": "-1.5,2.0,-2.0"}
POSE_N = {"default": 50, "offset": 155, "rotated": 50, "both": 131}
BANDS = {"near": ((0.6, 1.0), 50), "far": ((3.0, 5.0), 200)}
EDGES = {"near": [0.6, 0.8, 1.0], "far": [3.0, 3.5, 4.0, 4.5, 5.0]}
PAYLOADS = ("0.5", "1", "2")
COLLAPSE = 0.20   # m, as route_b_eval.py


def start_pose_jobs(pose):
    xy = np.array([float(v) for v in POSES[pose].split(",")[:2]])
    out = []
    for i in range(POSE_N[pose]):
        t = rb.sample_target(21, i)
        if 1.0 <= np.linalg.norm(t[:2] - xy) <= 3.0:
            out.append((i, t))
    return out


def oor_target(idx, radii):
    """run_batch.sample_target with another radius range (seed 31)."""
    rng = np.random.default_rng([31, idx])
    angle = rng.uniform(-np.pi, np.pi)
    radius = np.sqrt(rng.uniform(radii[0] ** 2, radii[1] ** 2))
    z = rng.uniform(*rb.HEIGHT_RANGE)
    return np.array([radius * np.cos(angle), radius * np.sin(angle), z])


def groups(model):
    """(folder, start pose, payload, jobs) for every test."""
    g = [(f"starts/{model}_{p}", POSES[p], None, start_pose_jobs(p)) for p in POSES]
    g += [(f"oor/{model}_{b}", "0,0,0", None, [(i, oor_target(i, r)) for i in range(n)]) for b, (r, n) in BANDS.items()]
    g += [(f"payload/{model}_{kg}kg", "0,0,0", kg, start_pose_jobs("default")) for kg in PAYLOADS]
    return g


def run(model, workers):
    os.environ["ROBOENV_POLICY"] = str(ROOT / "checkpoints" / f"{model}.pt")
    for folder, pose, kg, jobs in groups(model):
        out = ROOT / folder
        out.mkdir(parents=True, exist_ok=True)
        if kg is None:
            os.environ.pop("ROBOENV_PAYLOAD", None)
        else:
            os.environ["ROBOENV_PAYLOAD"] = kg
        args = Namespace(out_dir=str(out), start_pose=pose, noise=0.0, wheel_noise=0.0, noise_tau=0.1, cart_noise=0.0,
                         noise_fade="", script="robotnik_arm_policy_task.py", timeout=180.0)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda j: rb.run_episode(j[0], j[1], args, str(ROOT)), jobs))
        bad = [r for r in rows if r["status"] not in ("ok", "skipped_exists")]
        print(f"{folder}: {len(rows)} runs, {len(bad)} failed to run", flush=True)
    os.environ.pop("ROBOENV_PAYLOAD", None)


def counts(files):
    r = []
    for f in files:
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        s = run_stats(d)
        err = np.linalg.norm(d["ee_pos"] - d["target"], axis=1)
        s["reached"] = bool((d["nav_done"].astype(bool) & (err < 0.005)).any())
        s["collapsed"] = bool(err[np.argmin(err):].max() > COLLAPSE)
        s["final_dz"] = float(d["ee_pos"][-1, 2] - d["target"][2])
        r.append(s)
    return (f"{sum(x['reached'] for x in r)} / {sum(x['ever_1cm'] for x in r)} / {sum(x['held_1cm'] for x in r)} / "
            f"{sum(x['held_5cm'] for x in r)}" + (f" ({sum(x['collapsed'] for x in r)} coll.)" if any(x['collapsed'] for x in r) else "")
            + f"  [n {len(r)}, median final dz {100 * np.median([x['final_dz'] for x in r]):+.1f} cm]")


def evaluate(models):
    print("reached / ever 1 cm / held 1 cm / held 5 cm")
    for model in models:
        print(f"\n{model}")
        for folder, _, _, jobs in groups(model):
            files = [ROOT / folder / f"ep_{i:04d}.npz" for i, _ in jobs]
            files = [f for f in files if f.exists()]
            if folder.startswith("oor/"):
                band = folder.rsplit("_", 1)[1]
                dist = {i: np.linalg.norm(t[:2]) for i, t in jobs}
                e = EDGES[band]
                for lo, hi in zip(e[:-1], e[1:]):
                    sel = [f for f in files if lo <= dist[int(f.stem[3:])] < hi]
                    print(f"  {folder} {lo:.1f}-{hi:.1f} m: {counts(sel)}")
            else:
                print(f"  {folder}: {counts(files)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("run", "eval"))
    p.add_argument("models", nargs="+")
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    if a.mode == "run":
        for m in a.models:
            run(m, a.workers)
    else:
        evaluate(a.models)
