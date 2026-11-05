"""Run a trained policy with the mast camera on the 50 validation targets or the 150 test targets (PLAN.md step 29).

Same targets and start poses as `compare.py --val-set` (bc_data.quick_val_ids of the folder's split, targets from the
stored expert episodes), same 30 s cap, but the policy script is camera_policy_task.py (camera estimate instead of the
true target). Episode files go to <out-dir>/val_set/ep_XXXX.npz, so count_results.py reads them like compare.py's.
--set test: every test episode of the split (compare.py --test-set's targets), into <out-dir>/test_set/.

Usage (from robotnik_arm_task/):
    python ../concurrent_arm/camera_policy_eval.py --model split_conc_dartdagger_bb_plusCLD_r3_s0 \
        --data-dir data_concurrent_expert --out-dir compare_out/split_conc_dartdagger_bb_plusCLD_r3_s0_camera_val
Extra environment (e.g. ROBOENV_CAMERA_REFINE=0) is passed on to every run.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "robotnik_arm_task"))
from bc_data import quick_val_ids, split_episodes  # noqa: E402

POLICY_TIMEOUT = 600   # s of wall time per run, as a guard only


def run(i, expert_file, out_file, checkpoint):
    with np.load(expert_file) as d:
        target = d["target"]
        start = d["start_pose"] if "start_pose" in d.files else np.zeros(3)
    env = dict(os.environ)
    env.update(ROBOENV_HEADLESS="1", ROBOENV_POLICY=str(checkpoint), ROBOENV_OUT=str(out_file),
               ROBOENV_TARGET=",".join(f"{v:.6f}" for v in target), ROBOENV_START=",".join(f"{v:.6f}" for v in start))
    subprocess.run([sys.executable, "camera_policy_task.py"], cwd=HERE, env=env, timeout=POLICY_TIMEOUT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not out_file.exists():
        return i, "no result"
    with np.load(out_file) as d:
        return i, str(d["stop_message"])[:60]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint name in robotnik_arm_task/checkpoints")
    ap.add_argument("--data-dir", default="data_concurrent_expert")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--set", choices=("val", "test"), default="val")
    a = ap.parse_args()
    root = HERE.parent / "robotnik_arm_task"
    data_dir = root / a.data_dir
    checkpoint = root / "checkpoints" / f"{a.model}.pt"
    assert checkpoint.exists(), checkpoint
    _, val_ids, test_ids = split_episodes(data_dir)
    ids = quick_val_ids(val_ids) if a.set == "val" else list(test_ids)
    run_dir = Path(a.out_dir).resolve() / f"{a.set}_set"
    run_dir.mkdir(parents=True, exist_ok=True)
    for old in run_dir.glob("ep_*.npz"):
        old.unlink()
    print(f"{a.model}: {len(ids)} {a.set} targets with the camera, {a.workers} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs = [pool.submit(run, i, data_dir / f"ep_{i:04d}.npz", run_dir / f"ep_{i:04d}.npz", checkpoint) for i in ids]
        for n, job in enumerate(jobs, start=1):
            i, msg = job.result()
            print(f"  [{n}/{len(ids)}] episode {i}: {msg}", flush=True)
    print(f"episode files: {run_dir}")


if __name__ == "__main__":
    main()
