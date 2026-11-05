"""The expert reference for the sensor-noise tables (PLAN.md step 12): the sequential expert (robotnik_arm_task.py) on
compare.py's 50 validation targets (the same ids, targets and start poses as `compare.py --val-set`), with whatever
config ROBOENV_CONFIG names. Fresh runs are needed because compare.py scores the expert from the stored clean episodes.

This is the script behind the 4 Nov expert rows (compare_out/expert_clean_mac_val, expert_noise{1,5,10}_val), moved
into the repo unchanged except for the command-line options. Usage, from robotnik_arm_task/:

    python ../robustness/expert_noise_runs.py compare_out/expert_clean_val
    for n in 1 5 10; do
      ROBOENV_CONFIG=robotnik_arm_noise$n.json python ../robustness/expert_noise_runs.py compare_out/expert_noise${n}_val
    done

Count them with `python ../robustness/noise_recount.py expert` (folders compare_out/expert_<level>_val, one run per
level, no seeds). The noise isn't seeded, so a rerun gives slightly different numbers; the clean run repeats exactly
on the same machine.
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parent.parent / "robotnik_arm_task"
sys.path.insert(0, str(TASK_DIR))
from bc_data import quick_val_ids, split_episodes  # noqa: E402
from compare import run_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="folder for the expert's episode files (ep_XXXX.npz)")
    ap.add_argument("--data-dir", default="data_main", help="stored expert episodes: ids, targets, start poses")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = TASK_DIR / args.data_dir
    ids = quick_val_ids(split_episodes(data)[1])

    def one(i):
        d = np.load(data / f"ep_{i:04d}.npz", allow_pickle=True)
        start = d["start_pose"] if "start_pose" in d.files else (0.0, 0.0, 0.0)
        return run_episode("robotnik_arm_task.py", d["target"], out / f"ep_{i:04d}.npz", True, TASK_DIR, quiet=True,
                           start=start)

    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(one, ids))
    reached = sum(str(r["stop_message"]).startswith("Target reached") for r in res if r is not None)
    print(f"{out}: {sum(r is not None for r in res)} / {len(ids)} episodes, reached {reached}")


if __name__ == "__main__":
    main()
