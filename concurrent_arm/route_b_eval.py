"""Val-50 numbers for Route B and its references, from saved compare.py --val-set runs (no simulation).

Per model: reached (within 5 mm once the latch is set, compare.py's rule), ever / held within 1 cm and held within
5 cm (analyse_distances.run_stats: below the distance in one frame / for 0.5 s without a break), collapsed (the error
goes back above 20 cm after the closest approach, PLAN.md C18e), and the mean time to reach over the reached runs.

Usage (from robotnik_arm_task/):
    python ../concurrent_arm/route_b_eval.py compare_out/<model>_val ... [--data-dir data_route_b]
The 50 targets are quick_val_ids of the folder's split; data_route_b and data_main give the same 50 (ids 0-999).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "robotnik_arm_task"))
from analyse_distances import run_stats  # noqa: E402
from bc_data import quick_val_ids, split_episodes  # noqa: E402

COLLAPSE = 0.20   # m


def stats(path):
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    s = run_stats(d)
    err = np.linalg.norm(d["ee_pos"] - d["target"], axis=1)
    ok = d["nav_done"].astype(bool) & (err < 0.005)
    s["reached"] = bool(ok.any())
    s["t_reach"] = float(d["time"][np.argmax(ok)]) if ok.any() else np.nan
    s["collapsed"] = bool(err[np.argmin(err):].max() > COLLAPSE)
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="compare.py --out-dir folders (their val_set/ is read)")
    p.add_argument("--data-dir", default="data_route_b")
    p.add_argument("--set", choices=("val", "test"), default="val",
                   help="test: every test episode, from the folders' test_set/")
    a = p.parse_args()
    root = Path(__file__).resolve().parent.parent / "robotnik_arm_task"
    _, val_ids, test_ids = split_episodes(root / a.data_dir)
    ids = quick_val_ids(val_ids) if a.set == "val" else list(test_ids)
    cols = ("reached", "ever_1cm", "held_1cm", "held_5cm", "collapsed")
    print(f"{'runs':48s} {'n':>3s} " + " ".join(f"{c:>9s}" for c in cols) + "  t_reach (reached)")
    for run in a.runs:
        files = [Path(run) / f"{a.set}_set" / f"ep_{i:04d}.npz" for i in ids]
        rows = [stats(f) for f in files if f.exists()]
        t = [r["t_reach"] for r in rows if r["reached"]]
        print(f"{Path(run).name:48s} {len(rows):3d} " + " ".join(f"{sum(r[c] for r in rows):9d}" for c in cols)
              + f"  {np.mean(t):.2f} s" if t else "  -")


if __name__ == "__main__":
    main()
