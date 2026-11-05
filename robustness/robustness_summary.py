"""Counts for the start-pose, out-of-range and payload runs of robustness_runs.sh, a model next to the expert.

From the saved episode files, per run (one frame is 1 ms; analyse_distances.run_stats for ever / held):
- reached:    the stop message says "Target reached" (5 mm)
- ever 1 cm:  the hand-target error is below 1 cm at some frame
- held 1 / 5 cm: below that distance for 0.5 s in a row
- collapsed:  the error goes back above 20 cm after the closest approach
- timeout latch (out of range): the base's arrival was declared by its time limit, not by parking
- time to reach (payload): mean over the reached runs, without the 1 s settle
- torque at the limit (payload): share of frames with an arm joint torque at the 30 N*m limit
Start pose offset / both: only the runs whose target is 1-3 m (horizontal) from the start pose are counted (the rule
is on the targets only, never the outcomes; 50 per pose), so every pose has the training distance range.

Usage, from robotnik_arm_task/:
    python ../robustness/robustness_summary.py split_conc_drive_r1_s0
    python ../robustness/robustness_summary.py split_seqpre_ppdart_cld_r3_s0 --ref expert
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "robotnik_arm_task"))
from analyse_distances import run_stats  # noqa: E402

POSES = {"default": (0.0, 0.0), "offset": (2.0, 1.5), "rotated": (0.0, 0.0), "both": (-1.5, 2.0)}   # start x, y
BANDS = {"near": [(0.6, 0.8), (0.8, 1.0)], "far": [(3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.01)]}
TORQUE_LIMIT = 29.9   # N*m, just under the 30 N*m clip


def stats(path):
    d = dict(np.load(path, allow_pickle=True))
    s = run_stats(d)
    err = np.linalg.norm(d["ee_pos"] - d["target"], axis=1)
    msg = str(d["stop_message"])
    s["reached"] = msg.startswith("Target reached")
    s["coll"] = bool((err[int(np.argmin(err)):] > 0.2).any())
    s["timeout"] = str(d.get("arrived_source", "")) == "timeout"
    s["t_reach"] = float(d["time"][-1]) - 1.0 if s["reached"] else np.nan
    s["at_limit"] = float((np.abs(d["torque_arm"]) >= TORQUE_LIMIT).any(axis=1).mean())
    s["target"] = np.asarray(d["target"], dtype=float)
    return s


def load(folder):
    return [stats(p) for p in sorted(Path(folder).glob("ep_*.npz"))]


def line(R, extra=False):
    if not R:
        return "no episode files"
    c = lambda k: sum(r[k] for r in R)
    out = (f"n {len(R):3d} | reached {c('reached'):3d}  ever 1 cm {c('ever_1cm'):3d}  held 1 cm {c('held_1cm'):3d}  "
           f"held 5 cm {c('held_5cm'):3d}  collapsed {c('coll')}  timeout latch {c('timeout')}")
    if extra:
        t = [r["t_reach"] for r in R if r["reached"]]
        out += (f" | time to reach {np.mean(t):.2f} s" if t else " | time to reach -") + \
               f"  torque at the limit {100 * np.mean([r['at_limit'] for r in R]):.1f}% of frames"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="model name as in robustness_runs.sh, e.g. split_conc_drive_r1_s0")
    ap.add_argument("--ref", default="expert", help="the reference folders' name (default: expert)")
    a = ap.parse_args()
    who = [(a.model, "model "), (a.ref, "expert" if a.ref == "expert" else "ref   ")]

    print("START POSE (targets 1-3 m from the start pose, run_batch.py --seed 21)")
    for pose, xy in POSES.items():
        for name, tag in who:
            R = [r for r in load(f"starts/{name}_{pose}")
                 if 1.0 <= np.linalg.norm(r["target"][:2] - np.array(xy)) < 3.0]
            print(f"  {pose:8s} {tag} {line(R)}")

    print("OUT OF RANGE (default start, out_of_range.py --seed 31)")
    for band, sub in BANDS.items():
        for name, tag in who:
            R = load(f"oor/{name}_{band}")
            for lo, hi in sub:
                print(f"  {lo:.1f}-{min(hi, 5):.1f} m {tag} {line([r for r in R if lo <= r['dist'] < hi])}")

    print("PAYLOAD (extra mass at the hand, default start, the 50 start-pose targets; 0 kg = starts/<name>_default)")
    for kg in ("0", "0.5", "1", "2"):
        for name, tag in who:
            folder = f"starts/{name}_default" if kg == "0" else f"payload/{name}_{kg}kg"
            print(f"  {kg:>3} kg {tag} {line(load(folder), extra=True)}")


if __name__ == "__main__":
    main()
