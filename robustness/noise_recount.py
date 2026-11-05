"""Counts for the sensor-noise runs (PLAN.md step 12), recomputed from the saved policy episode files.

An independent check of compare.py / route_b_eval.py: nothing is taken from them except the stop message (for "reached"
and the time to reach). One frame is 1 ms.
- reached:   the stop message says "Target reached" (5 mm, the task scripts' REACH_TOLERANCE); also checked against
             the frames: the error is below 5 mm at some frame of every reached run
- ever 1 cm: the error is below 1 cm at some frame
- held 1 / 5 cm: below that distance for 0.5 s in a row
- collapsed: the error goes back above 20 cm after the closest approach
- off floor: the base height moves more than 10 cm from its start
- at cap:    the run lasted to the 30 s time cap
- time to reach: the time in the stop message ("at t=..."), i.e. without the 1 s settle; mean over the reached runs
  of each seed, then over the seeds (as route_b_eval.py)
- held 1 cm split by outcome (reached / not reached): shows whether a change in "held" comes from runs that hover at
  the time cap instead of reaching 5 mm

Usage, from robotnik_arm_task/ (folders as robustness/noise_runs.sh names them):
    python ../robustness/noise_recount.py split_conc_drive_r1
    python ../robustness/noise_recount.py split_seqpre_ppdart_cld_r3 --levels mac noise1 noise5 noise10
    python ../robustness/noise_recount.py expert     # robustness/expert_noise_runs.py's folders (one run per level)
"""
import argparse
import re
from pathlib import Path

import numpy as np


def held(err, thr, hold=0.5, dt=1e-3):
    best = cur = 0
    for inside in err < thr:
        cur = cur + 1 if inside else 0
        best = max(best, cur)
    return best * dt >= hold


def episode_row(f):
    d = np.load(f, allow_pickle=True)
    e = np.linalg.norm(d["ee_pos"] - d["target"], axis=1)
    z = d["base_pos"][:, 2]
    msg = str(d["stop_message"]) if "stop_message" in d.files else ""
    reached = msg.startswith("Target reached")
    t = re.search(r"at t=([0-9.]+)s", msg)
    if reached:
        assert e.min() < 0.005, (f, e.min())
    return dict(reached=reached, ever1=e.min() < 0.01, held1=held(e, 0.01), held5=held(e, 0.05),
                final=e[-1], collapsed=e[np.argmin(e):].max() > 0.2, off_floor=abs(z - z[0]).max() > 0.1,
                at_cap=float(d["time"][-1]) >= 29.9, t_reach=float(t.group(1)) if reached and t else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="model prefix, e.g. split_conc_drive_r1 (folders <model>_s<seed>_<level>_val)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--levels", nargs="+", default=["clean", "noise1", "noise5", "noise10"])
    ap.add_argument("--root", default="compare_out")
    args = ap.parse_args()

    ref_ids = None
    for level in args.levels:
        seeds = []
        # "expert": robustness/expert_noise_runs.py's folders, one per level, episode files at the top level
        for s in ([0] if args.model == "expert" else args.seeds):
            folder = (Path(args.root, f"expert_{level}_val") if args.model == "expert"
                      else Path(args.root, f"{args.model}_s{s}_{level}_val", "val_set"))
            files = sorted(folder.glob("ep_*.npz"))
            if not files:
                print(f"{level:8s} s{s}: no episode files")
                continue
            ids = [f.name for f in files]
            ref_ids = ref_ids or ids
            assert ids == ref_ids, f"{level} s{s}: different episodes from the first folder"
            rows = [episode_row(f) for f in files]
            c = {k: sum(r[k] for r in rows) for k in ("reached", "ever1", "held1", "held5", "collapsed",
                                                       "off_floor", "at_cap")}
            c["t_reach"] = np.nanmean([r["t_reach"] for r in rows]) if c["reached"] else np.nan
            c["held_if_reached"] = sum(r["held1"] for r in rows if r["reached"])
            c["n"] = len(rows)
            seeds.append(c)
            print(f"{level:8s} s{s}: n {c['n']}  reached {c['reached']:2d}  ever 1 cm {c['ever1']:2d}  "
                  f"held 1 cm {c['held1']:2d}  held 5 cm {c['held5']:2d}  "
                  f"median final {100 * np.median([r['final'] for r in rows]):.2f} cm  collapsed {c['collapsed']}  "
                  f"off floor {c['off_floor']}  at 30 s cap {c['at_cap']}  time to reach {c['t_reach']:.2f} s")
        if not seeds:
            continue
        m = {k: np.mean([c[k] for c in seeds]) for k in seeds[0]}
        tot = {k: sum(c[k] for c in seeds) for k in ("reached", "n", "held1", "held_if_reached", "at_cap")}
        print(f"{level:8s} mean of {len(seeds)}: reached {m['reached']:.1f}  ever {m['ever1']:.1f}  "
              f"held 1 cm {m['held1']:.1f}  held 5 cm {m['held5']:.1f}  collapsed {m['collapsed']:.1f}  "
              f"off floor {m['off_floor']:.1f}  time to reach {m['t_reach']:.2f} s  |  pooled: at cap "
              f"{tot['at_cap']} / {tot['n']}, held 1 cm among reached {tot['held_if_reached']} / {tot['reached']}, "
              f"among not reached {tot['held1'] - tot['held_if_reached']} / {tot['n'] - tot['reached']}")


if __name__ == "__main__":
    main()
