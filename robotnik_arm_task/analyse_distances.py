"""How close the final model gets, from its saved test-set runs (no simulation): the report's distances table and the
1 cm analyses (PLAN.md "Plan from 3 Nov", report structure).

For each run, the end-effector error to the target over time:
- ever within d: the error is below d in at least one frame (any time, or only after the latch);
- held within d: below d for 0.5 s without a break (the "held" rule of PLAN.md step 15);
at d = 5 mm, 1, 2, 5, 10 cm, per seed, the 3-seed mean and the expert on the same targets (data_main's stored runs).
Runs never within 1 cm are split into "still closing in" (the error over the last 0.5 s is at least 2 mm below the
error 5 s earlier, medians of 0.5 s windows; rule fixed before looking, 3 Nov) and "hovering" (everything else).
Ever / held within 1 cm per band of target distance (horizontal, from the start pose) and target height.

Usage (from the robotnik_arm_task folder):
    pixi run python analyse_distances.py                 # the final model (split_dartdagger_bb_plusCLD_r2_s0/1/2) on test 150
    pixi run python analyse_distances.py --models a_s0 b_s0 --run-suffix _test
Writes results/analysis/<name>_distances.csv and <name>_bands.csv (results.py doesn't read that folder).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from bc_data import split_episodes

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_main"
THRESHOLDS = [0.005, 0.01, 0.02, 0.05, 0.10]   # m
HOLD = 0.5                                      # s without a break, as PLAN.md step 15
CLOSING = 0.002                                 # m, "still closing in": this much lower than 5 s earlier
DIST_BANDS = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]   # m, horizontal target distance from the start pose
HEIGHT_BANDS = [(0.2, 0.5), (0.5, 0.8), (0.8, 1.1)]            # m, target height


def label(d):
    return f"{d * 1000:.0f}mm" if d < 0.01 else f"{d * 100:.0f}cm"


def longest_inside(inside, t):
    """Longest stretch (s) of consecutive frames with inside True."""
    if not inside.any():
        return 0.0
    edges = np.diff(np.r_[0, inside.astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return float((t[ends] - t[starts]).max())


def window_median(err, t, t_end):
    """Median error over the 0.5 s window ending at t_end."""
    m = (t > t_end - 0.5) & (t <= t_end)
    return float(np.median(err[m])) if m.any() else np.nan


def run_stats(d):
    """Per-run numbers from one saved episode (policy or expert)."""
    t = d["time"]
    err = np.linalg.norm(d["ee_pos"] - d["target"], axis=1)
    nav = d["nav_done"].astype(bool)
    start = d["start_pose"][:2] if "start_pose" in d else np.zeros(2)
    s = {"dist": float(np.linalg.norm(d["target"][:2] - start)), "height": float(d["target"][2])}
    for thr in THRESHOLDS:
        s[f"ever_{label(thr)}"] = bool((err < thr).any())
        s[f"held_{label(thr)}"] = longest_inside(err < thr, t) >= HOLD - 1e-9
    s["ever_1cm_after_latch"] = bool((err[nav] < 0.01).any())
    if not s["ever_1cm"]:
        end, before = window_median(err, t, t[-1]), window_median(err, t, t[-1] - 5.0)
        s["closing_in"] = bool(end <= before - CLOSING)
        s["final_err"] = float(err[-1])
        # Where and when it got closest: a run can get close, then drift back out ("hovering" by the rule above)
        s["closest"], s["closest_t"] = float(err.min()), float(t[np.argmin(err)])
    return s


def load_runs(run_dir, ids):
    runs = {}
    for i in ids:
        p = Path(run_dir) / f"ep_{i:04d}.npz"
        if p.exists():
            with np.load(p) as d:
                runs[i] = run_stats({k: d[k] for k in d.files})
    return runs


def in_band(v, band, last):
    lo, hi = band
    return lo <= v < hi or (last and v == hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Default = the final model chosen 4 Nov (CLD r2). Was split_dartdagger_r5_s{s}, the previous final model;
    # changed so a plain run regenerates the report's table for the model the report describes.
    # parser.add_argument("--models", nargs="+", default=[f"split_dartdagger_r5_s{s}" for s in range(3)],
    parser.add_argument("--models", nargs="+", default=[f"split_dartdagger_bb_plusCLD_r2_s{s}" for s in range(3)],
                        help="model names; runs read from compare_out/<model><suffix>/test_set")
    parser.add_argument("--run-suffix", default="_test")
    parser.add_argument("--name", default="final", help="prefix of the output files")
    parser.add_argument("--out", default=str(SCRIPT_DIR / "results" / "analysis"), help="folder for the two files")
    args = parser.parse_args()

    _, _, test_ids = split_episodes(DATA_DIR)
    rows = {}
    for m in args.models:
        rows[m] = load_runs(SCRIPT_DIR / "compare_out" / f"{m}{args.run_suffix}" / "test_set", test_ids)
        missing = [i for i in test_ids if i not in rows[m]]
        print(f"{m}: {len(rows[m])}/{len(test_ids)} runs" + (f", missing {missing}" if missing else ""))
    rows["expert"] = load_runs(DATA_DIR, test_ids)
    n = len(test_ids)

    # 1. Distances table: ever / held within each distance, per seed, mean of the seeds, expert
    cols = [f"{kind}_{label(thr)}" for thr in THRESHOLDS for kind in ("ever", "held")] + ["ever_1cm_after_latch"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = {}
    for m, runs in rows.items():
        table[m] = {c: sum(r[c] for r in runs.values()) for c in cols}
    table["mean of seeds"] = {c: np.mean([table[m][c] for m in args.models]) for c in cols}
    order = args.models + ["mean of seeds", "expert"]
    with open(out / f"{args.name}_distances.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "episodes"] + cols)
        for m in order:
            w.writerow([m, n] + [f"{table[m][c]:.1f}" if m == "mean of seeds" else table[m][c] for c in cols])
    print(f"\nTest {n}: ever / held ({HOLD} s) within each distance")
    print(f"{'':24s}" + "".join(f"{label(thr):>14s}" for thr in THRESHOLDS))
    for m in order:
        cells = "".join(f"{table[m]['ever_' + label(thr)]:>7.1f}/{table[m]['held_' + label(thr)]:<6.1f}"
                        for thr in THRESHOLDS)
        print(f"{m:24s}{cells}")
    print("\nEver within 1 cm: any time v after the latch")
    for m in order:
        print(f"  {m:24s} {table[m]['ever_1cm']:6.1f} v {table[m]['ever_1cm_after_latch']:6.1f}")

    # 2. Runs never within 1 cm: still closing in at the end v hovering
    print(f"\nNever within 1 cm: still closing in (last 0.5 s >= {CLOSING * 1000:.0f} mm below 5 s earlier) v hovering")
    for m in args.models + ["expert"]:
        never = [r for r in rows[m].values() if not r["ever_1cm"]]
        if not never:
            print(f"  {m:24s} never   0")
            continue
        closing = sum(r["closing_in"] for r in never)
        med = lambda k: np.median([r[k] for r in never])
        low = sum(r["height"] < HEIGHT_BANDS[0][1] for r in never)
        print(f"  {m:24s} never {len(never):3d}: closing in {closing:3d}, hovering {len(never) - closing:3d};"
              f" closest {100 * med('closest'):.2f} cm at {med('closest_t'):.1f} s,"
              f" final {100 * med('final_err'):.2f} cm"
              f" (medians); target below {HEIGHT_BANDS[0][1]} m: {low}")

    # 3. Bands of target distance and height: ever / held within 1 cm
    band_rows = []
    for key, bands, unit in (("dist", DIST_BANDS, "distance"), ("height", HEIGHT_BANDS, "height")):
        for j, band in enumerate(bands):
            last = j == len(bands) - 1
            ids = [i for i in test_ids if in_band(rows["expert"][i][key], band, last)]
            row = {"band": f"{unit} {band[0]:.1f}-{band[1]:.1f} m", "targets": len(ids)}
            for m in args.models + ["expert"]:
                for c in ("ever_1cm", "held_1cm", "ever_5mm"):
                    row[f"{m} {c}"] = sum(rows[m][i][c] for i in ids if i in rows[m])
            for c in ("ever_1cm", "held_1cm", "ever_5mm"):
                row[f"mean {c}"] = np.mean([row[f"{m} {c}"] for m in args.models])
                row[f"mean {c} %"] = 100 * row[f"mean {c}"] / len(ids) if ids else np.nan
            band_rows.append(row)
    fields = list(band_rows[0])
    with open(out / f"{args.name}_bands.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in band_rows:
            w.writerow({k: f"{v:.1f}" if isinstance(v, float) else v for k, v in row.items()})
    print("\nBy band (mean of seeds; expert): ever / held within 1 cm, and ever within 5 mm (= reached-like)")
    for row in band_rows:
        print(f"  {row['band']:22s} n={row['targets']:3d}  ever 1 cm {row['mean ever_1cm']:5.1f} "
              f"({row['mean ever_1cm %']:3.0f}%)  held 1 cm {row['mean held_1cm']:5.1f}"
              f" ({row['mean held_1cm %']:3.0f}%)"
              f"  5 mm {row['mean ever_5mm']:5.1f}   | expert ever 1 cm {row['expert ever_1cm']}/{row['targets']}")
    print(f"\nWritten: {out / (args.name + '_distances.csv')}, {out / (args.name + '_bands.csv')}")


if __name__ == "__main__":
    main()
