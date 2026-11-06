"""Policy v its own expert on test 150, with the same measures for both (report Table I and the cloning gap).

The measures, in the report's order (the team's metric rule, 5 Nov):
  ever within 1 cm   the error is below 1 cm in at least one frame;
  held within 1 cm   below 1 cm for 0.5 s without a break;
  within 5 mm        below 5 mm once the base has parked / the latch is set (the stop rule; scripts call it "reached");
  time to reach      time of that 5 mm frame (no 1 s settle), mean over the runs that get within 5 mm;
plus held within 1 cm on low targets (height < 0.5 m). Every count comes from route_b_eval.stats (count_results.stats
on master), the function behind the policy tables, applied unchanged to the expert's stored test episodes, so expert
and policy are scored by the same code.

Cloning gap = policy (3-seed mean) - its expert. Time is also given paired: policy minus expert on the targets both get
within 5 mm (per seed, then the mean over seeds), since the two means above are over different sets of runs.

Checks: the test ids come from the expert folder's split and must match the policy folders' files; each policy run's
target must equal the expert episode's target (same ids, same targets).

Usage (from robotnik_arm_task/; defaults = the report's two variants on Kris's Mac):
    python ../concurrent_arm/expert_gap.py [--csv results/analysis/expert_gap.csv]
    python ../concurrent_arm/expert_gap.py --pair NAME DATA_DIR RUN_PREFIX ...   # RUN_PREFIX + s0/s1/s2 + /test_set
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "robotnik_arm_task"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from bc_data import split_episodes  # noqa: E402

try:
    from count_results import stats  # noqa: E402  (master's name)
except ImportError:
    from route_b_eval import stats  # noqa: E402  (Kris branch's name)

LOW = 0.5   # m, "low targets": height below this
SEEDS = (0, 1, 2)
# name, expert data folder (old name, new name on master), policy runs prefix (+ s<seed>)
DEFAULT_PAIRS = [
    ("sequential", ("data_seqpre", "data_sequential_expert"), "compare_out/mac_split_seqpre_ppdart_cld_r3_"),
    ("concurrent", ("data_route_b", "data_concurrent_expert"), "compare_out/mac_split_conc_drive_r1_"),
]
COLS = ("ever_1cm", "held_1cm", "within_5mm", "t_reach", "low_held_1cm")


def find_dir(names):
    for n in names:
        if (ROOT / n).is_dir():
            return ROOT / n
    sys.exit(f"none of {names} found in {ROOT}")


def summarise(files):
    """Counts over a list of episode files, plus per-id stats for the paired time and the target check."""
    per_id = {}
    for f in files:
        s = stats(f)
        with np.load(f) as d:
            s["target"] = d["target"].copy()
        per_id[int(f.stem.split("_")[1])] = s
    rows = list(per_id.values())
    t = [r["t_reach"] for r in rows if r["reached"]]
    low = [r for r in rows if r["height"] < LOW]
    out = {"n": len(rows), "ever_1cm": sum(r["ever_1cm"] for r in rows),
           "held_1cm": sum(r["held_1cm"] for r in rows), "within_5mm": sum(r["reached"] for r in rows),
           "t_reach": float(np.mean(t)) if t else np.nan,
           "low_held_1cm": sum(r["held_1cm"] for r in low), "n_low": len(low)}
    return out, per_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", nargs=3, action="append", metavar=("NAME", "DATA_DIR", "RUN_PREFIX"),
                   help="replace the defaults: expert folder and policy runs prefix (seeds appended)")
    p.add_argument("--csv", help="also write the table to this CSV (relative to robotnik_arm_task/)")
    a = p.parse_args()
    pairs = [(n, (d,), r) for n, d, r in a.pair] if a.pair else DEFAULT_PAIRS

    table = []
    for name, data_names, prefix in pairs:
        data_dir = find_dir(data_names)
        test_ids = list(split_episodes(data_dir)[2])
        exp, exp_ids = summarise([data_dir / f"ep_{i:04d}.npz" for i in test_ids])
        seeds, paired = [], []
        for s in SEEDS:
            run = ROOT / f"{prefix}s{s}" / "test_set"
            files = [run / f"ep_{i:04d}.npz" for i in test_ids]
            missing = [f.name for f in files if not f.exists()]
            if missing:
                sys.exit(f"{run}: {len(missing)} test episodes missing, e.g. {missing[:3]}")
            res, pol_ids = summarise(files)
            bad = [i for i in test_ids if not np.allclose(pol_ids[i]["target"], exp_ids[i]["target"])]
            if bad:
                sys.exit(f"{run}: targets differ from {data_dir.name} on ids {bad[:5]}")
            both = [i for i in test_ids if pol_ids[i]["reached"] and exp_ids[i]["reached"]]
            paired.append(np.mean([pol_ids[i]["t_reach"] - exp_ids[i]["t_reach"] for i in both]))
            res["paired_n"] = len(both)
            seeds.append(res)
        mean = {c: float(np.mean([r[c] for r in seeds])) for c in COLS}
        gap = {c: mean[c] - exp[c] for c in COLS}
        table.append((name, data_dir.name, prefix, exp, seeds, mean, gap, float(np.mean(paired)),
                      [r["paired_n"] for r in seeds]))

    head = f"{'':24s} {'ever 1cm':>9s} {'held 1cm':>9s} {'5 mm':>7s} {'time (s)':>9s} {'low held':>9s}"
    for name, data, prefix, exp, seeds, mean, gap, paired, pn in table:
        print(f"\n{name}: expert {data}, policy {prefix}s0-2 (test 150)\n{head}")
        fmt = lambda r, dig: (f"{r['ever_1cm']:9.{dig}f} {r['held_1cm']:9.{dig}f} {r['within_5mm']:7.{dig}f} "
                              f"{r['t_reach']:9.2f} {r['low_held_1cm']:9.{dig}f}")
        print(f"{'expert':24s} " + fmt(exp, 0) + f"   (low n = {exp['n_low']})")
        for s, r in zip(SEEDS, seeds):
            print(f"{'policy s' + str(s):24s} " + fmt(r, 0))
        print(f"{'policy, 3-seed mean':24s} " + fmt(mean, 1))
        print(f"{'gap (policy - expert)':24s} " + fmt(gap, 1))
        print(f"paired time, policy - expert on targets both get within 5 mm: {paired:+.2f} s "
              f"(n = {' / '.join(map(str, pn))} per seed)")

    if a.csv:
        out = ROOT / a.csv
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant", "row", *COLS, "n_low", "paired_time_gap_s"])
            for name, _, _, exp, seeds, mean, gap, paired, _ in table:
                w.writerow([name, "expert", *[exp[c] for c in COLS], exp["n_low"], ""])
                for s, r in zip(SEEDS, seeds):
                    w.writerow([name, f"policy_s{s}", *[r[c] for c in COLS], r["n_low"], ""])
                w.writerow([name, "policy_mean", *[round(mean[c], 2) for c in COLS], exp["n_low"], ""])
                w.writerow([name, "gap", *[round(gap[c], 2) for c in COLS], "", round(paired, 3)])
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
