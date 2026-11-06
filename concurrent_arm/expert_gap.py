"""Policy v its own expert on test 150, with the same measures for both (report Table I and the cloning gap).

The measures, in the report's order (the team's metric rule, 5 Nov):
  ever within 1 cm   the error is below 1 cm in at least one frame;
  held within 1 cm   below 1 cm for 0.5 s without a break;
  within 5 mm        below 5 mm once the base has parked / the latch is set (the stop rule; scripts call it "reached");
  time to reach      time of that 5 mm frame (no 1 s settle), mean over the runs that get within 5 mm;
plus held within 1 cm on low targets (height < 0.5 m). Every count comes from count_results.stats, the function
behind the policy tables, applied unchanged to the expert's stored test episodes, so expert and policy are scored by
the same code.

Cloning gap = policy (3-seed mean) - its expert. Time is also given paired: policy minus expert on the targets both get
within 5 mm (per seed, then the mean over seeds), since the two means above are over different sets of runs.

With two variants it also compares the two policies target by target (paired_targets: per target and measure, how
many of the 3 seeds pass; majority yes / no with an exact sign test; per-target time difference; held within 1 cm by
height band).

Checks: the test ids come from the expert folder's split and must match the policy folders' files; each policy run's
target must equal the expert episode's target (same ids, same targets).

Usage (from robotnik_arm_task/; defaults = the report's two variants, test runs as in README section 6.1):
    python ../concurrent_arm/expert_gap.py [--csv results/analysis/expert_gap.csv]
    python ../concurrent_arm/expert_gap.py --pair NAME DATA_DIR RUN_PREFIX ...   # RUN_PREFIX + s0/s1/s2 + /test_set
"""
import argparse
import csv
import sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "robotnik_arm_task"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from bc_data import split_episodes  # noqa: E402

from count_results import stats  # noqa: E402

LOW = 0.5   # m, "low targets": height below this
SEEDS = (0, 1, 2)
# name, expert data folder, policy test runs prefix (+ s<seed>: README section 6.1's compare.py --test-set folders)
DEFAULT_PAIRS = [
    ("sequential", ("data_sequential_expert",), "compare_out/split_seqpre_ppdart_cld_r3_"),
    ("concurrent", ("data_concurrent_expert",), "compare_out/split_conc_drive_r1_"),
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


def sign_test(b, c):
    """Exact two-sided sign (McNemar) test on discordant counts b, c: P(|X - n/2| >= |b - n/2|), X ~ Bin(b + c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def paired_targets(first, second):
    """The two policies on the same targets (rules fixed before looking, 6 Nov).

    Per target, a policy's score on a measure is the number of its 3 seeds that pass (0-3). Counted: targets where
    the second policy scores higher, lower, the same. Majority (passes in >= 2 of 3 seeds) gives one yes / no per
    target and policy; the exact sign test (McNemar) on the discordant targets says whether the split could be chance.
    Time: per target the mean time to reach over the seeds that get within 5 mm; compared on the targets where both
    policies get within 5 mm in at least one seed.
    """
    (na, A), (nb, B) = first, second
    if A["ids"] != B["ids"]:
        sys.exit("paired comparison: the two variants have different test ids")
    ids = A["ids"]
    print(f"\nPer target, {nb} v {na} policy (same {len(ids)} test targets; score = seeds out of 3 that pass)")
    print(f"{'measure':12s} {nb + ' higher':>18s} {na + ' higher':>18s} {'same':>6s}   "
          f"majority: {nb} only / {na} only / both / neither   sign test p")
    for m, key in (("ever 1 cm", "ever_1cm"), ("held 1 cm", "held_1cm"), ("within 5 mm", "reached")):
        sa = {i: sum(s[i][key] for s in A["seeds"]) for i in ids}
        sb = {i: sum(s[i][key] for s in B["seeds"]) for i in ids}
        hi = sum(sb[i] > sa[i] for i in ids)
        lo = sum(sb[i] < sa[i] for i in ids)
        ma = {i: sa[i] >= 2 for i in ids}
        mb = {i: sb[i] >= 2 for i in ids}
        b_only = sum(mb[i] and not ma[i] for i in ids)
        a_only = sum(ma[i] and not mb[i] for i in ids)
        both = sum(ma[i] and mb[i] for i in ids)
        print(f"{m:12s} {hi:18d} {lo:18d} {len(ids) - hi - lo:6d}   "
              f"{b_only:>9d} / {a_only} / {both} / {len(ids) - b_only - a_only - both}"
              f"   p = {sign_test(b_only, a_only):.2g}")
    # held within 1 cm, by target height band
    print(f"held 1 cm by height band ({nb} higher / {na} higher / same, of n):")
    for lo_h, hi_h in ((0.2, 0.5), (0.5, 0.8), (0.8, 1.1)):
        band = [i for i in ids if lo_h <= A["seeds"][0][i]["height"] < hi_h + 1e-9]
        sa = {i: sum(s[i]["held_1cm"] for s in A["seeds"]) for i in band}
        sb = {i: sum(s[i]["held_1cm"] for s in B["seeds"]) for i in band}
        print(f"  {lo_h:.1f}-{hi_h:.1f} m: {sum(sb[i] > sa[i] for i in band)} / {sum(sb[i] < sa[i] for i in band)}"
              f" / {sum(sb[i] == sa[i] for i in band)}, of {len(band)}")

    def mean_t(P, i):
        t = [s[i]["t_reach"] for s in P["seeds"] if s[i]["reached"]]
        return float(np.mean(t)) if t else np.nan
    ta = {i: mean_t(A, i) for i in ids}
    tb = {i: mean_t(B, i) for i in ids}
    both = [i for i in ids if not (np.isnan(ta[i]) or np.isnan(tb[i]))]
    d = np.array([tb[i] - ta[i] for i in both])
    print(f"time to reach, {nb} - {na} per target (both within 5 mm in >= 1 seed, n = {len(both)}): mean {d.mean():+.2f} s,"
          f" median {np.median(d):+.2f} s; {nb} faster on {int((d < 0).sum())}, slower on {int((d > 0).sum())}"
          f" (sign test p = {sign_test(int((d < 0).sum()), int((d > 0).sum())):.2g})")


def distances(per_target, csv_path=None):
    """Ever / held (0.5 s) within 5 mm, 1, 2, 5 and 10 cm per policy (3-seed mean, seed range) and its own expert.

    Uses run_stats' ever_* / held_* (any time in the run). Also checks that "ever within 5 mm" here equals the
    metric rule's "within 5 mm" (below 5 mm once parked / latched) on every run, so the table's 5 mm column is that
    rule; any run that gets within 5 mm before parking is reported.
    """
    labels = ("5mm", "1cm", "2cm", "5cm", "10cm")
    rows = []
    print("\nDistances, test 150: ever / held (0.5 s) within each distance; policy = 3-seed mean [seed range]")
    print(f"{'':22s}" + "".join(f"{l:>20s}" for l in labels))
    for name, P in per_target.items():
        ids = P["ids"]
        for who, runs_list in (("expert", [P["expert"]]), ("policy", P["seeds"])):
            early = sum(r[i]["ever_5mm"] != r[i]["reached"] for r in runs_list for i in ids)
            if early:
                print(f"  note: {name} {who}: {early} runs within 5 mm before parking (ever_5mm != within 5 mm)")
            cells, row = "", {"variant": name, "row": who}
            for l in labels:
                ev = [sum(r[i][f"ever_{l}"] for i in ids) for r in runs_list]
                he = [sum(r[i][f"held_{l}"] for i in ids) for r in runs_list]
                row[f"ever_{l}"], row[f"held_{l}"] = round(float(np.mean(ev)), 2), round(float(np.mean(he)), 2)
                rng = f"[{min(he)}-{max(he)}]" if len(he) > 1 else ""
                cells += f"{np.mean(ev):>8.1f}/{np.mean(he):<5.1f}{rng:7s}"
            rows.append(row)
            print(f"{name + ' ' + who:22s}{cells}")
    if csv_path:
        out = ROOT / csv_path
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", nargs=3, action="append", metavar=("NAME", "DATA_DIR", "RUN_PREFIX"),
                   help="replace the defaults: expert folder and policy runs prefix (seeds appended)")
    p.add_argument("--csv", help="also write the table to this CSV (relative to robotnik_arm_task/)")
    p.add_argument("--distances-csv", help="also write the distances table to this CSV (relative to robotnik_arm_task/)")
    a = p.parse_args()
    pairs = [(n, (d,), r) for n, d, r in a.pair] if a.pair else DEFAULT_PAIRS

    table, per_target = [], {}
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
            per_target.setdefault(name, {"ids": test_ids, "seeds": [], "expert": exp_ids})["seeds"].append(pol_ids)
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

    distances(per_target, a.distances_csv)
    if len(per_target) == 2:
        paired_targets(*per_target.items())

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
