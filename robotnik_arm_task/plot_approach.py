"""Hand distance to the target over time, aligned at the moment the base parks: sequential v concurrent design.

One figure for the report (one IEEE column). No simulation: it reads saved episode files only.
- Policies: the saved test runs of each seed (`compare.py --test-set ... --out-dir <folder>`, files in
  <folder>/test_set/ep_XXXX.npz), all seeds pooled (3 x 150 = 450 runs per policy).
- Experts: their stored episodes on the same 150 test ids (bc_data.split_episodes, seed 0) from the data folder each
  policy was trained on (sequential pre-shaping expert / concurrent expert).
- t = 0 is the first frame with nav_done = 1: the policy's latch, or the expert's "base done".
- Each curve is the median over runs; the band is the 25-75% range (policies only).
- Runs end at different times (1 s after first getting within 5 mm, or at the 30 s cap): after its last saved frame
  each run keeps its last distance (carried forward). Before a run's first frame (a latch earlier than 4 s) it has
  no value and is left out of the median at those times.

Usage (from robotnik_arm_task/; defaults are the report's Mac test runs):
    python plot_approach.py
    python plot_approach.py --seq-runs compare_out/a_s0 compare_out/a_s1 --seq-data data_sequential_expert ...
Prints the cross-checks (median distance at t = 0, share within 1 cm at t = +6 s, held within 1 cm per seed).
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bc_data import split_episodes  # noqa: E402

HERE = Path(__file__).resolve().parent
T_MIN, T_MAX, T_STEP = -4.0, 6.0, 0.01        # s, shown window and plotting grid
GRID = np.round(np.arange(T_MIN, T_MAX + T_STEP / 2, T_STEP), 6)
SEQ_MODEL, CONC_MODEL = "split_seqpre_ppdart_cld_r3", "split_conc_drive_r1"


def held(err, thr=0.01, hold=0.5, dt=1e-3):
    """True if err stays below thr for `hold` seconds in a row."""
    best = cur = 0
    for inside in err < thr:
        cur = cur + 1 if inside else 0
        best = max(best, cur)
    return best * dt >= hold


def aligned(path):
    """Distance (cm) on GRID, t = 0 at the first nav_done frame; None if nav_done never becomes 1."""
    d = np.load(path, allow_pickle=True)
    dist = 100 * np.linalg.norm(d["ee_pos"] - d["target"], axis=1)
    nav = np.asarray(d["nav_done"]).astype(bool)
    if not nav.any():
        return None, dist
    t = np.asarray(d["time"], float)
    t_rel = t - t[int(np.argmax(nav))]
    k = np.searchsorted(t_rel, GRID - 1e-9)    # first frame at or after each grid time
    out = dist[np.minimum(k, len(dist) - 1)]   # after the last frame: last distance carried forward
    out[GRID < t_rel[0] - 1e-9] = np.nan       # before the first frame: no value
    return out, dist


def load_group(files):
    rows, missing, held_n, n = [], 0, 0, 0
    for f in files:
        a, dist = aligned(f)
        n += 1
        held_n += held(dist / 100)
        if a is None:
            missing += 1
        else:
            rows.append(a)
    return np.array(rows), missing, held_n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-runs", nargs="+", default=[f"compare_out/mac_{SEQ_MODEL}_s{s}" for s in range(3)])
    ap.add_argument("--conc-runs", nargs="+", default=[f"compare_out/mac_{CONC_MODEL}_s{s}" for s in range(3)])
    ap.add_argument("--seq-data", default="data_sequential_expert")
    ap.add_argument("--conc-data", default="data_concurrent_expert")
    ap.add_argument("--out", default="results/analysis/approach_seq_v_conc.png")
    args = ap.parse_args()

    groups = {}
    for name, runs, data in (("sequential", args.seq_runs, args.seq_data), ("concurrent", args.conc_runs, args.conc_data)):
        data = HERE / data
        test_ids = split_episodes(data)[2]
        expected = [f"ep_{i:04d}.npz" for i in test_ids]
        pol_files = []
        for r in runs:
            files = sorted((HERE / r / "test_set").glob("ep_*.npz"))
            assert [f.name for f in files] == expected, f"{r}: not the 150 test episodes of {data.name}"
            pol_files += files
        groups[f"{name} policy"] = load_group(pol_files) + (len(runs),)
        groups[f"{name} expert"] = load_group([data / e for e in expected]) + (1,)

    # Cross-checks
    i0, i6 = np.argmin(abs(GRID)), np.argmin(abs(GRID - T_MAX))
    for label, (a, missing, held_n, n, n_seeds) in groups.items():
        print(f"{label:18s} runs {n} (no nav_done: {missing})  median at t=0 {np.nanmedian(a[:, i0]):.2f} cm  "
              f"within 1 cm at t=+6 s {np.mean(a[:, i6] < 1):.1%} ({np.sum(a[:, i6] < 1) / n_seeds:.1f} of "
              f"{n // n_seeds} per seed)  held 1 cm {held_n / n_seeds:.1f} per seed  "
              f"runs with a value at t=-4 s {np.sum(~np.isnan(a[:, 0]))}")

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 8,
                         "ytick.labelsize": 8})
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    colours = {"sequential": "#D55E00", "concurrent": "#0072B2"}
    for label, (a, *_rest) in groups.items():
        design, kind = label.split()
        med = np.nanmedian(a, axis=0)
        if kind == "policy":
            lo, hi = np.nanpercentile(a, [25, 75], axis=0)
            ax.fill_between(GRID, lo, hi, color=colours[design], alpha=0.2, linewidth=0)
            ax.plot(GRID, med, color=colours[design], linewidth=1.4, label=label)
        else:
            ax.plot(GRID, med, color=colours[design], linewidth=1.0, linestyle=(0, (4, 1.5)), label=label)
    for y in (1.0, 0.5):
        ax.axhline(y, color="0.4", linestyle="--", linewidth=0.7)
    ax.axvline(0, color="0.6", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_xlabel("time since the base parks (s)")
    ax.set_ylabel("hand to target (cm)")
    ax.legend(loc="upper right", frameon=False, handlelength=2.2)
    fig.tight_layout(pad=0.3)
    out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
