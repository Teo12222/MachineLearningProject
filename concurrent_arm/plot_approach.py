"""Report figure: end-effector error around the moment the base parks, sequential v concurrent (test 150).

For every run, the hand's distance to the target is put on a common time axis with t = 0 at parking (the first frame
with nav_done: the policy's latch, the expert's own arrival). A run stops 1 s after its first 5 mm frame (or at 30 s);
after its last frame the run keeps its last error, so every run counts at every time (a stopped run has reached).
Policies: 3 seeds x 150 runs pooled, median and interquartile band. Experts: median of their 150 stored test episodes.
Same runs as expert_gap.py (report Table I).

Usage (from robotnik_arm_task/):
    python ../concurrent_arm/plot_approach.py [--out results/analysis/approach]   # writes .pdf and .png
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "robotnik_arm_task"
sys.path.insert(0, str(ROOT))
from bc_data import split_episodes  # noqa: E402

SEEDS = (0, 1, 2)
T = np.arange(-3.0, 5.0 + 1e-9, 0.01)   # s, relative to parking
# name, expert data folder, policy test runs prefix (+ s<seed>: README section 6.1's compare.py --test-set folders),
# colour (reference palette slots 1-2)
VARIANTS = [
    ("Sequential", ("data_sequential_expert",), "compare_out/split_seqpre_ppdart_cld_r3_", "#2a78d6"),
    ("Concurrent", ("data_concurrent_expert",), "compare_out/split_conc_drive_r1_", "#eb6834"),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def find_dir(names):
    for n in names:
        if (ROOT / n).is_dir():
            return ROOT / n
    sys.exit(f"none of {names} found in {ROOT}")


def aligned_error(path):
    """Error (m) on the grid T, t = 0 at the first nav_done frame; last value held after the run ends."""
    with np.load(path) as d:
        t, err = d["time"], np.linalg.norm(d["ee_pos"] - d["target"], axis=1)
        nav = d["nav_done"].astype(bool)
    if not nav.any():
        return None
    rel = t - t[np.argmax(nav)]
    out = np.interp(T, rel, err, left=np.nan, right=err[-1])
    return out


def curves(files):
    rows = [r for r in (aligned_error(f) for f in files) if r is not None]
    a = np.vstack(rows)
    return len(rows), np.nanmedian(a, 0), np.nanpercentile(a, 25, 0), np.nanpercentile(a, 75, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/analysis/approach", help="output path without extension")
    a = p.parse_args()

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    for y, lab in ((0.01, "1 cm"), (0.005, "5 mm")):
        ax.axhline(y, color=MUTED, lw=0.6, ls=":", zorder=1)
        ax.text(T[0] + 0.08, y * 1.08, lab, color=MUTED, ha="left", va="bottom", fontsize=7)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    ax.text(0.08, 0.0004, "base parks", color=MUTED, fontsize=7, va="bottom")

    summary = []
    for name, data_names, prefix, colour in VARIANTS:
        data_dir = find_dir(data_names)
        ids = list(split_episodes(data_dir)[2])
        pol = [ROOT / f"{prefix}s{s}" / "test_set" / f"ep_{i:04d}.npz" for s in SEEDS for i in ids]
        n, med, q1, q3 = curves(pol)
        ne, emed, _, _ = curves([data_dir / f"ep_{i:04d}.npz" for i in ids])
        ax.fill_between(T, q1, q3, color=colour, alpha=0.18, lw=0, zorder=2)
        ax.plot(T, med, color=colour, lw=1.6, zorder=4, label=f"{name} policy")
        ax.plot(T, emed, color=colour, lw=1.0, ls="--", zorder=3, label=f"{name} expert")
        i0 = np.searchsorted(T, 0.0)
        summary.append((name, n, ne, med[i0], emed[i0]))

    # Log above 5 mm, linear below it and compressed (linscale), so the experts' last millimetres don't take as much
    # height as a decade above: the step from 5 mm down to ~1 mm would otherwise look as large as 1 cm -> 5 mm.
    ax.set_yscale("symlog", linthresh=0.005, linscale=0.35)
    ax.set_ylim(0.0, 0.7)
    ax.set_xlim(T[0], T[-1])
    ax.set_yticks([0.0, 0.005, 0.01, 0.03, 0.1, 0.3])
    ax.set_yticklabels(["0", "0.5", "1", "3", "10", "30"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Time from parking (s)")
    ax.set_ylabel("Hand–target distance (cm)")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.grid(axis="y", which="minor", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    order = [0, 2, 1, 3]   # row 1 policies, row 2 experts
    fig.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=6.5, loc="upper center",
               ncol=2, handlelength=1.8, columnspacing=1.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.86))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    for name, n, ne, m0, e0 in summary:
        print(f"{name}: {n} policy runs, {ne} expert runs; median error at parking: policy {100 * m0:.1f} cm, "
              f"expert {100 * e0:.1f} cm")
    print(f"wrote {out.with_suffix('.pdf')} and .png")


if __name__ == "__main__":
    main()
