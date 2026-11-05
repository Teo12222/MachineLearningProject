"""Torque graph (PLAN.md step 25): the 7 arm torques of the policy vs the expert's on the same target, over time.

From the saved policy episode and the expert's episode in data_course_expert: no simulation. The latch (policy) and the
expert's arrival are marked; the EE error of both runs is the bottom panel, on the same time axis (a panel of its own
instead of a second y-axis, so each panel has one scale). Both runs start at t = 0, so they are aligned at the start,
not at the latch.

Usage (from the robotnik_arm_task folder):
    pixi run python plot_torques.py compare_out/split_arrived_s0/val_set 363 922   # -> compare_out/torque_plots/
    pixi run python plot_torques.py <run_dir> <episode> [...] --out <folder>
<run_dir> is a compare.py output folder (val_set / test_set); the episode numbers are data_course_expert's.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
EXPERT, POLICY = "#2a78d6", "#eb6834"   # categorical slots 1 and 2 of the dataviz palette (validated pair)
INK, MUTED = "#52514e", "#9a9893"
LIMIT = 30.0   # N*m, the torque clip of the expert and the policy


def first_true(mask, t):
    return t[np.argmax(mask)] if mask.any() else np.nan


def plot_episode(run_dir, i, out_dir, before=1.0, after=6.0):
    pol = np.load(Path(run_dir) / f"ep_{i:04d}.npz")
    exp = np.load(ROOT / "data_course_expert" / f"ep_{i:04d}.npz")
    t_latch = first_true(pol["nav_done"].astype(bool), pol["time"])
    t_arrive = first_true(exp["nav_done"].astype(bool), exp["time"])
    t0 = np.nanmin([t_latch, t_arrive]) - before
    t1 = min(max(pol["time"][-1], exp["time"][-1]), np.nanmax([t_latch, t_arrive]) + after)
    err_p = np.linalg.norm(pol["ee_pos"] - pol["target"], axis=1)
    err_e = np.linalg.norm(exp["ee_pos"] - exp["target"], axis=1)

    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    panels = axes.ravel()
    for j in range(7):
        ax = panels[j]
        for d, color, name in ((exp, EXPERT, "expert"), (pol, POLICY, "policy")):
            ax.plot(d["time"], d["torque_arm"][:, j], color=color, linewidth=1.2, label=name)
        ax.set_ylabel(f"J{j + 1} torque (N*m)", color=INK)
    ax = panels[7]
    ax.plot(exp["time"], 100 * err_e, color=EXPERT, linewidth=1.2, label="expert")
    ax.plot(pol["time"], 100 * err_p, color=POLICY, linewidth=1.2, label="policy")
    ax.set_yscale("log")
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1)
    ax.axhline(1.0, color=MUTED, linestyle=":", linewidth=1)
    ax.set_ylabel("EE error (cm, log)", color=INK)
    for ax in panels:
        ax.axvline(t_arrive, color=EXPERT, linestyle="--", linewidth=1)
        ax.axvline(t_latch, color=POLICY, linestyle="--", linewidth=1)
        ax.grid(True, alpha=0.25)
        ax.tick_params(colors=INK, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(t0, t1)
    for ax in panels[:7]:
        lo, hi = ax.get_ylim()
        if hi > LIMIT - 3 or lo < -LIMIT + 3:   # show the clip only where the torques get near it
            ax.axhline(LIMIT, color=MUTED, linewidth=0.8)
            ax.axhline(-LIMIT, color=MUTED, linewidth=0.8)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)", color=INK)
    handles = [plt.Line2D([], [], color=EXPERT, label="expert"), plt.Line2D([], [], color=POLICY, label="policy"),
               plt.Line2D([], [], color=EXPERT, linestyle="--", label=f"expert arrival {t_arrive:.2f} s"),
               plt.Line2D([], [], color=POLICY, linestyle="--", label=f"policy latch {t_latch:.2f} s"),
               plt.Line2D([], [], color=MUTED, linestyle=":", label="5 mm / 1 cm")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=5, frameon=False, fontsize=9)
    model = Path(run_dir).parent.name
    fig.suptitle(f"Arm torques, policy vs expert: {model}, episode {i} (closest {100 * err_p.min():.1f} cm)",
                 y=0.99, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model}_ep{i:04d}_torques.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("episodes", type=int, nargs="+")
    parser.add_argument("--out", default=str(ROOT / "compare_out" / "torque_plots"),
                        help="folder for the figures (default git-ignored, like the runs)")
    args = parser.parse_args()
    for i in args.episodes:
        print(plot_episode(args.run_dir, i, Path(args.out)))
