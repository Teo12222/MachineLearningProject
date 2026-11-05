"""Closed-loop diagnostics of a compare.py --test-set run: where the policy differs from the expert (PLAN.md step 14).

compare.py reports success and final error. While policies reach no targets those barely move, so this script reads
the saved policy runs and the stored expert runs and measures the failure itself: how late the turn -> drive switch
comes, how close the base gets to its parking spot, how close the end-effector gets to the target.

With --model the summary also goes into results/<model>.csv as quick_* columns (full_* with --full), with the machine
that ran the simulations, since runs differ slightly between machines. compare.py's sim_* columns stay full-test only.

Usage (from the robotnik_arm_task folder, after compare.py --test-set --quick --model X --out-dir compare_out/X):
    pixi run python analyse_temporal.py --out-dir compare_out/X
    pixi run python analyse_temporal.py --out-dir compare_out/X --full   # a full --test-set run instead of --quick
    pixi run python analyse_temporal.py --out-dir compare_out/X --model X   # also record the summary in results/X.csv
"""
import argparse
import platform
from pathlib import Path

import numpy as np

from compare import load_npz, outcome
from results import update_results

# The same drive-start rule for policy and expert (the policy has no phases). Above the lowest speeds the expert
# creeps at near a parking spot, and sustained, so a brief output spike while turning doesn't count.
WHEEL_RADIUS = 0.11     # m, BaseControllerParams.wheel_radius
DRIVE_SPEED = 0.1       # m/s, forward speed that counts as driving
DRIVE_SUSTAIN = 50      # frames (ms) it has to last
WHEEL_CLIP = 3.0        # rad/s, the policy's clip on the wheel commands
PARKED_WITHIN = 0.15    # m, "got to the parking spot"
TORQUE_CLIP = 30.0      # N*m, the policy's clip on the arm torques
ARM_DEV_TIME = 2.0      # s, when the arm is compared with the expert's (before the first drive in most episodes)
OFF_FLOOR = 0.3         # m, base height moved this far from its start: the physics has broken down


def drive_start(d):
    """Time (s) the base starts driving forward, nan if it never does."""
    speed = WHEEL_RADIUS * d["wheel_cmds"].mean(axis=1)
    driving = np.convolve(speed > DRIVE_SPEED, np.ones(DRIVE_SUSTAIN, dtype=int), mode="valid") == DRIVE_SUSTAIN
    return float(d["time"][np.argmax(driving)]) if driving.any() else np.nan


def run_metrics(policy, expert):
    """Numbers of one test episode; the parking spot and target come from the expert's file."""
    spot_dist = np.linalg.norm(policy["base_pos"][:, :2] - expert["base_goal"][:2], axis=1)
    spot_frame = int(np.argmin(spot_dist))
    at_clip = (np.abs(policy["torque_arm"]) >= TORQUE_CLIP - 1e-3).any(axis=1)
    i, j = np.searchsorted(policy["time"], ARM_DEV_TIME), np.searchsorted(expert["time"], ARM_DEV_TIME)
    return {
        "reached": outcome(policy) == "reached",
        "final_err": float(policy["final_error"]),
        "drive_start": drive_start(policy),
        "expert_drive_start": drive_start(expert),
        "spot_dist": float(spot_dist[spot_frame]),
        "spot_time": float(policy["time"][spot_frame]),
        "ee_dist": float(np.linalg.norm(policy["ee_pos"] - expert["target"], axis=1).min()),
        "wheels_at_clip": float((np.abs(policy["wheel_cmds"]) >= WHEEL_CLIP - 1e-3).any(axis=1).mean()),
        "torque_clip_time": float(policy["time"][np.argmax(at_clip)]) if at_clip.any() else np.nan,
        "arm_dev": float(np.abs(policy["q_mes"][i, 4:11] - expert["q_mes"][j, 4:11]).mean()),
        "off_floor": bool((np.abs(policy["base_pos"][:, 2] - policy["base_pos"][0, 2]) > OFF_FLOOR).any()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="compare_out", help="the --out-dir given to compare.py")
    parser.add_argument("--full", action="store_true", help="read a full --test-set run instead of --quick")
    parser.add_argument("--data-dir", default="data_course_expert", help="episode folder, relative to this script")
    parser.add_argument("--model", help="also write the summary to results/<model>.csv")
    args = parser.parse_args()

    run_dir = Path(args.out_dir).resolve() / ("test_set" if args.full else "test_set_quick")
    data_dir = Path(__file__).resolve().parent / args.data_dir
    files = sorted(run_dir.glob("ep_*.npz"))
    if not files:
        parser.error(f"no policy runs in {run_dir}")

    print(f"{'episode':>8} {'drive start (s)':>16} {'expert (s)':>11} {'closest spot (m)':>17} {'at (s)':>7} "
          f"{'closest EE (m)':>15} {'final err (m)':>14}  outcome")
    rows = []
    for f in files:
        policy = load_npz(f)
        row = run_metrics(policy, load_npz(data_dir / f.name))
        rows.append(row)
        print(f"{f.stem[3:]:>8} {row['drive_start']:16.2f} {row['expert_drive_start']:11.2f} {row['spot_dist']:17.3f} "
              f"{row['spot_time']:7.1f} {row['ee_dist']:15.3f} {row['final_err']:14.3f}  {outcome(policy)}")

    def col(name):
        return np.array([r[name] for r in rows])

    n = len(rows)
    delay = col("drive_start") - col("expert_drive_start")
    parked = col("spot_dist") < PARKED_WITHIN
    print(f"\nReached: {int(col('reached').sum())}/{n}, final error median {np.median(col('final_err')):.3f} m")
    print(f"Drive start: policy {np.nanmedian(col('drive_start')):.2f} s, expert "
          f"{np.nanmedian(col('expert_drive_start')):.2f} s (medians); delay median {np.nanmedian(delay):.2f} s, "
          f"policy never drove in {int(np.isnan(col('drive_start')).sum())}/{n}")
    print(f"Closest to the parking spot: median {np.median(col('spot_dist')):.3f} m; within {PARKED_WITHIN} m in "
          f"{int(parked.sum())}/{n}" + (f", at a median of {np.median(col('spot_time')[parked]):.1f} s" if parked.any()
                                        else ""))
    print(f"Closest EE error: median {np.median(col('ee_dist')):.3f} m, best {col('ee_dist').min():.3f} m")
    print(f"Wheels at the clip: {100 * np.mean(col('wheels_at_clip')):.1f}% of frames (mean over episodes)")
    print(f"Arm: at the torque clip first at {np.nanmedian(col('torque_clip_time')):.2f} s, {ARM_DEV_TIME:.0f} s from the "
          f"start {np.median(col('arm_dev')):.3f} rad from the expert (medians); base off the floor in "
          f"{int(col('off_floor').sum())}/{n}")

    if args.model:
        prefix = "full" if args.full else "quick"
        update_results(args.model, **{
            f"{prefix}_episodes": n, f"{prefix}_machine": f"{platform.system()}-{platform.machine()}",
            f"{prefix}_reached": int(col("reached").sum()),
            f"{prefix}_final_err_median": float(np.median(col("final_err"))),
            f"{prefix}_drive_delay_median": float(np.nanmedian(delay)),
            f"{prefix}_never_drove": int(np.isnan(col("drive_start")).sum()),
            f"{prefix}_spot_dist_median": float(np.median(col("spot_dist"))),
            f"{prefix}_within_spot": int(parked.sum()),
            f"{prefix}_ee_dist_median": float(np.median(col("ee_dist"))),
            f"{prefix}_ee_dist_best": float(col("ee_dist").min()),
            f"{prefix}_wheels_at_clip": float(np.mean(col("wheels_at_clip"))),
            f"{prefix}_torque_clip_time_median": float(np.nanmedian(col("torque_clip_time"))),
            f"{prefix}_arm_dev_median": float(np.median(col("arm_dev"))),
            f"{prefix}_off_floor": int(col("off_floor").sum()),
        })
        print(f"Summary written to results/{args.model}.csv")


if __name__ == "__main__":
    main()
