"""Camera expert v concurrent expert, from the saved episode files (PLAN.md step 29). No simulation.

Usage (from robotnik_arm_task/): python ../concurrent_arm/analyse_camera.py data_camera [data_camera_once ...]
  --ref data_concurrent   the concurrent expert on the same targets (same --seed and episode numbers)

Per folder: found / reached (the stop rule's own 5 mm, measured to the estimate) / true error ever and held within
5 mm and 1 cm, the estimate's error at the first sighting and at the end, search and reach times, paired with the
reference on the same episodes. "Held within 1 cm" = 0.5 s in a row within 1 cm (true error), as in PLAN.md.
"""
import argparse
import re
from pathlib import Path

import numpy as np

FRONT_HALF_ANGLE = np.deg2rad(40.0)   # targets within the camera's half field of view at t = 0


def reached_time(message):
    m = re.search(r"at t=([0-9.]+)s", message)
    return float(m.group(1)) if message.startswith("Target reached") and m else np.nan


def held(err, thr, dt=0.001, hold=0.5):
    """True if err stays below thr for `hold` seconds in a row."""
    n = int(round(hold / dt))
    run = 0
    for below in err < thr:
        run = run + 1 if below else 0
        if run >= n:
            return True
    return False


def episode_stats(path):
    with np.load(path) as d:
        msg = str(d["stop_message"])
        err = np.linalg.norm(d["ee_pos"] - d["target"][None, :].astype(np.float32), axis=1)
        out = {
            "msg": msg,
            "reached": msg.startswith("Target reached"),
            "t_reach": reached_time(msg),
            "final_err": float(d["final_error"]),
            "ever_5mm": bool(np.any(err < 0.005)),
            "ever_1cm": bool(np.any(err < 0.01)),
            "held_1cm": held(err, 0.01),
            "bearing": float(np.arctan2(d["target"][1], d["target"][0])),
        }
        if "target_est" in d.files:
            est = d["target_est"]
            seen = np.flatnonzero(~np.isnan(est[:, 0]))
            out["found"] = not msg.startswith("Target not found")
            out["t_seen"] = float(d["first_seen_time"])
            out["turned"] = float(np.rad2deg(d["search_turned"]))
            out["est_err_first"] = float(np.linalg.norm(est[seen[0]] - d["target"])) if len(seen) else np.nan
            out["est_err_final"] = float(d["est_error"])
            out["solves"] = int(d["n_q_final_solves"])
            out["n_det"] = int(d["n_detections_used"])
            status = d["cam_status"]
            out["hidden_frac"] = float(np.mean(status == 2)) if len(status) else np.nan
    return out


def load(folder):
    return {int(p.stem.split("_")[1]): episode_stats(p) for p in sorted(Path(folder).glob("ep_*.npz"))}


def pct(x):
    return f"{np.sum(x)}/{len(x)} ({100 * np.mean(x):.1f}%)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--ref", default="data_concurrent")
    args = ap.parse_args()
    ref = load(args.ref)
    print(f"reference {args.ref}: {len(ref)} episodes, reached {pct([r['reached'] for r in ref.values()])}, "
          f"ever 1 cm {pct([r['ever_1cm'] for r in ref.values()])}, held 1 cm {pct([r['held_1cm'] for r in ref.values()])}, "
          f"time to reach mean {np.nanmean([r['t_reach'] for r in ref.values()]):.2f} s")
    for folder in args.folders:
        eps = load(folder)
        ids = sorted(set(eps) & set(ref))
        E = [eps[i] for i in ids]
        R = [ref[i] for i in ids]
        print(f"\n== {folder}: {len(eps)} episodes ({len(ids)} paired with {args.ref})")
        print(f"  found              {pct([e['found'] for e in E])}")
        print(f"  reached (5 mm to estimate, stop rule) {pct([e['reached'] for e in E])}")
        print(f"  true: ever 5 mm    {pct([e['ever_5mm'] for e in E])}")
        print(f"  true: ever 1 cm    {pct([e['ever_1cm'] for e in E])}")
        print(f"  true: held 1 cm    {pct([e['held_1cm'] for e in E])}")
        fe = np.array([e["final_err"] for e in E])
        print(f"  true final error   median {np.median(fe)*1000:.1f} mm, mean {fe.mean()*1000:.1f} mm, max {fe.max()*1000:.1f} mm")
        ef = np.array([e["est_err_first"] for e in E])
        el = np.array([e["est_err_final"] for e in E])
        print(f"  estimate error     first sighting median {np.nanmedian(ef)*1000:.1f} mm (max {np.nanmax(ef)*1000:.1f}); "
              f"final median {np.nanmedian(el)*1000:.1f} mm (max {np.nanmax(el)*1000:.1f})")
        print(f"  q_final solves     {np.bincount([e['solves'] for e in E]).tolist()} (count of episodes with 0, 1, 2.. solves)")
        front = np.array([abs(e["bearing"]) < FRONT_HALF_ANGLE for e in E])
        ts = np.array([e["t_seen"] for e in E])
        print(f"  first sighting     front targets ({front.sum()}): mean {np.nanmean(ts[front]):.2f} s; "
              f"others ({(~front).sum()}): mean {np.nanmean(ts[~front]):.2f} s, max {np.nanmax(ts[~front]):.2f} s")
        both = [(e["t_reach"], r["t_reach"]) for e, r in zip(E, R) if e["reached"] and r["reached"]]
        dt = np.array([a - b for a, b in both])
        print(f"  time to reach      {np.mean([a for a, _ in both]):.2f} s v reference {np.mean([b for _, b in both]):.2f} s "
              f"on {len(both)} reached by both: +{dt.mean():.2f} s mean")
        # The search always turns counter-clockwise: targets on the right (bearing < 0) are found the long way round
        side = np.array([("front" if abs(e["bearing"]) < FRONT_HALF_ANGLE else "left" if e["bearing"] > 0 else "right")
                         for e, r in zip(E, R) if e["reached"] and r["reached"]])
        for s in ("front", "left", "right"):
            if np.any(side == s):
                print(f"    {s:5s} ({np.sum(side == s):3d}): +{dt[side == s].mean():.2f} s mean, "
                      f"max +{dt[side == s].max():.2f} s, min {dt[side == s].min():+.2f} s")
        bad = [(i, e) for i, e in zip(ids, E) if not e["held_1cm"] or not e["reached"]]
        for i, e in bad[:15]:
            print(f"    ep {i}: {e['msg'][:70]} | true {e['final_err']*1000:.1f} mm, est err {e['est_err_final']*1000:.1f} mm")


if __name__ == "__main__":
    main()
