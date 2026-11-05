"""Learned v classical detector on the held-out test images (shard 9 of cnn_data.py; PLAN.md step 29, option 2).

Position error (3D, mm) on the images with a usable target, by distance, for both detectors on the same images; the
CNN's detection counts (found / missed usable targets, false detections on images without one); CPU time per image.
The classical detector here is mast_camera.detect_target's maths on the stored RGB-D image (no self-filter: only
images whose target is usable are scored, so nothing is hidden).

Usage (from robotnik_arm_task/): python ../concurrent_arm/cnn_eval.py --model checkpoints/cnn_detector_mast_train.pt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_detector import DetectorNet, evaluate, load_shards, to_input  # noqa: E402
from mast_camera import MastCameraParams, PIXEL_OFFSET_U, PIXEL_OFFSET_V, red_mask, vertical_fov  # noqa: E402

BINS = [0.4, 1.0, 1.5, 2.0, 2.5, 3.0, 3.7]


def classical_cam(rgb, depth, params):
    """Classical estimate in the camera frame (right, down, forward), or None."""
    mask = red_mask(rgb)
    if mask.sum() < params.min_pixels:
        return None
    W, H = params.width, params.height
    fx = (W / 2) / np.tan(params.hfov / 2)
    fy = (H / 2) / np.tan(vertical_fov(params) / 2)
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float64)
    pts = np.stack([z * (u + PIXEL_OFFSET_U - W / 2) / fx, z * (v + PIXEL_OFFSET_V - H / 2) / fy, z], 1)
    s = pts.mean(0)
    return s + (2.0 / 3.0) * params.target_radius * s / np.linalg.norm(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="checkpoints/cnn_detector_mast_train.pt")
    ap.add_argument("--data", default="data_camera_images32")
    ap.add_argument("--shard", type=int, default=9)
    a = ap.parse_args()
    params = MastCameraParams()
    te = load_shards(a.data, [a.shard])
    model = DetectorNet()
    ck = torch.load(a.model, map_location="cpu")
    model.load_state_dict(ck["state"])
    mode = ck.get("range_mode", "direct")
    ev = evaluate(model, te, "cpu", params, mode=mode)
    use = te["usable"]
    print(f"test shard {a.shard}: {ev['n']} images, {ev['usable']} with a usable target (checkpoint epoch {ck['epoch']}, distance mode {mode})")
    print(f"CNN detection: found {ev['tp']}/{ev['usable']} ({100 * ev['tp'] / ev['usable']:.1f}%), missed {ev['fn']}, "
          f"false detections {ev['fp']} on {ev['n'] - ev['usable']} images without a usable target")
    fp_modes = np.bincount(te["mode"][ev["det"] & ~use], minlength=3)
    print(f"  false detections by placement (in view / at the gripper / out of view): {fp_modes.tolist()}")

    # Same images for both: usable targets the CNN found and the classical maths sees
    idx = np.flatnonzero(use & ev["det"])
    cnn_err, cls_err, dist = [], [], []
    for k in idx:
        c = classical_cam(te["rgb"][k], te["depth"][k], params)
        if c is None:
            continue
        cnn_err.append(np.linalg.norm(ev["pts"][k] - te["target_cam"][k]))
        cls_err.append(np.linalg.norm(c - te["target_cam"][k]))
        dist.append(te["range"][k])
    cnn_err, cls_err, dist = (np.array(x) * s for x, s in ((cnn_err, 1000), (cls_err, 1000), (dist, 1)))
    print(f"\n3D error on {len(dist)} usable targets (mm): mean / median / 95th pct / max")
    print(f"  CNN       {cnn_err.mean():6.2f} / {np.median(cnn_err):6.2f} / {np.percentile(cnn_err, 95):6.2f} / {cnn_err.max():7.2f}")
    print(f"  classical {cls_err.mean():6.2f} / {np.median(cls_err):6.2f} / {np.percentile(cls_err, 95):6.2f} / {cls_err.max():7.2f}")
    print("  by distance (m): n, CNN mean / median, classical mean / median")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.any():
            print(f"    {lo:.1f}-{hi:.1f}: {m.sum():4d}  CNN {cnn_err[m].mean():6.2f} / {np.median(cnn_err[m]):6.2f}   "
                  f"classical {cls_err[m].mean():6.2f} / {np.median(cls_err[m]):6.2f}")

    torch.set_num_threads(1)
    x = to_input(te["rgb"][:1], te["depth"][:1])
    with torch.no_grad():
        for _ in range(3):
            model(x)
        t = time.perf_counter()
        for _ in range(20):
            model(x)
    print(f"\nCPU inference, 1 thread: {(time.perf_counter() - t) / 20 * 1000:.1f} ms per image")


if __name__ == "__main__":
    main()
