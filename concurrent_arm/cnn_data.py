"""Training images for the learned target detector (PLAN.md step 29, option 2): rendered mast-camera views with labels.

Each sample is one mast-camera RGB-D image (320 x 240, the camera of mast_camera.py) of a scene set up directly (no
simulation steps): the robot at a random base pose with an arm configuration taken from real concurrent-expert frames
(data_concurrent: start pose, mid-drive, reaching), and the red target placed
- 55% anywhere in or just outside the field of view, 0.4-3.6 m from the camera (heights 0.1-1.3 m),
- 25% within 15 cm of the gripper (the end of a reach, where the arm hides part of it),
- 20% out of view (behind or beside the camera): nothing to find.
Labels (privileged, training only): the exact image position of the target centre (pixel-index coordinates of
TinyRenderer, see mast_camera.PIXEL_OFFSET_*), its distance from the camera, and "usable" = the classical detector's
rule for a clean detection (>= min_pixels red pixels, not touching the robot's own body or the image border), computed
with the segmentation mask. At run time the network gets only RGB + depth.

Usage (from robotnik_arm_task/): python ../concurrent_arm/cnn_data.py --n 12000 --out data_camera_images
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARM_JOINTS = list(range(9, 16))       # PyBullet joint ids of robot_j2s7s300_joint_1..7
FINGER_JOINTS = list(range(17, 23))   # finger_1, finger_tip_1, ... finger_tip_3
EE_LINK = 16                          # robot_j2s7s300_end_effector
FINGER_ANGLE = 0.1


def arm_pool(data_dir, n_episodes=80, every=40, seed=0):
    """Arm configurations from concurrent-expert episodes: every `every`-th frame of n_episodes random episodes."""
    rng = np.random.default_rng(seed)
    files = sorted(Path(data_dir).glob("ep_*.npz"))
    qs = []
    for f in rng.choice(files, size=min(n_episodes, len(files)), replace=False):
        with np.load(f) as d:
            qs.append(d["q_mes"][::every, 4:11])
    return np.concatenate(qs).astype(np.float64)


def generate(args):
    """Worker: n samples into one shard file."""
    worker, n, seed, arm_qs, out = args
    os.chdir(HERE)
    sys.path.insert(0, str(HERE))
    import pybullet as p
    from simulation_and_control import pb
    from mast_camera import (MastCameraParams, PIXEL_OFFSET_U, PIXEL_OFFSET_V, camera_pose, red_mask, render,
                             vertical_fov)

    sim = pb.SimInterface("robotnik_arm.json", conf_file_path_ext=str(HERE), use_gui=False)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    params = MastCameraParams()
    W, H = params.width, params.height
    fx = (W / 2) / np.tan(params.hfov / 2)
    fy = (H / 2) / np.tan(vertical_fov(params) / 2)
    vs = client.createVisualShape(p.GEOM_SPHERE, radius=params.target_radius, rgbaColor=[1.0, 0.0, 0.0, 0.85])
    body = client.createMultiBody(baseMass=0, baseVisualShapeIndex=vs, basePosition=[1, 0, 0.5])
    for j in FINGER_JOINTS:
        client.resetJointState(rid, j, FINGER_ANGLE)
    rng = np.random.default_rng(seed)

    rgb_all = np.zeros((n, H, W, 3), np.uint8)
    depth_all = np.zeros((n, H, W), np.float32)   # float16 would quantise ~4 mm beyond 2 m
    usable = np.zeros(n, bool)
    uv = np.full((n, 2), np.nan, np.float32)        # pixel-index coordinates of the centre's projection
    rng_lab = np.full(n, np.nan, np.float32)        # m, camera to target centre
    target_cam = np.zeros((n, 3), np.float32)       # centre in the camera frame (right, down, forward)
    mode_all = np.zeros(n, np.int8)
    for k in range(n):
        # Robot: random base pose (yaw anywhere, small chassis tilt), arm from the pool
        yaw = rng.uniform(-np.pi, np.pi)
        roll, pitch = rng.normal(0, np.deg2rad(0.7), 2)
        base_pos = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), 0.0])
        base_ori = np.array(p.getQuaternionFromEuler([roll, pitch, yaw]))
        client.resetBasePositionAndOrientation(rid, base_pos.tolist(), base_ori.tolist())
        q = arm_qs[rng.integers(len(arm_qs))]
        for j, a in zip(ARM_JOINTS, q):
            client.resetJointState(rid, j, float(a))
        eye, fwd, up, right = camera_pose(base_pos, base_ori, params)

        mode = rng.choice(3, p=[0.55, 0.25, 0.20])
        for _ in range(200):
            if mode == 0:     # in or just outside the view
                x = rng.uniform(-0.1 * W, 1.1 * W)
                y = rng.uniform(-0.1 * H, 1.1 * H)
                r = rng.uniform(0.4, 3.6)
                d = fwd + (x - W / 2) / fx * right - (y - H / 2) / fy * up
                tgt = eye + r * d / np.linalg.norm(d)
            elif mode == 1:   # next to the gripper
                ee = np.array(client.getLinkState(rid, EE_LINK)[0])
                off = rng.normal(0, 1, 3)
                tgt = ee + off / np.linalg.norm(off) * rng.uniform(0.0, 0.15)
            else:             # out of view: behind or beside the camera
                ang = yaw + rng.uniform(np.deg2rad(60), np.deg2rad(300))
                r = rng.uniform(0.6, 3.6)
                tgt = np.array([eye[0] + r * np.cos(ang), eye[1] + r * np.sin(ang), 0.0])
                tgt[2] = rng.uniform(0.1, 1.3)
            if 0.1 <= tgt[2] <= 1.3:
                break
        client.resetBasePositionAndOrientation(body, tgt.tolist(), [0, 0, 0, 1])

        rgb, depth, seg, _ = render(client, base_pos, base_ori, params)
        rel = tgt - eye
        zc = rel @ fwd
        mask = red_mask(rgb)
        npx = int(mask.sum())
        ok = False
        if zc > params.near and npx >= params.min_pixels:
            vv, uu = np.nonzero(mask)
            m = params.margin_px
            v0, v1 = max(vv.min() - m, 0), min(vv.max() + m + 1, H)
            u0, u1 = max(uu.min() - m, 0), min(uu.max() + m + 1, W)
            occluded = np.any(seg[v0:v1, u0:u1] == rid)
            truncated = uu.min() == 0 or vv.min() == 0 or uu.max() == W - 1 or vv.max() == H - 1
            ok = not (occluded or truncated)
        if zc > params.near:
            uv[k] = [W / 2 + fx * (rel @ right) / zc - PIXEL_OFFSET_U, H / 2 - fy * (rel @ up) / zc - PIXEL_OFFSET_V]
            rng_lab[k] = np.linalg.norm(rel)
            target_cam[k] = [rel @ right, -(rel @ up), zc]
        rgb_all[k], depth_all[k], usable[k], mode_all[k] = rgb, depth.astype(np.float32), ok, mode
    np.savez(out, rgb=rgb_all, depth=depth_all, usable=usable, uv=uv, range=rng_lab, target_cam=target_cam,
             mode=mode_all)
    client.disconnect()
    return out, int(usable.sum()), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--out", default="data_camera_images")
    ap.add_argument("--arm-data", default="data_concurrent")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    arm_qs = arm_pool(a.arm_data, seed=a.seed)
    print(f"{len(arm_qs)} arm configurations from {a.arm_data}", flush=True)
    per = int(np.ceil(a.n / a.workers))
    jobs = [(w, per, a.seed * 1000 + w + 1, arm_qs, str(out / f"shard_{w:02d}.npz")) for w in range(a.workers)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for path, n_ok, n in pool.map(generate, jobs):
            print(f"  {Path(path).name}: {n} images, {n_ok} with a usable target", flush=True)


if __name__ == "__main__":
    main()
