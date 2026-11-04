"""Mast-mounted RGB-D camera on the mobile base and a colour-based target detector (PLAN.md step 29).

The camera is not in the URDF: its pose is computed every frame from the base pose (full orientation, so chassis
pitch and roll move it too), and the Pinocchio model and joint order stay untouched. Images come from PyBullet's
TinyRenderer, which also works headless.

Assumption: the camera and its mast are weightless and have no collision shape. Nothing is added to the simulation, so
the physics are exactly those of every other expert and dataset (checked 4 Nov: 4 s of driving with random arm torques,
with and without rendering every 0.1 s, gave bit-identical base and joint states; robot mass 60.562 kg both times).
A real mast (~1-2 kg at 1 m) would raise the centre of mass a little; not modelled.

Mounting (probe renders, 4 Nov): on a mast behind the arm, 1.0 m above the floor, tilted 15 deg down. Behind and
above, so the arm, which reaches forward and below, blocks the view only at the end of the reach; 0.2 m to the right
of the centre line, because the arm's travel pose puts the hand (0.71, 0, 0.76) in the middle of a centred camera's
image, which is where the target sits once the base faces it. With an 80 deg x 64 deg field of view it sees every
target height (0.2-1.1 m) from 3 m away down to the parking distance.

Detection: red pixels -> 3D points from the depth image -> sphere centre. A detection is thrown away when the blob
touches the image border or the robot's own body in the image (the robot knows its own body; the segmentation mask
stands in for the self-filter a real robot would run on its known geometry): a partly hidden blob biases the centre.
"""
from dataclasses import dataclass

import numpy as np
import pybullet as p

# Image coordinates sampled by pixel (column u, row v) of a TinyRenderer image: x = u + 0, y = v + 1 (see detect_target)
PIXEL_OFFSET_U = 0.0
PIXEL_OFFSET_V = 1.0

@dataclass
class MastCameraParams:
    """Intrinsics, mounting and detection settings of the mast camera."""
    width: int = 320
    height: int = 240
    hfov: float = np.deg2rad(80.0)      # rad, horizontal field of view (vertical follows from the aspect: ~64 deg)
    near: float = 0.05                  # m
    far: float = 5.0                    # m, max detection range (targets start 1-3 m from the base)
    # Camera position in the base frame (x forward, y left, z up from the floor): behind the arm, to its right
    mount_xyz: tuple = (-0.10, -0.20, 1.00)
    pitch: float = np.deg2rad(-15.0)    # rad, tilt of the optical axis (negative = looking down)
    min_pixels: int = 6                 # blobs smaller than this are "not seen"
    period: float = 0.1                 # s, render rate (the control loop runs at 1 kHz)
    target_radius: float = 0.03         # m, radius of the red target sphere
    margin_px: int = 2                  # robot pixels this close to the blob mark it as (partly) hidden


def parse_mount(text):
    """'x,y,z,pitch_deg' (ROBOENV_CAMERA_MOUNT) -> (mount_xyz, pitch in rad)."""
    v = [float(s) for s in text.split(",")]
    return tuple(v[:3]), np.deg2rad(v[3])


def vertical_fov(params):
    return 2 * np.arctan(np.tan(params.hfov / 2) * params.height / params.width)


def camera_pose(base_pos, base_ori, params):
    """World position of the camera, unit optical axis, up and right vectors. base_ori: PyBullet [x, y, z, w]."""
    R = np.reshape(p.getMatrixFromQuaternion(list(base_ori)), (3, 3))
    eye = np.asarray(base_pos, dtype=float) + R @ np.asarray(params.mount_xyz, dtype=float)
    fwd = R @ np.array([np.cos(params.pitch), 0.0, np.sin(params.pitch)])
    up = R @ np.array([-np.sin(params.pitch), 0.0, np.cos(params.pitch)])
    right = np.cross(fwd, up)
    return eye, fwd, up, right


def render(client, base_pos, base_ori, params):
    """Render the camera. Returns (rgb HxWx3 uint8, depth HxW along the optical axis (m), segmentation HxW, pose)."""
    W, H = params.width, params.height
    eye, fwd, up, right = camera_pose(base_pos, base_ori, params)
    view = client.computeViewMatrix(eye.tolist(), (eye + fwd).tolist(), up.tolist())
    proj = client.computeProjectionMatrixFOV(np.rad2deg(vertical_fov(params)), W / H, params.near, params.far)
    _, _, rgba, depth_buf, seg = client.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.reshape(np.asarray(rgba, dtype=np.uint8), (H, W, 4))[..., :3]
    # The OpenGL depth buffer is non-linear: convert it to metric depth along the optical axis
    d = np.reshape(np.asarray(depth_buf, dtype=float), (H, W))
    depth = params.far * params.near / (params.far - (params.far - params.near) * d)
    seg = np.reshape(np.asarray(seg), (H, W))
    return rgb, depth, seg, (eye, fwd, up, right)


def red_mask(rgb):
    """Pixels that look like the red target sphere."""
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    return (r > 120) & (g < 60) & (b < 60)


@dataclass
class Detection:
    position: np.ndarray    # estimated sphere centre, world frame (m)
    pixel: np.ndarray       # blob centroid [u, v] (px)
    n_pixels: int
    distance: float         # m, camera to the estimated centre
    occluded: bool          # robot pixels next to the blob
    truncated: bool         # blob touches the image border

    @property
    def clean(self):
        return not (self.occluded or self.truncated)


def detect_target(client, base_pos, base_ori, params, robot_id):
    """Render, segment the red target and turn it into a world position.

    Returns a Detection (check .clean before using it), or None when the target is not in the image (outside the
    field of view, beyond `far`, fully hidden or too small).
    """
    rgb, depth, seg, (eye, fwd, up, right) = render(client, base_pos, base_ori, params)
    mask = red_mask(rgb)
    n = int(mask.sum())
    if n < params.min_pixels:
        return None
    v, u = np.nonzero(mask)
    z = depth[v, u]
    if np.median(z) >= 0.98 * params.far:
        return None

    W, H = params.width, params.height
    fx = (W / 2) / np.tan(params.hfov / 2)
    fy = (H / 2) / np.tan(vertical_fov(params) / 2)
    # TinyRenderer samples column u at x = u and row v at y = v + 1 (not at pixel centres, +0.5): measured on 33
    # targets 1-3 m away against their exact projection (offsets -0.04 / -0.96 px, spread 0.2 / 0.1 px).
    # With +0.5 instead, the estimate was off sideways by ~1 px, 12 mm at 3 m (PLAN.md step 29)
    x_right = (u + PIXEL_OFFSET_U - W / 2) / fx
    y_down = (v + PIXEL_OFFSET_V - H / 2) / fy
    # One 3D point per pixel on the visible surface; their mean sits 2r/3 in front of the centre along the
    # viewing ray (mean depth of a hemisphere seen along its axis, pixels sampling its projected area)
    points = eye + z[:, None] * (fwd + x_right[:, None] * right - y_down[:, None] * up)
    surface = points.mean(axis=0)
    ray = (surface - eye) / np.linalg.norm(surface - eye)
    centre = surface + (2.0 / 3.0) * params.target_radius * ray

    m = params.margin_px
    v0, v1, u0, u1 = max(v.min() - m, 0), min(v.max() + m + 1, H), max(u.min() - m, 0), min(u.max() + m + 1, W)
    occluded = bool(np.any(seg[v0:v1, u0:u1] == robot_id))
    truncated = bool(u.min() == 0 or v.min() == 0 or u.max() == W - 1 or v.max() == H - 1)
    return Detection(position=centre, pixel=np.array([u.mean() + PIXEL_OFFSET_U, v.mean() + PIXEL_OFFSET_V]), n_pixels=n,
                     distance=float(np.linalg.norm(centre - eye)), occluded=occluded, truncated=truncated)


@dataclass
class TargetEstimator:
    """Fuses clean detections into one target estimate.

    Weighted mean with weight 1 / distance^4: a pixel's error grows with distance, and depth error with its
    square, so close views dominate as the base approaches. refine=False keeps the first clean detection only
    (the "seen once" baseline).
    """
    refine: bool = True
    weight_sum: float = 0.0
    weighted_sum: np.ndarray = None
    n_used: int = 0
    first_seen_time: float = None

    def update(self, detection, current_time):
        """Adds a detection if it is clean; returns True when the estimate changed."""
        if detection is None or not detection.clean:
            return False
        if self.found and not self.refine:
            return False
        w = 1.0 / max(detection.distance, 0.1) ** 4
        if self.weighted_sum is None:
            self.weighted_sum = np.zeros(3)
            self.first_seen_time = current_time
        self.weighted_sum += w * detection.position
        self.weight_sum += w
        self.n_used += 1
        return True

    @property
    def found(self):
        return self.weight_sum > 0

    @property
    def estimate(self):
        return self.weighted_sum / self.weight_sum if self.found else None
