"""Learned target detector for the mast camera (PLAN.md step 29, option 2): a small CenterNet-style CNN.

RGB-D image (320 x 240) in; out, at half resolution (160 x 120): a heatmap of the target centre, a sub-pixel offset
and a distance channel. The peak gives the pixel; the distance comes, in the default "residual" mode, from the depth
image read at that pixel (the sphere's near surface on the ray through its centre) plus the network's learned
correction to the centre (~ the radius); the camera's intrinsics turn pixel + distance into a 3D point.
"direct" mode (the first model, kept for the report) regressed the whole camera-to-centre distance instead: pixel
error 0.07 px median but distance error 16 mm median (40 mm beyond 2 m), i.e. ~1 % relative precision, the usual
limit of a regressed absolute quantity; the residual mode leaves the metric part to the depth sensor. Direct xyz regression from the whole image would be simpler but typically cm-accurate, which is useless next
to a 1.6 mm classical detector; predicting where in the image and how far is the standard way to keep precision.
The network also learns when a detection is usable (target fully visible, not touching the arm or the image border):
at run time it gets RGB + depth only, no segmentation mask (the classical detector needs the mask as a self-filter).

Training (from robotnik_arm_task/, images from cnn_data.py):
    python ../concurrent_arm/cnn_detector.py --data data_camera_images32 --out checkpoints/cnn_detector.pt
Shards 0-7 train, 8 validation (early stopping), 9 test (reported only).
In the experts / policy runs: ROBOENV_DETECTOR=cnn (and ROBOENV_CNN_DETECTOR=<checkpoint>) makes
camera_expert.py / camera_policy_task.py use CNNDetector.detect instead of mast_camera.detect_target.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FAR = 5.0           # m, depth normalisation (MastCameraParams.far)
RANGE_SCALE = 4.0   # m, range normalisation ("direct" mode)
RESID_SCALE = 0.1   # m, normalisation of the surface-to-centre correction ("residual" mode)
SIGMA = 1.0         # output pixels, heatmap Gaussian


def block(cin, cout, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, padding=dilation, dilation=dilation, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DetectorNet(nn.Module):
    """Encoder-decoder, output stride 2. Channels: 0 heatmap logit, 1-2 sub-pixel offset (u, v), 3 range / 4 m."""

    def __init__(self, c=(16, 32, 64, 64)):
        super().__init__()
        self.e1 = nn.Sequential(block(4, c[0], 2), block(c[0], c[0]))
        self.e2 = nn.Sequential(block(c[0], c[1], 2), block(c[1], c[1]))
        self.e3 = nn.Sequential(block(c[1], c[2], 2), block(c[2], c[2]), block(c[2], c[2], dilation=2))
        self.e4 = nn.Sequential(block(c[2], c[3], 2), block(c[3], c[3], dilation=2))
        self.d3 = block(c[3] + c[2], c[2])
        self.d2 = block(c[2] + c[1], c[1])
        self.d1 = block(c[1] + c[0], 32)
        self.head = nn.Sequential(block(32, 32), nn.Conv2d(32, 4, 1))
        self.head[1].bias.data[0] = -4.6   # heatmap starts near 0.01 everywhere (CenterNet's prior)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        d = self.d3(torch.cat([up(e4, e3), e3], 1))
        d = self.d2(torch.cat([up(d, e2), e2], 1))
        d = self.d1(torch.cat([up(d, e1), e1], 1))
        return self.head(d)


def to_input(rgb, depth):
    """uint8 (N, H, W, 3) + depth (N, H, W) in m -> float tensor (N, 4, H, W)."""
    rgb = torch.as_tensor(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2).float() / 255.0
    dep = torch.as_tensor(np.asarray(depth, dtype=np.float32)).unsqueeze(1).clamp(max=FAR) / FAR
    return torch.cat([rgb, dep], 1)


def out_coords(uv):
    """Input pixel-index coordinates -> output-grid coordinates (output cell j covers input pixels 2j, 2j+1)."""
    return (uv - 0.5) / 2.0


def surface_range(depth, u, v, params):
    """Distance along the ray through input pixel-index coordinates (u, v), from the depth image (bilinear).
    depth: (N, H, W) m along the optical axis; u, v: (N,). Returns (N,) m."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mast_camera import PIXEL_OFFSET_U, PIXEL_OFFSET_V, vertical_fov
    depth = np.asarray(depth, dtype=np.float64)
    n, H, W = depth.shape
    u = np.clip(np.asarray(u, float), 0, W - 1.001)
    v = np.clip(np.asarray(v, float), 0, H - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    a, b = u - u0, v - v0
    ar = np.arange(n)
    z = ((1 - a) * (1 - b) * depth[ar, v0, u0] + a * (1 - b) * depth[ar, v0, u0 + 1]
         + (1 - a) * b * depth[ar, v0 + 1, u0] + a * b * depth[ar, v0 + 1, u0 + 1])
    fx = (W / 2) / np.tan(params.hfov / 2)
    fy = (H / 2) / np.tan(vertical_fov(params) / 2)
    xn, yn = (u + PIXEL_OFFSET_U - W / 2) / fx, (v + PIXEL_OFFSET_V - H / 2) / fy
    return z * np.sqrt(1 + xn ** 2 + yn ** 2)


def make_targets(uv, rng, usable, Ho, Wo, depth=None, params=None, mode="residual"):
    """Heatmap (N, Ho, Wo), peak cell indices (N, 2), offsets (N, 2), distance target (N,): range / RANGE_SCALE
    ("direct") or (range - depth-image range at the label pixel) / RESID_SCALE ("residual")."""
    n = len(uv)
    heat = np.zeros((n, Ho, Wo), np.float32)
    cells = np.zeros((n, 2), np.int64)
    offs = np.zeros((n, 2), np.float32)
    yy, xx = np.mgrid[0:Ho, 0:Wo]
    for k in np.flatnonzero(usable):
        uo, vo = out_coords(uv[k])
        cu, cv = int(np.clip(np.round(uo), 0, Wo - 1)), int(np.clip(np.round(vo), 0, Ho - 1))
        heat[k] = np.exp(-((xx - uo) ** 2 + (yy - vo) ** 2) / (2 * SIGMA ** 2))
        heat[k, cv, cu] = 1.0
        cells[k] = [cv, cu]
        offs[k] = [uo - cu, vo - cv]
    if mode == "direct":
        dist = rng / RANGE_SCALE
    else:
        dist = np.zeros(n)
        k = np.flatnonzero(usable)
        if len(k):
            dist[k] = (rng[k] - surface_range(depth[k], uv[k, 0], uv[k, 1], params)) / RESID_SCALE
    return heat, cells, offs, np.nan_to_num(dist).astype(np.float32)


def focal_loss(logit, heat):
    """CenterNet focal loss on the heatmap, normalised by the number of targets."""
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = heat.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - heat) ** 4 * neg
    return -(pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def decode(out, depth=None, params=None, mode="residual"):
    """Network output (N, 4, Ho, Wo) -> peak probability (N,), input-pixel u, v (N,), range m (N,).
    "residual" mode needs the depth images (N, H, W) and the camera params."""
    n, _, Ho, Wo = out.shape
    p = torch.sigmoid(out[:, 0]).reshape(n, -1)
    score, idx = p.max(1)
    cv, cu = idx // Wo, idx % Wo
    ar = torch.arange(n)
    du, dv = out[ar, 1, cv, cu], out[ar, 2, cv, cu]
    u = 2.0 * (cu.float() + du) + 0.5
    v = 2.0 * (cv.float() + dv) + 0.5
    if mode == "direct":
        rng = out[ar, 3, cv, cu] * RANGE_SCALE
    else:
        surf = surface_range(depth, u.numpy(), v.numpy(), params)
        rng = torch.as_tensor(surf, dtype=torch.float32) + out[ar, 3, cv, cu] * RESID_SCALE
    return score, u, v, rng


def load_shards(folder, ids):
    parts = [np.load(Path(folder) / f"shard_{i:02d}.npz") for i in ids]
    keys = ("rgb", "depth", "usable", "uv", "range", "target_cam", "mode")
    return {k: np.concatenate([d[k] for d in parts]) for k in keys}


def cam_point(u, v, rng, params):
    """Input pixel-index coordinates + range -> point in the camera frame (right, down, forward), m."""
    from mast_camera import PIXEL_OFFSET_U, PIXEL_OFFSET_V, vertical_fov
    W, H = params.width, params.height
    fx = (W / 2) / np.tan(params.hfov / 2)
    fy = (H / 2) / np.tan(vertical_fov(params) / 2)
    d = np.stack([(u + PIXEL_OFFSET_U - W / 2) / fx, (v + PIXEL_OFFSET_V - H / 2) / fy, np.ones_like(u)], -1)
    return rng[..., None] * d / np.linalg.norm(d, axis=-1, keepdims=True)


def evaluate(model, data, device, params, thr=0.5, batch=64, mode="residual"):
    """Detection counts and 3D error (mm) on the samples with a usable target."""
    model.eval()
    scores, us, vs, rs = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(data["rgb"]), batch):
            out = model(to_input(data["rgb"][s:s + batch], data["depth"][s:s + batch]).to(device)).float().cpu()
            sc, u, v, r = decode(out, data["depth"][s:s + batch], params, mode)
            scores.append(sc), us.append(u), vs.append(v), rs.append(r)
    score, u, v, r = (torch.cat(a).numpy() for a in (scores, us, vs, rs))
    det = score > thr
    use = data["usable"]
    pts = cam_point(u, v, r, params)
    err = np.linalg.norm(pts - data["target_cam"], axis=1)
    hit = use & det
    return dict(n=len(use), usable=int(use.sum()), detected=int(det.sum()), tp=int(hit.sum()),
                fp=int((det & ~use).sum()), fn=int((use & ~det).sum()),
                err_mm=err[hit] * 1000, range_true=data["range"][hit], score=score, det=det, pts=pts)


def train(a):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mast_camera import MastCameraParams
    params = MastCameraParams()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tr = load_shards(a.data, range(0, 8))
    va = load_shards(a.data, [8])
    print(f"train {len(tr['rgb'])} ({tr['usable'].sum()} usable), val {len(va['rgb'])} ({va['usable'].sum()}), "
          f"device {device}", flush=True)
    model = DetectorNet().to(device)
    print(f"{sum(p.numel() for p in model.parameters())} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(tr["rgb"]) / a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    Ho, Wo = params.height // 2, params.width // 2
    best, best_epoch, bad = np.inf, 0, 0
    for epoch in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        perm = np.random.permutation(len(tr["rgb"]))
        for s in range(0, len(perm), a.batch):
            b = np.sort(perm[s:s + a.batch])
            x = to_input(tr["rgb"][b], tr["depth"][b]).to(device)
            heat, cells, offs, rng = make_targets(tr["uv"][b], tr["range"][b], tr["usable"][b], Ho, Wo,
                                                  tr["depth"][b], params, a.range_mode)
            heat, cells, offs, rng = (torch.as_tensor(t).to(device) for t in (heat, cells, offs, rng))
            use = torch.as_tensor(tr["usable"][b]).to(device)
            out = model(x)
            loss = focal_loss(out[:, 0], heat)
            if use.any():
                ar = torch.arange(len(b), device=device)[use]
                cv, cu = cells[use, 0], cells[use, 1]
                loss = loss + F.l1_loss(out[ar, 1, cv, cu], offs[use, 0]) + F.l1_loss(out[ar, 2, cv, cu], offs[use, 1])
                loss = loss + (10.0 if a.range_mode == "direct" else 1.0) * F.l1_loss(out[ar, 3, cv, cu], rng[use])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        ev = evaluate(model, va, device, params, mode=a.range_mode)
        # Selection score: mean 3D error on the usable val targets plus a penalty per missed or false detection
        score = float(np.mean(ev["err_mm"])) if ev["tp"] else 1e9
        score += 10.0 * (ev["fn"] + ev["fp"]) / max(ev["usable"], 1)
        print(f"epoch {epoch + 1:2d}: train loss {tot / len(perm):.4f} | val: found {ev['tp']}/{ev['usable']}, "
              f"false {ev['fp']}, error mean {np.mean(ev['err_mm']):.2f} mm median {np.median(ev['err_mm']):.2f} mm "
              f"| {time.time() - t0:.0f} s", flush=True)
        if score < best:
            best, best_epoch, bad = score, epoch + 1, 0
            torch.save({"state": model.state_dict(), "epoch": epoch + 1, "val_score": score,
                        "range_mode": a.range_mode}, a.out)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop: best epoch {best_epoch}", flush=True)
                break
    print(f"saved {a.out} (epoch {best_epoch})", flush=True)


class CNNDetector:
    """Drop-in for mast_camera.detect_target in the experts / policy runs (CPU, one image at a time)."""

    def __init__(self, checkpoint, thr=0.5):
        torch.set_num_threads(1)
        self.model = DetectorNet()
        ck = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(ck["state"])
        self.mode = ck.get("range_mode", "direct")
        self.model.eval()
        self.thr = thr

    def detect(self, client, base_pos, base_ori, params, robot_id=None):
        from mast_camera import Detection, render
        rgb, depth, _, (eye, fwd, up, right) = render(client, base_pos, base_ori, params)   # segmentation unused
        with torch.no_grad():
            score, u, v, r = decode(self.model(to_input(rgb[None], depth[None])), depth[None], params, self.mode)
        if float(score[0]) < self.thr:
            return None
        pc = cam_point(u.numpy(), v.numpy(), r.numpy(), params)[0]   # right, down, forward
        world = eye + pc[0] * right - pc[1] * up + pc[2] * fwd
        return Detection(position=world, pixel=np.array([float(u[0]), float(v[0])]), n_pixels=0,
                         distance=float(r[0]), occluded=False, truncated=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_camera_images32")
    ap.add_argument("--out", default="checkpoints/cnn_detector.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--range-mode", choices=("residual", "direct"), default="residual")
    train(ap.parse_args())
