"""How well can the split arm network's inputs predict the expert's arm torques, per phase? (offline, no simulation)

Same inputs as the arm network (make_features columns 3:27 + the arrived flag), gravity-residual labels, a 3 x 256
ReLU MLP like the final model's arm. Variants add the concurrent expert's hidden state (drive progress, q_final) to
see how much of the drive-phase labels the arm can't see. Train / held-out split by episode.
"""
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, sys.argv[1])            # robotnik_arm_task folder
from bc_data import ARM_INPUTS, arm_gravity, make_features  # noqa: E402

DATA = {"sequential": sys.argv[2], "concurrent": sys.argv[3]}
STEP = 10                                  # every 10th frame (10 ms)
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
torch.manual_seed(SEED)
torch.set_num_threads(8)


def frames(path):
    d = np.load(path)
    ph, done = d["nav_phase"], d["nav_done"]
    keep = np.arange(0, len(ph), STEP)
    x = make_features(d["base_pos"][keep], d["base_bearing"][keep], d["q_mes"][keep, 4:11], d["qd_mes"][keep, 4:11],
                      d["target"], d["ee_pos"][keep])[:, ARM_INPUTS]
    flag = done[keep].astype(np.float32)[:, None]
    y = d["torque_arm"][keep] - arm_gravity(d["q_mes"][keep])
    # 0 align, 1 drive, 2 final turn (not done yet), 3 base done
    part = np.where(done[keep], 3, ph[keep]).astype(int)
    prog = d["arm_progress"][keep][:, None] if "arm_progress" in d.files else np.zeros((len(keep), 1), np.float32)
    qf = np.broadcast_to(d["q_final"], (len(keep), 7)) if "q_final" in d.files else np.zeros((len(keep), 7))
    return np.hstack([x, flag]), y, part, prog, np.sin(qf), np.cos(qf)


def fit(xtr, ytr, xva, yva, epochs=40):
    xm, xs = xtr.mean(0), xtr.std(0) + 1e-6
    ym, ys = ytr.mean(0), ytr.std(0) + 1e-6
    f = lambda a, m, s: torch.tensor((a - m) / s, dtype=torch.float32)
    Xt, Yt, Xv = f(xtr, xm, xs), f(ytr, ym, ys), f(xva, xm, xs)
    net = nn.Sequential(nn.Linear(Xt.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 7))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 2048):
            i = perm[b:b + 2048]
            opt.zero_grad()
            nn.functional.mse_loss(net(Xt[i]), Yt[i]).backward()
            opt.step()
    with torch.no_grad():
        return net(Xv).numpy() * ys + ym


for name, folder in DATA.items():
    rows = [frames(f"{folder}/ep_{i:04d}.npz") for i in range(300)]
    ids = np.random.default_rng(SEED).permutation(300)
    tr, va = ids[:240], ids[240:]
    cat = lambda which, k: np.concatenate([rows[i][k] for i in which])
    X = {s: cat(s_ids, 0) for s, s_ids in (("tr", tr), ("va", va))}
    Y = {s: cat(s_ids, 1) for s, s_ids in (("tr", tr), ("va", va))}
    P = {s: cat(s_ids, 2) for s, s_ids in (("tr", tr), ("va", va))}
    extra = {s: np.hstack([cat(s_ids, 3), cat(s_ids, 4), cat(s_ids, 5)]) for s, s_ids in (("tr", tr), ("va", va))}
    variants = {"arm inputs (as the final model)": (X["tr"], X["va"])}
    if name == "concurrent":
        variants["+ drive progress"] = (np.hstack([X["tr"], extra["tr"][:, :1]]), np.hstack([X["va"], extra["va"][:, :1]]))
        variants["+ progress + q_final"] = (np.hstack([X["tr"], extra["tr"]]), np.hstack([X["va"], extra["va"]]))
    print(f"\n== {name}: {len(X['tr'])} train / {len(X['va'])} held-out frames (every {STEP}th)")
    for vname, (xtr, xva) in variants.items():
        pred = fit(xtr, Y["tr"], xva, Y["va"])
        out = []
        for part, label in ((0, "turn"), (1, "drive"), (2, "final turn"), (3, "parked")):
            m = P["va"] == part
            if m.sum() < 50:
                continue
            err = pred[m] - Y["va"][m]
            r2 = 1 - (err ** 2).sum() / ((Y["va"][m] - Y["va"][m].mean(0)) ** 2).sum()
            out.append(f"{label}: RMSE {np.sqrt((err ** 2).mean()):.3f} N*m, label std {Y['va'][m].std():.3f}, R2 {r2:.3f}")
        print(f"  {vname}\n    " + "\n    ".join(out))
