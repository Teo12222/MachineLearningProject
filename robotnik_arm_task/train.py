"""Train the baseline MLP on the expert episodes and save the best checkpoint (lowest validation loss).

Afterwards the saved checkpoint is evaluated on the test episodes: per-episode and per-output errors are printed
and plotted (figures saved next to the checkpoint). Skip this with --no-test.

Usage (from the robotnik_arm_task folder):
    pixi run python train.py               # train, then test
    pixi run python train.py --no-test     # train only
    pixi run python train.py --test-only   # test the saved checkpoint, no training (checkpoint is not changed)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + numpy each ship an OpenMP runtime on Windows

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bc_data import split_episodes, load_episode, load_episodes
from bc_model import MLP, save_model, load_model

# Paths are relative to this file, so the script works from any working directory
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_main"
CHECKPOINT = SCRIPT_DIR / "checkpoints" / "mlp_baseline.pt"
EPOCHS = 20
BATCH = 2048
LR = 1e-3

OUTPUT_NAMES = ["wheel FR", "wheel FL", "wheel BL", "wheel BR"] + [f"J{j}" for j in range(1, 8)]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--no-test", action="store_true", help="skip the evaluation on the test episodes")
parser.add_argument("--test-only", action="store_true", help="skip training and only evaluate the saved checkpoint")
args = parser.parse_args()
if args.no_test and args.test_only:
    parser.error("--no-test and --test-only together would do nothing")

torch.manual_seed(0)
train_ids, val_ids, test_ids = split_episodes(DATA_DIR)

if not args.test_only:
    x_tr, y_tr = load_episodes(DATA_DIR, train_ids)
    x_va, y_va = load_episodes(DATA_DIR, val_ids)
    print(f"train frames {len(x_tr)}, val frames {len(x_va)}")

    model = MLP(x_tr.shape[1], y_tr.shape[1])
    model.set_normalisation(x_tr, y_tr)

    # Normalise once up front, so the training loop only indexes tensors
    xt = (torch.from_numpy(x_tr) - model.x_mean) / model.x_std
    yt = (torch.from_numpy(y_tr) - model.y_mean) / model.y_std
    xv = (torch.from_numpy(x_va) - model.x_mean) / model.x_std
    yv = (torch.from_numpy(y_va) - model.y_mean) / model.y_std

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    CHECKPOINT.parent.mkdir(exist_ok=True)
    best_val = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), BATCH):
            idx = perm[i:i + BATCH]
            loss = loss_fn(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        print(f"epoch {epoch + 1:2d}  train {loss.item():.4f}  val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            save_model(model, CHECKPOINT)
            print(f"   saved {CHECKPOINT}")
elif not CHECKPOINT.exists():
    parser.error(f"--test-only needs a trained checkpoint, but {CHECKPOINT} does not exist")

if args.no_test:
    raise SystemExit

# ------------------------------------------------------------------------------
# Test evaluation (offline): how closely the model reproduces the expert's commands on unseen episodes
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Evaluate the saved checkpoint (best validation epoch), not the weights left after the last epoch
best = load_model(CHECKPOINT)

wheel_rmse, torque_rmse = [], []
sq_err_sum, n_frames = np.zeros(len(OUTPUT_NAMES)), 0
print(f"\nTest set: {len(test_ids)} episodes, RMSE between model and expert commands")
print(f"{'episode':>8} {'wheels (rad/s)':>15} {'torques (N*m)':>14}")
for i in test_ids:
    x, y = load_episode(Path(DATA_DIR) / f"ep_{i:04d}.npz")
    err = best.predict(x) - y
    wheel_rmse.append(np.sqrt(np.mean(err[:, :4] ** 2)))
    torque_rmse.append(np.sqrt(np.mean(err[:, 4:] ** 2)))
    sq_err_sum += (err ** 2).sum(axis=0)
    n_frames += len(err)
    print(f"{i:8d} {wheel_rmse[-1]:15.4f} {torque_rmse[-1]:14.4f}")
wheel_rmse, torque_rmse = np.array(wheel_rmse), np.array(torque_rmse)
# Per-output RMSE over all test frames pooled together
channel_rmse = np.sqrt(sq_err_sum / n_frames)

print(f"{'average':>8} {wheel_rmse.mean():15.4f} {torque_rmse.mean():14.4f}   (mean of the per-episode values)")
print("\nPer-output RMSE over all test frames:")
for name, value in zip(OUTPUT_NAMES, channel_rmse):
    unit = "rad/s" if name.startswith("wheel") else "N*m"
    print(f"  {name:<9}{value:8.4f} {unit}")

# 1. Error per test episode (wheels and torques have different units, so one plot each)
labels = [str(i) for i in test_ids]
fig1, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
for ax, values, title, unit in ((axes[0], wheel_rmse, "Wheel command error", "rad/s"),
                                (axes[1], torque_rmse, "Arm torque error", "N*m")):
    ax.bar(labels, values, color="tab:blue")
    ax.axhline(values.mean(), color="gray", linestyle="--", linewidth=1)
    ax.text(len(labels) - 0.5, values.mean(), f" mean {values.mean():.3f}", va="bottom", ha="right", color="dimgray")
    ax.set_ylabel(f"RMSE ({unit})")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
axes[1].set_xlabel("Test episode")
axes[1].tick_params(axis="x", labelrotation=90, labelsize=7)
fig1.suptitle(f"Test error per episode ({CHECKPOINT.name})")
fig1.tight_layout()
fig1.savefig(CHECKPOINT.with_name(CHECKPOINT.stem + "_test_per_episode.png"), dpi=120)

# 2. Error per output, over all test frames
fig2, (ax_w, ax_t) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [4, 7]})
ax_w.bar(OUTPUT_NAMES[:4], channel_rmse[:4], color="tab:blue")
ax_w.set_ylabel("RMSE (rad/s)")
ax_w.set_title("Wheel commands")
ax_t.bar(OUTPUT_NAMES[4:], channel_rmse[4:], color="tab:blue")
ax_t.set_ylabel("RMSE (N*m)")
ax_t.set_title("Arm joint torques")
for ax in (ax_w, ax_t):
    ax.grid(True, axis="y", alpha=0.3)
fig2.suptitle(f"Test error per output ({CHECKPOINT.name})")
fig2.tight_layout()
fig2.savefig(CHECKPOINT.with_name(CHECKPOINT.stem + "_test_per_output.png"), dpi=120)

plt.show()
