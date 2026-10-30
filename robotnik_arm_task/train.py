"""Train the baseline MLP on the expert episodes and save the best checkpoint (lowest validation loss).

Afterwards the saved checkpoint is evaluated on the test episodes: per-episode and per-output errors are printed
and plotted (figures saved next to the checkpoint). Skip this with --no-test.

Usage (from the robotnik_arm_task folder):
    pixi run python train.py                       # train, then test (model mlp_baseline)
    pixi run python train.py --model mlp_robotframe # checkpoints/mlp_robotframe.pt + results/mlp_robotframe.csv
    pixi run python train.py --no-test             # train only
    pixi run python train.py --test-only           # test the saved checkpoint, no training (checkpoint is not changed)
    pixi run python train.py --model X --lr 3e-3 --hidden 512 --dropout 0.1   # other settings (tune.py uses these)
    pixi run python train.py --model gravres_s0 --gravity-residual   # learn torque minus gravity (PLAN.md step 12)
    pixi run python train.py --model temporal_window_k4_d50_s0 --history-k 4 --history-spacing 50   # past frames

Results go to results/<model>.csv (see results.py): training columns after training, test columns after testing.
Training stops early once the validation loss hasn't improved for --patience epochs (at most --max-epochs).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + numpy each ship an OpenMP runtime on Windows

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bc_data import (
    split_episodes, clean_ids, load_episode, load_episodes, decode_outputs, arm_gravity, with_history,
)
from bc_model import MLP, save_model, load_model
from results import SIM_COLUMNS, update_results

# Paths are relative to this file, so the script works from any working directory
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_main"
# Defaults of the training settings (all can be changed with options, see --help): cheap development defaults from
# the first tuning run (~4 min; lr 3e-3 with the 256 x 3 network got most of the tuning gain, PLAN.md). Models that
# go into the report are tuned with tune.py.
MAX_EPOCHS = 100
PATIENCE = 10
BATCH = 2048
LR = 3e-3
# Training frames: None = all, "uniform" or "adaptive" (see subsample_indices in bc_data.py). "adaptive" since the
# merge: more frames where the expert's commands change (phase switches, the reach), now that the expert's arm chatter
# is fixed. Models trained before used "uniform" (every 10th frame); the comparison is still to be run (PLAN.md
# step 7). All frames would be ~10x slower. Validation is subsampled uniformly and the test RMSE uses every frame.
SUBSAMPLE = "adaptive"
KEEP_FRAC = 0.1     # share of each episode kept when subsampling

OUTPUT_NAMES = ["wheel FR", "wheel FL", "wheel BL", "wheel BR"] + [f"J{j}" for j in range(1, 8)]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--no-test", action="store_true", help="skip the evaluation on the test episodes")
parser.add_argument("--test-only", action="store_true", help="skip training and only evaluate the saved checkpoint")
parser.add_argument("--model", default="mlp_baseline",
                    help="model name: checkpoints/<model>.pt and results/<model>.csv (put the seed in it, e.g. _s0)")
parser.add_argument("--lr", type=float, default=LR, help="learning rate")
parser.add_argument("--batch", type=int, default=BATCH, help="batch size")
parser.add_argument("--hidden", type=int, default=256, help="units per hidden layer")
parser.add_argument("--layers", type=int, default=3, help="number of hidden layers")
parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay (AdamW)")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout after each hidden layer")
parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="upper limit on the number of epochs")
parser.add_argument("--patience", type=int, default=PATIENCE,
                    help="stop once the validation loss hasn't improved for this many epochs (early stopping)")
parser.add_argument("--gravity-residual", action=argparse.BooleanOptionalAction, default=False,
                    help="torque outputs are torque minus arm gravity, added back when the model is run (PLAN.md "
                         "step 12); --no-gravity-residual for the ablation")
parser.add_argument("--history-k", type=int, default=0, help="past frames added to each input (PLAN.md step 14)")
parser.add_argument("--history-spacing", type=int, default=50, help="frames between them (1 frame = 1 ms)")
parser.add_argument("--seed", type=int, default=0, help="training seed (weights, batch order); put it in --model too")
parser.add_argument("--threads", type=int, help="torch CPU threads (several runs at once share the cores)")
args = parser.parse_args()
if args.no_test and args.test_only:
    parser.error("--no-test and --test-only together would do nothing")
CHECKPOINT = SCRIPT_DIR / "checkpoints" / f"{args.model}.pt"
if args.threads:
    torch.set_num_threads(args.threads)

torch.manual_seed(args.seed)
train_ids, val_ids, test_ids = split_episodes(DATA_DIR)
# Bad episodes are dropped after the split, so the good ones always stay in the same set
train_ids, val_ids, test_ids = (clean_ids(DATA_DIR, ids) for ids in (train_ids, val_ids, test_ids))

if not args.test_only:
    keep_frac = KEEP_FRAC if SUBSAMPLE else None
    x_tr, y_tr = load_episodes(DATA_DIR, train_ids, keep_frac=keep_frac, mode=SUBSAMPLE,
                               gravity_residual=args.gravity_residual,
                               history_k=args.history_k, history_spacing=args.history_spacing)
    # Validation is always subsampled uniformly, so runs with different SUBSAMPLE modes are judged on the same frames
    x_va, y_va = load_episodes(DATA_DIR, val_ids, keep_frac=keep_frac, mode="uniform",
                               gravity_residual=args.gravity_residual,
                               history_k=args.history_k, history_spacing=args.history_spacing)
    print(f"train frames {len(x_tr)}, val frames {len(x_va)} (subsampling: {SUBSAMPLE or 'off'}, "
          f"gravity residual: {'on' if args.gravity_residual else 'off'})")

    model = MLP(x_tr.shape[1], y_tr.shape[1], hidden=args.hidden, layers=args.layers, dropout=args.dropout,
                gravity_residual=args.gravity_residual, history_k=args.history_k,
                history_spacing=args.history_spacing)
    model.set_normalisation(x_tr, y_tr)

    # Normalise once up front, so the training loop only indexes tensors
    xt = (torch.from_numpy(x_tr) - model.x_mean) / model.x_std
    yt = (torch.from_numpy(y_tr) - model.y_mean) / model.y_std
    xv = (torch.from_numpy(x_va) - model.x_mean) / model.x_std
    yv = (torch.from_numpy(y_va) - model.y_mean) / model.y_std

    # AdamW: Adam with decoupled weight decay; with weight_decay 0 it is the same as Adam
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    best_val, best_epoch = float("inf"), 0

    start = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        perm = torch.randperm(len(xt))
        loss_sum = 0.0
        for i in range(0, len(xt), args.batch):
            idx = perm[i:i + args.batch]
            loss = loss_fn(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        # Training loss averaged over the epoch (not the last batch), so the train/val gap shows over-/underfitting
        print(f"epoch {epoch:3d}  train {loss_sum / len(xt):.4f}  val {val_loss:.4f}  "
              f"({time.perf_counter() - epoch_start:.1f} s)")
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            save_model(model, CHECKPOINT)
            print(f"   saved {CHECKPOINT}")
        elif epoch - best_epoch >= args.patience:
            print(f"   no improvement for {args.patience} epochs: stopping (best epoch {best_epoch})")
            break
    train_time = time.perf_counter() - start

    # New weights: the old test and simulation results no longer apply, so they are emptied ("not measured")
    # epochs = epochs actually run (where early stopping ended)
    update_results(
        args.model, train_episodes=len(train_ids), epochs=epoch, batch=args.batch, lr=args.lr,
        hidden=args.hidden, layers=args.layers, best_val_loss=best_val, best_epoch=best_epoch,
        train_time_s=train_time, parameters=sum(p.numel() for p in model.parameters()),
        subsample=SUBSAMPLE or "off", keep_frac=KEEP_FRAC if SUBSAMPLE else None,
        weight_decay=args.weight_decay, dropout=args.dropout, max_epochs=args.max_epochs, patience=args.patience,
        gravity_residual=args.gravity_residual, history_k=args.history_k,
        history_spacing=args.history_spacing if args.history_k else None, architecture="mlp", seed=args.seed,
        inference_ms=None, test_wheel_rmse=None, test_torque_rmse=None, **{c: None for c in SIM_COLUMNS},
    )
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
    path = Path(DATA_DIR) / f"ep_{i:04d}.npz"
    x, _ = load_episode(path)
    # Read from the checkpoint, so --test-only builds the inputs the model was trained on
    x = with_history(x, np.arange(len(x)), best.config.get("history_k", 0), best.config.get("history_spacing", 1))
    # Scored on the decoded commands against the expert's own, so every output format is measured the same way
    d = np.load(path)
    # Gravity-residual models: add gravity back, so the error is on the full torque like every other model.
    # Read from the checkpoint, so --test-only scores a saved model the way it was trained.
    tau_g = arm_gravity(d["q_mes"]) if best.config.get("gravity_residual") else None
    wheel_pred, torque_pred = decode_outputs(best.predict(x), tau_g)
    err = np.concatenate([wheel_pred - d["wheel_cmds"], torque_pred - d["torque_arm"]], axis=-1)
    wheel_rmse.append(np.sqrt(np.mean(err[:, :4] ** 2)))
    torque_rmse.append(np.sqrt(np.mean(err[:, 4:] ** 2)))
    sq_err_sum += (err ** 2).sum(axis=0)
    n_frames += len(err)
    print(f"{i:8d} {wheel_rmse[-1]:15.4f} {torque_rmse[-1]:14.4f}")
wheel_rmse, torque_rmse = np.array(wheel_rmse), np.array(torque_rmse)
# Per-output RMSE over all test frames pooled together
channel_rmse = np.sqrt(sq_err_sum / n_frames)

print(f"{'average':>8} {wheel_rmse.mean():15.4f} {torque_rmse.mean():14.4f}   (mean of the per-episode values)")

# Time of one single-frame prediction, as in the policy script (1 thread, one frame per simulation step)
torch.set_num_threads(1)
frame = x[0]
for _ in range(50):   # warm-up
    best.predict(frame)
times = []
for _ in range(1000):
    t0 = time.perf_counter()
    best.predict(frame)
    times.append(time.perf_counter() - t0)
inference_ms = 1000 * np.median(times)
print(f"\nInference: {inference_ms:.3f} ms per frame (median of 1000, 1 thread)")
update_results(args.model, test_wheel_rmse=wheel_rmse.mean(), test_torque_rmse=torque_rmse.mean(),
               inference_ms=inference_ms)
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
