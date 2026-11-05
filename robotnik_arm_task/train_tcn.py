"""Train the TCN (PLAN.md step 14) and test it offline; same data, split, early stopping and results as train.py.

A separate script because the TCN cuts its windows per batch instead of storing them (64 frames per sample would not
fit in memory), and so train.py and tune.py stay as they are. The gravity residual is on by default.

Usage (from the robotnik_arm_task folder):
    pixi run python train_tcn.py --model tcn_L6_d10_gravres_s0 --layers 6 --spacing 10
    pixi run python train_tcn.py --model X --no-gravity-residual --lr 1e-3 --max-epochs 20
    pixi run python train_tcn.py --model X --test-only     # test the saved checkpoint, no training

Results go to results/<model>.csv (see results.py); closed loop with compare.py --test-set --model <model>.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + numpy each ship an OpenMP runtime on Windows

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bc_data import split_episodes, clean_ids, load_episode, load_frames, window_batch, decode_outputs, arm_gravity
from bc_model import TCN, save_model, load_model
from results import SIM_COLUMNS, update_results

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_course_expert"
SUBSAMPLE = "adaptive"   # train.py's defaults: validation is subsampled uniformly, the test uses every frame
KEEP_FRAC = 0.1
EVAL_BATCH = 8192

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", default="tcn_baseline", help="model name: checkpoints/<model>.pt, results/<model>.csv")
parser.add_argument("--no-test", action="store_true", help="skip the evaluation on the test episodes")
parser.add_argument("--test-only", action="store_true", help="skip training and only evaluate the saved checkpoint")
parser.add_argument("--layers", type=int, default=6, help="number of layers L: the network sees 2^L frames")
parser.add_argument("--channels", type=int, default=64, help="channels per layer")
parser.add_argument("--spacing", type=int, default=10, help="frames between the frames it sees (1 frame = 1 ms)")
parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
parser.add_argument("--batch", type=int, default=2048, help="batch size")
parser.add_argument("--max-epochs", type=int, default=100, help="upper limit on the number of epochs")
parser.add_argument("--patience", type=int, default=10, help="stop after this many epochs without improvement")
parser.add_argument("--gravity-residual", action=argparse.BooleanOptionalAction, default=True,
                    help="torque outputs are torque minus arm gravity (PLAN.md step 12)")
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


def evaluate(model, x, y, frames, starts, loss_fn):
    """Mean loss over all samples, in batches: every window at once wouldn't fit in memory."""
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(frames), EVAL_BATCH):
            part = slice(i, i + EVAL_BATCH)
            windows = window_batch(x, frames[part], starts[part], model.n_frames, model.spacing)
            total += loss_fn(model(torch.from_numpy(windows)), y[part]).item() * len(windows)
    return total / len(frames)


if not args.test_only:
    x_tr, y_tr, frames_tr, starts_tr = load_frames(DATA_DIR, train_ids, KEEP_FRAC, SUBSAMPLE,
                                                   gravity_residual=args.gravity_residual)
    x_va, y_va, frames_va, starts_va = load_frames(DATA_DIR, val_ids, KEEP_FRAC, "uniform",
                                                   gravity_residual=args.gravity_residual)
    print(f"train frames {len(frames_tr)}, val frames {len(frames_va)} (subsampling: {SUBSAMPLE}, "
          f"gravity residual: {'on' if args.gravity_residual else 'off'})")

    model = TCN(x_tr.shape[1], y_tr.shape[1], layers=args.layers, channels=args.channels, spacing=args.spacing,
                gravity_residual=args.gravity_residual)
    print(f"sees {model.n_frames} frames, {model.n_frames * args.spacing} ms, "
          f"{sum(p.numel() for p in model.parameters())} parameters")
    # Normalised on the kept frames, like the MLP
    model.set_normalisation(x_tr[frames_tr], y_tr)
    x_mean, x_std = model.x_mean.numpy(), model.x_std.numpy()
    x_tr = (x_tr - x_mean) / x_std
    x_va = (x_va - x_mean) / x_std
    yt = (torch.from_numpy(y_tr) - model.y_mean) / model.y_std
    yv = (torch.from_numpy(y_va) - model.y_mean) / model.y_std

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    best_val, best_epoch = float("inf"), 0

    start = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        perm = torch.randperm(len(frames_tr))
        loss_sum = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            part = idx.numpy()
            windows = window_batch(x_tr, frames_tr[part], starts_tr[part], model.n_frames, model.spacing)
            loss = loss_fn(model(torch.from_numpy(windows)), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)

        model.eval()
        val_loss = evaluate(model, x_va, yv, frames_va, starts_va, loss_fn)
        print(f"epoch {epoch:3d}  train {loss_sum / len(perm):.4f}  val {val_loss:.4f}  "
              f"({time.perf_counter() - epoch_start:.1f} s)", flush=True)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            save_model(model, CHECKPOINT)
        elif epoch - best_epoch >= args.patience:
            print(f"   no improvement for {args.patience} epochs: stopping (best epoch {best_epoch})")
            break
    train_time = time.perf_counter() - start

    # New weights: the old test and simulation results no longer apply, so they are emptied ("not measured")
    update_results(
        args.model, train_episodes=len(train_ids), epochs=epoch, batch=args.batch, lr=args.lr,
        hidden=args.channels, layers=args.layers, best_val_loss=best_val, best_epoch=best_epoch,
        train_time_s=train_time, parameters=sum(p.numel() for p in model.parameters()),
        subsample=SUBSAMPLE, keep_frac=KEEP_FRAC, weight_decay=0.0, dropout=0.0, max_epochs=args.max_epochs,
        patience=args.patience, gravity_residual=args.gravity_residual, architecture="tcn", seed=args.seed,
        history_k=model.n_frames - 1, history_spacing=args.spacing,
        inference_ms=None, test_wheel_rmse=None, test_torque_rmse=None, **{c: None for c in SIM_COLUMNS},
    )
elif not CHECKPOINT.exists():
    parser.error(f"--test-only needs a trained checkpoint, but {CHECKPOINT} does not exist")

if args.no_test:
    raise SystemExit

# ------------------------------------------------------------------------------
# Test evaluation (offline), scored as in train.py: decoded commands against the expert's own
# ------------------------------------------------------------------------------
best = load_model(CHECKPOINT)
wheel_rmse, torque_rmse = [], []
print(f"\nTest set: {len(test_ids)} episodes, RMSE between model and expert commands")
for i in test_ids:
    path = Path(DATA_DIR) / f"ep_{i:04d}.npz"
    x, _ = load_episode(path)
    d = np.load(path)
    frames, starts = np.arange(len(x)), np.zeros(len(x), dtype=int)
    y = np.concatenate([best.predict(window_batch(x, frames[k:k + EVAL_BATCH], starts[k:k + EVAL_BATCH], best.n_frames,
                                                  best.spacing)) for k in range(0, len(x), EVAL_BATCH)])
    tau_g = arm_gravity(d["q_mes"]) if best.config["gravity_residual"] else None
    wheel_pred, torque_pred = decode_outputs(y, tau_g)
    wheel_rmse.append(np.sqrt(np.mean((wheel_pred - d["wheel_cmds"]) ** 2)))
    torque_rmse.append(np.sqrt(np.mean((torque_pred - d["torque_arm"]) ** 2)))
print(f"average RMSE: wheels {np.mean(wheel_rmse):.4f} rad/s, torques {np.mean(torque_rmse):.4f} N*m "
      "(mean of the per-episode values)")

# Time of one run-time step, as in the policy script (1 thread, one frame per simulation step)
torch.set_num_threads(1)
best.reset()
times = []
for frame in np.tile(x, (1 + 1050 // len(x), 1))[:1050]:
    t0 = time.perf_counter()
    best.step(frame)
    times.append(time.perf_counter() - t0)
inference_ms = 1000 * np.median(times[50:])   # after a short warm-up
print(f"Inference: {inference_ms:.3f} ms per step (median of 1000, 1 thread)")
update_results(args.model, test_wheel_rmse=np.mean(wheel_rmse), test_torque_rmse=np.mean(torque_rmse),
               inference_ms=inference_ms)
