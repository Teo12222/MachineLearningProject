"""Train the GRU (PLAN.md step 14) and test it offline; same data, split, early stopping and results as train.py.

A separate script because a GRU learns from contiguous windows, with a burn-in and truncated backpropagation through
time, and is scored on whole episodes with its memory carried along, as at run time. The gravity residual is on by
default.

Usage (from the robotnik_arm_task folder):
    pixi run python train_gru.py --model gru_h192_w1000_gravres_s0 --window 1000
    pixi run python train_gru.py --model gru_h192_w1000_vw_gravres_s0 --window 1000 --vw-outputs   # (v, w) (step 13)
    pixi run python train_gru.py --model X --test-only     # test the saved checkpoint, no training

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

from bc_data import split_episodes, clean_ids, load_episode, load_frames, decode_outputs, arm_gravity
from bc_gru import fit_gru
from bc_model import GRU, save_model, load_model
from results import SIM_COLUMNS, update_results

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_course_expert"
SUBSAMPLE = "adaptive"   # window ends are train.py's kept frames; validation is uniform, the test uses every frame
KEEP_FRAC = 0.1

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", default="gru_baseline", help="model name: checkpoints/<model>.pt, results/<model>.csv")
parser.add_argument("--no-test", action="store_true", help="skip the evaluation on the test episodes")
parser.add_argument("--test-only", action="store_true", help="skip training and only evaluate the saved checkpoint")
parser.add_argument("--hidden", type=int, default=192, help="size of the GRU memory")
parser.add_argument("--window", type=int, default=1000, help="frames per training window that count in the loss")
parser.add_argument("--burn-in", type=int, default=200, help="frames before them, to warm up the memory, without loss")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--batch", type=int, default=64, help="windows per batch")
parser.add_argument("--max-epochs", type=int, default=50, help="upper limit on the number of epochs")
parser.add_argument("--patience", type=int, default=10, help="stop after this many epochs without improvement")
parser.add_argument("--gravity-residual", action=argparse.BooleanOptionalAction, default=True,
                    help="torque outputs are torque minus arm gravity (PLAN.md step 12)")
parser.add_argument("--vw-outputs", action=argparse.BooleanOptionalAction, default=False,
                    help="wheel outputs are forward speed v and turn rate w instead of 4 wheel commands (PLAN.md step 13)")
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
    x_tr, y_tr, frames_tr, starts_tr = load_frames(DATA_DIR, train_ids, KEEP_FRAC, SUBSAMPLE,
                                                   gravity_residual=args.gravity_residual, all_labels=True,
                                                   vw_outputs=args.vw_outputs)
    x_va, y_va, frames_va, starts_va = load_frames(DATA_DIR, val_ids, KEEP_FRAC, "uniform",
                                                   gravity_residual=args.gravity_residual, all_labels=True,
                                                   vw_outputs=args.vw_outputs)
    print(f"train frames {len(x_tr)}, val frames {len(frames_va)} (window ends: {SUBSAMPLE}, "
          f"gravity residual: {'on' if args.gravity_residual else 'off'})")

    model = GRU(x_tr.shape[1], y_tr.shape[1], hidden=args.hidden, burn_in=args.burn_in,
                gravity_residual=args.gravity_residual, vw_outputs=args.vw_outputs)
    # Normalised on the kept frames, like the MLP
    model.set_normalisation(x_tr[frames_tr], y_tr[frames_tr])
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    best_val, best_epoch, epoch = fit_gru(model, x_tr, y_tr, frames_tr, starts_tr, x_va, y_va, frames_va, starts_va,
                                          nn.MSELoss(), window=args.window, batch=args.batch, lr=args.lr,
                                          max_epochs=args.max_epochs, patience=args.patience)
    save_model(model, CHECKPOINT)
    train_time = time.perf_counter() - start

    # New weights: the old test and simulation results no longer apply, so they are emptied ("not measured")
    update_results(
        args.model, train_episodes=len(train_ids), epochs=epoch, batch=args.batch, lr=args.lr,
        hidden=args.hidden, layers=1, best_val_loss=best_val, best_epoch=best_epoch,
        train_time_s=train_time, parameters=sum(p.numel() for p in model.parameters()),
        subsample=SUBSAMPLE, keep_frac=KEEP_FRAC, weight_decay=0.0, dropout=0.0, max_epochs=args.max_epochs,
        patience=args.patience, gravity_residual=args.gravity_residual, vw_outputs=args.vw_outputs,
        architecture="gru", seed=args.seed,
        window=args.window, burn_in=args.burn_in,
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
    y = best.predict(x[None])[0]
    tau_g = arm_gravity(d["q_mes"]) if best.config["gravity_residual"] else None
    # (v, w) models: converted back to 4 wheel commands, so test_wheel_rmse is measured the same way for every model
    wheel_pred, torque_pred = decode_outputs(y, tau_g, best.config.get("vw_outputs", False))
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
