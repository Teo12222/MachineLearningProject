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
    pixi run python train.py --model vw_s0 --vw-outputs              # (v, w) instead of 4 wheel speeds (PLAN.md step 13)
    pixi run python train.py --model split_arrived_s0 --split --gravity-residual   # split networks (PLAN.md step 15)
    pixi run python train.py --model X --split --gravity-residual --dagger dagger/split_dagger_r1   # + DAgger data
                                                   # for the arm network (dagger.py output, PLAN.md step 21)
    pixi run python train.py --model X --split --gravity-residual --base-arch gru --arm-from split_dagger_r5_s0
                                                   # GRU base behind a copied arm; --arm-arch gru, --base-from MODEL too
    pixi run python train.py --model temporal_window_k4_d50_s0 --history-k 4 --history-spacing 50   # past frames

Results go to results/<model>.csv (see results.py): training columns after training, test columns after testing.
Training stops early once the validation loss hasn't improved for --patience epochs (at most --max-epochs).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + numpy each ship an OpenMP runtime on Windows

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bc_data import (split_episodes, clean_ids, load_episode, load_episodes, load_frames, decode_outputs, arm_gravity,
                     with_history)
from bc_gru import fit_gru, normalised, predict_frames
from bc_model import GRU, MLP, save_model
from bc_split import (SplitPolicy, ArrivedLatch, arm_inputs, base_inputs, base_labels, base_loss, load_dagger_arm,
                      load_policy, net_settings)
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
# Removed (30 Oct): separate training frames for the split arm network. Uniform arm frames were tested and made the
# arm worse in closed loop, so both split networks use SUBSAMPLE (PLAN.md step 15). With it, also uncomment the block
# in the --split branch below and use x_arm / y_arm / a_arm in the arm's fit call.
# ARM_SUBSAMPLE = "uniform"
KEEP_FRAC = 0.1     # share of each episode kept when subsampling

OUTPUT_NAMES = ["wheel FR", "wheel FL", "wheel BL", "wheel BR"] + [f"J{j}" for j in range(1, 8)]
# Offline test of the "arrived" flag (split models): when the latch switches vs the expert's arrival, over the test
# episodes. Early is the dangerous direction: the base stops short and the arm reaches from the wrong place.
ARRIVE_COLUMNS = ["test_arrive_err_median", "test_arrive_early", "test_arrive_late", "test_arrive_timeout"]
EARLY_LATE_S = 0.1   # s, a latch this much before / after the expert's arrival counts as early / late

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
# Per-network size and dropout of a split model (default: the shared --hidden / --layers / --dropout), so the arm can
# be tuned on top of a copied base (PLAN.md 4a)
for _part in ("base", "arm"):
    parser.add_argument(f"--{_part}-hidden", type=int, help=f"with --split: the {_part} network's own --hidden")
    parser.add_argument(f"--{_part}-layers", type=int, help=f"with --split: the {_part} network's own --layers")
    parser.add_argument(f"--{_part}-dropout", type=float, help=f"with --split: the {_part} network's own --dropout")
parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="upper limit on the number of epochs")
parser.add_argument("--patience", type=int, default=PATIENCE,
                    help="stop once the validation loss hasn't improved for this many epochs (early stopping)")
parser.add_argument("--gravity-residual", action=argparse.BooleanOptionalAction, default=False,
                    help="torque outputs are torque minus arm gravity, added back when the model is run (PLAN.md "
                         "step 12); --no-gravity-residual for the ablation")
parser.add_argument("--vw-outputs", action=argparse.BooleanOptionalAction, default=False,
                    help="wheel outputs are forward speed v and turn rate w instead of 4 wheel commands, converted back "
                         "when the model is run (PLAN.md step 13); --no-vw-outputs for the ablation")
parser.add_argument("--split", action=argparse.BooleanOptionalAction, default=False,
                    help="separate base and arm networks with an 'arrived' flag (PLAN.md step 15); --no-split for "
                         "the ablation")
parser.add_argument("--dagger", nargs="+", metavar="FOLDER",
                    help="with --split: DAgger episode folders (dagger.py output) whose expert-labelled frames are added "
                         "to the arm network's training data (PLAN.md step 21); leave out for the ablation")
parser.add_argument("--arm-from", metavar="MODEL",
                    help="with --split: copy the arm network of this split checkpoint instead of training one, so two "
                         "models differ only in the base (e.g. vw vs 4 wheels, GRU vs MLP base)")
parser.add_argument("--base-from", metavar="MODEL",
                    help="with --split: copy the base network of this split checkpoint instead of training one, so two "
                         "models differ only in the arm")
parser.add_argument("--base-arch", choices=("mlp", "gru"), default="mlp", help="with --split: type of the base network")
parser.add_argument("--arm-arch", choices=("mlp", "gru"), default="mlp", help="with --split: type of the arm network")
# A GRU half trains like train_gru.py, whatever the settings of an MLP half (PLAN.md step 15)
parser.add_argument("--gru-hidden", type=int, default=192, help="GRU networks: size of the memory")
parser.add_argument("--gru-window", type=int, default=1000, help="GRU networks: frames per window that count in the loss")
parser.add_argument("--gru-burn-in", type=int, default=200, help="GRU networks: frames before them, without loss")
parser.add_argument("--gru-lr", type=float, default=1e-3, help="GRU networks: learning rate")
parser.add_argument("--gru-batch", type=int, default=64, help="GRU networks: windows per batch")
parser.add_argument("--gru-max-epochs", type=int, default=50, help="GRU networks: upper limit on the number of epochs")
parser.add_argument("--history-k", type=int, default=0, help="past frames added to each input (PLAN.md step 14)")
parser.add_argument("--history-spacing", type=int, default=50, help="frames between them (1 frame = 1 ms)")
parser.add_argument("--seed", type=int, default=0, help="training seed (weights, batch order); put it in --model too")
parser.add_argument("--threads", type=int, help="torch CPU threads (several runs at once share the cores)")
args = parser.parse_args()
if args.no_test and args.test_only:
    parser.error("--no-test and --test-only together would do nothing")
if args.dagger and not args.split:
    parser.error("--dagger needs --split: DAgger data only has arm labels (a single MLP would also need wheel labels)")
if args.arm_from and not args.split:
    parser.error("--arm-from needs --split")
if args.base_from and not args.split:
    parser.error("--base-from needs --split")
PER_NET = [f"{p}_{k}" for p in ("base", "arm") for k in ("hidden", "layers", "dropout")]   # base_hidden, ...
NET_OPTIONS = dict(hidden=args.hidden, layers=args.layers, dropout=args.dropout,
                   **{k: getattr(args, k) for k in PER_NET})
if not args.split and any(NET_OPTIONS[k] is not None for k in PER_NET):
    parser.error("--base-hidden / --arm-hidden etc. need --split")
if args.arm_from and args.dagger:
    # The copied arm keeps its own training data; DAgger frames would have nothing to train
    parser.error("--arm-from copies a trained arm: --dagger has no arm to train")
if args.dagger and args.arm_arch == "gru":
    # DAgger frames are single frames; a GRU arm needs the whole runs as windows (not built, PLAN.md step 15)
    parser.error("--dagger with --arm-arch gru isn't supported yet")
base_arch, arm_arch = args.base_arch, args.arm_arch
if args.arm_from:
    # Loaded and checked up front, so a wrong source stops the run before any training
    arm_source = load_policy(SCRIPT_DIR / "checkpoints" / f"{args.arm_from}.pt")
    if not isinstance(arm_source, SplitPolicy):
        parser.error(f"--arm-from {args.arm_from} is not a split model")
    if arm_source.config["gravity_residual"] != args.gravity_residual:
        # Its torques would be decoded with or without gravity added back, i.e. wrongly
        parser.error(f"--arm-from {args.arm_from} has gravity_residual={arm_source.config['gravity_residual']}")
    arm_arch = arm_source.config.get("arm_arch", "mlp")
if args.base_from:
    base_source = load_policy(SCRIPT_DIR / "checkpoints" / f"{args.base_from}.pt")
    if not isinstance(base_source, SplitPolicy):
        parser.error(f"--base-from {args.base_from} is not a split model")
    if base_source.config.get("vw_outputs", False) != args.vw_outputs:
        parser.error(f"--base-from {args.base_from} has vw_outputs={base_source.config.get('vw_outputs', False)}")
    base_arch = base_source.config.get("base_arch", "mlp")
# A checkpoint rebuilds both networks from its config, so a copied network must have the size this run gives that
# network. Dropout too: it adds layers, so a copied network without them couldn't be loaded back from the checkpoint
copied = ([("arm", arm_source)] if args.arm_from else []) + ([("base", base_source)] if args.base_from else [])
for part, source in copied:
    if (net_settings(source.config, part), source.config.get("gru_hidden", 192), source.config.get("burn_in", 200)) != (
            net_settings(NET_OPTIONS, part), args.gru_hidden, args.gru_burn_in):
        parser.error(f"--{part}-from: the copied {part} network's size or dropout {net_settings(source.config, part)} "
                     f"differs from this run's {net_settings(NET_OPTIONS, part)} (--{part}-hidden etc.)")
uses_gru = args.split and "gru" in (base_arch, arm_arch)
if args.split and args.history_k:
    # base_inputs / arm_inputs pick the current frame's columns, so the past frames would be silently dropped
    parser.error("--split with --history-k isn't supported yet (temporal + split: PLAN.md step 14)")
CHECKPOINT = SCRIPT_DIR / "checkpoints" / f"{args.model}.pt"
if args.threads:
    torch.set_num_threads(args.threads)

torch.manual_seed(args.seed)
train_ids, val_ids, test_ids = split_episodes(DATA_DIR)
# Bad episodes are dropped after the split, so the good ones always stay in the same set
train_ids, val_ids, test_ids = (clean_ids(DATA_DIR, ids) for ids in (train_ids, val_ids, test_ids))


def normalise(net, x, y):
    """Inputs and labels as tensors in the network's normalised units (normalised once up front, so the training
    loop only indexes tensors)."""
    return (torch.from_numpy(x) - net.x_mean) / net.x_std, (torch.from_numpy(y) - net.y_mean) / net.y_std


# Training and validation loss per epoch of each network trained by fit (key: its label), for the loss-curve plot
curves = {}


def fit(net, xt, yt, xv, yv, loss_fn, label=""):
    """Train one network with early stopping; afterwards it holds the weights of its best validation epoch.

    Returns (best validation loss, best epoch, epochs run). The single MLP and both split networks are trained by
    this same loop, so with and without --split differ only in the networks (ablation, PLAN.md step 23).
    """
    # AdamW: Adam with decoupled weight decay; with weight_decay 0 it is the same as Adam
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.perf_counter()
        net.train()
        perm = torch.randperm(len(xt))
        loss_sum = 0.0
        for i in range(0, len(xt), args.batch):
            idx = perm[i:i + args.batch]
            loss = loss_fn(net(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)

        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(xv), yv).item()
        # Training loss averaged over the epoch (not the last batch), so the train/val gap shows over-/underfitting
        print(f"{label}epoch {epoch:3d}  train {loss_sum / len(xt):.4f}  val {val_loss:.4f}  "
              f"({time.perf_counter() - epoch_start:.1f} s)")
        curves.setdefault(label.strip() or "model", []).append((epoch, loss_sum / len(xt), val_loss))
        if val_loss < best_val:
            # Kept in memory and saved once training ends (a split checkpoint holds both networks)
            best_val, best_epoch, best_state = val_loss, epoch, copy.deepcopy(net.state_dict())
        elif epoch - best_epoch >= args.patience:
            print(f"   no improvement for {args.patience} epochs: stopping (best epoch {best_epoch})")
            break
    net.load_state_dict(best_state)
    return best_val, best_epoch, epoch


if not args.test_only:
    keep_frac = KEEP_FRAC if SUBSAMPLE else None
    # a_tr / a_va: the expert's "arrived" flag of the same frames (split networks only)
    x_tr, y_tr, a_tr = load_episodes(DATA_DIR, train_ids, keep_frac=keep_frac, mode=SUBSAMPLE,
                                     gravity_residual=args.gravity_residual, vw_outputs=args.vw_outputs, arrived=True,
                                     history_k=args.history_k, history_spacing=args.history_spacing)
    # Validation is always subsampled uniformly, so runs with different SUBSAMPLE modes are judged on the same frames
    x_va, y_va, a_va = load_episodes(DATA_DIR, val_ids, keep_frac=keep_frac, mode="uniform",
                                     gravity_residual=args.gravity_residual, vw_outputs=args.vw_outputs, arrived=True,
                                     history_k=args.history_k, history_spacing=args.history_spacing)
    print(f"train frames {len(x_tr)}, val frames {len(x_va)} (subsampling: {SUBSAMPLE or 'off'}, "
          f"gravity residual: {'on' if args.gravity_residual else 'off'}, "
          f"wheel outputs: {'v, w' if args.vw_outputs else '4 wheels'}, split: {'on' if args.split else 'off'})")
    if uses_gru:
        # A GRU half needs every frame in order; the frames above are the same rows, for the normalisation
        xf_tr, yf_tr, fr_tr, st_tr, af_tr = load_frames(DATA_DIR, train_ids, keep_frac, SUBSAMPLE, all_labels=True,
                                                        gravity_residual=args.gravity_residual,
                                                        vw_outputs=args.vw_outputs, arrived=True)
        xf_va, yf_va, fr_va, st_va, af_va = load_frames(DATA_DIR, val_ids, keep_frac, "uniform", all_labels=True,
                                                        gravity_residual=args.gravity_residual,
                                                        vw_outputs=args.vw_outputs, arrived=True)
        gru_options = dict(window=args.gru_window, batch=args.gru_batch, lr=args.gru_lr, max_epochs=args.gru_max_epochs,
                           patience=args.patience)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    def val_output(net, x_all, x_kept):
        """Normalised output on the validation frames: whole episodes for a GRU (x_all), the frames for an MLP."""
        if isinstance(net, GRU):
            return normalised(net, predict_frames(net, x_all, fr_va, st_va))
        return net(x_kept)

    if args.split:
        model = SplitPolicy(hidden=args.hidden, layers=args.layers, dropout=args.dropout,
                            gravity_residual=args.gravity_residual, vw_outputs=args.vw_outputs, base_arch=base_arch,
                            arm_arch=arm_arch, gru_hidden=args.gru_hidden, burn_in=args.gru_burn_in,
                            **{k: NET_OPTIONS[k] for k in PER_NET})
        model.set_normalisation(x_tr, y_tr, a_tr)
        if args.base_from:
            model.base = base_source.base   # weights and its own normalisation, as for --arm-from
        # Removed (30 Oct, see ARM_SUBSAMPLE above): the arm's own training frames, loaded a second time
        # x_arm, y_arm, a_arm = load_episodes(DATA_DIR, train_ids, keep_frac=KEEP_FRAC if ARM_SUBSAMPLE else None,
        #                                     mode=ARM_SUBSAMPLE, gravity_residual=args.gravity_residual, arrived=True)
        # model.arm.set_normalisation(arm_inputs(x_arm, a_arm), y_arm[:, -7:])
        # Weight of the arrived=True frames, so both classes count equally in the flag's loss (computed on the frames
        # actually used: adaptive subsampling keeps relatively more navigation frames than the full episodes)
        pos_weight = torch.tensor((1 - a_tr).sum() / a_tr.sum())
        print(f"arrived in {100 * a_tr.mean():.1f}% of training frames, pos_weight {pos_weight:.2f}")
        xbv, ybv = normalise(model.base, base_inputs(x_va), base_labels(y_va, a_va))
        base_fn = lambda out, target: base_loss(out, target, pos_weight)
        x_base_va = base_inputs(xf_va) if uses_gru else None
        x_arm_va = arm_inputs(xf_va, af_va) if uses_gru else None
        if args.base_from:
            base_best = base_epochs = None
            print(f"base network copied from {args.base_from}")
        elif base_arch == "gru":
            _, base_best, base_epochs = fit_gru(model.base, base_inputs(xf_tr), base_labels(yf_tr, af_tr), fr_tr, st_tr,
                                                x_base_va, base_labels(yf_va, af_va), fr_va, st_va, base_fn,
                                                label="base ", **gru_options)
        else:
            _, base_best, base_epochs = fit(model.base, *normalise(model.base, base_inputs(x_tr), base_labels(y_tr, a_tr)),
                                            xbv, ybv, base_fn, "base ")
        dagger_frames = 0
        if args.arm_from:
            # The other model's arm network, unchanged (weights and its own normalisation): its training already
            # happened, so the two models' arms are identical, not just trained on the same data (PLAN.md step 15)
            model.arm = arm_source.arm
            arm_best = arm_epochs = None
            xav, yav = normalise(model.arm, arm_inputs(x_va, a_va), y_va[:, -7:])
            with torch.no_grad():
                arm_val = nn.functional.mse_loss(val_output(model.arm, x_arm_va, xav), yav).item()
            print(f"arm network copied from {args.arm_from}: val {arm_val:.4f}")
        elif arm_arch == "gru":
            # Every frame in order, flag 0 before the expert's arrival: the same inputs as at run time
            arm_val, arm_best, arm_epochs = fit_gru(model.arm, arm_inputs(xf_tr, af_tr), yf_tr[:, -7:], fr_tr, st_tr,
                                                    x_arm_va, yf_va[:, -7:], fr_va, st_va, nn.MSELoss(), label="arm  ",
                                                    **gru_options)
        else:
            # The arm is trained with the expert's true flag as input (at run time: the latched predicted one)
            x_arm, y_arm = arm_inputs(x_tr, a_tr), y_tr[:, -7:]   # torques: always the last 7 labels
            if args.dagger:
                # Frames the earlier policies visited, labelled with the expert's torques: thinned like the expert
                # data (every 1 / KEEP_FRAC-th labelled frame). Added after the normalisation, which stays the expert
                # data's, so best_val_loss stays comparable with models trained without DAgger; validation stays the
                # expert's
                x_dag, y_dag = load_dagger_arm(args.dagger, round(1 / KEEP_FRAC), args.gravity_residual)
                x_arm, y_arm = np.concatenate([x_arm, x_dag]), np.concatenate([y_arm, y_dag])
                dagger_frames = len(x_dag)
                print(f"arm training frames: {len(x_arm) - dagger_frames} expert + {dagger_frames} DAgger "
                      f"({', '.join(Path(f).name for f in args.dagger)})")
            arm_val, arm_best, arm_epochs = fit(
                model.arm, *normalise(model.arm, x_arm, y_arm),
                *normalise(model.arm, arm_inputs(x_va, a_va), y_va[:, -7:]), nn.MSELoss(), "arm  ")
        # Validation loss on the same scale as the single MLP's (mean squared error over all normalised outputs;
        # same frames, so the same normalisation): the base's wheel part (without the flag's loss) + the arm's.
        # Not comparable between wheel formats (different quantities): compare those on test_wheel_rmse
        with torch.no_grad():
            n_wheel = ybv.shape[1] - 1
            out_bv = val_output(model.base, x_base_va, xbv)
            wheel_val = nn.functional.mse_loss(out_bv[:, :-1], ybv[:, :-1]).item()
            # Recorded unweighted, so it reads as a plain cross-entropy (early stopping used the weighted one)
            flag_val = nn.functional.binary_cross_entropy_with_logits(out_bv[:, -1], ybv[:, -1]).item()
        best_val, best_epoch, epoch = (n_wheel * wheel_val + 7 * arm_val) / (n_wheel + 7), None, None
        split_columns = dict(base_epochs=base_epochs, base_best_epoch=base_best, arm_epochs=arm_epochs,
                             arm_best_epoch=arm_best, val_arrived_bce=flag_val,
                             dagger=";".join(Path(f).name for f in args.dagger) if args.dagger else "",
                             dagger_frames=dagger_frames, arm_from=args.arm_from or "", base_from=args.base_from or "",
                             base_arch=base_arch, arm_arch=arm_arch,
                             **{f"{p}_{k}": v for p in ("base", "arm")
                                for k, v in zip(("hidden", "layers", "dropout"), net_settings(model.config, p))},
                             **(dict(gru_hidden=args.gru_hidden, gru_window=args.gru_window, gru_burn_in=args.gru_burn_in,
                                     gru_lr=args.gru_lr) if uses_gru else {}))
    else:
        model = MLP(x_tr.shape[1], y_tr.shape[1], hidden=args.hidden, layers=args.layers, dropout=args.dropout,
                    gravity_residual=args.gravity_residual, vw_outputs=args.vw_outputs, history_k=args.history_k,
                    history_spacing=args.history_spacing)
        model.set_normalisation(x_tr, y_tr)
        best_val, best_epoch, epoch = fit(model, *normalise(model, x_tr, y_tr), *normalise(model, x_va, y_va),
                                          nn.MSELoss())
        split_columns = {}
    train_time = time.perf_counter() - start
    save_model(model, CHECKPOINT)
    print(f"saved {CHECKPOINT}")

    if curves:
        # Loss curves of the networks trained here (copied ones have none; GRU halves train in fit_gru, not recorded),
        # the kept (best validation) epoch marked: checkpoints/<model>_loss.png (PLAN.md step 25)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 4), squeeze=False)
        for ax, (name, rows) in zip(axes[0], curves.items()):
            ep, tr, va = np.array(rows).T
            k = np.argmin(va)
            ax.plot(ep, tr, label="training")
            ax.plot(ep, va, label="validation")
            ax.axvline(ep[k], color="gray", linestyle="--", linewidth=1)
            ax.plot(ep[k], va[k], "ko", label=f"kept: epoch {ep[k]:.0f}, val {va[k]:.4f}")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (normalised units)")
            ax.set_title(f"{name} network")
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle(f"Loss per epoch ({args.model})")
        fig.tight_layout()
        fig.savefig(CHECKPOINT.with_name(CHECKPOINT.stem + "_loss.png"), dpi=120)
        plt.close(fig)

    # New weights: the old test and simulation results no longer apply, so they are emptied ("not measured")
    # epochs = epochs actually run (where early stopping ended); split models record them per network instead
    update_results(
        args.model, train_episodes=len(train_ids), epochs=epoch, batch=args.batch, lr=args.lr,
        hidden=args.hidden, layers=args.layers, best_val_loss=best_val, best_epoch=best_epoch,
        train_time_s=train_time, parameters=sum(p.numel() for p in model.parameters()),
        subsample=SUBSAMPLE or "off", keep_frac=KEEP_FRAC if SUBSAMPLE else None,
        weight_decay=args.weight_decay, dropout=args.dropout, max_epochs=args.max_epochs, patience=args.patience,
        gravity_residual=args.gravity_residual, vw_outputs=args.vw_outputs, split=args.split, history_k=args.history_k,
        history_spacing=args.history_spacing if args.history_k else None, seed=args.seed,
        architecture=f"{base_arch}+{arm_arch}" if uses_gru else "mlp",
        **split_columns,
        inference_ms=None, test_wheel_rmse=None, test_torque_rmse=None, **{c: None for c in SIM_COLUMNS},
        **({c: None for c in ARRIVE_COLUMNS} if args.split else {}),
    )
elif not CHECKPOINT.exists():
    parser.error(f"--test-only needs a trained checkpoint, but {CHECKPOINT} does not exist")

if args.no_test:
    raise SystemExit

# ------------------------------------------------------------------------------
# Test evaluation (offline): how closely the model reproduces the expert's commands on unseen episodes
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Evaluate the saved checkpoint (best validation epoch), not the weights left after the last epoch.
# The checkpoint says whether it is a split model, so --test-only needs no --split.
best = load_policy(CHECKPOINT)
split = isinstance(best, SplitPolicy)


def predict_episode(x, t):
    """Model output for every frame of an episode, run the way the policy script runs it.

    Split models: the base network's p(arrived) goes through the same latch as in the policy, the latched flag is
    the arm network's input, and the wheels are 0 once it is set. Also returns the latch time (nan if never) and
    what latched it. Offline the states are still the expert's, so this tests the switch, not the closed loop.
    """
    if not split:
        return best.predict(x), np.nan, None
    wheels, p = best.predict_base(x)
    latch = ArrivedLatch()
    arrived = np.array([latch.update(pk, tk) for pk, tk in zip(p, t)])
    wheels[arrived] = 0.0
    t_latch = t[np.argmax(arrived)] if arrived.any() else np.nan
    return np.concatenate([wheels, best.predict_arm(x, arrived)], axis=-1), t_latch, latch.source


wheel_rmse, torque_rmse = [], []
arrive_err, n_timeout = [], 0
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
    # Gravity-residual models: add gravity back, so the error is on the full torque like every other model;
    # (v, w) models: converted back to 4 wheel commands, so the wheel error is measured the same way too.
    # Read from the checkpoint, so --test-only scores a saved model the way it was trained.
    tau_g = arm_gravity(d["q_mes"]) if best.config.get("gravity_residual") else None
    y_pred, t_latch, source = predict_episode(x, d["time"])
    wheel_pred, torque_pred = decode_outputs(y_pred, tau_g, best.config.get("vw_outputs", False))
    if split:
        # Latch time minus the expert's arrival (nan: never latched within the episode)
        arrive_err.append(t_latch - d["time"][np.argmax(d["nav_done"])])
        n_timeout += source == "timeout"
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

arrive_results = {}
if split:
    arrive_err = np.array(arrive_err)
    arrive_results = dict(
        test_arrive_err_median=np.nanmedian(np.abs(arrive_err)),        # s, over the episodes that latched
        test_arrive_early=int((arrive_err < -EARLY_LATE_S).sum()),      # episodes
        test_arrive_late=int(((arrive_err > EARLY_LATE_S) | np.isnan(arrive_err)).sum()),   # incl. never latched
        test_arrive_timeout=n_timeout,
    )
    print(f"\nArrived flag (latch vs the expert's arrival, {len(arrive_err)} episodes): median |error| "
          f"{arrive_results['test_arrive_err_median']:.3f} s, early (> {EARLY_LATE_S} s) "
          f"{arrive_results['test_arrive_early']}, late or never {arrive_results['test_arrive_late']}, "
          f"latched by the timeout {n_timeout}")

# Time of one single-frame prediction, as in the policy script (1 thread, one frame per simulation step; split
# models: both networks)
torch.set_num_threads(1)
frame = x[0]
step = (lambda f: (best.step_base(f), best.step_arm(f, 1.0))) if split else best.predict
for _ in range(50):   # warm-up
    step(frame)
times = []
for _ in range(1000):
    t0 = time.perf_counter()
    step(frame)
    times.append(time.perf_counter() - t0)
inference_ms = 1000 * np.median(times)
print(f"\nInference: {inference_ms:.3f} ms per frame (median of 1000, 1 thread)")
update_results(args.model, test_wheel_rmse=wheel_rmse.mean(), test_torque_rmse=torque_rmse.mean(),
               inference_ms=inference_ms, **arrive_results)
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
