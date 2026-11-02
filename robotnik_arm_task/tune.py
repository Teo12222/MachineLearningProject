"""Hyperparameter search for one model, on the validation set only (PLAN.md, "Hyperparameter tuning").

1. Staged grid search with train.py (no test evaluation): related settings are tried together, each stage starting
   from the best setting so far (lowest validation loss):
     1a. learning rate (3e-4, 1e-3, 3e-3) at batch 2048
     1b. batch 4096 and 8192, the learning rate scaled by sqrt(batch / 2048) (square-root rule for Adam)
     2.  hidden size (128, 256, 512) x layers (2, 3, 4)
     3.  weight decay (0, 1e-4) x dropout (0, 0.1)
     4.  the other two learning rates again, with the final size and regularisation (stage 1 chose them for the
         default network)
   Settings already trained are not trained again (~18 trainings in total).
2. Shortlist: the trials within TOLERANCE of the best validation loss (at most --top), checked in closed loop on a
   fixed subset of VALIDATION episodes, never test ones.
3. Winner: most validation targets reached, then the FASTEST TO TRAIN (differences in validation loss within the
   tolerance count as equal, so the cheaper setting is preferred). It is kept under the final name and its offline
   test is run; the trial checkpoints and results files are deleted.
The summary of all trials goes to results/tuning/<final>_tuning.csv (results.py doesn't read that folder).

Split models (PLAN.md 4a): --split, --gravity-residual, --base-from and --dagger are passed to every trial, as in
train.py. With --base-from only the arm is tuned on the copied base: the size and dropout stages set the arm's own
--arm-hidden / --arm-layers / --arm-dropout, and best_val_loss ranks the trials by the arm (the copied base's wheel
part is the same in every trial). Without --base-from every trial trains both networks with the trial's settings.
Epoch columns come from the arm (arm_best_epoch, arm_epochs). Tuning the base on its own: later (decided 2 Nov,
after the DART + DAgger comparison).

Usage (from the robotnik_arm_task folder):
    pixi run python tune.py --final mlp_robotframe_tuned
    pixi run python tune.py --final X --top 3 --parallel 4
    pixi run python tune.py --final split_dart_tuned_s0 --split --gravity-residual --base-from split_arrived_s0
        --dagger dart/a60 dart/a60b                # one command: the arm of a DART model (PLAN.md 4a)
Afterwards, the full closed-loop test of the tuned model (separately, it takes longest):
    pixi run python compare.py --test-set --model <final>
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from bc_data import quick_val_ids, split_episodes
from compare import POLICY_SCRIPT, POLICY_TIMEOUT, load_npz, outcome, run_episode, stored_start
from results import RESULTS_DIR, read_results, update_results

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_main"
CHECKPOINTS = SCRIPT_DIR / "checkpoints"

# Search space, by stage (PLAN.md). DEFAULTS are train.py's defaults, the starting point.
DEFAULTS = {"lr": 3e-3, "batch": 2048, "hidden": 256, "layers": 3, "weight_decay": 0.0, "dropout": 0.0}
BASE_LRS = [3e-4, 1e-3, 3e-3]            # learning rates at the reference batch size
REF_BATCH = 2048
BIG_BATCHES = [4096, 8192]
HIDDEN = [128, 256, 512]
LAYERS = [2, 3, 4]
WEIGHT_DECAY = [0.0, 1e-4]
DROPOUT = [0.0, 0.1]
# Validation losses within this fraction of the best count as equally good (then the faster setting wins)
TOLERANCE = 0.02
OPTIONS = {"lr": "--lr", "hidden": "--hidden", "layers": "--layers", "batch": "--batch",
           "weight_decay": "--weight-decay", "dropout": "--dropout"}


def scaled_lr(base_lr, batch):
    """Learning rate for a batch size, by the square-root scaling rule (4x the batch -> 2x the learning rate)."""
    return float(f"{base_lr * np.sqrt(batch / REF_BATCH):.3g}")


def run_train(name, setting, extra, options=OPTIONS):
    """Run train.py for one setting; returns True if it finished. Plots are saved, never shown."""
    cmd = [sys.executable, "train.py", "--model", name] + extra
    for key, value in setting.items():
        cmd += [options[key], str(value)]
    proc = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True, env=dict(os.environ, MPLBACKEND="Agg"))
    if proc.returncode != 0:
        print(f"  {name} FAILED:\n{proc.stderr[-1500:]}", flush=True)
    return proc.returncode == 0


def closed_loop(name, episode_ids, workers):
    """Success rate and median final error of checkpoint <name> in simulation on the given (validation) episodes."""
    os.environ["ROBOENV_POLICY"] = str(CHECKPOINTS / f"{name}.pt")   # run_episode passes the environment on
    run_dir = SCRIPT_DIR / "compare_out" / "tuning" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    experts = {i: load_npz(DATA_DIR / f"ep_{i:04d}.npz") for i in episode_ids}
    with ThreadPoolExecutor(workers) as pool:
        runs = list(pool.map(lambda i: run_episode(POLICY_SCRIPT, experts[i]["target"], run_dir / f"ep_{i:04d}.npz",
                                                   True, str(SCRIPT_DIR), POLICY_TIMEOUT, True, stored_start(experts[i])),
                             episode_ids))
    reached = sum(r is not None and outcome(r) == "reached" for r in runs)
    errors = [float(r["final_error"]) for r in runs if r is not None]
    return reached / len(episode_ids), float(np.median(errors)) if errors else float("inf")


class Search:
    """Trains settings stage by stage, never the same setting twice, and remembers every trial."""

    def __init__(self, final, extra, parallel, split=False, options=OPTIONS):
        self.final, self.extra, self.parallel, self.options = final, extra, parallel, options
        # Split models record their epochs per network; the tuned one is the arm
        self.epoch_cols = ("arm_best_epoch", "arm_epochs") if split else ("best_epoch", "epochs")
        self.rows = []                       # one per trial: name, stage, setting, results
        self.start = time.time()

    def best(self):
        ok = [r for r in self.rows if r["status"] == "ok"]
        return min(ok, key=lambda r: float(r["best_val_loss"])) if ok else None

    def stage(self, label, settings):
        """Train the settings not tried yet (in parallel); returns the best setting over all trials so far."""
        tried = [{k: r[k] for k in DEFAULTS} for r in self.rows]
        new = []
        for s in settings:
            if s not in tried and s not in new:
                new.append(s)
        names = [f"{self.final}_t{len(self.rows) + j:02d}" for j in range(len(new))]
        print(f"\nStage {label}: {len(new)} new settings ({len(settings) - len(new)} already trained)", flush=True)

        def train_trial(j):
            ok = run_train(names[j], new[j], self.extra, self.options)
            r = read_results(names[j]) if ok else {}
            print(f"  [{time.time() - self.start:5.0f} s] {names[j]} {new[j]} -> "
                  f"{'val ' + r['best_val_loss'] + ' (best epoch ' + r[self.epoch_cols[0]] + ')' if r else 'failed'}",
                  flush=True)
            return r

        with ThreadPoolExecutor(self.parallel) as pool:
            results = list(pool.map(train_trial, range(len(new))))
        for name, s, r in zip(names, new, results):
            self.rows.append({"trial": name, "stage": label, **s, "best_val_loss": r.get("best_val_loss", ""),
                              "best_epoch": r.get(self.epoch_cols[0], ""), "epochs": r.get(self.epoch_cols[1], ""),
                              "train_time_s": r.get("train_time_s", ""), "status": "ok" if r else "failed",
                              "val_success_rate": "", "val_final_err_median": "", "chosen": ""})
        best = self.best()
        if best is None:
            sys.exit("No setting finished training")
        print(f"  best so far: {best['trial']} (val {best['best_val_loss']})", flush=True)
        return {k: best[k] for k in DEFAULTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final", required=True, help="name of the tuned model (checkpoint + results file)")
    parser.add_argument("--top", type=int, default=3,
                        help="at most this many shortlisted settings (within the tolerance) checked in closed loop")
    # 50, the same episodes as compare.py --val-set (20 until 30 Oct)
    parser.add_argument("--val-episodes", type=int, default=50, help="validation episodes for the closed-loop check")
    parser.add_argument("--parallel", type=int, default=4, help="trainings at once (they share the CPU cores)")
    parser.add_argument("--workers", type=int, default=16, help="simulations at once in the closed-loop check")
    parser.add_argument("--max-epochs", type=int, help="passed to train.py (only to make quick test runs shorter)")
    # The model being tuned, passed to every trial as in train.py (PLAN.md 4a)
    parser.add_argument("--split", action="store_true", help="split networks (train.py --split)")
    parser.add_argument("--gravity-residual", action="store_true", help="train.py --gravity-residual")
    parser.add_argument("--base-from", metavar="MODEL", help="with --split: the copied base, so only the arm is tuned")
    parser.add_argument("--dagger", nargs="+", metavar="FOLDER", help="with --split: extra arm data (DART / DAgger)")
    parser.add_argument("--seed", type=int, default=0, help="training seed of every trial")
    args = parser.parse_args()
    if (args.base_from or args.dagger) and not args.split:
        parser.error("--base-from and --dagger need --split")

    threads = max(1, (os.cpu_count() or 4) // 2 // args.parallel)   # physical cores split between the trainings
    extra = ["--no-test", "--threads", str(threads)] + (["--max-epochs", str(args.max_epochs)] if args.max_epochs else [])
    extra += ["--seed", str(args.seed)] + (["--split"] if args.split else [])
    extra += ["--gravity-residual"] if args.gravity_residual else []
    extra += ["--base-from", args.base_from] if args.base_from else []
    extra += ["--dagger", *args.dagger] if args.dagger else []
    print(f"train.py options of every trial: {' '.join(extra)}", flush=True)
    print(f"Staged search, {args.parallel} trainings at a time ({threads} threads each)", flush=True)
    # With a copied base, size and dropout are the arm's own (train.py --arm-hidden etc.); the base keeps its own
    options = dict(OPTIONS, hidden="--arm-hidden", layers="--arm-layers", dropout="--arm-dropout") if args.base_from \
        else OPTIONS
    search = Search(args.final, extra, args.parallel, args.split, options)

    # 1. Search, stage by stage
    best = search.stage("1a lr", [dict(DEFAULTS, lr=lr, batch=REF_BATCH) for lr in BASE_LRS])
    base_lr = best["lr"]
    best = search.stage("1b batch", [dict(best, batch=b, lr=scaled_lr(base_lr, b)) for b in BIG_BATCHES])
    best = search.stage("2 size", [dict(best, hidden=h, layers=n) for h in HIDDEN for n in LAYERS])
    best = search.stage("3 regularisation", [dict(best, weight_decay=w, dropout=d) for w in WEIGHT_DECAY for d in DROPOUT])
    best = search.stage("4 lr re-check", [dict(best, lr=scaled_lr(lr, best["batch"])) for lr in BASE_LRS])
    ranked = sorted((r for r in search.rows if r["status"] == "ok"), key=lambda r: float(r["best_val_loss"]))

    # 2. Shortlist (validation loss within the tolerance of the best), checked in closed loop on validation episodes
    best_loss = float(ranked[0]["best_val_loss"])
    shortlist = [r for r in ranked if float(r["best_val_loss"]) <= best_loss * (1 + TOLERANCE)][:args.top]
    _, val_ids, _ = split_episodes(DATA_DIR)
    check_ids = quick_val_ids(val_ids, args.val_episodes)
    print(f"\nShortlist: {len(shortlist)} within {TOLERANCE:.0%} of the best validation loss; closed-loop check on "
          f"{len(check_ids)} validation episodes ...", flush=True)
    for r in shortlist:
        r["val_success_rate"], r["val_final_err_median"] = closed_loop(r["trial"], check_ids, args.workers)
        print(f"  {r['trial']}: val loss {r['best_val_loss']}, trained in {float(r['train_time_s']) / 60:.1f} min, "
              f"reached {r['val_success_rate']:.0%}, median final error {r['val_final_err_median']:.3f} m", flush=True)

    # 3. Winner: most targets reached, then the fastest to train
    winner = min(shortlist, key=lambda r: (-r["val_success_rate"], float(r["train_time_s"])))
    winner["chosen"] = "yes"
    summary = RESULTS_DIR / "tuning" / f"{args.final}_tuning.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(search.rows[0]))
        writer.writeheader()
        writer.writerows(search.rows)                # in training order, with the stage of each trial
    print(f"\nChosen: {winner['trial']} {', '.join(f'{k}={winner[k]}' for k in DEFAULTS)}\nSummary: {summary}")

    # The checked model itself becomes the final one (no retraining, so it is exactly the model that was checked),
    # then its offline test fills the test columns of results/<final>.csv
    shutil.copyfile(CHECKPOINTS / f"{winner['trial']}.pt", CHECKPOINTS / f"{args.final}.pt")
    update_results(args.final, **{k: v for k, v in read_results(winner["trial"]).items() if k != "model"})
    run_train(args.final, {}, ["--test-only"])

    # Remove the trial runs (their numbers are in the summary)
    for r in search.rows:
        (CHECKPOINTS / f"{r['trial']}.pt").unlink(missing_ok=True)
        (CHECKPOINTS / f"{r['trial']}_loss.png").unlink(missing_ok=True)   # train.py's loss plot of each trial
        (RESULTS_DIR / f"{r['trial']}.csv").unlink(missing_ok=True)
    print(f"\nDone in {(time.time() - search.start) / 60:.0f} min: checkpoints/{args.final}.pt, "
          f"results/{args.final}.csv")
    print(f"Next, the full closed-loop test: pixi run python compare.py --test-set --model {args.final}")


if __name__ == "__main__":
    main()
