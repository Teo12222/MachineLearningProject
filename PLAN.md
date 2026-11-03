# Project plan: behaviour cloning for the Robotnik arm task

COMP0245 assignment: train a PyTorch model that copies the expert controller, i.e. outputs the wheel commands
(4) and arm torques (7) from the robot state and the target. Report + code due **6 Nov**.

## Files

The shared pipeline lives on `master`, in `robotnik_arm_task/` (updated 31 Oct: everyone's work is merged, i.e. the
split networks and DAgger (15, 21), the (v, w) outputs (13) and the history windows, TCN and GRU (14)). Run everything
with `pixi run python <script>` from there.

| File | What it does |
|---|---|
| `robotnik_arm_task.py`, `robotnik_arm_control.py` | The expert (simulation + base and arm controllers, stop rules); saves one `.npz` per episode, including its hidden state (`nav_phase`, `nav_done`, `q_des_arm`) |
| `run_batch.py` | Runs the expert on many random targets in parallel to collect data (defaults: `data_main`, from episode 0) |
| `data_main/` | The dataset: 1,000 expert episodes after the expert fixes (git-ignored, ~3 GB; recollect with `run_batch.py`) |
| `data_pilot/` | The first 350 episodes, before the expert fixes: no longer used for training (still committed) |
| `bc_data.py` | Model inputs (27 robot-frame features) and outputs, `decode_outputs`, episode loading, adaptive / uniform subsampling, episode cleaning, gravity torques (`arm_gravity`), the train/val/test split by episode (fixed seed) and the fixed quick-test / validation subsets, (v, w) wheel outputs (`wheels_to_vw` / `vw_to_wheels`, 13), past-frame inputs (`with_history`, and `FrameHistory` at run time, 14), and per-batch windows for the TCN / GRU (`load_frames`, `window_batch`) |
| `bc_model.py` | The MLP (size set by options) and the TCN and GRU (14), each with its normalisation stored inside, plus save/load (`load_model` picks the class from the checkpoint) |
| `bc_split.py` | Split networks (15): base net (wheels + "arrived" flag) and arm net in one checkpoint, the arrival latch, the DAgger data loader, `load_policy` (loads any checkpoint: split, or MLP / TCN / GRU via `load_model`) |
| `train.py` | Trains one model (early stopping on the **validation** loss), then tests it offline on the test episodes; writes `results/<model>.csv`. Options: `--model`, settings, `--seed`, `--gravity-residual` (12), `--vw-outputs` (13), `--history-k` / `--history-spacing` (14), `--split` (15), `--dagger` (21), `--arm-from` (copy a trained split arm, so models differ only in the base), `--no-test` / `--test-only`; `--split` doesn't combine with `--vw-outputs` or `--history-k` yet. Saves the loss curves as `checkpoints/<model>_loss.png` (25) |
| `train_tcn.py` | Trains a TCN (14): same data, split, early stopping and results as `train.py`; windows cut per batch. Options: `--layers`, `--channels`, `--spacing` |
| `train_gru.py` | Trains a GRU (14): scored on whole windows, memory warmed up at the episode start. Options: `--hidden`, `--window`, `--burn-in` |
| `analyse_temporal.py` | Failure metrics for the temporal models from their simulation runs (first torque-clip hit, arm deviation, base off the floor), in `quick_*` / `full_*` results columns |
| `tune.py` | Hyperparameter search for a final model (staged grid via `train.py`, closed-loop check of the best few on the 50 validation episodes) |
| `robotnik_arm_policy_task.py` | Same simulation as the expert, but a trained model drives the robot (`ROBOENV_POLICY` picks the checkpoint; single MLP, history window, TCN / GRU or split) |
| `plot_torques.py` | Torque graphs (25): a policy run's 7 arm torques vs the expert's on the same target, latch / arrival and EE error, from saved files (no simulation) |
| `compare.py` | Expert vs model in simulation: `--target x,y,z`, `--episode N`, `--test-set` (all 150 test targets, writes the `sim_*` results), `--test-set --quick` (50 test targets, for looking only), `--val-set` (50 validation targets, for every decision) |
| `expert_relabel.py` | The expert's arm controller run offline along any saved run: what the expert would have commanded there (DAgger labels, 21) |
| `dagger.py` | DAgger for the split arm (21): runs a split model on training targets, labels it with `expert_relabel.py`, retrains via `train.py --dagger`, several rounds (data in the git-ignored `dagger/`) |
| `results.py` | Results recording (`update_results`); `python results.py` stacks every `results/<model>.csv` into `results_combined.csv` with seed means |

**Results:** every model's numbers are in `results/<model>.csv` (one file per model; see "Recording results" below).
Look there for the current state of each model instead of this file.

## How we work

- `master` = data collection, preprocessing and the shared pipeline. Each idea gets its own **branch**.
- **Keep the split and the evaluation identical on every branch** (`split_episodes` with the default seed, same
  test set, same `compare.py`), otherwise results can't be compared in the report.
- Fixes to shared files go to `master` first, then get merged into the branches.
- `checkpoints/` is committed (31 Oct: everyone's models and test plots in one place for the report); don't commit
  `compare_out/` (in `.gitignore`). Name every model uniquely (results and checkpoints share the name): git can't
  merge two versions of a checkpoint, so two people retraining the same name means picking one.
- Run with `pixi run python <script>` from `robotnik_arm_task/`. In PyCharm, scripts that import torch set
  `KMP_DUPLICATE_LIB_OK` themselves (needed on Windows).
- **Success at 1 cm, next to 5 mm, for every closed-loop test** (the brief's tolerance; decided 2 Nov, before the
  DART / DAgger results, so it doesn't move the goalposts). **Computed once, after the final comparison (B7)**, by one
  script over all saved runs; until then the only thing to do is keep the files (`--out-dir`, below).
  Defined as **held within 1 cm for 0.5 s** (the same "held" rule as the 2 / 5 / 10 cm columns; "ever within 1 cm"
  would count arms that swing through the target). 5 mm stays the main number, so every model stays comparable with
  the earlier ones and the expert; 1 cm is an extra column. Older models: recompute it from their saved policy episodes
  (`ee_pos`, `target` in each `.npz`) where the files still exist on that machine (`compare_out/` is git-ignored;
  quick runs without their own `--out-dir` were overwritten); leave it empty where they don't.
- **Every closed-loop run goes into its own output folder: `compare.py ... --out-dir compare_out/<model>_<set>`**
  (e.g. `compare_out/split_dart_a60_s1_val`, `..._test`; `run_batch.py --out-dir` likewise for the start-pose and
  noise runs). Without it, runs overwrite each other's episode files (`compare_out/test_set_quick/` etc.), and those
  files are the only source for anything computed later (1 cm, held, latch times). Seeds and machines too: one folder
  per model name, and model names already include the seed (and `_mac_` / `_pc_` where both trained it).
- **When you start or finish a test, message the others** (what ran, on which machine, where the files are, the
  numbers), so nobody waits on a result that is already in, or runs the same thing twice.

## Who does what

### Plan from 2 Nov (decided 2 Nov, Matteo and Kris; overrides "Plan from 1 Nov" where they differ)

Why a new order: DART and DAgger fix different things (DART: recovery near the expert's states, a stable arm that
never spins; DAgger: the states the policy actually visits, the last 2 cm), so the final model combines them. DAgger's
corrections only fit the policy they were collected behind (step 13's vw result), so **the DART amount is fixed first
and DAgger runs behind that exact model**. The seed spread (DART 16-30/50 on val) is bigger than the gaps being
decided, so the final choice uses 3 seeds on one machine. Seed repeats only for the candidates and the final model:
rejected models lost by more than the seed spread (see "Evidence per feature" under step 23).

**A. Before any run**
1. `data_main` verified on the machine that runs (1,000 episodes, 700 / 150 / 150 split, the 6 known drops). (Matteo,
   Kris)
2. **`dagger.py` accepts the DART data** (Matteo): a `--extra-data <folders>` option whose folders go into every round's
   `train.py --dagger`. Why: `dagger.py` starts every run with only its own round folders, so behind a DART model it
   would silently retrain without the DART data and "DART + DAgger" would really be DAgger alone.
   **Catch (2 Nov):** the default `--name split_dagger` reuses `dagger/split_dagger_r<k>` and `split_dagger_r<k>_s0`,
   which exist for rounds 1-5 on Matteo's PC, and finished steps are skipped: without its own `--name` the chain would
   collect and train nothing and "finish" with the old r1-r5. So the overnight run gets its own name (e.g. `--name
   split_dartdagger`), and `dagger.py` refuses a name whose round 1 was started from another `--start`.
   **Built (2 Nov), smoke test pending:** `--extra-data`, the name guard (round 1's folder records `--start` and
   `--extra-data` in `dagger_info.json`; a clash, or an old run without the record such as the 30 Oct r1-r5, stops
   before anything runs), and two more fixes found on the way: the rounds now copy the base from `--start`
   (`--base-from`; it was retrained, the same weights at seed 0 but a different base for the seed repeats, while every
   DART model copies `split_arrived_s0`'s), and they take lr, batch and weight decay from `--start`'s results file
   (only the sizes were passed, so a tuned start model's rounds would have trained with the defaults).
3. The DART-amount runs (B4) are the arm half of the learning curve (17), not run twice. (Matteo, Kris)

**B. Choose the final model**
4. **DART rounds: amount x seeds** (Kris, **by the end of 2 Nov**): **start high and go down** (decided 2 Nov, to
   save trainings: 2-3 amounts x 3 seeds instead of 5 x 3). Noise 60, 3 training seeds each, val 50:
   - First **600 episodes** (the 300 at `--seed 2` / `3` plus 300 new on `--seed 4` / `5`; check the new targets are
     not next to validation ones, as before), then **300**, then **150**.
   - **Stopping rule, fixed before looking:** compare seed means of "reached" (and held within 5 cm). If the lower
     amount is within the seed spread of the higher one, the higher one isn't worth it: go down another step. As soon
     as a lower amount is clearly worse, stop and take the amount above it. **On a tie, take the smaller amount:** the
     DART data goes into every later training (tuning, all DAgger rounds, all seeds), so less data makes the whole
     chain faster.
   - Hand over the chosen amount: **all 3 seeds' checkpoints, their `results/` files and val numbers, and the DART
     data folders** (Matteo reuses them, see 7).
   - Fallback, only if something crashes or runs long: if not all seeds are done by 3 Nov midday, Kris picks the best
     amount among the seeds that are finished and hands that one over; the remaining seeds still finish and still
     count for the DART-alone row (7).
4a. **Fix `tune.py` for split models, then tune the arm of the chosen DART model** (Matteo; reverses "Dropped:
   tuning of the final model" of 1 Nov). Fix first: `tune.py` doesn't pass `--split`, `--gravity-residual` or the
   `--dagger` folders, and its printout and summary use `best_epoch` / `epochs`, which are empty for split models
   (use `arm_epochs` / `arm_best_epoch`); tune the arm only (`--base-from split_arrived_s0`, the base is already
   good). **Run it before the DAgger rounds (6), on the DART model:** DART data is labelled by the expert on its own
   runs, so it fits any retuned arm, while DAgger data only fits the policy it was collected behind, so tuning after
   the rounds would leave them collected behind a different policy; tuning after the seed runs (7, 8) would invalidate
   those too. Judge in closed loop (`tune.py` already checks the best few on validation targets): the first tuning run
   improved the validation loss but made closed loop worse (step "Hyperparameter tuning"). ~1 h; if it doesn't fit by
   3 Nov evening, skip it and say so in the report.
   **Built (2 Nov, Matteo), smoke-tested:** `tune.py --split --gravity-residual --base-from split_arrived_s0 --dagger
   <DART folders>` passes these to every trial and reads the arm's epoch columns. Two things found on the way:
   - **Dropout with a copied base made unloadable checkpoints** (dropout adds layers; one `--dropout` built both
     networks, the copied base has none), and `tune.py` counted the failed closed-loop runs as 0% reached instead of an
     error. `train.py` now refuses a copied network whose size or dropout differs.
   - **Per-network size and dropout** (`train.py --arm-hidden / --arm-layers / --arm-dropout`, `--base-…` likewise,
     stored in the checkpoint; old checkpoints build as before), so the full grid (size and dropout too, ~18
     trainings) tunes the arm on the same copied base, and every model keeps `split_arrived_s0`'s base (checked: all
     DART models, r5 and the 300-episode seeds have it bit for bit). `dagger.py` passes the start model's per-network
     settings on, so the rounds train the tuned arm size.
   **Base tuning: later (decided 2 Nov),** after the DART + DAgger comparison (7): then every later model uses the
   tuned base. Why not now: the base already parks within 2-4 cm in every run, failed runs park as well as reached
   ones, and a new base now would mean new val / test runs for every row of the comparison.
   **Result (2 Nov, Matteo's PC, 19:28-21:23, 116 min):** arm only, DART-300 data (`dart/a60`, `dart/a60b`), base
   copied, seed 0, 18 trials; summary `results/tuning/split_dart_tuned_s0_tuning.csv`, log
   `compare_out/overnight/tune.log`.

   | arm | lr | val loss | reached (val 50) | arm training time |
   |---|---|---|---|---|
   | 256 x 3 (default) | 3e-3 | 0.0080 | (untuned DART: 32 PC seed 0; 30 / 16 / 25 Mac) | ~5 min |
   | 256 x 4 | 3e-3 | 0.0066 | not checked | ~7 min |
   | 512 x 3 | 3e-3 | 0.0061 | not checked | ~16 min |
   | 512 x 4, weight decay 0 (t12, `tune.py`'s pick) | 3e-3 | 0.0058 | 29 | ~27 min |
   | 512 x 4, weight decay 1e-4 (t14) | 3e-3 | 0.0058 | 26 | ~25 min |

   Learning rate 3e-3 and batch 2048 (the defaults) stayed best; dropout 0.1 made it worse (0.0076-0.0078).
   - **Validation loss falls steadily with arm size** (0.0080 -> 0.0066 -> 0.0061 -> 0.0058, -27%): a trend, not noise.
     **In closed loop there is no sign of it yet:** neither 512 x 4 trial beats the default's 32 on this PC. The same
     pattern as the first tuning run (better validation loss, no better driving): judge in closed loop only.
   - **t12 vs t14 is noise** (val loss 0.8% apart, reached 29 v 26 against a seed spread of ~14 for this recipe). Picked
     by the tie rule fixed before the results (decided 2 Nov, Matteo delegated the picks overnight): shortlist within
     5/50 on reached = tie -> the default settings if shortlisted, else the best validation loss -> **t14**. Trained
     again under `split_dart_512x4wd_s0/_s1/_s2` (the trial checkpoints are deleted by `tune.py`); `split_dart_tuned_s0`
     stays as `tune.py`'s own pick (t12), for the record.
   - **Which arm size for the DAgger rounds (rule fixed before the seeds' results):** three candidates, each with 3
     seeds
     on val 50 on this PC: 256 x 3 (the default; Kris's `split_dart_a60x2_s0/_s1/_s2`, run here as `..._pc_val`),
     256 x 4 (`split_dart_256x4_s*`, lr 3e-3, no weight decay: the size stage's setting; added on Matteo's suggestion as
     the middle ground, val loss -17% for ~1.4x the training time) and 512 x 4 (`split_dart_512x4wd_s*`). **Take the
     smallest whose 3-seed mean reached is within 2/50 of the best mean.** Why it leans small: the 512 x 4 arm trains
     ~5x slower (every DAgger round and every later seed), inference is fine for all (~0.17-0.19 ms per step), and a
     gain that only shows offline isn't worth that. The DAgger start is then the best seed of the chosen recipe (most
     reached, then closest approach); in
     the final comparison that recipe's row is the 3-seed **mean**, not the best seed.
   - **Result (2 Nov, 22:09), val 50, this PC, 3 seeds each:**

     | arm | reached | mean | held 1 cm | mean | held 2 cm | collapsed / spins |
     |---|---|---|---|---|---|---|
     | **256 x 3 (default)** | 32 / 18 / 25 | **25.0** | 37 / 31 / 37 | **35.0** | 46 / 43 / 48 | 0 / 0 |
     | 256 x 4 | 26 / 19 / 18 | 21.0 | 39 / 30 / 25 | 31.3 | 48 / 43 / 41 | 0 / 0 |
     | 512 x 4 (tuned) | 26 / 20 / 16 | 20.7 | 30 / 29 / 27 | 28.7 | 45 / 49 / 45 | 1 / 0 |

     **The default 256 x 3 wins outright** (best mean and smallest; the 2/50 margin never came into it), also within
     1 cm. The larger arms have the lower validation loss but are worse in closed loop (clearly for 512 x 4, mildly for
     256 x 4). Seed 0 of 512 x 4 reproduced tuning's own trial exactly (26), and the default's seeds here match Kris's
     Mac (30 / 16 / 25). **DAgger start: `split_dart_a60x2_s0`** (the default's best seed, 32/50; Kris's checkpoint).
   - **Report line either way:** "Tuning reduced the validation loss by 27% with a larger arm network but made
     closed-loop success worse (3 seeds each: 20.7 v 25.0 reached of 50)" (confirmed by the seeds), the offline vs
     closed-loop gap
     again.
5. **DAgger seed rounds** (Matteo, from 2 Nov): `split_dagger_r5` retrained with training seeds 1 and 2 on its existing
   DAgger data (seed repeats of the DAgger candidate).
   **Started 2 Nov 19:08 (Matteo's PC):** `split_dagger_r5_s1` / `_s2`, same data and default settings as s0, but
   the base copied (`--base-from split_arrived_s0`) instead of retrained: s0's retrained base came out identical
   only because it was seed 0, and copying keeps every model on one base (as Kris's DART seeds). Val 50 into
   `compare_out/split_dagger_r5_s<k>_val/`, logs `compare_out/r5_seeds/`.
   **Result (2 Nov, val 50, Matteo's PC):**

   | seed | reached | held 1 / 2 / 5 cm for 0.5 s | closest (median) | collapsed | arm spins in drive |
   |---|---|---|---|---|---|
   | 0 (30 Oct) | 26 | 13 / 32 / 42 | 0.50 cm | 8 | 9 |
   | 1 | 25 | 19 / 36 / 49 | 0.50 cm | 0 | 0 |
   | 2 | 24 | 11 / 36 / 50 | 0.59 cm | 0 | 0 |
   | mean | **25.0** (24-26) | 14.3 / 34.7 / 47.0 | | | |

   - r5 is very steady on reached (24-26, against DART's 16-30 over seeds). Part of it is probably that seeds 1 and 2
     retrain on data collected behind seed 0's policies (see B7, "What the seeds measure").
   - **The arm spins were seed 0's, not DAgger's:** seeds 1 and 2, same data and base, never spin or collapse and hold
     within 5 cm in 49-50/50, like DART (correction at step 15's "New problem found").
   - Against DART-300 on val: level on reached (25.0 v 23.7 mean), far behind within 1 cm (held 11-19 v 37 for
     `split_dart_a60x2_s0` on this PC; DART's seeds 1 and 2 not yet run here).
6. **DAgger behind the chosen (tuned) DART model** (Matteo): `dagger.py --start <DART model> --extra-data <DART folders> --name <new>`, rounds
   until the gain flattens (~5).
   **Done (2 Nov, 22:09-22:51, Matteo's PC):** behind `split_dart_a60x2_s0` (untuned: tuning made closed loop worse,
   see 4a; the default's best seed), `--extra-data dart/a60 dart/a60b --name split_dartdagger --rounds 5 --val-set`.
   200 policy runs per round, all latched; 0.76-0.85 M labelled frames per round; ~8 min per round in total.

   | round | DAgger frames (DART + rounds) | reached | held 1 / 2 / 5 cm | ever 1 cm | closest (median) |
   |---|---|---|---|---|---|
   | 0 (DART start) | 379 k | 32 | 37 / 46 / 50 | 38 | 0.41 cm |
   | 1 | 465 k | 29 | 25 / 46 / 50 | 39 | 0.48 cm |
   | 2 | 544 k | 31 | 35 / 49 / 50 | 43 | 0.42 cm |
   | 3 | 620 k | 19 | 26 / 45 / 50 | 36 | 0.67 cm |
   | 4 | 704 k | 31 | 27 / 43 / 50 | 40 | 0.39 cm |
   | **5** | 781 k | **35** | **41** / 47 / 50 | 44 | 0.35 cm |

   **Chosen: round 5** by the rule fixed before looking (most reached, then closest approach). The rounds are noisy
   (round 3: 19): each is a fresh training, so round-to-round differences include seed-like noise; the seeds (7)
   decide. No collapse or arm spin in any round.
7. **Final comparison** (Matteo): DART alone vs DAgger r5 vs DART + DAgger, 3 seeds each, val 50. **DART alone is
   not retrained:** its 3 seeds are Kris's runs from 4 (3 trainings saved). Consequence: that row comes from Kris's
   machine and the other two from Matteo's, so mark the machine in the table; Mac vs PC differed by a few episodes
   before (r5: 23 vs 26/50), smaller than the seed spread, so it doesn't decide the comparison on its own. If the
   DART row and the winner are within that difference, rerun the DART seeds on Matteo's machine before choosing. Chosen by "reached" (seed mean) and stability (held within 5 cm, arm spins, base off the
   floor).
   **What the seeds measure (2 Nov), for the report:** *"Seeds vary the arm's training; all models share one trained
   base (`split_arrived_s0`'s), so the reported spread excludes base variation."* Why one base: every row then differs
   only in the arm, which is what the comparison is about (all DART, DAgger and the 300-episode seeds have it bit for
   bit, checked 2 Nov). Two caveats to state with it:
   - The spread is the arm's only; a base retrained per seed would add some (how much is unknown; the base parks within
     2-4 cm in every run and failed runs park as well as reached ones, so probably little).
   - **DAgger seed repeats retrain on data collected behind seed 0's policies** (the rounds aren't rerun per seed: ~3 h
     each). That data fits seed 0 best, so seeds 1 and 2 may come out a little lower and the spread a little narrower
     than with full reruns. DAgger (and less so DART, where the expert drove) data also fits this base's habits (parks
     2-3 cm off, latches ~0.25 s early).
   - **Cheap check of the base's share, later** (with the base tuning after 7): train the base with seeds 1 and 2 (base
     only), put the final arm on each with `--arm-from` (no arm retraining), val 50 each: ~10 min of runs, shows how
     much a different base moves the result.
   **Result (2 Nov, val 50, all on Matteo's PC, 3 seeds each):**

   | model | reached | mean | held 1 cm (mean) | held 2 / 5 cm (mean) | collapsed / spins |
   |---|---|---|---|---|---|
   | DART alone (`split_dart_a60x2_s*`, run here as `_pc_val`) | 32 / 18 / 25 | 25.0 | 35.0 | 45.7 / 50 | 0 / 0 |
   | DAgger alone (`split_dagger_r5_s*`) | 26 / 25 / 24 | 25.0 | 14.3 | 34.7 / 47.0 | 8 / 9 (all seed 0) |
   | **DART + DAgger (`split_dartdagger_r5_s*`)** | 35 / 39 / 39 | **37.7** | **39.0** | 48.3 / 50 | 0 / 0 |

   **Winner: DART + DAgger**, by far more than the seed spread on reached (+12.7 over both, ranges don't overlap) and
   also the steadiest (narrowest spread, nothing lost). All three rows from one machine, so the Mac v PC caveat above
   doesn't apply. Seeds 1 and 2 of DART + DAgger retrain round 5's recipe on the same data (collected behind seed 0's
   policies, see "What the seeds measure"); they came out higher than seed 0 (39, 39 v 35), so no sign that this
   favoured seed 0. DAgger on top of DART helps far more than DAgger alone (+12.7 v +0 over DART; DAgger alone is
   level with DART on reached but far worse within 1 cm).
8. **Full test set (150) on the winner x 3 seeds** (Matteo; seeds shared with Kris's machine).
   **Done (2 Nov, 22:59-23:08, Matteo's PC), `compare_out/split_dartdagger_r5_s*_test/`:**

   | model (test 150) | reached | held 1 cm | ever 1 cm | held 2 / 5 cm | collapsed / spins |
   |---|---|---|---|---|---|
   | DART + DAgger r5, seed 0 | 99 | 98 | 121 | 144 / 150 | 0 / 0 |
   | seed 1 | 109 | 114 | 141 | 148 / 150 | 0 / 0 |
   | seed 2 | 103 | 103 | 136 | 145 / 150 | 0 / 0 |
   | **mean** | **103.7 (69.1%)** | **105.0 (70.0%)** | **132.7 (88.4%)** | 145.7 / 150 | 0 / 0 |
   | DART alone, 3 seeds (Kris, Mac) | 64.3 (42.9%) | 95.3 (63.6%) | | 138.0 / 149.3 | 0 / 0 |
   | DAgger alone r5, seed 0 (PC) | 62 (41.3%) | 36 (24%) | 90 | 85 / 121 | 25 / 38 |

   Expert 149/150. Picked on validation only (val 75%, test 69%: the usual small drop of a best-of-several pick). On
   the brief's 1 cm: held 70%, ever within 1 cm 88% (the other group's rule; their 149/150 is still well above).

**If B isn't enough: ideas to improve the final model further (2 Nov, cheapest first; only on 4 Nov, cut-off 4 Nov
midday, then report only).** Where the best models fail now is **precision, not collapse**: DART holds within 5 cm in
50/50 and within 2 cm in 45-47/50 but reaches 5 mm in 16-30/50; r5 was ever within 1 cm in 90/150 on test but within
5 mm in 62/150. So the fixes aim at the last 5-20 mm:
- **Report success at 1 cm too (14):** no cost. The brief's tolerance is 1 cm, ours is 5 mm to match the expert;
  show both with the reason. Already decided (2 Nov, see "How we work"); computed after 7 from the saved runs.
- **Seed ensemble:** average the arm outputs of the 3 seeds trained anyway (7, 8). Attacks the seed spread (16-30)
  directly and usually helps fine precision; no new training, ~0.5 ms per step (under the 1 ms step); a small change
  in `bc_split.py` (a policy holding several arms). Probably the best value.
- **Weight the frames near the target in the arm loss:** adaptive subsampling keeps only 6% of frames within 2 cm of
  the target vs 16% of all frames, exactly where it fails now. Uniform frames lost overall (step 15: they also thin
  out the start of the reach), but a loss weight on "EE within 2 cm" keeps the reach start and adds precision. One
  option in `train.py`, one retrain per seed.
- **More DAgger rounds** behind the final model: gains were flattening (+2, +2 at rounds 4-5), expect a few episodes.
- **Arm tuning (4a)** if it was skipped: last resort, the first tuning run improved validation but made closed loop
  worse.
- **Not in the time left:** joint-target outputs, base DAgger, new architectures (each needs new code plus its own
  DAgger rounds).
  Joint-target outputs are the most likely reason another group reaches 149/150 at 1 cm without DART / DAgger (to
  confirm with them; their success is "ever within 1 cm"): the net predicts `q_des`, `qd_des` and a fixed PD formula
  inside the model gives the torques (the expert's `kp (q_des - q) + kd (qd_des - qd)`, like the gravity residual adds
  g(q)), so damping and stiffness are structure, not learned, and an error is a steady offset instead of a growing
  drift. Labels exist: `q_des_arm` is saved, `qd_des` follows exactly from the saved torque. Design in `test_notes.md`
  on `test/arm-features` (2 Nov); a report experiment on expert data only if time allows, not the final model.
A report with ~40-50% at 5 mm, the 1 cm number, and a clear analysis of why the rest fail (the last 2 cm, seed
variance, offline vs closed loop) is a strong report: the analysis is worth more than a few extra percent.

**Overnight 2 -> 3 Nov (decided 2 Nov):** the 3 Nov work only waits for Kris's handover, not for the date, so it runs
overnight and the final comparison (7) moves to 3 Nov.

**Week from 3 Nov (decided 2 Nov):**

| Day | Matteo | Kris | Jamie |
|---|---|---|---|
| Tue 3 Nov | Final comparison (7), full test 150 on the winner x 3 seeds (8) | Helps get to the final model (runs part of 8's seeds); **once the 150 tests on the winner are done, starts the camera**. Noise-config change for (12), smoke-tested | Results table, big table, efficiency, plots, out-of-range targets |
| Wed 4 Nov | **Report: a good full draft** (own sections + the camera section written as future work / in progress; start-pose table filled in when the runs finish). **In the background: start-pose test (11)** | **Report: a good full draft** (own sections; sensor-noise table filled in when the runs finish). **In the background: sensor-noise runs (12)** | Own sections, results table with the final model |
| Thu 5 Nov | Final edit + README; joins the camera once the edit is done | **Finishes the camera**; its results replace the future-work section only if it works by **Thursday evening** (cut-off) | Conclusion, contribution statement, results table final |
| Fri 6 Nov | Buffer + submission | Buffer | Buffer |

Why the camera section is written on Wednesday as future work: the camera finishes the day before the deadline, so the
report must already be complete without it; Thursday only swaps in results, nobody rewrites sections on Friday.
- **Matteo, before the handover:** fix `tune.py` and add `--extra-data` to `dagger.py`, then **smoke-test both** (1 epoch,
  2 validation episodes, 1 DAgger round on a few targets): a chain that crashes at night loses the night. Run the r5
  seed repeats (5) while the machine is idle.
- **Matteo, overnight:** tuning (4a) and the DAgger rounds (6) chained in one command (`tune.py ... && dagger.py
  --start <tuned> --extra-data <DART folders> --name <new>`), one job at a time (parallel jobs share the CPU). PC awake and plugged in;
  both scripts skip finished steps, so an interruption resumes with the same command. Risk accepted: if tuning picks
  a bad setting, the rounds run behind it; check `results/tuning/<name>_tuning.csv` in the morning, and if tuning
  didn't help in closed loop, redo the rounds behind the untuned DART model on 3 Nov.
  **As run (2 Nov, Matteo's PC):** one script (kept outside the repo, decided 2 Nov) chains: (0) check every DART
  episode loads (a copy of `dart/a60/ep_0087.npz` arrived empty: 0 bytes; it would have crashed every training, so
  the chain stops at once on any unreadable file), (1) `tune.py --final split_dart_tuned_s0 --split --gravity-residual
  --base-from split_arrived_s0 --dagger dart/a60 dart/a60b`, (2) `dagger.py --start split_dart_tuned_s0 --extra-data
  dart/a60 dart/a60b --name split_dartdagger --rounds 5 --val-set`, (3) the winning round by the rule fixed before
  looking (most reached on val 50, then the smallest median closest approach; checked on the 30 Oct rounds, it picks
  r5 as then), (4) `compare.py --test-set` on it into `compare_out/<winner>_test/`, (5) its seeds 1 and 2: the same
  data (DART + rounds 1..k) and the tuned settings, base copied, val 50 each. Logs `compare_out/overnight/`
  (`chain.log` is the summary).
- **Kris, overnight:** the DART-side removal runs (9).

**C. Experiments on the final model**
9. Removal runs (23): DAgger off (Matteo); gravity residual off and other DART-side removals (Kris, **overnight
   2 -> 3 Nov**, at the chosen DART amount, once 4 is done; DART data is labelled by the expert on its own runs, so it
   fits any retrained model, and these runs don't wait for the final model). They measure each component in the DART
   model, not in DART + DAgger: say so next to the numbers. Evidence table from existing comparisons (Jamie).
10. Learning curve (17), base half (Kris): **only after the report drafts, if time is left** (moved 2 Nov). Why: the
    base already parks like the expert, so it can't change the final model; the arm half is 4.
11. **Start-pose test** (Matteo; **runs on Wed 4 Nov in the background while we write the report**, on the final model, with 12) and out-of-range targets (Jamie) (24). No new code: `run_batch.py --script
    robotnik_arm_policy_task.py --start-pose x,y,yaw --n 50 --out-dir starts/<model>_<pose>` with `ROBOENV_POLICY` set,
    and the same without `--script` for the expert as the reference; ~3 poses (offset, rotated, both). The world-frame
    baseline can no longer run (its inputs were removed in step 6), so its expected failure stays as words.
12. **Robustness: sensor noise only** (Kris; **runs on Wed 4 Nov in the background while we write the report**, on the final model, with 11;
    the noise-config change is done and smoke-tested on Tuesday; decided 2 Nov:
    friction and shoves dropped, the start-pose test (11) is the other robustness result). Uses the simulator's own
    noise: `noise_flag` and `robot_noise` in the config (noise on joint angles / velocities and base position /
    orientation, i.e. exactly what the policy reads); `collision_enable` (self-collision: the arm can hit the base) as
    one extra "different physics" run if time allows. Catches:
    - The `*_cov` values are **standard deviations**, not variances (passed straight to `np.random.normal` as the
      spread): the default 0.0001 is 0.1 mrad. Pick levels as standard deviations, e.g. 0.001 / 0.005 / 0.01 rad.
    - **Don't edit the shared `robotnik_arm.json`**: every script reads it (expert, `run_batch.py`, DART / DAgger
      collection, `compare.py`), so data collected meanwhile would silently be noisy. Use a copy
      (`robotnik_arm_noise.json`) picked by an environment variable (small change: both task scripts hard-code the
      file name).
    - The noise is not seeded (global `np.random`): runs aren't repeatable; use the 50 validation targets.
    - The expert reference needs **fresh expert runs under the same noise** (`run_batch.py` with the noisy config
      into its own folder, never `data_main`): `compare.py --test-set` scores the expert from the stored clean
      episodes.
13. Efficiency table (Jamie) (24).
14. Success at 1 cm next to 5 mm: decided 2 Nov ("How we work"); computed once after 7 from the saved runs (Matteo).

**D. Extensions:** camera (Kris from 3 Nov once the winner's test 150 is done, finished 5 Nov; Matteo joins after the
final edit; cut-off Thursday evening, see the week table); parking pose + arm early: future work in the report. Dropped unless time: parked-start arm episodes, randomised arm start
(20), more low targets (18).

**E. Report:** results table and the big table from "Evidence per feature" (Jamie); analysis plots (Jamie); own
sections drafted by **4 Nov evening** (Matteo: introduction, data collection (26), pipeline and evaluation method,
input features, split + DAgger, the 15 vs 14 point; Kris: subsampling, cleaning, temporal models, DART, learning
curve, robustness, offline vs closed loop; Jamie: gravity residual, (v, w), efficiency); **5 Nov** final edit + README
(Matteo), conclusion + contribution statement (Jamie); **6 Nov** buffer and submission.

**If it slips, cut in this order:** camera -> the 0 / 75 / 600 DART points -> tuning (4a) -> robustness down to sensor noise only.
Never cut: the final comparison with seeds, the full test set, the report.

### Plan from 1 Nov (decided 1 Nov; overrides the lists below where they differ)

Everything is merged into `master`. Commit small changes straight to `master` and pull often; use a branch only for
bigger code (Kris's GRU in the split model, Matteo's camera work) and merge it as soon as it is tested.

| When | Matteo | Kris | Jamie |
|---|---|---|---|
| **1 Nov** | GRU vw results (14); send Kris the GRU-in-split brief | **GRU in the split model** (15): answer by the end of the day (1) is split worth it compared with the temporal model, (2) can the GRU be used inside the split model, (3) which configuration is best (GRU base / GRU arm / both). Then **DART (19)** straight after | Quick **decision tree** for the results table (start from current `master`, not the old `DecisionTree` branch) |
| **2 Nov** | Pick the final model (rule below); **DAgger behind that exact model**; full test set (150) on it; start the **camera extension** | DART; then the **learning curve (17)**, incl. base vs arm data separately | Results table (22) with the numbers so far |
| **3-4 Nov** | Camera extension; start-pose test (24) | Robustness (24); learning curve | Out-of-range targets (24); efficiency table (24) |
| **all** | Seeds: 3 per final candidate, for your own models; the final model's seeds and removal runs (23) shared over the three machines | | |
| **by 4 Nov evening** | Own report sections drafted (as listed below) | | |
| **5 Nov** | Final edit, README | Fixes to own sections | Results table final, conclusion, contribution statement |
| **6 Nov** | Buffer only (deadline) | | |

- **Choosing the final model:** the best on the 50 validation episodes; if no GRU-in-split variant beats
  `split_dagger_r5_s0` (26/50), r5 stays. Then DAgger behind the winner (its corrections only fit the policy they were
  collected from, step 13's vw result), then the full test set on it.
  **Status 1 Nov evening (Kris):** no GRU-in-split variant beats r5 (step 15), but DART does: `split_dart_a60x2_s0`,
  30/50 validation (r5: 23 on the same Mac run), 76/150 test (r5: 62/150) (step 19). Proposal, to agree on 2 Nov: it
  becomes the candidate, and DAgger behind it keeps the DART data in training (it stops the arm spinning in the drive, step 15).
  **Update 2 Nov:** seed 1 of the same model gets 16/50, so on "reached" DART is about level with r5, not ahead; its clear
  gain is stability (always within 5 cm, no spinning arm). The seed spread (16-30) is bigger than most gaps we compare.
- **Ablations: light version** (step 23, replaces the full grid). The report shows a "component evidence" table built
  from the paired comparisons already made (robot frame, adaptive subsampling, gravity residual, split, DAgger rounds,
  (v, w), temporal models), each labelled with the model and seeds it comes from, plus on the final model only: 3 seeds
  and 2-3 removal runs (gravity residual off, DAgger off, GRU vs MLP base if the GRU wins).
- **Extensions:** camera first (Matteo). Then, **if the final model works well**, parking-pose prediction + starting
  the arm early (Matteo, the rest of 15). Why second: they change when and where the arm starts, the part that still
  fails (step 13's vw result showed how sensitive the arm is to that), for a gain in speed rather than success.
  **If either extension isn't done, describe it in the report as future work** (design, why it should work, what it
  would need, e.g. its own DAgger rounds behind the early-starting arm), using the designs in steps 15 and the
  camera section.
- **Dropped:** shallow vs deep check (16), base DAgger (21), tuning of the final model, DAgger rounds behind the vw base.
- **Open, to decide later:** report success at 1 cm too (the brief's tolerance) next to our 5 mm.

Numbers refer to the steps in the plan below. Everyone tunes the hyperparameters of their own models (on validation
only), runs several seeds for their final candidates, records everything in `results/`, and writes the report sections
for their own work as they go.

**Matteo: pipeline, input features, split networks, integration**
1. **27 Oct, critical path:** expert fixes 1-6, recollect 1,000-1,500 episodes (in the background), results
   recording + `train.py --name`, retrain the world-frame baseline (`mlp_worldframe`), then robot-frame inputs (6)
   and retrain. If it slips: expert fixes, new data and results recording first, robot-frame the next day. Until subsampling exists, train the baselines on a
   uniform subsample (e.g. every 10th frame) and note it in their results.
2. **Then:** quick test subset in `compare.py` (fixed ~50 test episodes, early: everyone uses it). The other shared
   improvements on master are Kris's.
3. **Input features** (all in `bc_data.py`, which Matteo owns anyway): robot-frame polar vs Cartesian (6) and the
   feature checks from the preprocessing slides (after 11).
4. **Then:** multi-model / split networks (15): base and arm networks, "arrived" flag, then the phase head,
   parking-pose prediction, starting the arm early. **Then DAgger for the split arm (21)** (decided 30 Oct; the
   phase head is parked): the arm drifts off the expert's states ~1 s into the reach, and the expert's arm controller
   can label the policy's own states offline, so no expert re-runs. Noise injection (19) stays Kris's: the two are
   compared in the report.
   **Order after DAgger (decided 30 Oct):** (a) if DAgger makes the arm reach, the full test set on the best round
   (the first real success rate); (b) integrate early: merge the split + DAgger into `master` and agree the temporal
   interfaces with Kris, so his model plugs into the split (temporal base / arm) when ready; (c) final model,
   ablations with tuning, seeds, full test sets (point 5); (d) the report. Parking-pose prediction and starting the
   arm early wait: the base already parks 9 cm from its spot, and starting the arm early adds new failure modes to
   the arm, the part that still fails, for a gain in speed only.
5. **Once the ideas have results:** combine the winning ideas into the final model (components as switches),
   ablations (23), start-pose test (24).
6. **Extensions, once a model works:** first parking-pose prediction + starting the arm early (the rest of 15; could
   reach targets faster than the expert), then straight after the camera extension (final section of the plan,
   shared with Kris: whoever is free).
7. **Report:** introduction, data collection (26), pipeline and evaluation methodology, input features, split-network
   results, README, and the final edit of the whole report.

**Kris: shared improvements, temporal models, data, camera**
1. **Shared improvements on master** (coordinate with Matteo, who owns `bc_data.py`): first `decode_outputs()` (10,
   needed early by Jamie's `(v, w)` outputs), then adaptive subsampling (7; the past-frame windows need its frame
   indices) and episode cleaning (8). Faster training + `--seed` (11) last: seed repeats only start once the main
   improvements are done.
2. Temporal data (14): past-frame windows, a TCN, then the RNN (GRU).
3. Noise injection, DART (19); randomised arm starting pose (20) optional (DAgger, 21, moved to Matteo, 30 Oct).
   Moved up (30 Oct): the split model's arm drifts off the expert's states ~1 s into the reach and the data has no
   examples of correcting from there (step 15 results); noise injection adds exactly those.
4. Learning curve (17), which decides whether to extend the dataset; more low-height targets (18).
5. Robustness tests (24: sensor noise, friction, shoves), on the final model.
6. Camera extension (final section of the plan), shared with Matteo: whoever is free once the final model works.
7. **Report:** shared improvements (subsampling, cleaning), temporal models, data experiments, robustness, the
   offline vs closed-loop discussion, camera extension (with Matteo).

**Jamie: outputs, results and report tables**
1. Gravity-residual torques (12) and `(v, w)` outputs (13).
2. Analysis plots (25); low priority: RANSAC preprocessing (9).
3. **Report:** output-format results and efficiency (parameters, inference time per step, 24), plus:
   - the **overall results table** (22) and keeping all tables/figures consistent (same units, names, seeds format);
   - the **conclusion** (limitations of the method and experiments, how they could be improved);
   - the **contribution statement** (draft from what everyone did, checked by all).

**Phases**

| Phase | Matteo | Kris and Jamie |
|---|---|---|
| 1. Critical path (27 Oct) | New data, results recording, both baselines | Read the plan, set up branches, start the code for your first idea; report outline |
| 2. Ideas | Input features, then split networks; merge master fixes into branches | Experiments + tuning on the new data, robot-frame baseline as the starting point; write method sections as you go |
| 3. Combine and test | Combine winners, ablations | Seeds, ablation runs, generalisation / robustness tests (on all three machines) |
| 4. Report (before the 6 Nov deadline) | Final edit, README, introduction/methodology polish, buffer | Own results and discussion sections; Jamie: results table, conclusion, contribution statement; proofreading |

**Ground rules**
- `bc_data.py` is the most shared file (features, subsampling, `decode_outputs`): Matteo owns it on `master`; keep
  branch changes there small or in separate functions to avoid merge conflicts.
- Model names include the branch (e.g. `temporal_gru_s0`, `split_arrived_s1`) so results files never clash.
- Report: ~1.5 pages of method + results per person, plus shared introduction, conclusion and tables (6 pages max).
- Short daily check-in: what's merged, what's blocking, which results are in `results/`.

## Plan

### 0. Expert fixes (first: they change the training data)
1. **`check_stop`: check "reached" before "stalled".** Currently a run already within 1 cm can be labelled
   stalled (happened to episode 10).
2. **Stop on reaching the target instead of the fixed 8 s minimum.** `min_time` makes the expert sit idle after
   reaching early, adding useless frames (every pilot episode lasts at least 8 s). **Decided:** drop `min_time`
   and end the episode the moment the end-effector is within the 1 cm tolerance, no hold time: reaching the target
   is all the data we need.
3. **Fix the stall rule** (3 s window now). **Decided:**
   - **Expert:** keep a stall check, but only once the base has arrived (as now), with a shorter window (e.g. 2 s;
     we checked: the expert never goes more than 1.75 s without improving, so no successful run is cut off).
   - **Policy: no stall check, only a maximum time.** The policy is checked from t = 0 (`check_stop(..., True)` in
     the policy script), and while the base turns in place the EE distance can go well over 2 s without improving,
     so a stall rule could stop a run that would have succeeded. A time cap is simple and can't do that. Pilot expert
     runs take 8-15.7 s (median 10.8 s, with the 8 s minimum), so e.g. **30 s** (about 2x the longest) is generous;
     anything not reached by then counts as a failure. It also replaces the 40,000-step cap (40 s) for policy runs,
     so a failing policy no longer runs for long.

   **Done (27 Oct), fixes 1-3:** `check_stop` checks reached -> time cap -> stall; no `min_time`; the step cap is now a
   time cap (`ROBOENV_MAX_TIME`: expert 40 s, policy 30 s); expert stall window 2 s, policy no stall check; outcome
   label `step_cap` renamed `time_cap`. Why: (a) **consistency**: a run is a success by one rule (within 1 cm), for
   the expert and the policy alike, so a reached run can no longer be labelled stalled, and success rates of the
   expert and every model are measured the same way; (b) **no useless training data**: frames of the expert sitting
   idle at the target (up to the old 8 s minimum) taught the model nothing about reaching but made up part of every
   short episode. (Ending right at 1 cm left the arm still moving with no "settled at the target" frames: changed
   on 28 Oct, see the settle update below.)
   Why a time cap instead of the step cap: the limits are decided in seconds (30 s = ~2x the longest expert run), and
   a step count would silently change meaning if the simulation time step changed. `ROBOENV_MAX_STEPS` is no longer
   read (the old line is left commented out in both task scripts); use `ROBOENV_MAX_TIME` (seconds) instead.
   The outcome label in `compare.py` / `run_batch.py` follows the new stop message (`time_cap`); old "Reached step
   cap" messages (e.g. in `data_pilot`) map to `time_cap` too, so old and new episode files read the same way.

   **Update (28 Oct), 1 s settle after reaching:** success is still the *first* time the EE gets within 1 cm, but
   the run then goes on for a fixed 1 s (`StopCriteria.settle_time`) before it stops. The stop message still starts
   with "Target reached ... at t=<time of reaching>". Same rule for the expert and the policy, so their final errors
   are measured the same way; during that second the time cap and the stall check are skipped (the run already
   counts as reached), and since it only starts after reaching it can't stop a policy early while the base turns.
   Why: (a) the model sees the arm **converging and holding** at the target, which it never saw when runs ended at
   1 cm; if the final tests don't stop the instant the EE crosses 1 cm (e.g. a fixed run time), a policy that never
   saw "hold" could drift or keep pushing, and a policy that drifts shows it in its final error; (b) more data close
   to the target, where the final precision comes from. Unlike the old 8 s idle tail these frames still show the
   error going down, and there are few of them (~1,000 per episode, ~10%; adaptive subsampling (7) can thin them).
   Pilot data (29 episodes that ran on >= 1 s after reaching): error 4.9 mm after 0.5 s, **2.5 mm after 1 s** (90%:
   4.8 mm), 0.6 mm after 2 s. Why fixed and not "until the error stops improving": we tried that first (stop when no
   0.5 mm improvement in 0.3 s, 2 s cap; ~1 mm final error) but it needed three settings and a separately saved
   reach time, because the run length varied. With a fixed second, a reached run always lasts its time to reach +
   1 s, so the sim time still measures the time to reach (minus 1 s, the same for every model) and nothing extra is
   saved.

   **Update (28 Oct), tolerance 5 mm for the expert and the policy:** a run counts as reached (and its 1 s settle
   starts) at 5 mm instead of 1 cm, the course's tolerance from the starter code (`REACH_TOLERANCE` in both task
   scripts, `SUCCESS_TOLERANCE` in `compare.py`). Why: the expert clearly can do better than 1 cm (in pilot runs that
   kept going it got from 1 cm to 5 mm within 0.5 s), so the data shows a more precise final approach; and the
   policy is judged by the **same rule** as the expert, so expert and model success rates in `results/` mean the
   same thing (we first considered keeping 1 cm for the policy, but then the expert's "reached" would be stricter
   than the models'). Our success test is stricter than the brief's 1 cm: say so in the report; the policy's frames
   are saved, so success at 1 cm (ever within 1 cm) can still be computed. Checked on all 350 pilot targets at 5 mm
   + 1 s settle:
   **349/350 reached**, final error median 1.25 mm (90%: 1.46 mm), sim time median 12.5 s (5.8-17.5 s), 4.30 M
   frames vs 3.81 M for the pilot (~13% more, now spent approaching and holding instead of idling). The one failure,
   episode 77 (target 0.36, -1.07, 0.50), can't get below 5.9 mm and is stopped by the stall rule (at 1 cm it reached
   with 6.2 mm): such rare "stalled" episodes are filtered by episode cleaning (8). The old world-frame baseline
   reached 0/52 test targets at either tolerance, so the policy side couldn't be compared yet.
4. **Make the code work from any start pose** (we assume the final tests may start the cart away from (0, 0)
   and/or rotated). Now we only make it *possible*; all training and testing stays at the (0, 0, 0) start until
   the final evaluation (step 24), where other start poses are tested.
   - `robotnik_arm_task.py` chooses the parking spot using the target's direction **from the origin**
     (`arctan2(target_y, target_x)`), not from the robot. Harmless now (the cart always starts at the origin, so all
     current data is correct), but wrong for any other start. Fix: measure from the start position,
     `to_target = TARGET_EE_POS[:2] - init_base_pos[:2]`, and use its angle. Gives identical results for a start
     at the origin, so it is safe to apply now.
   - Add a start-pose option to **both** task scripts (e.g. `ROBOENV_START=x,y,yaw`, like `ROBOENV_TARGET`) so the
     expert and the policy can be started anywhere, and pass it through `compare.py` / `run_batch.py` (needs a
     way to reset the base pose before the loop, e.g. PyBullet's `resetBasePositionAndOrientation`). When the
     option is not set, the start stays (0, 0, 0), so data collection and current runs are unchanged.

   **Done (28 Oct), fix 4:** `ROBOENV_START=x,y,yaw` in both task scripts (`set_start_pose` in `robotnik_arm_control.py`
   moves the base before the first step; it does nothing for the default origin start), `compare.py --start` (with
   `--target`; `--episode` / `--test-set` start the policy from each stored episode's start pose, so it always
   matches the expert run) and `run_batch.py --start-pose` (`--start` was already the first episode index). The parking
   spot is measured from the start position. Why: the final tests may start the cart elsewhere, and we need the
   start-pose test (24) without touching the training data: pilot episodes start at exactly (0, 0, 0), so all
   current runs give identical results. With `--start-pose`, `run_batch.py` still draws targets around the origin
   (not shifted to the start: the start-pose test in step 24 picks its own targets). The base height is taken from the raw sim
   state, not `GetBasePosition()`, so sensor-noise tests (24) don't add noise to the start pose.
5. **Review other expert quirks** (e.g. the base skids sideways while turning in place): fix or document.

   **Done (28 Oct), documented, nothing changed:**
   - **Sideways skid while turning in place:** in the pilot data the base drifts 9.3 cm (median, max 13.3 cm) during
     its first turn in place, 11.9 cm median for turns over 90 deg. Cause: the wheels have almost no sideways friction
     (`anisotropicFriction=[1, 0.01, 1]`). **Not changed**, because: the friction, contact and motor-gain settings are
     the course's own (identical to RoboEnv's `tests/mobile_base_kinematic_controller.py`), so the final tests most
     likely use them, and a policy trained on other physics would meet a different robot; a skid-steer base needs
     low sideways friction to turn at all; the base controller's `wheel_base_width` was calibrated for it; and the
     skid causes no failures (all 350 pilot episodes reach), since the expert re-aims at the parking spot every step.
     If it ever matters, turning more slowly in place (expert-only) would reduce it without changing the physics.
   - **Navigation timeout (25 s, absolute time):** after it the base counts as arrived wherever it is and the arm
     starts reaching.
   - **The base can circle a close parking spot until the timeout** (`data_main` episode 774, target -0.77, 0.87,
     0.61; watched in the GUI): the spot is only 0.52 m from the start, behind-left. After the turn in place the
     base passes it ~9-10 cm away, just outside the 8 cm capture radius (`pos_tol`), and the "passed the goal" rule
     doesn't fire. With the drive phase's steering capped at 0.3 rad/s and its minimum forward speed of 0.05 m/s, it
     can't turn in, so it loops (~1 m across, faster as it gets farther) until the 25 s timeout, then the arm
     reaches from 1.15 m away and the stall rule ends the run (27.2 s). After the timeout the base still coasts
     ~0.3 m with zero wheel commands (sliding on the low-friction wheels). 1 of 1,000 episodes; **documented, not
     fixed**: episode cleaning (8) drops it. If it ever needs fixing (e.g. for other start poses in 24): in the drive
     phase, go back to `align` (turn in place) when the heading error exceeds ~0.5 rad, but only farther than
     ~2 x `pos_tol` from the spot (closer, a small sideways offset already gives a large heading error, and
     `passed_goal` ends the drive). Not a wider capture radius instead: the base would park up to 15 cm off and
     some targets would be out of the arm's 0.85 m reach. Any such fix changes the data, so it needs a recollection
     (~8 min).
   - **Phase after arrival:** `base_state.phase` stays `"orient"` once `done` is set (also after the timeout), so
     the phase and the "arrived" flag are logged separately (fix 6).
   - **Last frame not logged:** the frame where a stop condition fires is not saved; `final_error` comes from it.
   - **`base_pos`** is PyBullet's base-link position (the sim's comment suggests the centre of mass, not the frame
     origin); expert, policy and features all use the same value, so it is consistent.
   - **The base parks up to 8 cm / 0.05 rad off its parking spot:** drive ends within `pos_tol = 0.08` m of the goal
     and orient within `ang_tol = 0.05` rad, with no later correction; the arm makes up the difference (test run on
     episode 10's target: parked 7.8 cm short). So the real parking pose differs from the saved `base_goal` by up to
     that much (matters for the parking-pose prediction in 15).
   - **The expert's final error depends on the stop rule:** it is the error 1 s after first reaching 5 mm (~1.25 mm,
     see the settle and tolerance updates in fixes 1-3), not at a crossing nor after the old 8 s hold. Say so in the
     report, and don't compare `sim_final_err` with pilot-era numbers.
   - **Phases that finish instantly have no frames:** `nav_phase` is logged after each step's update, so a target
     almost straight ahead has no `align` frames at all (the first frame is already `drive`), and a base already facing
     the target goes to `orient` and arrival on the same frame. Phase frame counts are uneven; keep it in mind when
     using `nav_phase` as RNN ground truth or class labels.
   - **`orient` mostly means "the arm is reaching":** the phase stays `orient` after arrival (all 979 reach frames in the
     test run). The real orient-phase frames are `nav_phase == orient` and `nav_done == False`.

   **Done (28 Oct), FIXED: the arm torques chattered** (flipped every 1 ms step, e.g. J6 between -5.5 and +10.5 N*m,
   J2 between 19.6 and 27.2, while the joint angles barely moved). **Why it had to be fixed before training anything
   further:** a model can't predict a torque that flips every step from the current state, so those labels are pure
   noise (part of the ~0.2 N*m torque RMSE was unlearnable); adaptive subsampling (7) would oversample exactly these
   frames because the torques change so much there, teaching the noise more strongly; and every model trained on it
   would have to be retrained after a later fix, breaking the comparison between models, so it was fixed once, now,
   before the one recollection. How common it was in the old `data_main`: in a sample of 250 episodes, heavy chatter
   in 19 of 24 targets at 0.2-0.3 m height, 31 of 31 at 0.3-0.4 m, 10 of 65 at 0.4-0.6 m, none above 0.6 m.
   Found by logging every controller step of one episode (580) in a copy of `arm_controller` (identical run to
   `data_main`'s). No contacts were involved (logged). **Two separate causes:**
   1. **Velocity gain too high for the 1 ms step in low arm poses** (e.g. episode 580): the set of frozen joints stayed
      constant, the planned joint velocities were tiny and smooth, but the measured velocities zig-zagged every step
      on all 7 joints and `kd * (qd_des - qd)` turned that into torque flips (a feedback loop that overreacts each
      step). Lowering `kd` removes it.
   2. **A joint held exactly on the edge of the joint-limit freeze** (e.g. episodes 231, 130): the reference of J2 is
      clipped to limit + margin, so the arm holds J2 on that line (within 1e-5 rad), and the old on/off rule ("frozen if
      within the margin and pushing into it") flipped it between frozen and free hundreds of times per reach (582 in
      231). Each flip changed the whole joint solution at once and kicked the torques of the arm. A lower `kd` doesn't
      help, and widening the freeze zone only moved the edge (J2 then sat on the new edge, same number of flips).
   **Fix (`robotnik_arm_control.py`):** `kd` 30 -> 20, and the on/off freeze replaced by **smooth joint-limit weighting**
   in the damped least-squares solve: a joint moving into a limit is slowed from weight 1 to 0 over `limit_band` =
   0.05 rad above the margin (and only as far as it actually pushes into the limit, fully from `limit_push_eps` = 0.01
   rad/s), its share of the motion going to the other joints. Weight 0 is the old freeze, but nothing switches
   abruptly. Stateless (depends only on the current angles), so it adds no hidden state the model would have to guess.
   **Tested before changing the expert** (headless expert runs, same targets each time):

   | Expert version (31 targets: 580, the 5 stalls, 10 low, 15 others) | Reached | Mean time to reach | Median final error | Chatter (share of reach) | Max arm sag while driving |
   |---|---|---|---|---|---|
   | Old: on/off freeze, kd 30 | 26/31 | 11.62 s | 1.34 mm | 4.1% | 0.058 rad |
   | kd 25 only | 29/31 | 11.24 s | 1.32 mm | 1.0% | 0.062 rad |
   | Smooth weighting, kd 25 | 29/31 | 11.45 s | 2.15 mm | 0.8% | 0.062 rad |
   | **Smooth weighting, kd 20 (chosen)** | 28/31 | 11.69 s | 2.03 mm | **0.0%** | 0.071 rad |

   On 20 of the 52 lowest targets (below 0.25 m), where cause 1 lives: smooth + kd 25 still chattered heavily in 3
   (episodes 383, 851, 580: 23-34% of the reach), smooth + kd 20 in none; both reached 20/20 with the same time
   (12.4 s) and final error (2.1 mm). Hence kd 20, not 25. Lower gains were also tried: kd 15 settles worse (some
   arms overshoot after reaching, episode 170 ends 15 mm off). The clean implementation was checked to give exactly
   the tested runs (5 targets, identical torques frame by frame).
   **Costs, accepted:** final error ~2 mm instead of ~1.3 mm (joints near a limit now slow down gradually, so the last
   millimetres take a little longer; still well inside 5 mm), a few runs end 4-6 mm off after the settle second, and
   slightly more arm sag while driving (0.071 vs 0.058 rad, lower damping). **Gains:** no chatter, and more targets
   reached (738, 752 and 774 now reach; 77 and 799 still stall).
   **Consequence:** `data_main` has to be recollected with the fixed expert (once; this is why both causes were
   fixed together). Every result recorded before (expert, `mlp_worldframe`, `mlp_robotframe`) is on the old data:
   the expert and `mlp_robotframe` get recomputed; `mlp_worldframe` can't be (its inputs are gone) and stays as the
   old-data reference row, marked as such.
6. **Log the expert's hidden state** in the `.npz`: navigation phase (`base_state.phase`), the "arrived" flag
   (`base_state.done`) and the arm joint reference (`q_des_arm`). These are the only things the expert carries over
   between steps that affect its commands and can't be rebuilt from saved data (time is already saved, the parking
   spot can be recomputed). Not model inputs: labels for the "arrived" flag (15) and ground truth to check whether
   the RNN (14) learned the phase. Also save the **start pose** (base x, y, yaw at t = 0): always (0, 0, 0) for
   now, but once `ROBOENV_START` exists the parking spot can't be recomputed without it. Only before recollecting,
   otherwise it costs another full collection.

   **Done (28 Oct), fix 6:** new fields in the expert's `.npz`: `nav_phase` (int8 index into `nav_phase_names` =
   align / drive / orient), `nav_done` (the "arrived" flag), `q_des_arm` (N x 7), `start_pose` (x, y, yaw measured at
   t = 0) and `base_goal` (parking spot x, y, yaw; saved although it could be recomputed, so nobody has to get the
   recomputation right). Policy files get `start_pose` too. Each frame stores the state *after* that step's controller
   update, i.e. the state that produced that frame's commands (on the arrival frame `nav_done` is already True and the
   arm already reaches). Per-frame arrays are saved as **float32, uncompressed** (`np.savez`). Why: half the size (1.87
   instead of 3.73 MB per pilot episode, ~2.8 GB for 1,500), and the model trains in float32 anyway (`bc_data` casts),
   so it sees exactly the same numbers. Compression would save another ~25% (1.41 MB) but makes every load slower;
   not worth it at this size. `target`, `final_error` and the poses stay float64; old float64 files load the same way.

   **Checked (30 Oct): `q_des_arm` is hardly hidden in practice.** Before arrival it is the arm's start pose (the
   same in every episode); during the reach the current 27 inputs predict `q_des_arm - q` with R2 0.98-1.00 per joint
   (validation episodes), leaving 0.01-0.15 N*m of torque, about the current torque error. So the arm needs no
   memory for it: a single-frame arm network is fine (see 14 and 15).

**Only the expert fixes have to come before recollecting.** Input features are built by `make_features` when
episodes are loaded, not when they are recorded, and everything they need is already saved (`base_pos`,
`base_bearing`, `q_mes`, `qd_mes`, `ee_pos`, `time`, `target`): robot-frame inputs, polar vs Cartesian, sin/cos
joints, base velocity, scaling can all change later on the same data. The quick test subset is taken from the new
test split, so it comes after the data.

Then **recollect the dataset** with `run_batch.py` into a new folder (models are retrained on it once results
are recorded, see below).
- Pass `--start 0` (the default is 50; `data_pilot` uses episodes 0-349) and a new `--out-dir` (the default is
  `data_pilot`).
- It is quick: the pilot took ~2.5 s wall time per episode with 16 workers, so 1,500 episodes is a few minutes. Run
  it in the background while building results recording.
- Then point the scripts at the new folder: `DATA_DIR` in `train.py` and the `--data-dir` default in `compare.py`
  (both `data_pilot` now).

**Done (28 Oct):** collected **1,000 episodes** into `robotnik_arm_task/data_main` (`run_batch.py --n 1000 --start 0
--seed 0 --out-dir data_main`, 16 workers, after expert fixes 1-6, the 1 s settle and the 5 mm expert tolerance).
Why 1,000: the low end of the 1,000-1,500 range; the learning curve (17) decides whether to extend it (same seed,
`--start 1000`). `data_main` is git-ignored (~3 GB). Everything now points at it: `DATA_DIR` in `train.py`, the
`--data-dir` default in `compare.py`, the quick check in `bc_data.py`, and `run_batch.py`'s defaults (`--out-dir
data_main`, `--start 0`, so running it without options can only add missing episodes to `data_main`). Seed 0 as for the
pilot, so episodes 50-349 have the same targets as the pilot ones. Note: `data_pilot` (351 files, ~1.6 GB) is
**committed to git**, against the "don't commit the dataset" rule; removing it (`git rm --cached`) is a team decision,
since it stays in the history either way.

**Superseded (28 Oct):** this first `data_main` had the arm torque chatter (see the fix in the expert quirks, fix 5),
so it is recollected with the fixed expert (same command, same seed, same targets). The numbers below are for the
first collection.

**Recollected (28 Oct), the current `data_main`:** 1,000 episodes, no crashes or timeouts, 12.3 M frames; **995
reached, 5 stalled**; final error median 1.65 mm (90%: 2.42 mm; 9 reached runs end above 5 mm after the settle
second); sim time median 12.4 s. **Heavy chatter in 0 of 1,000 episodes** (no 100 ms window above 20 N*m per step
in any height band; before the fix: most targets below 0.4 m). Stalled: 77, 752, 799 (close mid-height targets, the
joint-limit dead ends from before) and two new ones, 297 and 791: far targets (~2.8 m) that got to 5.3 mm and were
stopped by the stall rule (2 mm in 2 s) just short of 5 mm, since joints near a limit now slow down gradually. Drop
all 5 in episode cleaning (8). 774 (the circling base) now reaches.

**Result:** 1,000 episodes in 8.4 min, no crashes or timeouts, 2.8 GB, 12.1 M frames; **995 reached, 5 stalled**
(final error median 1.25 mm, sim time median 12.2 s, max 27.2 s). The 5 stalled episodes (drop them in episode
cleaning, 8):
- 77, 738, 752, 799: close targets (~1.0-1.2 m from the start, mid height) where the arm ends 6-11 mm from the
  target. **Episode 77 checked (GUI + saved frames): a real dead end, not the stall rule cutting off a slow
  approach.** The base parks normally; the error falls to 5.9 mm at ~7.5 s and then creeps back up to 6.9 mm. Three
  joints sit at their lower limits (J2 0.056 rad, J4 0.057 rad, J6 0.024 rad above them, inside the controller's
  0.05 rad safety margin, so they are frozen), and the arm can't get closer with the other joints. Visually it
  looks reached (the target sphere has a 3 cm radius), and it would count as reached at 1 cm; at our 5 mm rule it's
  0.9 mm short. **Episode 799 checked too, same cause:** best 9.6 mm; J4 and J6 at their lower limits (0.050 and
  0.024 rad above them). The base parked ~8 cm from its spot (within `pos_tol`), and the folded arm can't pull the
  end-effector back in: it stays ~1 cm too far out, not too high or low. 738 and 752 (closest 7.0 and 7.1 mm, also
  close mid-height targets) are assumed to be the same, not checked individually. These are not low targets (all 0.50-0.61 m high; the 105 targets below 0.3 m all reached): the weak spot
  is close + mid height (radius < 1.2 m, height 0.45-0.60 m: 4 of 11 episodes stall). If it matters for the
  models, collect more close targets (like 18), rather than keep the stalled episodes.
- 774: the base circles its parking spot until the 25 s timeout (see the expert quirks, fix 5).

**Dataset size: start with 1,000-1,500 episodes, extend only if it helps.**
- Why more than now: the model has to generalise over *targets*, and training currently sees only ~245 different
  targets spread over a 3D region (full circle, 1-3 m out, 0.2-1.1 m high). ~1,000+ episodes cover it far more
  densely (probably part of why low and close targets do badly now).
- Why not 3,000 straight away: disk (~3-4 MB per episode, so ~10-12 GB for 3,000), memory (~30 million frames, ~4 GB),
  and slower test runs. Start smaller and let the **learning curve (step 17)** decide: extend towards ~3,000 only if
  performance is still improving with more episodes. Report this ("we stopped at N because the curve flattened").
- Extending is cheap: each target depends only on the seed and the episode number, so
  `run_batch.py --seed <same> --start <N> --n <more>` adds new targets without repeating old ones.
- Note: pilot episodes 0-49 have different targets from what `run_batch.py` now draws for those numbers (same angle
  and height, smaller radius: collected before the radius range changed). Doesn't matter for the new dataset, but
  compare new and pilot episodes only from episode 50 on.
- **Don't commit the dataset** (too big for git; the brief doesn't require submitting data). Consider saving float32
  instead of float64 or using `np.savez_compressed` (part of the expert fixes) to cut the size.
- At this size **subsampling (step 7) is needed**, not optional: the full data won't fit comfortably in memory.
- **Test set:** 15% is ~150-225 test targets (more reliable success rates than 52). Keep a **fixed smaller subset**
  (e.g. ~50 test episodes, always the same ones) for quick checks while developing, and use the **full test set for
  the final numbers**. Only full-test-set runs write the `sim_*` columns in `results/`, so quick checks never mix
  with reported results.

### Recording results (build before retraining anything on the new dataset)
- **One results file per model:** `robotnik_arm_task/results/<model>.csv`, named after the checkpoint (one header row
  + one row of values). Each training run writes only its own file, so parallel runs can't overwrite each other and
  branches never get merge conflicts (as long as model names differ per branch). The `results/` folder is committed.
- **The seed is part of the model name** (e.g. `robotframe_s0`, `robotframe_s1`), not a separate column.
- **`train.py` needs a `--name` option now:** it always saves to `checkpoints/mlp_baseline.pt`, so `mlp_worldframe`
  and the robot-frame model can't be told apart. Add `--name` (checkpoint + results file) with results recording;
  `--seed` stays with step 11.
- `train.py` fills the training and offline-test columns; `compare.py` fills the `sim_*` columns **only with
  `--test-set`** (single `--target` / `--episode` runs are for looking, not reporting). The expert gets its own
  `results/expert.csv` with the `sim_*` columns.
- A small **combine step** stacks all files into one table (one row per model) and groups seeds into mean +- std.
  Empty cells mean "not measured yet" (e.g. trained with `--no-test`), not zero.
- **Columns:**

  | Column | From | Meaning |
  |---|---|---|
  | `model` | `train.py` | Checkpoint name (also the file name), including the seed |
  | `train_episodes` | `train.py` | Number of training episodes (learning curve, step 17) |
  | `epochs`, `batch`, `lr`, `hidden`, `layers` | `train.py` | Hyperparameters used (plus any model-specific ones, see "Hyperparameter tuning") |
  | `best_val_loss`, `best_epoch` | `train.py` | Lowest validation loss and the epoch it happened |
  | `train_time_s`, `parameters`, `inference_ms` | `train.py` | Efficiency (`inference_ms` = one single-frame prediction, 1 thread, must be well under 1 ms) |
  | `test_wheel_rmse`, `test_torque_rmse` | `train.py` | Offline test: mean per-episode RMSE (rad/s, N*m) |
  | `sim_success_rate` | `compare.py --test-set` | Fraction of test targets reached within 5 mm (same rule as the expert) |
  | `sim_final_err_mean`, `sim_final_err_median` | `compare.py --test-set` | Final distance to the target (median is robust to runaway runs) |
  | `sim_time_to_reach` | `compare.py --test-set` | Mean episode length of successful runs: time to reach + the fixed 1 s settle (same offset for all) |

- **Wheel RMSE is always measured on the 4 wheel speeds**, even for models that predict `(v, w)`: convert their
  output back to 4 wheel speeds with `decode_outputs()` (step 10) first, so the column stays comparable.
- The new dataset (1,000-1,500 episodes after the expert fixes, possibly extended later) replaces `data_pilot`; results recorded on `data_pilot`
  are not comparable and get recomputed or deleted when we switch.

**Done (28 Oct):** `results.py` (`update_results`), `train.py --model <name>` and `compare.py --model <name>` (one option
name in both, matching the `model` column; the plan above said `--name`). Each script only updates its own columns
of `results/<model>.csv`, so they never overwrite each other. Details:
- `train.py` writes the training columns after training (plus `frame_step`, see below) and **empties the old test
  and `sim_*` columns**: they belonged to the previous weights. After testing it writes `test_wheel_rmse`,
  `test_torque_rmse` and `inference_ms` (median of 1,000 single-frame predictions, 1 thread, as in the policy
  script). `train_time_s` is the training loop only, `best_epoch` counts from 1.
- `compare.py --test-set` writes the `sim_*` columns of `results/<model>.csv` and of `results/expert.csv` (the stored
  expert runs on the same test episodes). A test episode without a policy result counts as failed;
  `sim_time_to_reach` is the mean episode length of reached runs, so it includes the 1 s settle (the same for every
  model and the expert; subtract 1 s in the report if an absolute time to reach is quoted).
  `compare.py` now stops if the checkpoint doesn't exist (before, it ran the expert alone).
- `python results.py` stacks every `results/<model>.csv` into `results_combined.csv` (next to `results/`, so it is
  never read back as a model): one row per model, plus one "`<name>` (mean +- std, N seeds)" row per model trained
  with several seeds (`<name>_s0`, `_s1`, ...). A setting the seeds share is shown once; a mean over only some
  seeds is marked, e.g. `0.5 (1/2)`.
- **Quick test subset:** `compare.py --test-set --quick` runs only a fixed 50 of the test episodes
  (`bc_data.quick_test_ids`, drawn with a fixed seed, so everyone checks the same ones), for quick closed-loop checks
  while developing (about a third of the time). It prints the same table and plot (`test_set_quick_final_error.png`,
  its own `test_set_quick/` folder) but **never writes results**: only the full `--test-set` fills the `sim_*`
  columns, so quick checks never mix with reported numbers.
- **Decisions on validation, from 30 Oct: `compare.py --val-set`** (the same check on a fixed 50 validation
  episodes, `bc_data.quick_val_ids`, its own `val_set/` folder, never writes results). Why: choosing between models on
  the quick test subset (which model, setting or DAgger round to keep) is tuning on test episodes, and those 50 are
  part of the 150 the final numbers come from. So every decision now uses `--val-set`; the quick test subset is
  still useful to see how a model does, but never to choose; the full test set is only for the final numbers.
  `tune.py`'s closed-loop check uses the same 50 (it used 20: `--val-episodes` default and `quick_val_ids` now 50,
  so differences aren't lost in the noise; the check takes ~2.5x longer) and `dagger.py --val-set` runs it after each
  round. **Earlier decisions were made on the quick test subset** (gravity residual on, uniform arm frames rejected;
  step 15): re-checked on the validation 50, see step 15. Tell the team: `--quick` was everyone's development check.
- **The test set depends on the dataset:** `split_episodes` shuffles the episode ids found in `data_main`, so if the
  dataset changes (e.g. extended with `--start 1000` after the learning curve), the train/val/test split, and with it
  the quick 50, change too. Then **every model's results must be recomputed** (retrain + `--test-set`), as when we
  switched from `data_pilot`.
- **Uniform subsample for now:** `train.py` trains and validates on every 10th frame (`FRAME_STEP = 10`, recorded
  as the `frame_step` column), via a crude `every` argument of `load_episodes` (thinned per episode while loading,
  so the full 12 M frames never sit in memory). Placeholder until adaptive subsampling (7) replaces it. The test
  RMSE always uses every frame. **Replaced at the merge (29 Oct), see below.**
- The training seed is still fixed (`torch.manual_seed(0)`) until `--seed` (11): put the seed in the model name
  yourself (e.g. `mlp_worldframe_s0`). **Since 28 Oct: `train.py --seed N` (default 0).**

**Merge of Matteo's and Kris's `master` work (29 Oct).** Both sides kept; how the overlaps were resolved and why:
- **Subsampling:** Kris's `subsample_indices` / `load_episodes(..., keep_frac, mode)` replace the crude `every=` /
  `FRAME_STEP` (they were its placeholder). **`train.py` now uses `SUBSAMPLE = "adaptive"`, `KEEP_FRAC = 0.1`** (same
  number of frames as before; Kris's default `None` = all 12 M frames, ~10x slower). Why adaptive now: it samples
  more frames where the expert's commands change (phase switches, the start of the reach), which is where the policy
  fails (late turn -> drive switch, imprecise reach); the reason to hold back was the expert's arm chatter, which
  adaptive sampling would have oversampled, and that is fixed. **Consequence:** models trained from now on use
  adaptive frames, all earlier models (`mlp_robotframe`, `mlp_robotframe_tuned`) used uniform ones (`frame_step = 10`
  in their results files; new ones record `subsample` / `keep_frac`), so they differ in the training data too.
  **To do: test the difference at some point** (step 7's "compare against uniform"): train the same model with
  `SUBSAMPLE = "uniform"` and `"adaptive"` and compare offline and closed loop (`--quick`, then the full test set),
  so the choice is backed by a result in the report. Validation stays uniformly subsampled in both, so the
  validation losses are comparable.
- **Episode cleaning:** Kris's `clean_ids` runs after the split in `train.py`. On the recollected `data_main` it
  drops 6 episodes: the 5 stalled ones (77, 297, 752, 791, 799) and 834 (reached, then drifted to 10.1 mm in the
  settle second, above the 1 cm check). So training now uses 696 / 149 / 149 episodes instead of 700 / 150 / 150
  (`train_episodes` = 696 in new results; earlier models used 700). The cleaning threshold stays 1 cm (the expert stops
  at 5 mm, but runs that drift to 5-10 mm while settling are still good demonstrations; the constant's comment says
  so). **`compare.py --test-set` still runs all 150 test targets** (it doesn't clean): a target the expert couldn't
  reach is still a valid closed-loop test, and every model and the expert are judged on the same 150.
- **Offline test scored through `decode_outputs`** (Kris, step 10): predictions are decoded into 4 wheel speeds and 7
  torques and compared with the expert's own commands from the episode file, so `test_wheel_rmse` /
  `test_torque_rmse` mean the same for every output format (e.g. Jamie's `(v, w)`). Same numbers as before for
  models that output wheels + torques directly.
- **Policy script:** robot-frame `make_features(..., ee_pos)`, then Kris's `decode_outputs(model.predict(x))`, then
  both safety clips (wheels +-3 rad/s, torques +-30 N*m) on the decoded commands.
- **Two merge slips caught and fixed:** `torch.manual_seed(0)` had replaced `torch.manual_seed(args.seed)` (so
  `--seed` was ignored), and `FRAME_STEP` was still recorded although no longer used. Checked after the merge: all
  scripts compile, `bc_data.py`'s self-check runs, and a 1-epoch training gives 856 k training frames (every 10th),
  27 inputs, and the new results columns.

Then **retrain the current world-frame baseline on the new dataset** under its own name (e.g. checkpoint
`mlp_worldframe.pt`, so `results/mlp_worldframe.csv`), with `train.py` and `compare.py --test-set`. This is the
reference row: the first results recorded, on the same data as every later model.

### Baseline fix: robot-frame inputs (after the world-frame retrain, on master)
6. **Robot-frame inputs** in `bc_data.py`: target (and other positions) relative to the base, in the base's own
   frame, instead of absolute world coordinates. Needed by every branch, because the final tests may start the
   cart away from (0, 0, 0):
   - The current baseline uses world-frame inputs (absolute base x, y, heading and target), and every training
     episode starts at (0, 0) facing +x, so it will very likely fail from any other start pose.
   - On a flat floor only the target's position relative to the robot matters, and our targets already cover the
     full circle and 1-3 m around the cart. With robot-frame inputs, random cart starts would mostly repeat
     situations we already have, so **no extra start-pose data is needed** (teammate's point; only true with
     robot-frame inputs).
   - **Test it instead of collecting it, at the end** (step 24): evaluate on targets with random cart starts. The
     robot-frame model should do as well as from the origin; the world-frame baseline should fail. Evidence for
     the report at no data cost. Until then everything runs from (0, 0, 0).
   - The world-frame model retrained on the new data (`mlp_worldframe`) stays as a row in the results table and is
     kept for the start-pose test at the end (step 24); give the robot-frame model a new name so it doesn't overwrite
     it. Robot-frame inputs become a switch for the ablation.
   - **Form of the robot-frame inputs (switch, compare both):** Cartesian (target dx, dy, dz in the base frame) vs
     polar (horizontal distance to the target, bearing error as sin/cos, target height). Polar matches the expert's
     own decision variables (it turns until the bearing error is small, drives until the distance is small), which
     should help around the phase switches where the world-frame baseline failed, and especially helps tree models.
     - The bearing must be sin/cos, not a raw angle (no jump at +-pi).
     - Distance and bearing alone are not enough for the arm: keep the target height and the 3D target relative to
       the end effector either way. `ee_pos` is not a model input yet: pass it to `make_features` (it is saved in
       the `.npz`, and the policy script already computes it every step).
     - The expert steers to the parking spot, which depends on the target height and the start-to-target line: keep
       the target height, or use the distance and bearing to the parking spot instead.
     - A switch in `make_features` (`bc_data.py`, e.g. `ROBOT_FRAME_POLAR`), Cartesian by default. Matteo runs the
       comparison and records both in `results/`.

Then **retrain the baseline** with robot-frame inputs (on the recollected data); all branches start from this.

**Done (28 Oct): robot-frame inputs, Cartesian only, replacing the world-frame ones** (`bc_data.make_features`,
27 inputs): target forward/left of the base in the base's frame (2), target height (1), target relative to the end
effector in the base's frame (3), arm joint angles as sin/cos (14, see "Feature checks") and velocities (7). No absolute base position or heading.
`make_features` now also takes `ee_pos` (saved in every episode, computed every step in the policy script). Checked
on a saved episode: moving and rotating the whole scene (base, EE, target together) changes the inputs by at most
1.3e-7 (float32 rounding). Decisions and why:
- **World-frame inputs removed, not kept as an option:** `mlp_worldframe`'s results are recorded (0/150 in closed
  loop) and go into the report as they are; running it again from other start poses (24) could only show 0% again,
  and "without robot-frame inputs" in the ablation (23) is that same recorded row. The world-frame code is in git
  history (before this change) if ever needed. Consequence: `checkpoints/mlp_worldframe.pt` (and the old
  `mlp_baseline.pt`, 23 inputs) can no longer be run by the policy script; their recorded results stay valid.
- **Polar not built now, only Cartesian:** comparing them on a baseline that may still fail in closed loop is weak
  evidence, and the winner could change once the model works. Polar (distance + bearing error as sin/cos + height)
  becomes one of the ablation variants (23) on the combined final model, where the choice is measured properly.
- **No parking-spot feature (yet):** the expert steers to a spot on the start-to-target line, which the model can't
  see after the base has moved; but while driving the base is roughly on that line anyway, and the feature would
  need the start pose remembered in the policy and builds the expert's design into the model. Try it only if the
  robot-frame models struggle while driving.
- Fits the camera extension: an estimated target (seen once at t = 0, turned into a world position with the base
  pose at t = 0) simply replaces `TARGET_EE_POS`; `make_features` turns it into robot-frame inputs every step.

**Result and diagnosis (28 Oct), `mlp_robotframe`** (27 inputs, batch 2048, otherwise the same settings as
`mlp_worldframe`):
- Offline it is the best so far: validation loss 0.0089 (world frame 0.0095), test RMSE wheels 0.086 rad/s (0.094),
  torques 0.209 N*m (0.221). Expected to be close: from the origin both input sets hold the same information; robot
  frame matters in closed loop and from other start poses.
- Closed loop (quick subset, with the wheel clip): **0/50**, every run at the 30 s cap, final error 1.05-8.6 m
  (mean 5.1 m; before the clip, mean 222 m). Closest approach to the parking spot: median 0.76 m at ~5 s; a wheel is
  at the clip in ~65% of frames and the base turns ~610 deg in total (the expert at most ~180 deg).
- **How it fails** (episode 558 watched in the GUI and compared frame by frame with the expert): the first ~3 s match
  the expert almost exactly (turn in place, same heading to 0.05 rad; on the expert's own states the model gives
  wheels 2.98 / -3.08 vs 3.0 / -3.0, torques within 0.18 N*m). Then (1) **the turn -> drive switch comes late**
  (expert drives at 2.93 s; at 3.0 s the policy commands 0.04 vs 0.25 m/s): the hidden-phase problem, the model
  can't see the expert's decision; and (2) **the arm slowly sags** away from its holding pose (largest joint
  deviation 0.128 rad at 6 s vs the expert's 0.051, which never exceeds 0.06 while driving) until it loses
  control. Once the arm is in a pose never seen in training, all inputs are unfamiliar, so the wheel commands (same
  network) go wrong too and the base slows and backs off.
- **Why the arm sags:** holding it still takes large torques that are mostly gravity (J2 ~18-20 N*m); a torque
  error of a few tenths of a N*m (test RMSE ~0.2) is enough to let it drift, and the data never shows a drifted arm
  to learn the correction from.
- **What follows (priorities):** (a) **gravity-residual torques (12, Jamie) first**: gravity computed exactly by
  Pinocchio for the actual pose, the network only learns the small remainder, so the arm can't sag from missed
  gravity; (b) **split base / arm networks + "arrived" flag (15, Matteo)**: arm mistakes no longer feed the wheel
  commands, and the flag tells the arm when to reach. **Corrected (29 Oct):** the flag flips at the *end* of orient,
  so it does not fix the late turn -> drive switch; the phase head (15, v2) and temporal inputs (14, Kris) address
  that; (c) tuning
  (`tune.py`) still worth doing, but a better validation loss alone won't fix these closed-loop failures.
- (The numbers above are on the first `data_main`, with the expert's arm chatter.)

**After the expert chatter fix and recollection (28 Oct), `mlp_robotframe` retrained** (same settings):

| `mlp_robotframe` | Validation loss | Test wheels | Test torques | Quick test reached | Final error (median) | Closest to parking spot (median; within 15 cm) | Closest EE error (median; best) | Wheel at clip | Total turning |
|---|---|---|---|---|---|---|---|---|---|
| Old data (chatter) | 0.0089 | 0.086 rad/s | 0.209 N*m | 0/50 | ~5 m | 0.76 m; 0/50 | 0.72 m; 155 mm | 65% of frames | ~610 deg |
| **New data** | **0.0038** | 0.083 rad/s | **0.116 N*m** | 0/50 | **0.78 m** | **0.09 m; 35/50** | **0.17 m; 21 mm** (3/50 under 5 cm) | 7% | ~270 deg |

- **The chatter was a large part of the torque error:** torque RMSE fell from 0.209 to 0.116 N*m and the validation
  loss more than halved, with the same model and settings. Those were unlearnable labels.
- **Closed loop is far better, still 0/50:** the base now gets near its parking spot in most runs (35/50 within
  15 cm) and the arm stops sagging early (largest joint deviation in the first 6 s: median 0.062 rad, the expert's
  level), but the base gets there late (closest approach at a median of ~20 s, the expert arrives at ~8-10 s) and
  the arm's reach then doesn't get within 5 mm before the 30 s cap (closest median 17 cm). So the failure has moved
  from "the arm sags and the base runs off" to "slow navigation + an imprecise reach".
- **Renamed (30 Oct): `mlp_robotframe` -> `mlp_robotframe_old`** (checkpoint, test plots and `results/` file; the
  numbers are unchanged, and it still runs with `compare.py --model mlp_robotframe_old`). Why: it was trained with
  the old settings (lr 1e-3, a fixed 20 epochs, uniform every-10th frames, 700 episodes), so it isn't comparable with
  models trained with the current defaults; the robot-frame single MLP is retrained as `mlp_robotframe_s0` (the
  reference for 15). This section and the tuning section still call it `mlp_robotframe`: they describe what it did
  at the time. `mlp_robotframe_tuned` also used the old settings (name kept).
- **Arm input check (2 Nov, Matteo, branch `test/arm-features`, not merged): keep the full inputs.** Asked because
  another group suggested fewer inputs. Arm only, expert data only, seed 0, base copied from `split_arrived_s0`, val 50
  (closest approach median / under 5 cm / error back above 20 cm after the latch): full 25 inputs 4.7 cm / 27 / 2.9 s;
  raw joint angles instead of sin/cos (18) 8.9 cm / 14 / 1.7 s despite a slightly lower val loss, but calmer once it
  fails (torques at the limit in 0% vs 80% of post-latch frames, base pushed > 0.5 m in 4 vs 14 runs, off the floor 1
  vs 7); without the joint velocities (18, 11) the arm spins in 50/50 and the robot gets thrown 10-50 m: the expert's
  arm law is a PD and its D term needs qd (arm val loss 25x higher). Report point: precision vs stability, and why
  the velocities are needed. Also found: the same weights rerun gave 4.73 vs 4.85 cm, so the simulation isn't exactly
  repeatable (~1 episode on 50); training on this PC is (the rerun was bit-identical). Details: `test_notes.md` there.

### Master: shared improvements
7. **Adaptive subsampling** (after the split): the base/wheel data sampled densely at the start and at phase
   switches but sparsely while driving straight; the arm data sampled densely while it moves (reach) and lightly
   while it holds; a small floor rate everywhere (never zero). Per episode, keeping frame indices so past-frame
   windows stay possible. Compare against uniform subsampling.
   - **Done (Kris, 28 Oct, pushed 28 Oct). Still to run: the comparison against uniform.**
   - **Default since the merge (29 Oct): `SUBSAMPLE = "adaptive"`, `KEEP_FRAC = 0.1` in `train.py`** (see the merge note
     under "Recording results" for why). **To do: test the difference** against uniform: train the same model with both
     settings (only `SUBSAMPLE` changes) and compare offline (validation loss, test RMSE), then in closed loop
     (`compare.py --test-set --quick`, then the full test set), so the choice is backed by a result in the report.
     Validation is subsampled uniformly in both, so their validation losses are directly comparable. (A 1-epoch
     check already hinted at it: validation loss 0.0091 adaptive vs 0.0135 uniform, too short to count.)
   - `afafe7a` (`bc_data.py`): `subsample_indices(wheel_cmds, torque_arm, keep_frac, mode, floor=0.3, smooth=50, seed)`
     returns the sorted frame indices to keep from one episode (indices, not copies, so the temporal models (14) can
     still look up past frames in the full episode).
     - `"uniform"`: evenly spaced frames. `"adaptive"`: frames drawn at random, weighted by how much the expert's
       commands change around each frame (smoothed over 50 ms, so the frames around a jump count too). Chance of a
       frame being kept = `floor * uniform + (1 - floor) * change score`, so 30% of the kept frames are spread over
       the whole episode and no part is left out. Wheels and torques get half of the change score each. Seeded per
       episode `(seed, episode id)`, so repeatable.
     - `load_episodes(..., keep_frac=None, mode="adaptive", seed=0)`: `None` keeps every frame (callers unchanged).
     - Why change-based and not phase rules: on the pilot data 90% of the wheel-command change happens in ~10% of the
       frames (~3 jumps per episode), but the torques change over ~44% of the frames, almost as much while driving as
       while reaching, so "sample the arm lightly while it holds" would drop useful torque data.
     - Checked on 50 pilot episodes: exactly 10% kept, sorted, no repeats, repeatable; frames within 50 ms of a wheel
       switch kept ~4.5x more densely than uniform; largest gap between kept frames 128 ms median (160 ms max).
     - Adaptive also oversamples torque chatter such as the arm wiggle near joint limits: the expert fix for the
       wiggle should be in before the recollection.
   - `bd71ddc` (`train.py`): `SUBSAMPLE` (`None`, `"uniform"`, `"adaptive"`) and `KEEP_FRAC` (default 0.1) next to
     `EPOCHS` / `BATCH` / `LR`; default `None` = all frames, so training is unchanged until switched on. Validation is
     always subsampled **uniformly** with the same `KEEP_FRAC`, so different modes are judged on the same frames. Test
     still uses every frame. Checked: 2-epoch runs in all three modes work end to end, ~6 s instead of ~29 s.
8. **Episode cleaning:** trim the idle tail (unless step 2 removes it), filter failed episodes, sanity checks
   (NaNs, torques at the +-30 N*m limit).
   - **Done (Kris, 28 Oct, pushed 28 Oct).** No tail trimming: expert fix 2 ends episodes on reaching (on `data_pilot`
     the tail is 2% of all frames: 54 episodes > 0.1 s, up to 3.85 s).
   - `d308f75` (`bc_data.py`): `check_episode(d)` lists what makes an episode unusable: not reached (stop message),
     final error above 1 cm, arrays of different lengths, NaN/inf, shorter than 1 s. `clean_ids(data_dir, ids)` keeps
     the episodes that pass, counts missing or truncated files (crashed runs) as dropped, and prints what was dropped
     and why (for the report). `python bc_data.py` also prints the check: run it once after recollecting.
     - Torques at the 30 N*m limit are **not** dropped: on `data_pilot` 349/350 episodes touch the limit, but only
       ~0.3% of frames, in bursts of <= 32 ms at the start of the reach (the expert's kick right after the base parks;
       the expert clips at 30 too, so they are correct labels). Only a warning above 10% of an episode's frames.
     - Checked: broken copies of a real episode (stalled, NaN, length mismatch, 0.5 s, truncated file, missing file,
       final error 2 cm) are all dropped with the right reason; a 23% saturated one is kept with a warning. On
       `data_pilot` all 350 pass (0.8 s).
     - Expert quirk found (for 5): the saved `ee_pos` ends 10.01-10.09 mm from the target although the expert stops
       below 10 mm, so it seems to lag the stop check by a step. Checking "reached" from saved data needs a small margin.
   - `6f7def7` (`train.py`): `clean_ids` runs on the train, validation and test episodes right after `split_episodes`.
     Filtering **after** the split means dropping a bad episode never moves a good one to another set. No change on
     `data_pilot` (nothing is dropped).
   - **Matteo:** `compare.py --test-set` should call `clean_ids` on its test ids too, so the policy isn't scored on
     targets the expert itself failed. Left to you since you're editing `compare.py`.
9. **RANSAC / outlier preprocessing (experiment):** keep both raw and preprocessed data and compare models trained
   on each. Note: the simulator is noise-free, so we expect little difference on clean data; it becomes more
   relevant once sensor noise is added (robustness tests).
   - **Decided (29 Oct, not done yet; Jamie's item, low priority): an experiment on a branch, not part of the shared
     data, and it doesn't block branching.** Only changes to the data or inputs everyone trains on had to come before
     branching, and RANSAC is unlikely to be used on the shared data: (a) the data has no outliers in the usual sense
     (noise-free simulator, no NaN/inf in the episode checks, and the one real data problem, the expert's arm chatter,
     was fixed at the source); (b) **what it would remove could be the most important frames**: fitted to our data, the
     frames that don't fit the overall trend are where the expert's command changes abruptly (turn -> drive switch,
     the base stopping, the start of the reach), i.e. exactly the decisions the policy fails to copy (the late turn ->
     drive switch), so removing them could make the model worse at the failure we diagnosed. Check closed-loop
     results before applying it to any training data. (c) Where it does make sense is on noisy measurements in the
     robustness tests (24), as a filter on the inputs at run time (part of the final model), and in the camera
     extension (fitting and removing the floor plane). Expected result on clean data: no difference, still worth a
     sentence in the report (the "clean" step of the lecture pipeline).
10. **`decode_outputs()` helper** in `bc_data.py`, so the policy script doesn't change when the output format does.
    - **Done (Kris, 27 Oct, pushed 28 Oct).**
    - `96f3ca3` (`bc_data.py`, `robotnik_arm_policy_task.py`, `train.py`): `decode_outputs(y)` returns
      `(wheel_cmds[4], torque_arm[7])` from the model output, the inverse of `make_labels`, for one frame or a whole
      episode. A new output format (e.g. `(v, w)`, 13) only changes `make_labels` and `decode_outputs`. The policy
      script runs the model through it (the +-30 N*m clip comes after decoding). `train.py`'s offline test now scores
      the decoded commands against the expert's saved `wheel_cmds` / `torque_arm` instead of the training labels, so
      every output format is measured on the same 4 wheel speeds and 7 torques. Results on the current format are
      unchanged (checked on a test episode).
11. **Faster training (optional):** changes how quickly we get results, not what the model learns.
   - Print the time per epoch, to see whether changes actually help.
   - Bigger batches (e.g. 8192 instead of 2048): fewer, smoother weight updates per epoch and better use of the CPU
     cores, so epochs finish sooner; usually needs a slightly higher learning rate (e.g. 2e-3 to 3e-3).
   - `--seed` / `--checkpoint` options so several trainings can run at once (e.g. 4 runs x 4 threads), each saving
     its own checkpoint. Needed later for seed repeats and ablations.
   - Torch already uses all 16 physical cores within one run; the biggest saving is less data per epoch
     (subsampling, step 7).
   - **Early stopping:** `train.py` already keeps the epoch with the lowest validation loss (done; describe it as
     early stopping in the report). Add a patience (stop after N epochs without improvement) to save time.
   - Print the training loss averaged over the epoch (now the last batch only), so the train/val gap shows
     over- or underfitting.
   - **Done (28 Oct, Matteo, for `tune.py`; Kris hadn't started step 11):** early stopping with `--patience`
     (default 10) and `--max-epochs` (default 100) replaces the fixed 20 epochs; the `epochs` column now records
     where training stopped. Training loss averaged over the epoch, seconds per epoch in the printout, `--seed`
     (default 0) and `--threads`. Still open from step 11: bigger batches / faster training (batch is in the
     tuning space instead).

**Feature checks from the preprocessing slides** (small, on `make_features`):
**Decided (28 Oct): do these now (Matteo), before the split networks, then freeze the inputs.** Why: every model on
every branch is trained on `make_features`' inputs, so each later change there forces everyone to retrain, re-tune
and redo their results; settling them while the branches are just starting avoids that. **Outcome: the inputs stay
the current 27 robot-frame inputs** (base velocity dropped, scaling left as it is, polar an ablation variant, 23);
announce to the team that the inputs are fixed once `master` is merged.
- Arm joints without limits (`has_limits` is false) can wrap: encode them as sin/cos, like the heading.
  **Done (28 Oct), as a safety measure:** all seven arm joint angles go in as sin/cos (27 inputs instead of 20).
  Only J1, J3, J5, J7 ("continuous" in the URDF) could wrap; J2, J4, J6 have limits, but sin/cos describes their
  angles just as well, and doing all seven keeps the code simple (no joint lists). Checked first: over all 12.1 M
  frames of `data_main` the limitless joints stay within -0.67..+0.52 rad (J7 within +-0.02), nowhere near +-pi, so
  nothing wraps in the current data. Kept anyway because it is cheap and protects against wrapping once the arm
  start pose is randomised (20) or a policy drifts. Checked: J1 + 2 pi gives identical inputs. `mlp_robotframe` has
  to be retrained after this change (its checkpoint expects 20 inputs).
- Add the base velocity (linear + turn rate, from consecutive poses): not an input now, unlike the arm velocities.
  **Decided (28 Oct): not added.** The expert's wheel commands don't depend on the base velocity (only on the pose,
  the parking spot and its hidden phase), unlike the arm, whose PD law uses the joint velocities (so `qd` stays an
  input). Its only value would be a hint about the hidden phase, and it is a known behaviour-cloning trap (the
  "copycat" problem / causal confusion, de Haan et al. 2019, Wen et al. 2020): the velocity is almost exactly the
  result of the previous wheel command, so the model can lower its loss by "keep doing what I'm doing", which looks
  good offline (the validation loss improves) but stops it from switching phases in closed loop. The hidden phase
  is addressed directly instead: "arrived" flag + split networks (15), temporal models (14).
- Min-max instead of z-score scaling for bounded quantities (joint angles, +-30 N*m torques): quick comparison.
  **Decided (28 Oct): not now.** For an MLP trained with Adam the difference is usually small and z-score is the
  standard; the joint angles are already sin/cos (bounded). Possible later as an option once a model works.

### Hyperparameter tuning (for every model)
Each model is tuned separately with sensible values before it is compared, so a model isn't judged on settings
that only suited the baseline. Tune on the **validation** set (`best_val_loss`, and closed-loop success on a few
validation targets if needed), never on the test set; then report the chosen settings and test results.
- **Current hyperparameters:** epochs, batch size, learning rate, hidden size, number of layers.
- **Other ones worth tuning:** weight decay (L2) and dropout (against overfitting), learning-rate schedule (e.g.
  reduce on plateau or cosine), loss function (MSE vs Huber) and the weighting of wheel vs torque errors, early-stopping
  patience, activation function (ReLU vs e.g. SiLU/Tanh), optimiser (Adam vs AdamW; Adam's betas usually less
  important), weight initialisation (PyTorch default vs He/Kaiming for ReLU; still compared over seeds, see 24:
  the initialisation is random either way).
- **Model-specific ones:** history length k and frame spacing (temporal models); TCN layers (receptive field
  2^L frames), frame spacing and channels per layer; RNN hidden size, training window and burn-in length;
  subsampling rates and floor (adaptive subsampling); noise level (DART data).
- Keep the search small (a few values per hyperparameter, most important first: learning rate, network size, batch);
  record each tried setting as its own model name so it lands in `results/`.

**Decided (28 Oct): a `tune.py` script, after the robot-frame inputs (6).** Not before: the best settings depend on
the inputs, and every branch starts from the robot-frame baseline, so that's the model whose tuning matters (tuning
the world-frame model can't fix its closed-loop failure: 0/150 at a good validation loss).
- **Workflow:** develop with `train.py`'s default settings and quick checks (`compare.py --test-set --quick`);
  **only the final models are tuned** (the combined final model, and any model whose result the report depends on,
  once they reach targets): **first `tune.py`, then the full test set** (`compare.py --test-set`) on the tuned model.
  The other rows (baselines, branch ideas) are reported with the default settings. **Updated 28 Oct, after the first
  tuning run** (see below): tuning every model costs ~1 h of CPU each and, while policies reach 0 targets, improved
  only the validation loss, not closed loop (median final error 1.90 m tuned vs 0.86 m untuned); most of its offline
  gain is cheap. So `train.py`'s defaults are now cheap development defaults from that run, the same for every
  branch: **lr 3e-3, 256 x 3, batch 2048, early stopping (patience 10, up to 100 epochs)**, ~4 min per training, val
  loss 0.00261 vs 0.00217 for the full search's winner. The time goes into the closed-loop failure modes instead
  (split networks 15, gravity residual 12, temporal models 14). The tuned model keeps the model's normal name
  (e.g. `<final>_s0`); `results/tuning/<model>_tuning.csv` keeps the search behind it.
- Trains a batch of settings for one model and keeps the best; the report says the settings were "selected by
  `tune.py` on the validation set".
- **Never uses the test episodes.** Ranked by validation loss (the validation episodes already used in training).
  Because a low validation loss doesn't guarantee closed-loop success (world-frame baseline), the best few are also
  checked in closed loop on a fixed subset of **validation** episodes, not `compare.py --quick` (which uses test ones).
- Tuning runs go to `results/tuning/<model>/` (one file per setting + a summary ranked by validation loss), so they
  don't bury the real models; only the chosen setting is retrained under the real name and lands in `results/`.
- Early stopping (generous max epochs + patience, stop epoch recorded) instead of tuning the number of epochs.
- First round, random sample of ~20-30 combinations: learning rate (3e-4, 1e-3, 3e-3), hidden (128/256/512), layers
  (2/3/4), batch (2048/8192), weight decay (0/1e-4), dropout (0/0.1). Second round if useful: Huber vs MSE, wheel vs
  torque loss weighting, SiLU.
- Needs the settings of `train.py` to be settable from outside (options or a `train(config)` function): coordinate
  with Kris, who owns `--seed` and faster training (11).
- **Seeds:** tune with seed 0 only, then retrain the chosen setting unchanged with seeds 1 and 2 (same
  hyperparameters, only the seed differs), so the spread measures that one setting. Not done yet for the baselines.
  The baselines (`mlp_worldframe`, then the robot-frame one) also go into the report if there is space, to show the
  progression, **marked as single-seed in the table** (no seed repeats for them).

**Done (28 Oct): `tune.py`**, `pixi run python tune.py --final <name>` (from `robotnik_arm_task/`). What it does:
1. **Staged grid search** with `train.py --no-test` (no test evaluation during tuning), 4 trainings at a time
   sharing the CPU. Related settings are tried together (all combinations within a stage), each stage starting from
   the best setting so far (lowest validation loss); settings already trained are skipped (18 trainings; **since
   trimmed to 14, see "Changes to `tune.py` after it" below**):

   | Stage | Settings | Trainings |
   |---|---|---|
   | 1a optimiser | learning rate 3e-4 / 1e-3 / 3e-3 at batch 2048 | 3 |
   | 1b batch | batch 4096 and 8192 with the best learning rate x sqrt(batch / 2048) | 2 |
   | 2 size | hidden 128 / 256 / 512 x layers 2 / 3 / 4 | 8 new |
   | 3 regularisation | weight decay 0 / 1e-4 x dropout 0 / 0.1 | 3 new |
   | 4 lr re-check | the other two learning rates (scaled to the chosen batch) with the final size + regularisation | 2 |

   Why staged and not the full grid (216 combinations, ~2-4.5 h per model) or a random sample: grouping settings
   that interact (learning rate with batch, width with depth, weight decay with dropout) tests those interactions
   fully, costs fewer runs than 24 random ones, and gives one small table per stage for the report. Batch is not
   crossed with every learning rate: the learning rate is matched to it by the **square-root scaling rule**
   (bigger batch -> fewer, more reliable steps -> higher learning rate; sqrt suits Adam), so a bigger batch mostly
   changes speed. The weak point is greediness across stages (stage 1 picks the learning rate for the default
   network), which stage 4 checks. The number of epochs is not tuned: early stopping (patience 10, at most 100)
   ends each trial where its validation loss stops improving. Patience 10 rather than 5: safer for slowly improving
   settings (e.g. the low learning rate), at the cost of a longer run.
2. Ranks all trials by validation loss.
3. Checks the best 3 in closed loop on 20 fixed **validation** episodes (`bc_data.quick_val_ids`), never test ones.
4. Winner: most validation targets reached, then lowest median final error, then lowest validation loss.
5. The winner's checkpoint becomes `checkpoints/<name>.pt` (copied, not retrained, so the final model is exactly the
   one the closed-loop check chose) and its results `results/<name>.csv`, then `train.py --test-only` adds the
   offline test columns. The trial checkpoints and results files (`<name>_t00` ... `_t23`) are deleted.
6. Summary of all trials (settings, validation loss, best / stop epoch, training time, closed-loop numbers of the
   top 3, which one was chosen): `results/tuning/<name>_tuning.csv`. `results.py` doesn't read that subfolder, so
   only the final model is in the combined table. The full test set is run separately afterwards
   (`compare.py --test-set --model <name>`). Why in this shape: tuning runs in the normal folders keep things
   simple (no extra folder options), and deleting them at the end keeps `results/` to the models that matter.
   Smoke-tested (staged search, 1 epoch per trial, 2 validation episodes): all 5 stages ran, 18 trainings, the
   duplicates skipped, and it left exactly the final model and the summary (with a `stage` column). Time for a full
   run: ~20-40 min.
- **Loss and activation: later, as a separate test, once a model actually works** (not a `tune.py` stage, no
  options added now). Then 4 full trainings, MSE / Huber x ReLU / SiLU, with the tuned settings. Why separate: they
  are design choices for the method (fixed for all models so their numbers stay comparable), not per-model settings;
  and why later: while every policy still reaches 0 targets, small differences can't be judged in closed loop.
  **Compare them on the same measure** (offline test RMSE in real units and closed loop), never on `best_val_loss`:
  a Huber run's validation loss is a Huber loss, not on the same scale as MSE. For an RNN (14) only the loss result
  carries over; LSTM/GRU have their own built-in gate activations.
- First use: `tune.py --final mlp_robotframe_tuned`, so the untuned `mlp_robotframe` stays as its own row (the
  progression: world frame -> robot frame -> tuned robot frame).

**First tuning run (28 Oct), `mlp_robotframe_tuned`** (the 18-training version described above; summary in
`results/tuning/mlp_robotframe_tuned_tuning.csv`). Took **65 min** wall time (137 min of training, 4 at a time):
- Stage results: lr 3e-3 best at the default size; batches 4096 / 8192 (lr scaled) both worse than 2048; size
  512 x 3 best (512 x 4 stopped at epoch 11, unstable at lr 3e-3); **dropout 0.1 much worse** (val 0.0058-0.0063 vs
  0.0024); weight decay 1e-4 neutral; the lr re-check (stage 4) moved the learning rate down to 3e-4 / 1e-3 for the
  bigger network, exactly the greediness it was there to catch.
- Chosen: lr 3e-4, 512 x 3, batch 2048, no regularisation: val 0.00217 (untuned `mlp_robotframe`: 0.00377), test
  RMSE wheels 0.058 rad/s (0.083), torques 0.098 N*m (0.116); 545 k parameters, inference 0.11 ms.
- **Most of the gain came from early stopping, not the settings:** the defaults trained with early stopping (44
  epochs instead of 20) already reach val 0.00279, ~60% of the improvement.
- **Closed loop got worse, not better:** full test set 0/150 for both, final error median **1.90 m** tuned vs
  **0.86 m** untuned (base near the parking spot in 71/150 runs, closest EE approach median 16 cm). A lower
  validation loss does not mean better closed-loop behaviour (the offline vs closed-loop gap again, a report point).
  So while every policy is at 0%, tuning on validation loss can't be trusted to improve what matters; tune the
  models that go into the report, and judge them in closed loop.
- **Chosen model was the slowest for a noise-level difference:** t16 (lr 3e-4, 23.3 min, hit the 100-epoch cap at
  epoch 98, not converged) beat t17 (lr 1e-3, 14.1 min, converged at epoch 58) on validation loss by 0.5% (0.00217
  vs 0.00218), and on the closed-loop tie-break by median final error 2.05 vs 2.95 m with both at 0/20 reached, which
  at 0% mostly reflects where the policy wandered.

**Changes to `tune.py` after it (28 Oct):**
- **(a) Time in the selection:** trials within **2%** of the best validation loss count as equally good (the
  shortlist, at most `--top`); after the closed-loop check the winner is the one with most validation targets
  reached, then the **fastest to train** (no longer the median final error, which is noise at 0%). Why: a 0.5%
  validation difference isn't worth 65% more training time, and the cheaper setting is easier to repeat for seeds.
  On the first run this would have chosen t17 (lr 1e-3, 14 min) instead of t16. `mlp_robotframe_tuned` is kept as
  chosen by the old rule (t16) and not re-tuned: both reach 0 targets, so it makes no practical difference, and
  only final models are tuned from now on.
- **(b) Search kept at 18 trainings:** dropout and the bigger batches were worse in the first run, but they stay in
  the search: only 4 extra trainings, and they may matter once a model works (e.g. dropout against overfitting with a
  bigger network or more data). Smoke-tested with the new selection rule: all 5 stages, 18 trainings, shortlist and
  winner as intended.

### Branches: model ideas (rough order)
12. **Gravity-residual torques:** learn torque minus the gravity torque, add it back at run time (aimed at low targets).
13. **`(v, w)` wheel outputs:** forward speed + turn rate instead of 4 wheel speeds.
    - **Code (31 Oct):** `--vw-outputs` in `train.py` (single MLP, Jamie), with `--split` (labels and base outputs
      indexed from the end: torques = last 7, logit = last base output, so one code path serves both formats) and in
      `train_gru.py`. `test_wheel_rmse` is always measured on the decoded 4 wheels; `best_val_loss` is not comparable
      between formats (different quantities).
    - **Split + vw (Matteo, 31 Oct; validation, 50 episodes, one seed): better base, much worse reach.**

      | Model | Reached | Closest median | Ever < 1 cm | Parking vs expert, median / 90% | Latch vs expert | Wheel RMSE |
      |---|---|---|---|---|---|---|
      | `split_dagger_r5_s0` (4 wheels) | 26/50 | 0.5 cm | 32 | 2.4 / 27.9 cm | 0.25 s early | 0.181 |
      | `split_dagger_vw_s0` (vw base + its own arm, DAgger r1-r5) | 12/50 | 1.8 cm | 16 | - | - | 0.156 |
      | `split_vwbase_r5arm_s0` (vw base + r5's exact arm) | 6/50 | 1.8 cm | 13 | 1.6 / 3.2 cm | 0.40 s late | 0.156 |

      - First guess, wrong: the drop came from the arm being retrained with a different random state (the base's
        size and epoch count shift the shared seed). `--arm-from` (copies a trained arm unchanged) ruled it out:
        with r5's exact arm the vw base does worse still.
      - Likely reason: the arm's DAgger data was all collected behind 4-wheel bases, which latch early and sometimes
        park badly; its corrections fit those starting states. The vw base parks far more accurately and latches
        ~0.65 s later, handing the arm states its DAgger data barely covers. Not proven.
      - **Decision: keep the 4-wheel base** (deadline). Report point: DAgger's corrections fit the policy whose
        mistakes they were collected from; a better base doesn't help an arm trained behind a different one.
      - Not done (if time is left): DAgger rounds behind the vw base (`dagger.py` needs `--vw-outputs` pass-through
        and a `--prior` option to keep r1-r5), ~30 min per round.
14. **Temporal data** (aimed at the early stop of turning and the phase switches). Also a discussion point: why
    temporal data is needed (the expert has hidden state: navigation phase, accumulated joint targets).
    - First, **past-frame windows** into the MLP: stack k frames with a spacing, sweep both.
    - Then a **temporal convolutional network (TCN)** (Lecture 07; added 29 Oct): causal dilated 1-D convolutions
      directly on the 27 inputs of the last N frames give the commands at time t. Plain first, no per-frame MLP
      encoder: the lecture uses one to compress wide inputs (many channels, images) into features, and ours are only
      27 numbers. If it trains badly, add residual connections around each layer first (standard in TCNs; deep plain
      dilated stacks are hard to train), and the encoder only if it underfits the per-frame mapping. Unlike
      past-frame windows (separate weights per past frame, cost grows with the window), the TCN shares weights over
      time and its reach grows exponentially with dilation: windows for short histories, the TCN for a long
      receptive field at low cost. Settings and why:
      - **Causal only** (no future frames at run time; for the same reason no bidirectional RNN), **stride 1**
        (one prediction per 1 ms step), **kernel 2 with dilation 1, 2, 4, ...** as in the lecture: the receptive
        field doubles per layer (2^L frames), so the number of layers L is the main knob to sweep.
      - Frames inside the window at a **spacing** (e.g. 10 ms): neighbouring 1 ms frames are almost identical, so
        without it even 6 layers (64 frames) only see 64 ms; with 10 ms, 6 layers see 0.64 s.
      - **No max pooling** by default: pooling makes the output ignore *when* something happened in the window (the
        lecture's translation invariance, useful for recognising whether an event happened), but our commands
        depend on exactly that timing (time since the turn stopped, progress of the reach), and it lowers the time
        resolution. At most an ablation.
      - **Episode start:** the first frames have no past; pad by repeating the first frame (not zeros: zero in
        normalised units means "an average state"), **identically in training and in the policy script**.
      - **No repeated computation:** training computes `make_features` once per episode and builds windows by
        indexing (the frames from `subsample_indices` + fixed offsets), not by copying N frames per sample (all 12 M
        frames x 27 inputs is ~1.3 GB in float32: fine on the desktop, tight on the laptop). In the policy, each
        layer keeps a ring buffer of its past outputs and computes only the newest node each step: the lecture's
        dilation tree, where each node is computed once and reused by later steps (same cost per step as the MLP).
        Measure `inference_ms` either way.
      - Expected to help the **base** (the phase switch shows in the recent history) more than the **arm**: the
        arm's internal reference `q_des_arm` is predictable from the current frame (checked under fix 6), so its
        memory is short. With split networks (15) a temporal model can go into one net only: try base-only first.
      - **Code done (Kris, 30 Oct, on `Kris`, not pushed yet); sweep below.** Decided: ring buffers as planned,
        a separate `train_tcn.py`, gravity residual on by default.
      - `728a157` (`bc_model.py`): `TCN(n_in, n_out, layers, channels, spacing, gravity_residual)`: kernel 2, written as a
        linear layer on the two time points; 2^L frames, spacing ms apart. `forward` walks the tree (training), `step`
        computes the newest node of each layer with one ring buffer per layer (run time). `load_model` picks the class
        from the checkpoint. Dilation counted in ms, so 10 ms spacing needs no staggered buffers.
      - `51ed5fa` (`bc_data.py`): `load_frames` stores each frame once, `window_batch` cuts windows per batch (first frame
        repeated at the episode start), since stored windows would take ~6 GB at 64 frames.
      - `ada87a4` (`robotnik_arm_policy_task.py`): `model.step(x)` for TCNs; other models as before.
      - `b4c51d3` (`train_tcn.py`, new): same data, split, early stopping, results; records `architecture=tcn`.
      - Checked: `forward` and `step` agree to 1e-7 over 3,000 real frames; `window_batch` identical to `with_history`,
        padding included; a frame just beyond the reach and a future frame change nothing; a tiny batch overfits; saved
        checkpoints reload; `load_frames` matches `load_episodes`; 1 epoch end to end for L = 6 (13 s per epoch) and L = 8
        (37 s); 0.05 ms per step at either size; TCN, MLP and window checkpoints all run in the policy script.
      - **Sweep (Kris, 30 Oct; 10 ms spacing, 64 channels, gravity residual, lr 3e-3, quick test on 50 targets, Mac):
        all 0/50, and closed loop far worse than the MLP.** No TCN beat `mlp_gravres_s0` on validation loss.

        | Model | Reach | Parameters | Val loss | Wheel / torque RMSE | Max torque first hit (s) | Arm dev. at 2 s (rad) | Base off the floor | Spot within 15 cm |
        |---|---|---|---|---|---|---|---|---|
        | `mlp_gravres_s0` | 1 frame | 141,579 | 0.00131 | 0.057 / 0.104 | 8.73 | 0.001 | 12/50 | 23/50 |
        | `mlp_gravres_lr1e3_s0` | 1 frame | 141,579 | 0.00211 | 0.075 / 0.125 | 8.97 | 0.002 | 2/50 | 44/50 |
        | `tcn_L4_d10_gravres_s0` | 160 ms | 29,003 | 0.00174 | 0.059 / 0.117 | 1.25 | 0.109 | 47/50 | 4/50 |
        | `tcn_L6_d10_gravres_s0` | 640 ms | 45,515 | 0.00227 | 0.064 / 0.130 | 1.09 | 5.78 | 50/50 | 2/50 |
        | `tcn_L8_d10_gravres_s0` | 2.56 s | 62,027 | 0.00184 | 0.050 / 0.120 | 0.58 | 5.86 | 50/50 | 0/50 |

        - Arm deviation: median over episodes of the mean |q - q_expert| over the 7 joints. Max torque first hit: median
          first time a torque reaches the 30 N*m clip. Off the floor: base height moved over 0.3 m from its start.
        - **Do not read `quick_final_err_median` (30 / 371 / 1,032 m) as distances.** The wheels allow at most 10 m in
          30 s. The arm is at the torque clip almost throughout, the physics blows up, and in the worst runs the robot
          ends over 1 km below the floor. In the median run it leaves the floor at 7.2 / 1.9 / 2.4 s.
        - Not a run-time bug: `step` and `predict` agree within 5e-6 over 4,000 frames of a real episode for all three.
        - A longer reach hits the torque clip sooner. L6 and L8 get there before the expert's drive start (1.36 s), L4 at
          about the same time here but at 2.59 s on the full test below. Why is not established.
        - Not comparable with the MLP on size: these TCNs have 20-45% of its parameters, and val loss is not monotonic in
          L (L6 is worse than L4 and L8), which suggests the plain deep stack trains badly (no residual connections yet).
        - **Old settings on L4, `tcn_L4_d10_gravres_lr1e3_s0` (lr 1e-3, max 20 epochs): no rescue, unlike the MLP.** Val
          0.00408, wheel / torque RMSE 0.105 / 0.166, 0/50; max torque first hit 1.89 s (L4: 1.25 s), arm dev. at 2 s 0.38
          rad, off the floor 49/50, spot within 15 cm 6/50. Val loss was still
          falling at epoch 20, so this model is underfit; it is not a test of a well-trained TCN at lr 1e-3.
      - **Temporal arm too? Test it, don't assume it** (decided 30 Oct). Once a temporal base works, compare
        temporal base + single-frame arm vs temporal base + temporal arm (closed loop, and the torque error during
        the reach). Why not by default: what memory could fix in the arm is small (above), the flag already handles
        the hold -> reach switch, and a temporal arm sees its own recent motion (mostly the result of its own last
        torques), the copycat risk (see "Feature checks"). Keep the simpler arm unless the temporal one clearly wins.
        Report point either way: memory helps where the expert has hidden state (base), not where it doesn't (arm).
    - Then a **recurrent neural network (RNN)**: a GRU (`torch.nn.GRU`; why not an LSTM: below). It keeps an
      internal memory that is updated every step, so it can learn the expert's hidden state (which phase it is in)
      instead of us choosing a fixed window. Training uses short sequences cut from the episodes; in the policy
      script the memory is carried from step to step and reset at the start of each episode.
      - **Plain GRU first, trained in one go** (decided 29 Oct): one GRU layer stepping every 1 ms on the 27
        inputs, linear output (the lecture's basic RNN; simplest, and the cleanest comparison with the MLP and the
        TCN). Only if its offline torque RMSE is clearly worse than the MLP's (underfitting the pose-dependent
        torques: a GRU has one nonlinear layer per step, the MLP three), add a small MLP head on [GRU output,
        current frame]; gravity-residual torques (12) would remove most of that nonlinearity anyway. Not the
        lecture's MLP encoder + GRU (encoder before the GRU, trained in stages): more parts and more training steps
        for inputs that are already small.
      - **Training data:** **contiguous windows** cut from the episodes (not `subsample_indices`' scattered
        frames; window starts can still favour phase switches), with a **burn-in** at the start of each window
        without loss, since a window cut mid-episode starts from a state the policy would never have; **truncated
        backpropagation through time** over the window (long sequences risk vanishing/exploding gradients).
      - **Why a GRU and not an LSTM (decided 29 Oct):** an LSTM adds a separate cell memory next to the hidden
        state (input, forget and output gates: 4 weight blocks instead of the GRU's 3, ~33% more parameters and
        compute per step), so it can keep information for much later without it affecting the current output
        (Lecture 07). Our hidden state is the opposite case: the phase and the joint reference matter for the
        commands *now*, over a few seconds, not information stored for later. The lecture also notes the GRU was
        designed as a simpler LSTM with negligible loss, and in 15 the forward-only phase latch already carries the
        longest memory (the phase) by hand. So no separate LSTM model; since it is a drop-in swap (`nn.GRU` ->
        `nn.LSTM`, same training code, the state becomes (h, c)), **one LSTM run** only if the GRU clearly fails to
        hold the phase (poor phase accuracy near the switches, or validation loss still falling with longer
        training windows), or as an ablation if time is left at the end.
      - Optional later: the GRU also predicts the phase (as in 15's phase head) to check whether it learned the
        hidden state; 15 builds the phase head first.
      - **Code done (Kris, 30 Oct, on `Kris`, not pushed yet); sweep below.** Decided: 192 memory units (129,419 parameters, close to the MLP's 141,579), separate `train_gru.py`, gravity
        residual on by default, lr 1e-3, gradient norm clipped at 1.0, max 50 epochs, stepping every 1 ms as planned.
      - `bc_model.py`, `robotnik_arm_policy_task.py`: `GRU(n_in, n_out, hidden, burn_in, gravity_residual)`, one layer and a
        linear output, `step` at run time; stateful models (TCN, GRU) both use `model.step(x)`. The memory starts warmed
        up on `burn_in` copies of the first frame, matching the training windows at an episode's start.
      - `bc_data.py`, `train_gru.py` (new): `load_frames(..., all_labels=True)` keeps every frame's labels. Window ends are
        the adaptive kept frames; 200 ms burn-in without loss; repeated first frames have no loss; an epoch is about as
        many frames as the training set. Validation and test run whole episodes from the start, as at run time.
      - `analyse_temporal.py`: three failure metrics in `quick_*` / `full_*` columns: when the torque clip is first hit,
        arm deviation at 2 s, base off the floor. Needed since `quick_final_err_median` means nothing after a blow-up.
      - Checked: `predict` and `step` agree to 2e-6 over 4,000 real frames; changing future frames changes nothing earlier;
        a tiny batch overfits; saved checkpoints reload; `all_labels` keeps the same frames and labels; 1 epoch end to end
        takes 75 s on 3 threads; 0.04 ms per step; a 1-epoch checkpoint runs in the policy script on 50 targets.
      - **Sweep (Kris, 30 Oct; window = frames in the loss, 192 units, gravity residual, lr 1e-3, quick test on 50 targets,
        Mac): still 0/50, but the first temporal models that are stable, and the best base so far.** Best validation
        does not mean best closed loop; the GRU's validation runs whole episodes, so it is not the same test as the MLP's.

        | Model | Val loss | Wheel / torque RMSE | Max torque first hit (s) | Arm dev. at 2 s (rad) | Base off the floor | Spot within 15 cm | Closest EE median (m) | EE within 10 / 5 / 2 / 1 cm | Final err median (m) |
        |---|---|---|---|---|---|---|---|---|---|
        | `mlp_gravres_lr1e3_s0` | 0.00211 | 0.075 / 0.125 | 8.97 | 0.002 | 2/50 | 44/50 | 0.102 | 22 / 12 / 2 / 1 | 1.66 |
        | `tcn_L4_d10_gravres_s0` | 0.00174 | 0.059 / 0.117 | 1.25 | 0.109 | 47/50 | 4/50 | 0.884 | not counted | blow-up |
        | `gru_h192_w250_gravres_s0` | 0.00521 | 0.075 / 0.189 | 9.89 | 0.003 | 6/50 | 40/50 | 0.101 | 24 / 12 / 4 / 0 | 1.90 |
        | `gru_h192_w1000_gravres_s0` | 0.00306 | 0.077 / 0.150 | 11.43 | 0.002 | 0/50 | 49/50 | 0.087 | 31 / 11 / 1 / 1 | 1.15 |
        | `gru_h192_w2000_gravres_s0` | 0.00390 | 0.094 / 0.168 | 13.30 | 0.003 | 0/50 | 43/50 | 0.152 | 19 / 7 / 2 / 1 | 1.09 |

        - **Base solved on these targets, reach not:** `w1000` parks within 15 cm in 49/50, with wheels at the clip in 7% of
          frames (MLP 43%). The arm comes to about 9 cm (median closest) but no run gets within the 5 mm target, and the
          torque clip is first hit at 10-13 s across the windows, around the end of the expert's reach (median 12.1 s).
        - **Not finished training:** `w1000` and `w2000` stopped at the 50-epoch cap with validation still improving (best
          epochs 46 and 49). One seed each; differences between the three windows may be noise.
      - **Full test (Kris, 31 Oct; all 150 test targets, one seed each, Mac).** The MLP row is the best MLP on the same run.

        | Model | Reached | Spot within 15 cm | Closest EE median (m) | EE within 10 / 5 / 2 / 1 cm | Max torque first hit (s) | Arm dev. at 2 s (rad) | Base off the floor | Wheels at clip | Final err median (m) |
        |---|---|---|---|---|---|---|---|---|---|
        | `mlp_gravres_lr1e3_s0` | 0 | 127 | 0.116 | 61 / 28 / 6 / 1 | 9.15 | 0.002 | 6 | 43% | 1.49 |
        | `gru_h192_w250_gravres_s0` | 0 | 107 | 0.111 | 68 / 39 / 16 / 4 | 9.60 | 0.003 | 18 | 3% | 1.87 |
        | `gru_h192_w1000_gravres_s0` | 0 | 148 | 0.076 | 107 / 31 / 3 / 1 | 11.20 | 0.002 | 0 | 9% | 1.08 |
        | `gru_h192_w2000_gravres_s0` | 1 | 114 | 0.156 | 54 / 23 / 7 / 2 | 13.36 | 0.003 | 0 | 4% | 1.25 |
        | `tcn_L4_d10_gravres_s0` | 0 | 12 | 0.877 | 5 / 2 / 0 / 0 | 2.59 | 0.082 | 139 | 89% | blow-up |
        | `temporal_window_k4_d50_gravres_s0` | 0 | 15 | 0.333 | 27 / 12 / 1 / 0 | 6.30 | 0.003 | 144 | 78% | blow-up |

        - `gru_h192_w1000_gravres_s0` is the best for the base and for stability: 148/150 parked and no blow-ups.
        - **No model has really reached a target.** The one reached run (`gru_h192_w2000_gravres_s0`, `ep_0069`, 28.0 s, 15 s
          after the expert) is a fluke: the end-effector swung through the target from 1.2 m away, under 5 mm for 1 frame,
          and was 1.67 m away 1 s later. compare.py counts it, so the 1 stays in the results file; the real count is 0/150.
        - **Closest approach flatters a swinging arm, and the MLP holds position better than the GRU.** Runs that stay
          within 10 / 5 / 2 cm of the target for 0.5 s: MLP 42 / 13 / 1; GRU 250 ms 26 / 10 / 0; GRU 1000 ms 33 / 1 / 0; GRU 2000
          ms 20 / 5 / 1; TCN L4 0 / 0 / 0; window 4 / 0 / 0. So the GRU's gain is the base and stability, not the reach: a
          reason to try it in the base network only (15).
        - **Correction (31 Oct):** the arm deviations first written here counted all 17 `q_mes` columns, wheel angles
          included. All tables now use the 7 arm joints (`analyse_temporal.py`). So the GRU's arm at 2 s is level with the
          MLP's, not 7x closer. The window model's arm is also level at 2 s, and hits the torque clip at 6.3 s here: the
          "2-3 s" arm failure above does not show in this measure. Sweep tables are the 50-target quick test.
        - The 6 s arm deviations in the earlier tables (0.75, 0.84, 0.056, 0.54 rad) are a different, older measure that
          could not be reproduced exactly. Mean over the 7 joints at 6 s, quick test: `mlp_robotframe_s0` 0.346,
          `mlp_robotframe_uniform_s0` 0.520, `mlp_gravres_s0` 0.012, `mlp_gravres_lr1e3_s0` 0.005, windows d10 / d50 / d125
          6.06 / 18.3 / 2.62, window d50 + residual 0.246, GRU 1000 ms 0.004. The ordering and the gravity-residual gain hold.
        - The test scripts differ slightly: `compare.py` runs every test episode, including `ep_0297` (expert failed, dropped
          from training tests), so the full test has 150 targets, not 149.
        - `compare.py --test-set` rewrites `results/expert.csv`: restored, since it is Matteo's file.
      - **GRU with (v, w), and a PC retrain of w1000 (Matteo, 1 Nov; validation, 50 episodes, one seed; same settings as
        Kris's w1000: 192 units, 1000-frame windows, 200 burn-in, lr 1e-3, gravity residual, 50 epochs).** The retrain
        is named `gru_h192_w1000_gravres_pc_s0` so it doesn't overwrite Kris's Mac results file.

        | Model | Reached | Closest EE median | Ever within 1 / 5 / 10 cm | Held within 5 cm for 0.5 s | Base within 15 cm of the expert's spot | Max torque first hit (median) | Wheel / torque RMSE | Final err median |
        |---|---|---|---|---|---|---|---|---|
        | `split_dagger_r5_s0` (reference) | 26/50 | 0.5 cm | 32 / 43 / 44 | 42 | 45/50 | 3.7 s (reach start) | 0.181 / 0.802 | 3.1 cm |
        | `gru_h192_w1000_gravres_pc_s0` | 0/50 (1 fluke) | 7.9 cm | 2 / 15 / 31 | 4 | **50/50** | 10.6 s | 0.078 / 0.150 | 95 cm |
        | `gru_h192_w1000_vw_gravres_s0` | 0/50 | 23.4 cm | 0 / 4 / 8 | 1 | 35/50 | 7.2 s | 0.091 / 0.141 | 135 cm |

        - **The PC retrain reproduces Kris's Mac result:** val loss 0.00300 (Mac 0.00306), closest EE 7.9 cm (Mac 8.7 cm,
          quick test), the base reaches the expert's spot in every run (Mac 49/50). Best epochs 47 / 48 of 50: still
          improving at the cap, as on the Mac.
        - The one "reached" run (`ep_0403`) is a swing-through: under 5 mm for 2 frames at 6.5 s, 54 cm away at the end.
          compare.py counts it; the real count is 0/50.
        - **(v, w) hurts the GRU too:** the base reaches the expert's spot in 35/50 instead of 50/50, closest EE 23 vs 8
          cm, worse wheel RMSE (0.091 vs 0.078). With the split model (13) vw made the base better and the reach worse;
          here it makes the base worse. **Decision: no vw for the GRU** (and so not for a GRU base in the split model).
        - **For the GRU in the split model (15, Kris, 1 Nov):** the GRU's base is the best we have (50/50 at the expert's
          spot vs r5's 45/50), its arm is not (0/50, 4 runs held within 5 cm vs r5's 42). So the most promising
          variant is GRU base + MLP arm. Caveat from 13: r5's arm only works well behind the base it was trained with,
          so the GRU base may need DAgger rounds behind it (dagger.py with the combined model) before it pays off.
    - **Compare TCN vs RNN** on the same inputs, split and metrics (validation loss, test RMSE, closed loop,
      inference time per step). Report points: (a) the MLP's only temporal information is the joint velocity
      (one-step memory; the lecture's "RNNs vs MLPs"); (b) the TCN needs a fixed window, while the arrival time
      varies 3.6-12.8 s, which the RNN handles without choosing a window (the lecture's "TCN limitations"); the TCN
      is simpler to train; (c) **14 vs 15**: the same hidden phase, learned here vs built by hand in 15 (the
      forward-only latch), see the report point at the end of 15.
    - Owner: Kris (step 14); the TCN can move to Matteo if 15 finishes first (decide at the check-in).
15. **Multi-model / split networks** (mirrors the expert's structure), from simple to more structured. Matteo,
    models `split_*` (e.g. `split_arrived_s0`). Design agreed 29 Oct; v1 built 30 Oct (see "Done" under v1):
    - **Facts from 150 training episodes (`data_main`)**, behind the design: `nav_done` is True in ~31% of frames
      (median; 15-58%), only ever goes False -> True, and the wheels are exactly 0 after it. Arrival at a median
      8.7 s (3.6-12.8 s), wheel speed just before ~0.86 rad/s (the orient turn in place). Before arrival the arm is
      a PD hold of its start pose + gravity (drift <= 0.067 rad), and its torques step by 6.5-11 N*m at arrival.
      **The flag flips at the end of orient, so it does not fix the late align -> drive switch** (the diagnosis
      under step 6 said it would; corrected there): it mainly helps the arm (hold vs reach), and base-only inputs
      help the base. The class imbalance is mild (~2:1); the hard part is the timing of the switch.
    - **v1: separate base and arm networks + "arrived" flag.**
      - Base net: `make_features` columns 0-2 (target forward/left in the base frame, target height) -> 4 wheels
        (MSE, all frames including the post-arrival zeros) + 1 "arrived" logit (`BCEWithLogitsLoss`, `pos_weight`
        = negatives / positives on the frames actually used). Loss = MSE + BCE, 1:1 to start.
      - Arm net: columns 3-26 (target relative to the EE, sin/cos q, qd: 24) + the flag (0/1) = 25 inputs -> 7
        torques. Trained with the true `nav_done`, run with the latched predicted flag. Columns 0-2 left out on
        purpose: the arm doesn't need them (target - shoulder = EE-relative target + a function of q), and they
        would let it guess the phase from the base position instead of using the flag.
      - Both nets on the same adaptive frames at first, so `best_val_loss` stays comparable with the single MLP
        ((4 x wheel MSE + 7 x torque MSE) / 11 on the same normalisation); per-net subsampling is a later switch.
      - Policy: the flag latches once p > 0.5 for ~20 ms and never unlatches; wheels forced to 0 after the latch
        (a hand-coded rule mirroring the expert: say so in the report). Success is still `check_stop(..., True)`,
        not the flag. p(arrived) and the latched flag are saved in the policy `.npz`. **Decided (30 Oct): also latch
        at 25 s** like the expert's navigation timeout, and log whether the network or the timeout latched. Why: the
        expert's own rule (the arm always gets a try), and late arrivals are common (the old robot-frame model got
        closest to the parking spot at a median ~20 s); logged so a timeout latch can't pass as a working flag.
      - Code (the single-MLP files stay as they are): `bc_split.py` reuses `bc_model.MLP` unchanged (base
        `MLP(3, 5)`, arm `MLP(25, 7)`), saved as **one** checkpoint so `ROBOENV_POLICY` / `compare.py --model`
        work; the logit column gets `y_mean` 0 / `y_std` 1 so the output normalisation doesn't rescale it.
        `bc_data.py`: additions only (`nav_done` labels aligned with the subsampled frames, `BASE_INPUTS` /
        `ARM_INPUTS` slices). The policy script branches on the checkpoint type; the single-MLP path is unchanged.
        **Decided (30 Oct): a `--split` option in `train.py`**, not a separate `train_split.py` (first recommended).
        Why: the ablation (23) switches one component off in the same script with everything else identical; with a
        separate script "without split networks" would be a different script that could quietly differ (loading,
        early stopping, the test). Merge-conflict risk kept small by putting the split code in `bc_split.py`.
      - **Gravity residual (12) on from the start** (decided 30 Oct): split models are trained with
        `--gravity-residual`. Not the class default (off, like the MLP's), so `--split` alone never switches gravity
        on and the ablation changes one thing at a time. **Reminder: switching it off is an ablation run (23).**
      - Evaluation: **a 2 x 2 for the results table** (decided 30 Oct; not a report table, but every row is in
        `results/`): single MLP / split x without / with gravity residual = `mlp_robotframe_s0`, Jamie's
        gravity-residual MLP (still to come; check it used the current defaults: lr 3e-3, early stopping, adaptive
        frames, seed 0, 696 episodes), `split_arrived_nograv_s0`, `split_arrived_s0`. Each row of it then shows the
        split's effect with and without gravity, each column the gravity residual's. The old `mlp_robotframe` isn't
        a fair reference (old settings; renamed `mlp_robotframe_old`, see step 6). Offline: `best_val_loss`, test
        RMSE replayed with the predicted, latched flag feeding the arm net, median arrival-time error, number of
        latches > 0.1 s early (early is the dangerous direction). Closed loop (`compare.py --test-set --quick`):
        reached, median final error; the latch time and where the base was at the latch are in the policy `.npz`
        files (`compare.py` doesn't print them, see "Done"). Old reference: 0/50, 0.78 m, 35/50 within 15 cm at
        ~20 s, 17 cm.
      - **Done (30 Oct): v1 code** (`bc_data.py`, new `bc_split.py`, `train.py`, `robotnik_arm_policy_task.py`).
        Usage: `train.py --model split_arrived_s0 --split --gravity-residual`, then `compare.py --model ...` as for
        any model. What was built and why:
        - `bc_data.py`: `BASE_INPUTS` / `ARM_INPUTS` (column slices of `make_features`, so the inputs stay frozen) and
          `load_episodes(..., arrived=True)`, which also returns `nav_done` of the same (subsampled) frames.
          `train.py` always loads it (cheap, unused without `--split`), so it only runs on data with `nav_done`
          (`data_main`; not the old `data_pilot`).
        - `bc_split.py`: `SplitPolicy` holds two unchanged `bc_model.MLP`s (base `MLP(3, 5)`, arm `MLP(25, 7)`) and is
          saved with the existing `save_model`; its config has `kind: "split"`, which `load_policy` reads to build the
          right class (old checkpoints have no `kind` and load as MLPs, so every existing model still runs).
          `ArrivedLatch` is the switching rule, shared by the offline test and the policy so both switch the same way.
        - `train.py`: the training loop moved unchanged into `fit()`, used for the single MLP and for each split
          network, so with and without `--split` differ only in the networks. The two networks train one after the
          other (no shared weights; the arm uses the true flag), each with its own early stopping. **Consequence:**
          the checkpoint is saved once at the end (a split checkpoint holds both networks), no longer at every
          improvement: the weights are still the best validation epoch's, but an interrupted run leaves none.
          Checked: a 1-epoch single-MLP run gives validation loss 0.0091, as before the change.
        - **`pos_weight` from the frames actually used:** adaptive subsampling keeps relatively more navigation frames
          (the wheel switches), so "arrived" is only **17.4%** of the training frames (31% of all frames): `pos_weight`
          4.74, not ~2.
        - Results columns of split models: `best_val_loss` = (4 x wheel MSE + 7 x torque MSE) / 11 on validation
          (comparable with the single MLP); `epochs` / `best_epoch` empty, per network instead (`base_epochs`,
          `base_best_epoch`, `arm_epochs`, `arm_best_epoch`); `val_arrived_bce` (unweighted, so it reads as a plain
          cross-entropy; early stopping used the weighted one); `split`. Offline test (each test episode replayed as
          the policy runs it: latch, flag into the arm net, wheels 0 after the latch): `test_arrive_err_median` (s,
          |latch - expert arrival|, episodes that latched), `test_arrive_early` / `test_arrive_late` (episodes more
          than 0.1 s early / late or never latched), `test_arrive_timeout`.
        - Policy script: loads any checkpoint with `load_policy`; for split models the latch, wheels 0 after it, a
          printed "Arrived flag latched at t=..." line, and `p_arrived`, `nav_done` (the latched flag, under the
          expert's name so the two compare frame by frame), `arrived_at` and `arrived_source` (network / timeout /
          never) in the `.npz`. Single-MLP models run the old line unchanged.
        - **`compare.py` unchanged:** a table of latch time vs expert arrival and the base's distance to the expert's
          parking spot at the latch was built, then removed to keep `compare.py` simple (it runs split models without
          any change; the numbers are in the policy `.npz` files if a failure needs diagnosing). Not in `results/`
          either (closed-loop latch numbers would need new `sim_*` columns that `train.py` also clears).
        - Smoke test (1 epoch, scratch copy): both modes run end to end; inference 0.17 ms per frame for both networks
          (single MLP ~0.07 ms), well under the 1 ms step.
        - **Before tuning a split model:** `tune.py` doesn't pass `--split` yet, and its printout and summary use
          `best_epoch` / `epochs`, which are empty for split models.
      - **Result (30 Oct), first trainings + quick tests** (default settings, seed 0): all 0/50.
        - Offline: `mlp_robotframe_s0` val 0.0014, test wheels 0.063 rad/s, torques 0.111 N*m. Both split models
          (with / without gravity residual) wheels 0.181, torques 0.77: an offline check on 38 test episodes showed
          the **arm net is twice as accurate as the single MLP given the true flag** (torques 0.056 vs 0.108 N*m), and
          the torque error comes from the latch: p(arrived) crosses 0.5 a median 87 ms before the expert's arrival
          (`pos_weight` 4.74 likely shifts it early), so the arm already reaches while the expert still holds. The
          3-input base net is worse at the wheels (0.21 vs 0.07 rad/s before arrival, even with the true flag).
          Gravity residual: no difference (arm val 0.0008 both; base identical, same seed).
        - Closed loop (final error median): `mlp_robotframe_s0` ~8 m (7 runs flung 400-1,100 m; worse than
          `mlp_robotframe_old`'s 0.78 m despite the lower val loss), `split_arrived_nograv_s0` ~1.1 m,
          `split_arrived_s0` ~0.86 m (no blow-ups).
        - **`split_arrived_s0` navigates like the expert:** the flag latched by the network in 50/50 runs at a median
          7.7 s (expert 7.9 s; the old robot-frame model got closest to the spot at ~20 s), the base 9 cm from the
          parking spot (all within 15 cm; expert 7.4 cm), heading error 0.011 rad. The early latch doesn't matter in
          closed loop. **The arm then fails:** it follows the expert's reach almost exactly for ~1 s, gets to
          1.5-2.5 cm (closest median 5.1 cm, 25/50 under 5 cm, best 6 mm), drifts off and collapses (error back above
          20 cm a median 3 s after the latch, torques at the limit). Final errors only measure the collapse: compare
          models on the closest approach.
        - **Uniform arm frames tested and rejected** (`split_arrived_armuniform_s0`; base adaptive, arm uniform). Why
          tried: adaptive keeps only 6.1% of frames within 2 cm of the target vs 16.4% of all frames, where the arm
          fails. Result: worse, closest approach median 8.5 cm (7/50 under 5 cm), collapse sooner (1.65 s), 2 runs
          flung > 100 m. The arm learns better from adaptive frames (dense at the start of the reach), so the drift
          isn't a data-balance problem but compounding error away from the expert's states: next candidates are noise
          injection (19, Kris) or joint-target outputs (team decision). The `ARM_SUBSAMPLE` switch in `train.py` is
          commented out (not deleted, it was never committed).
        - **Gravity residual helps in closed loop** (`split_arrived_s0` vs `split_arrived_nograv_s0`, quick test rerun
          with its own `--out-dir`): closest approach median **5.1 vs 13.2 cm**, under 5 cm 25 vs 11 of 50, under 2 cm 6
          vs 1, error back above 20 cm 3.0 vs 1.35 s after the latch; latch time and park error identical (same base
          net). Offline the two are identical (arm val 0.0008, test torques 0.77 N*m): offline every state is one the
          expert visited, but in closed loop the arm drifts into poses it never visited, where gravity computed
          exactly for the actual pose leaves only the small remainder to guess. So the split models keep
          `--gravity-residual` (`split_arrived_s0` is the main variant), and step 12 has a first closed-loop result in
          its favour. One seed, quick subset: the number for the report comes from the gravity ablation (23) with
          several seeds. (The earlier "Gravity residual: no difference" above is offline only.)
        - **Both decisions re-checked on the 50 validation episodes (30 Oct, `compare.py --val-set`), both hold**
          (they were made on the quick test subset, see "Recording results"): closest approach median **4.9 cm**
          with gravity residual and adaptive arm frames (26/50 under 5 cm, 5 under 2 cm, error back above 20 cm
          2.9 s after the latch) vs **10.9 cm** without gravity residual (13/50, 1; collapse 0.9 s) and **9.4 cm**
          with uniform arm frames (11/50, 2). Navigation the same in all three (latch ~7.9 s, 9.2 cm from the spot).
          `split_arrived_s0` stays the main variant and DAgger starts from it.
        - Keep each model's quick-test files with `compare.py --out-dir compare_out/<model>` (otherwise each run
          overwrites `compare_out/test_set_quick/`).
    - **v2: phase head: parked (30 Oct), not worth doing for now.** Why: what it was for is already fixed by v1. In
      the `split_arrived_s0` quick runs the turn -> drive switch comes a median 23 ms after the expert's (80% within
      -54..+57 ms, worst 190 ms; detected from the commanded forward speed, a detector that finds the expert's logged
      switch to the millisecond), the base arrives at 7.7 s (expert 7.9 s) and parks 9 cm from its spot (expert
      7.4 cm). The remaining failure is the arm, which a better base can't change, and the time goes there. The
      "15 vs 14" report point still works with v1's arrival latch (hand-built memory) against a GRU's timing.
      Revisit only if navigation breaks later, e.g. from other start poses (step 24: the 3-input base net has only
      seen origin starts). The design below is kept as it was.
    - v2 design (added 29 Oct; after v1, so the comparison shows what it adds). The expert's phases only
      move forward (align -> drive -> orient -> arrived; phases can be skipped, never revisited), so the latch
      generalises: the base net outputs 4 phase logits (from `nav_phase` + `nav_done`; "orient" = `orient` and not
      `nav_done`; class-weighted cross-entropy, since phases that finish instantly have no frames) instead of the
      1 arrived logit, and gets the current phase (one-hot) as input: the true phase in training, the latched one
      at run time. The policy only advances the phase (p(a later phase) > 0.5 for ~20 ms); "arrived" is v1's flag.
      - Why per frame is enough here: the latch is the memory, so the net only has to spot the next transition, and
        the transitions are mostly readable from the base inputs (align ends when the heading to the parking spot
        is within 0.05 rad, and since the base turns in place on the start-to-target line that is the heading to
        the target, up to the ~9 cm skid; drive ends within 8 cm of the spot, a standoff from the target set by its
        height).
      - **This is the part that addresses the late align -> drive switch:** a regression blurs the jump in the wheel
        commands (turning at +-3 rad/s -> driving forward), a classifier + latch makes it a sharp decision.
      - Not a per-frame phase classifier feeding a GRU (14): without the latch the phase can't be read from one
        frame near the switches (the expert's switch rules have hysteresis), and a front end trained on true phases
        but run on predicted ones adds a train/run mismatch.
      - **Report point: 15 vs 14.** Both go after the same problem, the expert's hidden "which phase am I in"
        (the single MLP sees one frame and can't know it). Here the memory is **built by hand** (the forward-only
        latch, using the logged phases as labels); in 14 the network **learns** it (TCN window, GRU state) from the
        commands alone. Compare them on the same metrics, especially the timing of the align -> drive switch and of
        arrival: if the hand-built latch wins, the structure of the expert is worth building in; if the GRU matches
        it, the memory can be learned without phase labels.
    - **The next two parts are the first extension (decided 30 Oct), once a model reaches targets, before the camera
      extension:** not needed for the main result (the base already parks 9 cm from its spot) and risky while the arm
      still fails; see "Who does what", Matteo point 4 for the order of work until then.
    - a model that **predicts the base's final pose** (where to park, from the target) that the base network then
      drives to;
    - **start the arm before the base arrives** (no expert change needed): train the arm network only on the reach
      phase (base parked), so the pre-arrival "arm holds" data isn't needed. At run time start the arm at t = 0 and
      give it the *predicted* final base pose instead of the current one, so it reaches as if the base had already
      parked; switch to the real base pose once the "arrived" flag is true. This goes beyond copying the expert
      (arm and base move together), so check: arm torques while the base drives (never in the training data),
      re-aiming if the base parks slightly off, and stability with a stretched arm on a moving base. If it works it
      should reach targets faster than the expert (time to reach is in `compare.py`'s table).
    **GRU in the split model (1 Nov, Kris): done, the MLP pair stays.** Code: `SplitPolicy(base_arch, arm_arch)`, `--base-from`,
    `bc_gru.py`; `step_*` matches `predict_*` for all three layouts. All runs: 50 validation episodes, Mac, seed 0, one
    seed per model. Counts recomputed from the episode files and equal to `compare.py`'s logs; same 50 ids for all.
    "Held" = stayed within that distance of the target for 0.5 s without a break, because closest approach flatters a
    swinging arm. "Parked" = base came within 15 cm of the expert's final spot. "Off floor" = base height moved > 10 cm.

    | model | reached | held 10 / 5 / 2 cm | parked | off floor |
    |---|---|---|---|---|
    | `mlp_gravres_lr1e3_s0` (single) | 0 | 9 / 3 / 0 | 44 | 4 |
    | `gru_h192_w1000_gravres_s0` (single) | 0 | 16 / 2 / 0 | 50 | 0 |
    | `split_arrived_s0` (MLP + MLP, no DAgger) | 0 | 38 / 16 / 0 | 50 | 3 |
    | `split_mlpbase_gruarm_s0` (same base, GRU arm, no DAgger) | 0 | 33 / 15 / 5 | 50 | 7 |
    | `split_grubase_gruarm_s0` (GRU + GRU, no DAgger) | 0 | 23 / 7 / 0 | 50 | 12 |
    | `split_dagger_r5_s0` (MLP + MLP, DAgger arm) | 23 | 43 / 43 / 33 | 47 | 1 |
    | `split_grubase_r5arm_s0` (GRU base + the same DAgger arm) | 11 | 47 / 39 / 18 | 48 | 0 |

    1. **Is split worth it compared with the temporal model? Yes.** With no DAgger neither reaches, but the split holds
       within 10 cm in 38 episodes against the GRU's 16, and within 5 cm in 16 against 2. The only models that reach
       use DAgger, so the jump to 23 is DAgger's, on top of the split; we have no single model with DAgger to separate
       the two.
    2. **Can the GRU go in the split model? Yes, it runs and it is stable, but it gives nothing here.** The reason for
       trying it has gone: the split MLP base parks as well as the GRU (50/50, median closest 0.8 cm).
    3. **Best configuration: MLP base + MLP arm with DAgger (23/50).** GRU arm against MLP arm on the same base, both
       without DAgger, is 0 v 0, and the GRU arm leaves the floor more often (7 v 3). GRU + GRU is the worst (12 off floor).
       GRU base + DAgger arm loses to MLP base + the same arm 11 v 23 (r5 only 14, GRU only 2, both 9).
       **That last comparison is not fair to the GRU base:** the DAgger corrections were collected behind the MLP base.
       Its parking is a little wider (median 3.4 v 2.4 cm from the expert's spot at the latch; 15 v 7 beyond 5 cm), and
       it latches later (+0.73 v -0.23 s against the expert). Even so, parked within 5 cm it reaches 7/35 against 22/43,
       so parking distance alone doesn't explain the drop. Not tested: DAgger rounds behind the GRU base. Given (2),
       not worth the time unless the MLP base turns out to be the limit.
    Mac v PC: `split_dagger_r5_s0` is 23/50 here, 26/50 on Matteo's PC. Differences of that size aren't meaningful.

    **New problem found (1 Nov, Kris): the DAgger arm sometimes spins during the drive.** In `split_dagger_r5_s0` a joint
    moves over 1 rad from its start before the latch in 10/50 episodes (worst 63 rad, about 10 turns; from 1 to 1.6 s
    in the worst three), and only 2 of those reach, against 21/40 for the others. `split_arrived_s0`, same base, never does (0/50;
    its arm stays within 0.06 rad), and its parking stays within 3.8 cm of the expert's spot, against up to 136 cm with the
    r5 arm, so the spinning arm also disturbs the drive. Cause not known. It isn't DAgger as such: the no-noise DART control
    (19) spins in 15/50 and has flag-0 frames in its extra data. The noisy DART arms spin in 0/50 (19). So the arm
    needs recovery examples in the drive too: use DART data alongside the final DAgger.
    **Correction (2 Nov, Matteo): the spinning was one seed's, not DAgger's.** r5 retrained with seeds 1 and 2 on the
    same DAgger data and base never spins (0/50 each) or collapses (B5). So DAgger as such doesn't make the arm spin;
    seed 0's arm does. DART + DAgger is still the plan (DART's precision within 1 cm), but "DAgger spins" is no longer a
    reason for it, and the report shouldn't claim it.
16. **Dropped (1 Nov, Matteo): shallow vs deep check.** Was: train one shallower and one deeper variant of the final
    model to show its size is reasonable. Why dropped: the time goes to the final model and its tests instead;
    efficiency (parameters, inference time per step) is still reported from the existing models (24).

### New data (`run_batch.py` into new folders)
17. **Minimum data needed:** train on 25 / 50 / 100 / 200 / 500 / all episodes and plot test performance (learning
    curve). Also decides whether to extend the dataset towards ~3,000 episodes (see the dataset-size note after
    the expert fixes).
    - **Base and arm data separately** (split networks, 15): also train the base on fewer episodes than the arm, and
      the other way round, to show which network needs the data. Possible because the two are trained separately;
      needs a `train.py` option per network (e.g. `--base-episodes` / `--arm-episodes`).
    - **No data augmentation, and why** (report point): the simulator gives as many expert episodes as we want, so
      there is no shortage to make up for. The one imbalance, the "arrived" flag (~17% of used frames), is handled by
      `pos_weight`. SMOTE / ADASYN would blend existing states into states the expert never visited, with blended
      labels the expert never gave, and target class imbalance, not our real problem (the policy drifting off the
      expert's states). The augmentation that fits behaviour cloning is new states labelled by the real expert:
      DAgger (21, done) and DART (19). The learning curve backs this: if it flattens, more data isn't the bottleneck.
18. More low-height targets.
19. Noise injection (DART): noisy commands applied, clean expert commands recorded as labels. Chosen instead of
    plain input-noise augmentation, whose labels would be wrong for the perturbed states.
    **Done (1 Nov, Kris): it works, 0 to 20/50 without DAgger.** `run_batch.py --noise A --wheel-noise W`: slowly varying
    noise (correlation 0.1 s) on the applied commands, clean expert commands stored as labels. Scale 1 = the split MLP's
    error on expert frames (0.01-0.08 N*m by joint, 0.24 rad/s; 20 validation episodes), 0.1 s is that error's
    correlation time. The frames go in through `--dagger` (`load_dagger_arm`): all frames, the expert's own flag.
    - **Pilot, 16 targets per level:** wheel noise is the wrong tool. Scale 5 on the wheels shifts the parking spot by a
      median 1.7 m and only 6/16 reach, because the expert has to replan a whole drive. Arm noise alone keeps parking within
      2 cm: 16/16 at 5 and 20, 12/16 at 60. So the runs use arm noise only. The expert is stiff, so even at 60 its arm
      is only 0.015 rad off its clean run; the policy drifts 0.03 rad at 1 s and 0.17 at 2 s after the latch.
    - **Runs:** 150 expert episodes on new targets (`--seed 2`, nearest is 12 cm from any validation target) added to the
      arm's training; base copied from `split_arrived_s0` (checked identical). Same validation and "held" as step 15.
      Failed noisy episodes stay in: their labels are still the expert's.

    | model | extra data | reached | held 10 / 5 / 2 cm | arm spins in drive | off floor | median final error |
    |---|---|---|---|---|---|---|
    | `split_arrived_s0` | none | 0 | 38 / 16 / 0 | 0 | 3 | 0.892 m |
    | `split_dart_a0_s0` | 150 clean (control) | 1 | 10 / 3 / 0 | 15 | 1 | 0.926 m |
    | `split_dart_a20_s0` | 150 at 20 (149 reached) | 5 | 50 / 48 / 33 | 0 | 0 | 0.042 m |
    | `split_dart_a60_s0` | 150 at 60 (116 reached) | 20 | 50 / 50 / 47 | 0 | 0 | 0.014 m |
    | `split_dart_a60_s1` | same, training seed 1 | 21 | 50 / 50 / 39 | 0 | 0 | 0.026 m |
    | `split_dart_a100_s0` | 150 at 100 (69 reached) | 15 | 50 / 50 / 39 | 0 | 0 | 0.039 m |
    | `split_dart_a60x2_s0` | 300 at 60 (`--seed 2` and `3`; 228 reached) | **30** | 50 / 50 / 45 | 0 | 0 | 0.007 m |
    | `split_dart_a60x2_s1` | same, training seed 1 (2 Nov) | 16 | 50 / 50 / 43 | 0 | 0 | 0.021 m |
    | `split_dagger_r5_s0` | 5 DAgger rounds | 23 | 43 / 43 / 33 | 10 | 1 | 0.034 m |

    The gain is the noise, not the extra data alone: the same targets without noise give 1, and success rises with the
    noise up to 60, then falls at 100, where over half the expert's runs fail. **More noisy data: not shown.** 300 episodes
    at 60 gave 30/50 with seed 0 but 16/50 with seed 1 (2 Nov); 150 episodes gave 20 and 21. The training seed alone moves
    "reached" by 14 episodes here, so 150 v 300 can't be told apart, and the 30 was a lucky seed. "Reached" (5 mm) is the
    noisy measure; "held within 5 cm" is 50/50 for every seed. On reached, DART is level with DAgger (23 here, 26 on
    Matteo's PC, one seed each), not ahead. DART also never loses the arm: 50/50 stay
    within 5 cm in every noisy model, against 43. What's left is the last 2 cm down to 5 mm. It needs no policy runs, and
    150 episodes take under 2 minutes.
    **Full test set, `split_dart_a60x2_s0`: 76/150 reached (50.7%)** (expert 149/150), Mac, recomputed from the episode
    files. Against `split_dagger_r5_s0`'s 62/150 (Matteo's PC, a machine difference of a few episodes is possible). Final
    error median 1.1 cm, mean 2.5 cm. Held 10 / 5 / 2 cm: 150 / 148 / 136. Parked 150, arm spins 0, off floor 0,
    no reach shorter than 50 ms within 5 mm, no latch by timeout. Mean time to reach 15.5 s. One seed. Picked on validation
    before this test ran. **Caveat (2 Nov):** seed 1 of the same recipe gets 16/50 on validation against seed 0's 30, so
    the 76/150 is the best seed's number, not what the recipe gives on average. For the report: test several seeds and
    give the mean and spread (step 23's 3 seeds).
    **Same model on Matteo's PC (2 Nov), val 50** (`compare_out/split_dart_a60x2_s0_pc_val/`): **32/50** reached (Mac
    30/50: the machine difference is small, as for r5's 23 vs 26), held within 1 / 2 / 5 cm for 0.5 s **37** / 46 / 50,
    closest approach median 0.41 cm, collapsed 0, arm spins 0. Against `split_dagger_r5_s0` on the same PC and targets:
    26 reached, held 1 / 2 / 5 cm 13 / 32 / 42, 8 collapsed, 9 spins. **On the brief's 1 cm the gap is much bigger than
    on reached (37 vs 13 held):** r5 swings around the target (e.g. val 429: between 0.5 and 4 cm every ~1.3 s, never
    settles; torques normal), DART comes in smoothly and stays. One seed each.
    **DART amount x 3 training seeds (B4, 2 Nov, Kris):** noise 60, validation 50, Mac. 600 episodes = the 300 above
    plus 300 new (`--seed 4` / `5` into `dart/a60c` / `dart/a60d`; the expert reached 112 and 117 of 150; the nearest new
    target is 5.0 cm from a validation target, against 3.8 cm for `data_main`'s nearest). Models `split_dart_a60_s*`
    (150), `split_dart_a60x2_s*` (300), `split_dart_a60x4_s*` (600), runs in `compare_out/<model>_val` (the older four
    in `compare_out/<model>/val_set`). Recomputed from the episode files and checked against the logs; all 9 on the same
    50 targets, base copied from `split_arrived_s0`.

    | DART episodes | reached, seeds 0 / 1 / 2 | mean (range) | held 5 cm | held 2 cm | held 1 cm | median final error |
    |---|---|---|---|---|---|---|
    | 150 | 20 / 21 / 19 | 20.0 (19-21) | 50 / 50 / 50 | 47 / 39 / 45 | 27 / 28 / 28 | 1.4 / 2.6 / 2.0 cm |
    | 300 | 30 / 16 / 25 | 23.7 (16-30) | 50 / 50 / 50 | 45 / 43 / 47 | 39 / 30 / 41 | 0.7 / 2.1 / 0.9 cm |
    | 600 | 30 / 24 / 26 | 26.7 (24-30) | 50 / 50 / 50 | 46 / 48 / 47 | 36 / 39 / 35 | 0.6 / 1.3 / 1.2 cm |

    "Held" = within that distance for 0.5 s without a break (1 cm here is this check, not yet the shared 1 cm number of
    14). No arm spins in the drive and no base off the floor in any of the 9. Training takes 1.3-3 min at every amount (600:
    2.0-2.7 min; early stopping ends it sooner, 44-53 arm epochs against 59-77 at 300), so more data costs almost no
    training time here.
    **Chosen: 300 (Kris, 2 Nov).** 600 v 300 is a tie under the stopping rule (300's mean, 23.7, is just under 600's
    lowest seed, 24, and its seeds overlap all of 600's), and a tie goes to the smaller amount. Read step by step the rule
    would go on to 150 (20.0 is inside 300's 16-30), but 150 is clearly worse than 600 (no seed overlap, 19-21 against
    24-30) and lower within 1 cm (27-28 against 30-41 at 300 and 35-39 at 600), so it stops at 300. 600 would have had
    the best mean and the tightest spread; the tie-break's reason (less data trains faster) didn't show in training time.
    **Full test set, 300 x 3 seeds (2 Nov, Kris, Mac), runs in `compare_out/split_dart_a60x2_s*_test`:**

    | seed | reached (of 150) | held 10 / 5 / 2 / 1 cm | median / mean final error | reach shorter than 50 ms |
    |---|---|---|---|---|
    | 0 | 76 | 150 / 148 / 136 / 102 | 1.1 / 2.5 cm | 0 |
    | 1 | 56 | 150 / 150 / 139 / 90 | 2.1 / 2.6 cm | 1 |
    | 2 | 61 | 150 / 150 / 139 / 94 | 1.9 / 2.4 cm | 2 |

    **Mean 64.3 / 150 (42.9%), range 56-76**; expert 149/150. Every run parked, no arm spins, nothing off the floor,
    no latch by timeout. Mean time to reach 15.5 / 16.2 / 14.9 s. Seed 0 rerun from scratch is identical to the 1 Nov
    run, arm trajectory included, so the test repeats exactly on this machine. The short reaches only pass through 5 mm
    (3 to 21 ms, final error 1.1-3.2 cm) but count as reached in `compare.py`, as for every model; without them 76 / 55 /
    59. Against `split_dagger_r5_s0`'s 62/150 (Matteo's PC, one seed): about level, not ahead.
20. **Randomised arm starting pose:** genuinely new situations that changing the target can't produce (the cart
    start is already covered by robot-frame inputs, see 6). Randomise it and collect data for it if we want to
    claim generalisation to new initial conditions.
21. DAgger (optional): the expert relabels states the policy visits.
    **Done (30 Oct, Matteo): DAgger for the split arm network** (why: the split arm drifts off the expert's states
    ~1 s into the reach and the data has no examples of correcting from there; step 15 results). Code, not run yet:
    - `expert_relabel.py`: the expert's own `arm_controller` stepped offline along any saved run, carrying its joint
      reference `q_des`, with the robot model rebuilt from the URDF: the torques the expert would have commanded at
      each frame, with no simulator and no expert re-runs. Check (`python expert_relabel.py`): on 5 expert episodes it
      gives back the saved torques to 0.0007-0.016 N*m median (worst 0.37 N*m; the base is taken as upright, only its
      yaw is saved).
    - `dagger.py --start split_arrived_s0 --rounds 3 [--quick]`: each round runs the previous model on 200 new
      **training** targets (never validation or test), labels the arm phase, saves the episodes to
      `dagger/split_dagger_r<k>/` (git-ignored) and calls `train.py --split --dagger <all rounds so far>`
      (models `split_dagger_r<k>_s0`, settings and gravity residual from the start model; `--quick` adds a quick test
      per round into `compare_out/<model>/`). Steps already done are skipped, so an interrupted run continues.
      Retrained from scratch each round on all data (standard DAgger: every round's model is a clean, comparable
      training); the base gets no new data and comes out identical (same data and seed), so every change is the arm's.
    - **Labelled part: from the latch until 5 s later, or until the expert's label stays at its 30 N*m limit for
      50 ms (not counted in the first 0.5 s, where the expert's own reach starts with a short kick at the limit).**
      Why: first tried until the arm's error climbed back above 20 cm, but then most late labels sat at the limit
      (60-100% of frames), a median 6-8 standard deviations outside the expert's data, and 1,137 such frames pushed
      the arm's training loss from 0.04 to 4.3, i.e. they would dominate training. Before the limit the labels are
      the expert's normal size (median 1-3 standard deviations); with the rule the labelled part ends at 1.7-2.4 s on
      the test runs and the loss stays normal (0.045). As the policy drifts less in later rounds, the part grows.
    - Episode format: `data_main`'s fields without the expert-only ones (`nav_phase`, `q_des_arm`, `base_goal`);
      `torque_arm` = the expert's labels (nan where unlabelled), `arm_label_mask` = which frames are labelled,
      `policy_wheel_cmds` / `policy_torque_arm` = what the policy did. Full rate and the full run, so base DAgger
      (wheel labels, later) and Kris's temporal models (windows) can use the same files.
    - `train.py --dagger <folders>` (with `--split` only; a single MLP would also need wheel labels): adds every
      10th labelled frame (like the expert data) to the arm's training frames. Full torques are stored and gravity is
      subtracted at training time, so the same files serve the gravity ablation. Normalisation and validation stay
      the expert data's (so `best_val_loss` stays comparable; it can't show DAgger's benefit: judge in closed loop).
      New results columns `dagger` (folders) and `dagger_frames`. Leaving `--dagger` out is the ablation.
    - Checked offline: labelling 3 policy runs takes 3.3 s (200 per round: ~1 min with 16 processes); the labels
      differ from the policy's torques by 0.4-1.7 N*m in the first 0.5 s, as in the diagnosis; a 1-epoch training with
      DAgger data runs and records the new columns. Not yet run: the policy runs on training targets (round 1).
    - **Dropped (1 Nov): base DAgger** (was: replay the expert's base state machine along policy runs for wheel and
      "arrived" labels; DAgger on Kris's temporal single model once base labels exist). Why: the base already parks
      well, the arm is the bottleneck, and it needs new code with 5 days left. Ablation: DAgger on / off.
    - **Result (30 Oct): DAgger works, the first targets reached.** `dagger.py --start split_arrived_s0 --rounds 3
      --val-set` (default settings, seed 0), closed loop on the 50 validation episodes:

      | Round (model) | DAgger frames (total) | Reached | Closest approach (median) | Under 1 cm | Collapsed (error back > 20 cm) |
      |---|---|---|---|---|---|
      | 0 (`split_arrived_s0`) | 0 | 0/50 | 4.9 cm | 1 | 50/50, at 2.9 s |
      | 1 (`split_dagger_r1_s0`) | 32.5 k | **10/50** | 1.2 cm | 22 | 16/50 |
      | 2 (`split_dagger_r2_s0`) | 111.6 k | **15/50** | 0.8 cm | 32 | 21/50 |
      | 3 (`split_dagger_r3_s0`) | 180.6 k | **22/50** | 0.7 cm | 31 | 7/50, at 12.6 s |

      - Navigation unchanged (latch ~7.9 s, ~9 cm from the spot): the gain is all the arm's, as intended. The arm no
        longer saturates (torques at the limit in 0-1% of post-latch frames, 80% before) and stays near the target
        (median time within 5 cm 5.4 s in round 3, 0.15 s in round 0). Final error median 0.92 m -> 0.04 m.
      - Round 2 labelled far more frames than round 1 (79 k vs 32.5 k): its policy drifted less, so the labelled
        part (until the expert's label saturates) got longer, as expected.
      - **Offline numbers got worse while closed loop got much better:** `best_val_loss` 0.0058 / 0.0125 / 0.0132 and
        test torque RMSE ~0.79 N*m, both measured on the expert's states, which the DAgger frames are not. The
        offline vs closed-loop gap again (a report point): judge DAgger in closed loop only.
      - Possible improvements: more rounds (the trend is still rising); early stopping judges on the expert's
        validation states only, and stopped the arm at epochs 11-27 (40-54 without DAgger): validating on held-out
        DAgger frames too could let it train longer. Choose the round on the validation 50, then the full test set.
    - **Rounds 4-5 and the first full test set (30 Oct).** `dagger.py` first allowed each training target only once
      (696 targets: at most 3 rounds of 200); later rounds now wrap around and reuse targets (rounds 1-3 unchanged).
      Why that's fine: a target is only a starting point, and a newer policy visits different states from it.
      Validation 50: round 4 (218.6 k DAgger frames) **24/50** reached, closest median 0.5 cm, 2/50 collapsed;
      round 5 (282.1 k) **26/50**, 0.5 cm, 8/50 collapsed. Gains flattening (+10, +5, +7, +2, +2).
      **Chosen: round 5**, by the rule set before looking (most reached, then closest approach). Round 4 is the steadier
      model (all 50 within 5 cm vs 42, 2 collapses vs 8) and 24 vs 26 is within run-to-run noise: seeds can decide.
      **Full test set, `split_dagger_r5_s0`: 62/150 reached (41.3%)** (expert 149/150), final error median 3.9 cm
      (mean 0.20 m), time to reach 11.2 s for reached runs (expert 12.2 s, both incl. the 1 s settle; the reached
      targets are likely the easier ones, so not a like-for-like speed comparison). 90/150 got within 1 cm at some
      point (the brief's 1 cm tolerance, "ever within 1 cm"). Validation (52%) was higher than test (41%): expected,
      since the round was picked as the best of five on validation. Offline as before: test wheels 0.181 rad/s,
      torques 0.80 N*m; inference 0.17 ms per step.
      **Recomputed 2 Nov from the saved files** (`compare_out/split_dagger_r5_s0/test_set/`): held within 1 cm for
      0.5 s **36/150** (held 2 / 5 cm 85 / 121). "Reached" is a first crossing too (`check_stop` doesn't check the 1 s
      settle), and half the reached runs swing through: only 30 of the 62 hold 1 cm for 0.5 s, 33 end the settle more
      than 1 cm away.

### Evaluation and report
22. **Compare all models:** one results table with every model (baseline, each branch, final), offline error and
    closed-loop success / final error, all on the same test set.
23. **Light version decided 1 Nov (see "Plan from 1 Nov"):** paired comparisons already made + 3 seeds and 2-3
    removal runs on the final model, instead of the full grid below (kept for the reasoning).
    **Ablations** (done once the winning branches are combined into one final model): remove one component at a
    time (robot-frame inputs, robot-frame polar vs Cartesian, gravity residual, history window, adaptive subsampling,
    noise data, DAgger data, split networks),
    retrain, and report offline error + closed-loop success for each vs the full model. This shows how much each
    design choice contributes in combination (branch results show each idea alone; the ablation shows it still
    helps with everything else, and reveals ideas that overlap).
    - While combining, make each component a switch (e.g. `ROBOT_FRAME = True`, `HISTORY_FRAMES = 10`), so every
      ablation run is a one-setting change and everything else is guaranteed identical.
    - **Also check the losers** (Matteo, 30 Oct): the best temporal model of 14 goes into the split networks (15), and the
      temporal models that lost the comparison are swapped in as ablation runs on the combined model, so a model that
      lost alone gets tested in combination too.
    - Same split, seeds and test set for all runs.
    - Budget time: e.g. 5 components x 3 seeds + the full model x 3 = ~18 trainings plus a `--test-set` run each, so
      aim to have the combined model ready several days before 6 Nov.
    **DART-side removal runs (C9, 2 Nov, Kris), done:** one part removed at a time from the chosen DART model
    (`split_dart_a60x2_s*`: 300 episodes at noise 60, gravity residual, base copied from `split_arrived_s0`), 3 training
    seeds each, val 50, Mac. Noise off = the same 300 targets without noise (`dart/a0` + `dart/a0b`, `--seed 2` / `3`,
    targets checked identical; the expert reached all 300). Recomputed from the episode files and checked against the
    logs; runs in `compare_out/<model>_val`. Subsampling not run (decided 2 Nov).

    | model | reached, seeds 0 / 1 / 2 | mean (range) | held 10 / 5 / 2 / 1 cm, seed mean | arm spins | off floor | median final error |
    |---|---|---|---|---|---|---|
    | full (`split_dart_a60x2_s*`) | 30 / 16 / 25 | 23.7 (16-30) | 50 / 50 / 45.0 / 36.7 | 0 | 0 | 0.7 / 2.1 / 0.9 cm |
    | gravity residual off (`split_dart_a60x2_nograv_s*`) | 20 / 23 / 25 | 22.7 (20-25) | 50 / 49.7 / 45.3 / 31.7 | 0 | 0 | 1.5 / 2.5 / 1.3 cm |
    | noise off (`split_dart_a0x2_s*`) | 0 / 1 / 0 | 0.3 (0-1) | 22.7 / 4.0 / 0 / 0 | 1 / 7 / 0 | 1 / 8 / 1 | 98 / 105 / 104 cm |

    - **The noise is the whole gain:** the same 300 targets without noise collapse like the model without extra data
      (0-1 reached, about 1 m off at the end, arm spins and the base off the floor in some runs). Confirms step 19's
      one-seed result (`split_dart_a0_s0`, 150 episodes) with 300 episodes and 3 seeds.
    - **The gravity residual no longer matters on reached:** 22.7 against 23.7, well inside the seed spread, and the
      same at 2 cm; slightly lower within 1 cm (31.7 against 36.7, ranges 30-34 and 30-41 overlap). It mattered
      without DART (step 12: closest approach 4.9 against 10.9 cm on `split_arrived_s0`), so DART's recovery data
      covers what the residual helped with. For the report: a feature that helped a weak model can stop mattering in
      the final one, which is why removal runs are done on the final recipe. Measured in the DART model, not in DART +
      DAgger.
    - **Rerun on the merged code (2 Nov, Kris), as Matteo asked:** all 6 retrained after merging master (per-network
      settings, 4a): arm weights bit for bit identical, the 50 validation trajectories identical, so the table stands
      (reruns not kept). Redo only if tuning (4a) gives the DART arm non-default settings: the removals belong on the
      final recipe.
**Evidence per feature (drafted 2 Nov, Matteo; for the report and Jamie's tables, 22).** Every feature we added,
whether a test already shows its effect, and the two-row "without / with" table for the report where one exists.
One seed unless stated. Sets: "val 50" = `compare.py --val-set`, "quick 50" = `--test-set --quick`, "test 150" =
`--test-set`; rows of one table always come from the same set. Only `--test-set` writes `sim_success_rate` to
`results/`: val 50 and quick 50 numbers are in this file and the `compare.py` printouts only. Features without a test
are listed after the tables: **justify them in words in the report, or give them a removal run if time allows.**

- **Expert chatter fix (fix 5)**, `mlp_robotframe` retrained, quick 50:

  | Training data | Val loss | Test torque RMSE | Closest to parking spot (median) | Closest EE (median) |
  |---|---|---|---|---|
  | Old expert (torque chatter) | 0.0089 | 0.209 N*m | 0.76 m | 0.72 m |
  | Fixed expert | 0.0038 | 0.116 N*m | 0.09 m | 0.17 m |

- **Robot-frame inputs (6)**, old data:

  | Inputs | Val loss | Test wheels / torques RMSE | Closed loop |
  |---|---|---|---|
  | World frame (`mlp_worldframe`) | 0.0095 | 0.094 / 0.221 | 0/150 |
  | Robot frame (`mlp_robotframe`) | 0.0089 | 0.086 / 0.209 | 0/50 (quick) |

  From the origin both hold the same information, so no difference is expected here: the real test is the start-pose
  test (24, still to do), where the robot frame should hold up and the world frame should fail.
- **Adaptive subsampling (7)**, split arm only, val 50 (`split_arrived_armuniform_s0` vs `split_arrived_s0`):

  | Arm training frames | Closest EE (median) | Ever under 5 cm | Ever under 2 cm |
  |---|---|---|---|
  | Uniform | 9.4 cm | 11/50 | 2 |
  | Adaptive | 4.9 cm | 26/50 | 5 |

  Supporting, single MLP (quick 50): arm deviation at 6 s 0.520 rad uniform (`mlp_robotframe_uniform_s0`) vs 0.346
  adaptive (`mlp_robotframe_s0`).
- **Gravity residual (12)**, split, val 50 (`split_arrived_nograv_s0` vs `split_arrived_s0`):

  | Arm torques learned | Closest EE (median) | Ever under 5 cm | Ever under 2 cm | Error back > 20 cm after the latch |
  |---|---|---|---|---|
  | Full torque | 10.9 cm | 13/50 | 1 | 0.9 s |
  | Residual (gravity from Pinocchio) | 4.9 cm | 26/50 | 5 | 2.9 s |

  Offline the two are identical (arm val 0.0008): an offline vs closed-loop point for the report.
- **(v, w) wheel outputs (13)**, split with the same DAgger r5 arm, val 50: 4 wheel speeds (`split_dagger_r5_s0`)
  26/50 reached vs (v, w) base (`split_vwbase_r5arm_s0`) 6/50. Its base parks better (step 13), so the drop is the
  arm meeting states its DAgger data wasn't collected in: also the evidence that DAgger data only fits its own policy.
- **Temporal model (14)**, single models, test 150:

  | Model | Reached | Parked within 15 cm | Closest EE (median) | Base off the floor |
  |---|---|---|---|---|
  | MLP (`mlp_gravres_lr1e3_s0`) | 0 | 127 | 11.6 cm | 6 |
  | GRU, 1000 ms windows (`gru_h192_w1000_gravres_s0`) | 0 | 148 | 7.6 cm | 0 |

  TCN (`tcn_L4_d10_gravres_s0`) and past-frame window (`temporal_window_k4_d50_gravres_s0`) left the floor in 139
  and 144 of 150: text, not a row.
- **Split networks (15)**, no DAgger, val 50 (Mac):

  | Model | Reached | Held within 10 / 5 / 2 cm for 0.5 s | Parked | Base off the floor |
  |---|---|---|---|---|
  | Single MLP (`mlp_gravres_lr1e3_s0`) | 0 | 9 / 3 / 0 | 44 | 4 |
  | Split (`split_arrived_s0`) | 0 | 38 / 16 / 0 | 50 | 3 |

- **DAgger (21)**, val 50 (PC):

  | Model | Reached | Closest EE (median) | Collapsed (error back > 20 cm) |
  |---|---|---|---|
  | No DAgger (`split_arrived_s0`) | 0/50 | 4.9 cm | 50/50 |
  | 5 rounds (`split_dagger_r5_s0`) | 26/50 | 0.5 cm | 8/50 |

  Test 150: 62/150. All five rounds are in step 21 (big table).
- **DART (19)**, 150 extra arm episodes on the same targets, val 50 (Mac):

  | Extra arm data | Reached | Held within 10 / 5 / 2 cm | Arm spins in the drive |
  |---|---|---|---|
  | 150 clean expert episodes (`split_dart_a0_s0`) | 1 | 10 / 3 / 0 | 15 |
  | 150 episodes, arm noise 60 (`split_dart_a60_s0`) | 20 | 50 / 50 / 47 | 0 |

  The gain is the noise, not the extra data. Test 150 (`split_dart_a60x2_s0`, 300 episodes): 76/150, but its seed 1
  gets 16/50 on val against seed 0's 30: report the seed mean (B).
- **GRU base in the split (15)**, same DAgger r5 arm, val 50 (Mac):

  | Base network | Reached | Held within 10 / 5 / 2 cm | Base off the floor |
  |---|---|---|---|
  | MLP (`split_dagger_r5_s0`) | 23 | 43 / 43 / 33 | 1 |
  | GRU (`split_grubase_r5arm_s0`) | 11 | 47 / 39 / 18 | 0 |

  Not fair to the GRU (the DAgger data was collected behind the MLP base): say so next to the table.
- **Tuning vs development defaults** (`tune.py`), test 150:

  | Settings | Val loss | Reached | Final error (median) |
  |---|---|---|---|
  | Defaults, early stopping (`mlp_robotframe`) | 0.00377 | 0/150 | 0.86 m |
  | Tuned (`mlp_robotframe_tuned`) | 0.00217 | 0/150 | 1.90 m |

  A lower validation loss with a worse closed loop: the offline vs closed-loop gap.

**No test yet: justify in words in the report, or a removal run if time allows** (Jamie: decide with the owner which
of these get a run; the rest get one sentence each):
- **DART + DAgger combined:** to do, it is the final-model comparison (B), not optional.
- **Episode cleaning (8):** drops 6 of 1,000 episodes (5 stalled, 1 drifted to 10 mm). Words: a failed run is a wrong
  demonstration. Run (cheap): train without `clean_ids`; expected no visible change at 6 episodes.
- **sin/cos joint angles:** words: no jump at +-pi; in `data_main` all joints stay within +-0.7 rad, so no effect is
  expected on this data (it matters for randomised arm starts, 20). No run.
- **Robot frame, polar vs Cartesian:** only Cartesian was built. Words: Cartesian chosen for simplicity, polar
  (the expert's own distance + bearing) as future work. A run needs new code: skip.
- **Hand-coded rules in the split policy** (wheels forced to 0 after the latch, latch at the 25 s timeout): words:
  they mirror the expert's own rules. Evidence that they don't carry the result: no latch by timeout in the DART
  full test (150/150 latched by the network). No run.
- **Safety clips** (wheels +-3 rad/s, torques +-30 N*m): words: the expert's own ranges. Evidence (old data,
  `mlp_robotframe`, quick 50): final error mean 222 m without the wheel clip, 5.1 m with. No run.
- **Learning curve (17), start pose, out-of-range targets, robustness (24):** planned (C), not done.

**Big table for Jamie (22): every test not shown in a two-row table above but worth showing** (appendix or one large
results table; one row per model; take the numbers from `results/` where `sim_*` exists, otherwise from the step
named; mark the set, the machine and the number of seeds in every row):

| Group | Models (rows) | Set | Where the numbers are |
|---|---|---|---|
| Baselines | `mlp_worldframe`, `mlp_robotframe_old`, `mlp_robotframe_s0`, `mlp_robotframe_mac_s0`, `mlp_robotframe_tuned` | test 150 / quick 50 | step 6, tuning, `results/` |
| Gravity residual, single | `mlp_gravres_s0`, `mlp_gravres_lr1e3_s0`, `nogravres_s0` | quick 50 / test 150 | step 12, step 14 |
| (v, w) | `vw_s0`, `gru_h192_w1000_vw_gravres_s0`, `split_vwbase_r5arm_s0` | val 50 | step 13, step 14 |
| Past-frame windows | `temporal_window_k4_d10_s0`, `_d50_s0`, `_d125_s0`, `_d50_gravres_s0` | quick 50 / test 150 | step 14 |
| TCN | `tcn_L4_d10_gravres_s0`, `_lr1e3_s0`, `tcn_L6_...`, `tcn_L8_...` | quick 50 | step 14 |
| GRU | `gru_h192_w250/w1000/w2000_gravres_s0`, `gru_h192_w1000_gravres_pc_s0` (Mac vs PC) | quick 50 / test 150 / val 50 | step 14 |
| Split variants | `split_arrived_s0`, `_nograv_s0`, `_armuniform_s0`, `split_mlpbase_gruarm_s0`, `split_grubase_gruarm_s0`, `split_grubase_r5arm_s0` | val 50 | step 15 |
| DAgger rounds | `split_dagger_r1` ... `r5_s0`, `split_dagger_vw_s0` | val 50, r5 also test 150 | step 21 |
| DART | `split_dart_a0/a20/a60/a100_s0`, `split_dart_a60_s1`, `split_dart_a60x2_s0/_s1` | val 50, a60x2_s0 also test 150 | step 19 |
| Final model | the winner of B, 3 seeds + removal runs (23) | test 150 | to come |
| Expert | `expert` | test 150 | `results/expert.csv` |

24. **Experiments:**
    - **Several seeds per model, reported as mean +- spread** (e.g. 3 seeds each, mean +- standard deviation).
      Training is random (weight initialisation, batch order), so one run can be lucky or unlucky; a difference
      between two models only counts as a real improvement if it is bigger than the run-to-run spread. Applies to
      every number in the results tables (offline error, success rate, final error), including the ablations.
      Only the training seed changes between repeats; the episode split and test set stay fixed.
    - Generalisation: targets outside the training range, held-out directions, and **cart not starting at
      (0, 0, 0)** (the first time other start poses are tested, using the start-pose option from 4, for the final
      model and the world-frame baseline, see 6).
    - Robustness (sensor noise, friction, shoves).
    - Efficiency (parameters, inference time per step).
25. **Analysis plots:** final error vs target height and distance (once some targets succeed).
    - **Loss curves (added 2 Nov, Matteo):** training and validation loss per epoch for the base and the arm network,
      the kept (best validation) epoch marked, saved as `checkpoints/<model>_loss.png` by `train.py` (it only prints
      them now; saving them changes nothing that is trained). Why: shows over- / underfitting and where early stopping
      cut, and is the offline half of the offline vs closed-loop point (DAgger: val loss worse, closed loop better).
      Copied networks (`--base-from` / `--arm-from`) have no curve of their own; GRU halves (`fit_gru`) aren't
      recorded. **Built (2 Nov):** in `train.py` on master (every training saves it); first used on `test/arm-features`
      (step 6).
    - **Torque graphs (added 2 Nov, Matteo):** for 2-3 validation episodes, the 7 arm torques of the policy vs the
      expert's on the same target over time, with the latch time and the expert's arrival marked, and the EE error on
      a second axis. From the saved policy `.npz` (`compare_out/<model>_val/val_set/`) and the expert episode in
      `data_main`: no new simulation. One episode that gets close and one that collapses. Why: shows where the policy
      leaves the expert (the drift ~1 s into the reach, step 15) and whether it ends at the torque limit.
      Note: the two runs are only aligned at t = 0, not at the latch (the latches differ by ~0.2 s), so the torque
      jump at arrival shows up twice. **Built (2 Nov):** `plot_torques.py <run_dir> <episodes>` (figures in
      `compare_out/torque_plots/`); the EE error is its own bottom panel on the same time axis, not a second y-axis.
      Seen so far (`split_arrived_s0`): the policy follows the expert for ~1 s after the latch, J3 / J4 pull away, and
      it ends switching between +-30 N*m 4-5 s after the latch.
26. **Data collection rationale (report):** why targets are sampled as they are (radius 1-3 m, height 0.2-1.1 m,
    full circle, uniform over the area), why one fixed start pose, how many episodes and why.
    Note (2 Nov): the pilot's episodes 0-49 used the same range but drew the radius uniformly (median 2.0 m) instead of
    uniformly over the area (median 2.3 m); "smaller radius" in step 0 is misleading. No target in any of our data is
    closer than 1 m.
27. **Report:** IEEE template, max 6 pages + contribution statement; code README must say what was written from
    scratch vs adapted. Due **6 Nov**. Discussion points found so far:
    - **No test data in any choice (Matteo, 2 Nov; say it in the report):** every decision (settings, tuning, DART
      amount, DAgger round, seed for the DAgger start, which model wins) used the 150 validation episodes or their
      fixed 50; the 150 test episodes were only run to fill the tables, to show how the chosen models do on unseen
      targets. Why it matters: picking on test would make the test numbers optimistic (r5 was 52% on validation, where
      it was picked as the best of 5 rounds, and 41% on test).
    - **Offline vs closed-loop gap:** the first (world-frame) baseline had low offline error (wheels ~0.08 rad/s,
      torques ~0.19 N*m RMSE on the test set) but failed in closed loop: it stopped turning too early, drifted into
      states the expert never visited, and its commands blew up. This is the behaviour-cloning "compounding error"
      (covariate shift) problem, and it motivates the temporal models (14) and noise injection (19).
    - **Where offline errors are largest:** torque error is highest for low targets, wheel error highest for close
      targets (short drives, so phase switches dominate).
    - **Model selection methodology:** chosen on validation loss (best-epoch checkpoint), never training loss
      (rewards memorising) or the test set. No k-fold: enough episodes for a reliable fixed split, k-fold would
      multiply every training and closed-loop test; seed repeats give the error bars instead.
    - **Preprocessing, following the lecture pipeline:** clean (episode cleaning, RANSAC), scale (z-score from training
      data only), engineer (sin/cos angles, robot-frame inputs, velocities), balance (adaptive subsampling around
      phase switches), format (sequences for the RNN). Timestamp alignment doesn't apply: one synchronous sim loop.

### Extension: concurrent expert, arm moves while the base drives (professor, 3 Nov)
28. **Concurrent expert** (Kris, folder `concurrent_arm/`, nothing in `robotnik_arm_task/` changed). Suggested by the
    professor on 3 Nov: the model should estimate the arm's final pose for the target and move the arm there before the
    base has arrived. Order: 1. turn in place to face the target (no drive, no arm motion), 2. drive and move the arm
    at the same time. This is "parking pose + arm early" from 15, so far left as future work in the report (D under
    "Plan from 2 Nov"; Matteo's in "Plan from 1 Nov"): **agree with Matteo how far it goes before the 6 Nov deadline.**
    - **Files:** `concurrent_arm/concurrent_expert.py` (a copy of `robotnik_arm_task.py` with the changes below) and
      `concurrent_arm/concurrent_control.py` (the new arm functions). The shared `robotnik_arm_control.py` is imported
      from `../robotnik_arm_task`, not copied or edited.
    - **Run:** `cd concurrent_arm && ROBOENV_TARGET=x,y,z python concurrent_expert.py` (GUI; add `ROBOENV_HEADLESS=1`
      for none), or in batches from `robotnik_arm_task/` with `run_batch.py --script ../concurrent_arm/concurrent_expert.py
      --out-dir data_concurrent` (`data_concurrent/` is git-ignored). In a fresh checkout, `concurrent_arm/models` and
      `configs` are symlinks to `robotnik_arm_task/`'s (git-ignored; create them with `ln -s`).

#### Concurrent expert: exact changes vs the sequential expert (`robotnik_arm_task.py`)
The sequential expert runs the base through align (turn in place) -> drive -> orient -> done, and the arm calls
`arm_controller(..., q_hold_arm=q_init_arm, reach_active=base_navigation_done)` every step: it holds its start pose
until the base is done, then reaches with Cartesian IK. The concurrent expert changes only the arm, as follows:

1. **New: final arm pose, solved once at t = 0** (`solve_final_arm_pose`, after the parking spot is planned, before
   the loop). The base is placed virtually at its planned parking spot (`base_goal_pos`, `base_goal_bearing`, start
   height, no tilt), and `arm_controller` with `reach_active=True` is iterated with no simulation: the joints follow
   their reference exactly (q = q_des, zero velocity), dt = 0.01 s, from the start joint angles. It stops at 1 mm EE
   error, after 200 iterations without 0.01 mm of improvement, or at 3,000 iterations, and returns the best pose seen
   (`q_final`) and its EE error. Same IK as the reach: damping, joint-limit band, reach limit 0.85 m (`q_final_error`
   stays above 1 mm for targets beyond it). One dynamics model, so this overwrites `pin_data`, which the loop
   recomputes every step anyway.
2. **New: drive progress per step** (`drive_progress`): 0 in align, `1 - distance to parking spot / distance when
   the drive phase started` in drive (clipped to 0..1), and 1 in orient and once done. Uses the same
   `base_goal_pos` and phase as the base controller, which is unchanged.
3. **Changed: the arm while the base is not done.** Instead of holding `q_init_arm` with zero velocity reference:
   - reference `q_ref = q_init + (q_final - q_init) * b(u)`, with `b(u) = 10u^3 - 15u^4 + 6u^5` (minimum jerk: zero
     velocity and acceleration at both ends) and `u = (progress - ARM_PATH_START) / (1 - ARM_PATH_START)`, clipped
     to 0..1 (`arm_path_reference`). `ARM_PATH_START` comes from `ROBOENV_ARM_PATH_START`, default 0 (the arm moves
     over the whole drive);
   - `q_des_arm` steps towards `q_ref`, at most `max_joint_vel * dt` per step (1 rad/s on the fastest joint, the
     same cap as the IK; scaled so all joints keep their direction). `qd_des_arm` = that step / dt, so the PD gets
     a velocity feed-forward (the hold had 0) (`arm_path_controller`);
   - torque = `kp (q_des - q) + kd (qd_des - qd) + gravity`, clipped to +-30 N*m: the same gains (100, 20), gravity
     compensation and clip as `arm_controller`.
   So during align u = 0 and the arm holds its start pose exactly as before; during drive it moves to `q_final`;
   during orient progress is 1 and it finishes the path (if the rate cap left some of it).
4. **Unchanged once the base is done:** the same `arm_controller` call with `reach_active=True`, continuing from the
   current `q_des_arm` (no jump in the reference): Cartesian IK corrects for where the base really parked.
5. **Saved per episode, new fields** (all old fields unchanged, same names and types): `q_final` (7, float32),
   `q_final_error` (m), `arm_progress` (per frame, float32), `arm_path_start`. Printed at the start: `q_final`, its
   error and the IK iterations. Added with the re-solve option (7; `data_concurrent_v1` was collected before, so its
   files don't have them): `arm_mode` (per frame, int8: 0 path to `q_final`, 1 blend to `q_final_parked`,
   2 Cartesian IK), `arm_resolve`, `q_final_parked` (7; nan without the re-solve), `q_final_parked_error`,
   `resolve_time` (s; nan without).
6. **Removed:** the commented-out plotting block at the end (a string literal, never ran).
7. **Option, off by default: re-solve at the end of the drive** (`ROBOENV_ARM_RESOLVE=1`; tried and not used, see
   "Tried" below). On the first step after the drive phase, `solve_final_arm_pose` runs again for the measured base
   position and the goal heading, from the current reference (then forward kinematics are recomputed for the step,
   since the solve overwrites `pin_data`). The arm blends from its reference to that pose by minimum jerk over
   `blend_duration` (1.875 x the largest joint change / 1 rad/s, at least 0.2 s), and the Cartesian IK takes over
   only when the blend is over, even if the base is done before. Off, the expert is unchanged (checked: bit for bit).
8. **Cartesian IK from the end of the drive, on by default from 3 Nov** (`ROBOENV_ARM_EARLY_IK=0` turns it off; see "Tried"
   below). The reach (`arm_controller`, `reach_active=True`) runs whenever the base is done *or* its phase is past
   "drive" (orient; the phase stays "orient" once done), instead of only once done. Saved: `arm_early_ik`. With both
   options on, the blend of 7 still comes first. Off, the expert is unchanged (checked: bit for bit).
9. **Option: the Cartesian correction's gain** (`ROBOENV_CARTESIAN_KP`, default 1.5 = `ArmControllerParams`' default,
   so unchanged; tried higher and not used, see "Tried" below). Passed to the expert's own `ArmControllerParams`, so
   the shared code is untouched. The final-pose solves (1 and 7) keep the default gain (`solve_params`), so `q_final`
   doesn't depend on it (checked: identical at 2 and 3). Saved: `cartesian_kp`.

**Not changed:** simulation setup, target, start pose, standoff and parking spot, the base controller and its
phases (so `nav_phase` / `nav_done` mean the same), the stop rules (5 mm + 1 s settle; "reached" and the stall rule
still count only once the base is done, so an EE passing the target during the drive does not end the run), the
time cap, DART noise, and what is logged per frame. **Differs as a side effect:** the wheel labels come from the same
controller, but the base now carries a moving arm, so its motion and the wheel commands along it are not identical
to the sequential expert's on the same target. Two things the path phase does not do, unlike `arm_controller`:
clip `q_des_arm` to within 0.15 rad of the measured angles, or to the joint limits. Neither is needed for
`q_final` (inside the limits by the IK), but the path starts from the start pose, whose J6 (1.0 rad) is below its
lower limit (see "Found on the way" below).
    - **Done (Kris, 3 Nov, not pushed):** a184e79. Smoke test against the sequential expert on 4 targets (side, far behind, low
      behind, high near):

      | target | sequential: EE error when parked / reached at | concurrent: EE error when parked / reached at |
      |---|---|---|
      | (0, 1.1, 0.5) | 22.5 cm / 7.38 s | 5.7 cm / 6.26 s |
      | (2, -1, 0.9) | 18.0 cm / 9.66 s | 5.0 cm / 9.27 s |
      | (-1.5, -1.5, 0.2) | 53.6 cm / 13.22 s | 11.0 cm / 10.87 s |
      | (1.2, 0.3, 1.2) | 52.2 cm / 7.93 s | 4.2 cm / 5.69 s |

      All 8 runs reach 5 mm. The arm stays still while turning (0.06 rad of sag, the same as the sequential expert's),
      the base never leaves the floor, and the IK reaches `q_final` within 1 mm in 300-1,500 iterations. The final
      correction moves the joints up to 0.12-0.21 rad from `q_final`: that is the parking error.
    - **Found on the way, also in the sequential expert (not changed, it is shared):** the arm's start pose has J6 at
      1.0 rad, below its URDF lower limit (1.134). When J6 first moves up to about 1.095, PyBullet jolts it
      (about 8 rad/s) and the torques clip for about 20 ms. The sequential expert hits this at the start of its reach,
      the concurrent one mid-drive, so the arm net sees the jolt in different states.
    - **Done (Kris, 3 Nov, not pushed): `data_concurrent`, 300 episodes, first version (no early IK; renamed
      `data_concurrent_v1` when early IK became the default, see below)** (`run_batch.py --n 300 --seed 0 --workers 10
      --script ../concurrent_arm/concurrent_expert.py --out-dir data_concurrent`, Mac, 138 s, 796 MB, git-ignored).
      Episodes 0-299 have the same targets as `data_main`'s 0-299 (checked per episode), so the two experts compare
      target by target. Checked twice: `index.csv`, then recomputed from the `.npz` files (reached = within 5 mm once
      the base is done, as `check_stop`); both agree.

      | expert, same 300 targets | reached | time to reach, mean of the 298 both reach | EE error when parked, median / max | clipped-torque frames per episode, median / max |
      |---|---|---|---|---|
      | sequential (`data_main` 0-299) | 298 (stalled: 77 at 6.3 mm, 297 at 5.3 mm) | 11.38 s | 23.6 / 53.2 cm | 23 / 34 |
      | concurrent, first version (`data_concurrent_v1`) | 300 (77, 297 reached at 6.09 s, 12.41 s) | 10.29 s | 7.2 / 11.1 cm | 18 / 30 |

      - **Faster by 1.10 s per target** (mean; median 1.09 s; paired over the 298). The base parks at the same time
        (8.52 s mean in both): all of the gain is in the shorter correction after parking.
      - **Slower on 32 targets** (by 0.38 s on average, at most 1.44 s). **Looked at (3 Nov):** all 32 are at heights
        0.72-0.82 m (the 266 others span the whole range), and there are two reasons:
        1. **10 of the 32 are swing-throughs by the sequential expert, not faster reaches.** Its EE passes through the
           5 mm ball, counts as reached on that pass, and leaves it again during the 1 s settle (5 of them by more
           than 1 cm, at most 20.7 mm). Example: episode 170, 7.1 -> 0.5 cm in 0.4 s, then back out to 2.1 cm, still
           1.1 cm off 1 s later. All 10 of the sequential expert's swing-throughs on these 298 targets are among the
           32. The concurrent expert holds within 5 mm through the settle on all 298 (it approaches smoothly).
        2. **At these heights the sequential expert's start pose is already almost the final pose.** In the start pose
           the EE is 0.71 m ahead of the base at 0.757 m height; for targets at 0.72-0.82 m the standoff is 0.65 m,
           so the parked base has the target about 6 cm from the EE before the arm moves (EE error when parked:
           1.3-10.4 cm on the 32). The concurrent expert always parks with 3.9-11.1 cm of EE error (7.2 cm median),
           and from there its correction takes ln(error / 5 mm) / 1.5 s, the Cartesian IK's exponential approach
           (`cartesian_kp` 1.5): the measured times match that formula to within 1% (median ratio 0.99, r = 0.91),
           1.39-2.21 s. So any target where the start pose happens to land closer than that is corrected faster by
           the sequential expert. **Where the 7.2 cm come from** (corrected 3 Nov: first written as "the parking miss"
           alone, which is one of three parts). Medians over the 300, in the base's frame when parked:
           - forward -5.9 cm: the base stops 7.3 cm short of its planned spot (the drive ends inside `pos_tol`,
             8 cm), and the arm was aimed at the planned spot (r = 0.50);
           - sideways 2.9 cm: the final turn stops at its tolerance (`ang_tol`, 0.05 rad: the median heading error is
             exactly 0.050), which swings the EE by reach x angle = 3.6 cm (r = 0.82);
           - vertical -3.3 cm: the arm sags below its joint reference (PD tracking lag, up to 0.05 rad).

        Counted by "stays within 5 mm from here on" instead of the first pass (the swing-throughs no longer count
        early), the concurrent expert is faster on 269 of 295 paired targets (11.44 -> 10.31 s mean) and slower on 26,
        by 0.21 s on average (all 26 among the 32).
      - **Tried (3 Nov): re-solve `q_final` at the end of the drive, for where the base really is
        (`ROBOENV_ARM_RESOLVE=1`, off by default). Not better, so not used.** On the first step after the drive (the
        base is within 0.6 cm, median, of where it finally stops; the final turn lasts 0.35 s, median), `q_final` is
        solved again from the current reference for the real base position and the goal heading, and the arm blends
        to it (minimum jerk, the fastest joint peaking at 1 rad/s: 0.4-0.5 s) before the Cartesian IK takes over.
        Same 300 targets (run on the Mac into the scratchpad, not kept):

        | expert, same 300 targets | reached | time to reach, mean of the 298 the sequential reaches | EE error when parked, median | slower than sequential on |
        |---|---|---|---|---|
        | concurrent | 300 | 10.29 s | 7.2 cm | 32 |
        | concurrent + re-solve | 300 | 10.36 s | 5.9 cm | 49 |

        It removes only the forward part (the parking miss), not the sideways (heading) or vertical (sag) parts, and
        the correction time only grows with the log of the error. The blend then starts the Cartesian IK later, which
        costs more than that saves: +0.07 s mean (+0.035 s median), faster on 129 targets, slower on 171. Checked
        twice: the reach times from the `.npz` files (+ the 1 s settle) match `index.csv`'s `sim_time` within 6 ms
        (1 ms steps), and with the switch off the script reproduces `data_concurrent_v1` bit for bit (10 episodes rerun
        through `run_batch.py`).
      - **Tried (3 Nov): start the Cartesian IK when the drive ends (`ROBOENV_ARM_EARLY_IK`, now on by default). Better:
        -0.22 s per target, every reach still held.** Instead of holding `q_final` during the final turn in place
        (0.35 s median), the arm runs the same `arm_controller` reach from the first step after the drive. It steers
        from the measured EE, so it shrinks all three parts of the parking error at once. Same 300 targets, Mac,
        into the scratchpad (not kept):

        | expert, same 300 targets | reached | time to reach, mean of the 298 the sequential reaches | held within 5 mm through the settle | EE error when parked, median | parked -> reached, median | slower than sequential on |
        |---|---|---|---|---|---|---|
        | sequential (`data_main`) | 298 | 11.38 s | 288 | 23.6 cm | 2.77 s | - |
        | concurrent, first version (`data_concurrent_v1`) | 300 | 10.29 s | 300 | 7.2 cm | 1.74 s | 32 |
        | concurrent + early IK | 300 | 10.07 s | 300 | 5.2 cm | 1.55 s | 21 |

        - Against the concurrent expert, target by target: -0.22 s mean (-0.25 s median), faster on 269, slower on 21
          (at most +0.29 s; not looked at), the same on 10. Against the sequential expert: -1.31 s mean (was -1.09 s).
        - Below the 0.35 s the turn lasts, as expected: while the base rotates it sweeps the EE sideways, and the IK
          (capped at 0.25 m/s) can't fully keep up. No run reaches 5 mm before the base is done (the stop rule only
          counts "reached" from then on, so it couldn't end earlier anyway).
        - Clipped-torque frames unchanged (median 18, max 29 vs 30).
        - Checked twice: the `.npz` reach times match `index.csv` within 5 ms, and with the switch off the script
          still reproduces `data_concurrent_v1` bit for bit (10 episodes).
        - **Done (Kris, 3 Nov, not pushed): made the default and `data_concurrent` recollected with it.** The first
          version moved to `data_concurrent_v1` (+ `data_concurrent_v1.log`; git-ignored, kept for comparison). Same
          command (`run_batch.py --n 300 --seed 0 --workers 10 --script ../concurrent_arm/concurrent_expert.py
          --out-dir data_concurrent`), Mac, 129 s, 785 MB. Checked twice: all 300 reached in `index.csv` and
          recomputed from the `.npz` files (10.07 s mean over the 298, as in the table), `arm_early_ik` set in all
          300, and every episode identical bit for bit to the early-IK test run above. **`data_concurrent` is now
          the dataset for the concurrent model.**
      - **Tried (3 Nov): a higher gain for the Cartesian correction (`ROBOENV_CARTESIAN_KP`). Faster first pass, but
        the reaches stop holding: not used, the default stays 1.5.** Correction time is ln(error / 5 mm) / gain, so a
        higher gain shortens it as expected, but the joint PD (kp 100, kd 20) can't follow the faster reference, and
        the EE overshoots: the swing-throughs we found in the sequential expert. All with early IK, same 300 targets,
        Mac, into the scratchpad (not kept):

        | gain | reached | time to reach, mean of the 298 | parked -> reached, median | held within 5 mm / 1 cm through the settle | worst EE error after the first pass |
        |---|---|---|---|---|---|
        | 1.5 (default, `data_concurrent`) | 300 | 10.07 s | 1.55 s | 300 / 300 | 5.0 mm |
        | 1.75 | 300 | 9.83 s | - | 288 / 299 | 12.6 mm |
        | 2 | 300 | 9.60 s | 1.08 s | 240 / 296 | 11.9 mm |
        | 3 | 300 | 9.35 s | 0.89 s | 167 / 268 | 18.8 mm |
        | (sequential, `data_main`) | 298 | 11.38 s | 2.77 s | 288 / 293 | 20.7 mm |

        - First pass faster on every one of the 300 targets: -0.48 s mean at 2, -0.72 s at 3.
        - Already at 1.75, 12 runs leave the 5 mm ball after the first pass (288, as many as the sequential expert),
          so 1.5 is at the edge. A first pass that doesn't hold flatters the reach time and gives the policy worse
          labels during the settle: the reason the expert stops at 5 mm + 1 s is precise final-approach data.
        - No clipped torques during the correction at any gain (0, all runs; the 18 per run, median, are the J6 jolt
          mid-drive, about 20 ms), the torques change faster per step (0.010 -> 0.014 -> 0.017 N*m), the fastest joint during the
          correction 0.35 -> 0.43 -> 0.56 rad/s (median). `q_final` identical at every gain.
        - Checked twice: `index.csv` and the `.npz` files agree (all 300 reached at each gain, times within 6 ms), and
          with the default gain the script still reproduces `data_concurrent` bit for bit (10 episodes).
        - What a faster correction would need: a faster joint loop (higher PD gains, or a torque feed-forward from the
          arm dynamics, so the joints track a faster reference without lag). That changes the arm's torque labels
          everywhere, not just the last few cm, so it is a bigger change than a gain.
      - The arm moves at most 0.063 rad while the base turns (the same sag as the sequential expert's). The base never
        leaves the floor (lowest base z -0.004 m in both). `q_final`'s IK error is at most 1 mm on all 300 targets:
        none is beyond the reach limit.
    - **Next:** the model: a goal network (target -> `q_final`; its labels need no simulation) plus the split arm
      net with `q_final` and progress as inputs. Expect the goal network to be easy: with perfect parking the target
      is always `standoff(z)` straight ahead at height z, so `q_final` depends on the target height alone.

### Low priority: other model families (comparison only, unassigned: only if time is left at the end)
Purpose: show in the report why a neural network is the right choice. One or two of these in the results table is
enough. All from scikit-learn (already installed); reuse `make_features` from `bc_data.py` so inputs are identical.
- **Decision tree** (existing `DecisionTree` branch): interpretable, step-shaped outputs.
- **Random forest:** many averaged trees, more accurate and stable than one tree, supports all 11 outputs at once.
- **Gradient boosting** (`HistGradientBoostingRegressor`): often the strongest non-neural model on tabular data;
  predicts one output at a time, so 11 models (or `MultiOutputRegressor`).

Practical notes: subsample the training data first (they are slow / memory-hungry on millions of rows); they need
their own save/load (e.g. `joblib`) and a matching variant of the two model lines in the policy script; check the
time per prediction, since the policy must answer every 1 ms; like the single-frame MLP they have no memory of past
frames unless stacked frames are given as inputs.

### Parked (only if needed)
- Wheel-command safety clip in the policy script. **Done (28 Oct):** `np.clip(y[:4], -3.0, 3.0)` in
  `robotnik_arm_policy_task.py`, next to the existing +-30 N*m torque clip. 3.0 rad/s is the expert's largest wheel
  command over all of `data_main` (turning in place at its 1 rad/s limit). Why: the first robot-frame quick test
  (0/50) showed the base getting near its parking spot (median 0.33 m, 15 of 50 within 15 cm, best EE error 2.1 cm),
  then the wheel outputs exploding in unfamiliar states (median max ~1,700 rad/s) and flinging the robot ~47 m, so
  final errors measured how far it flew, not how close it got. Part of the policy script, not the model: training
  and offline test errors are unchanged, no retraining needed, and it applies equally to every model. It doesn't fix
  the drift into unfamiliar states (compounding error), only stops one extreme output from ending the run. All
  closed-loop results from here on use both clips; say so in the report.
  **Alternative considered, maybe later:** bound the outputs inside the model (`limit * tanh(...)` instead of the
  linear output layer; a sigmoid rescaled to +-limit is the same curve). Changes training and every model, and
  frames at the expert's limit (turns in place, saturated torques) learn slowly near the bound unless the limit
  has a margin (e.g. 10%). Reconsider as a model experiment, judged on validation + closed loop.
- `--expert-only` option for `compare.py`.
- Silence the expert's output in `compare.py`'s parallel run.
- Switch OpenBLAS to its pthreads build to drop the OpenMP workaround.

### Final extension (once the final model is done): camera input instead of the target position
The final policy no longer gets `TARGET_EE_POS` from the simulator: a depth camera on the robot finds the target (the
red sphere) instead. Closer to a real robot, which wouldn't be told the target's coordinates. Start this only once the
main model works and the ablations are running. In the report it is an **extension** on top of the main solution (the
brief's model input includes the target), not a replacement.

**Design: two models in a chain**
1. **Perception model (new):** camera image(s) in, estimated target position out.
2. **The finished policy (unchanged):** robot state + the *estimated* target in, wheel commands + torques out, exactly
   as now. It already knows how to reach a given target, so it **does not need retraining**. Optional refinement:
   fine-tune it on estimated targets so it tolerates small estimation errors.

**Sensor**
- A depth (RGB-D) camera: depth gives the distance directly, so the target's 3D position can be recovered (with RGB
  only, distance would have to be guessed from the sphere's size). `simulation_and_control` already provides
  `GetCameraImageFromLink(...)` (RGB, depth, segmentation mask from a camera on any robot link); example:
  `tests/eye_in_hand_camera.py` in RoboEnv.
- **Field of view:** targets are all around the robot, and the expert never demonstrates "look around until you find
  it" (it turns towards the target only because it knows where it is), so search behaviour can't be learned by
  imitation. Solve it with sensors: e.g. **four cameras on the base** (front, back, left, right) so one snapshot
  covers 360 degrees.
- **Don't use PyBullet's segmentation mask** to find the target: it labels the target's pixels exactly, which is
  essentially being given the answer. Use colour (RGB) + depth.

**See the target once, at the start**
- The target never moves, so it only has to be seen once, at t = 0. After that the robot knows its own pose
  (`base_pos`, `base_bearing`), so the target's position relative to the robot is updated every step from the robot's
  own movement, with no new images.
- This removes most camera problems: occlusion by the arm during the reach doesn't matter, and only one snapshot (no
  10-30 Hz rendering) is needed.
- It fits the robot-frame inputs (step 6): a camera on the robot sees the target relative to the robot, so
  perception can output the target directly in the robot's frame.

**Perception model: build and train**
- **Classical reference first (no learning):** RANSAC to fit and remove the floor plane (where the slides use it:
  3D spatial noise), then find the red pixels, read their depth, convert to a 3D point with the
  camera's intrinsics and its known pose on the robot. Shows how good a learned model would need to be to be worth it.
- **Learned version:** a small CNN in PyTorch, image(s) in, target position out. Training data is cheap: place the
  target at random, render the snapshot and save it with the true target position as the label. No expert runs
  needed, so thousands of examples are quick to generate. Store images small (e.g. 128x128).
- Optional realism: add noise to the depth image and test how robust the estimate is (fits the robustness tests).

**End-to-end (not recommended before the deadline)**
- A CNN mapping images straight to commands. Strictly harder than the main problem (which already struggles in
  closed loop with the exact target), needs far more data and training, makes the dataset much bigger, and failures
  are hard to diagnose.

**Evaluation** (same test set)
- **Perception error:** estimated vs true target position, in cm.
- **Closed-loop success** of the full chain vs the same policy given the true target. If success drops, the two
  numbers show whether perception or the policy is to blame.
