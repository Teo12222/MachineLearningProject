# Project plan: behaviour cloning for the Robotnik arm task

COMP0245 assignment: train a PyTorch model that copies the expert controller, i.e. outputs the wheel commands
(4) and arm torques (7) from the robot state and the target. Report + code due **6 Nov**.

## Files

The shared pipeline lives on `master`, in `robotnik_arm_task/`:

| File | What it does |
|---|---|
| `robotnik_arm_task.py`, `robotnik_arm_control.py` | The expert (simulation + controllers); saves one `.npz` per episode |
| `run_batch.py` | Runs the expert on many random targets in parallel to collect data |
| `data_pilot/` | 350 expert episodes (all reach the target), listed in `index.csv` |
| `bc_data.py` | Model inputs/outputs, episode loading, train/val/test split by episode (fixed seed) |
| `bc_model.py` | Baseline MLP (3 x 256) with its normalisation stored inside, plus save/load |
| `train.py` | Trains, keeps the epoch with the lowest **validation** loss, then tests it offline on the test episodes. `--no-test`, `--test-only` |
| `robotnik_arm_policy_task.py` | Same simulation as the expert, but the trained model drives the robot (`ROBOENV_POLICY` picks the checkpoint) |
| `compare.py` | Expert vs model: `--target x,y,z`, `--episode N`, or `--test-set` (success rate + final error on all test targets) |

**Results:** every model's numbers are in `results/<model>.csv` (one file per model; see "Recording results" below).
Look there for the current state of each model instead of this file.

**Progress:** what has been done so far, commit by commit, is in the "Progress log" at the end of this file.

## How we work

- `master` = data collection, preprocessing and the shared pipeline. Each idea gets its own **branch**.
- **Keep the split and the evaluation identical on every branch** (`split_episodes` with the default seed, same
  test set, same `compare.py`), otherwise results can't be compared in the report.
- Fixes to shared files go to `master` first, then get merged into the branches.
- Don't commit `checkpoints/` or `compare_out/` (already in `.gitignore`); name checkpoints after the branch.
- Run with `pixi run python <script>` from `robotnik_arm_task/`. In PyCharm, scripts that import torch set
  `KMP_DUPLICATE_LIB_OK` themselves (needed on Windows).

## Who does what

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
4. **Then:** multi-model / split networks (15): base and arm networks, "arrived" flag, parking-pose prediction,
   starting the arm early.
5. **Once the ideas have results:** combine the winning ideas into the final model (components as switches),
   ablations (23), start-pose test (24).
6. Camera extension (final section of the plan), shared with Kris: whoever is free once the final model works.
7. **Report:** introduction, data collection (26), pipeline and evaluation methodology, input features, split-network
   results, README, and the final edit of the whole report.

**Kris: shared improvements, temporal models, data, camera**
1. **Shared improvements on master** (coordinate with Matteo, who owns `bc_data.py`): first `decode_outputs()` (10,
   needed early by Jamie's `(v, w)` outputs), then adaptive subsampling (7; the past-frame windows need its frame
   indices) and episode cleaning (8). Faster training + `--seed` (11) last: seed repeats only start once the main
   improvements are done.
2. Temporal data (14): past-frame windows, then the RNN (LSTM / GRU).
3. Learning curve (17), which decides whether to extend the dataset; more low-height targets (18).
4. Noise injection, DART (19); randomised arm starting pose (20) and DAgger (21) optional.
5. Robustness tests (24: sensor noise, friction, shoves), on the final model.
6. Camera extension (final section of the plan), shared with Matteo: whoever is free once the final model works.
7. **Report:** shared improvements (subsampling, cleaning), temporal models, data experiments, robustness, the
   offline vs closed-loop discussion, camera extension (with Matteo).

**Jamie: outputs, results and report tables**
1. Gravity-residual torques (12) and `(v, w)` outputs (13).
2. Analysis plots (25); low priority: RANSAC preprocessing (9).
3. One shallow vs deep check (16) on the final model, at the end.
4. **Report:** output-format results, shallow vs deep check and efficiency, plus:
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
5. **Review other expert quirks** (e.g. the base skids sideways while turning in place): fix or document.
6. **Log the expert's hidden state** in the `.npz`: navigation phase (`base_state.phase`), the "arrived" flag
   (`base_state.done`) and the arm joint reference (`q_des_arm`). These are the only things the expert carries over
   between steps that affect its commands and can't be rebuilt from saved data (time is already saved, the parking
   spot can be recomputed). Not model inputs: labels for the "arrived" flag (15) and ground truth to check whether
   the RNN (14) learned the phase. Also save the **start pose** (base x, y, yaw at t = 0): always (0, 0, 0) for
   now, but once `ROBOENV_START` exists the parking spot can't be recomputed without it. Only before recollecting,
   otherwise it costs another full collection.

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

**Dataset size: start with 1,000-1,500 episodes, extend only if it helps.**
- Why more than now: the model has to generalise over *targets*, and training currently sees only ~245 different
  targets spread over a 3D region (full circle, 1-3 m out, 0.2-1.1 m high). ~1,000+ episodes cover it far more
  densely (probably part of why low and close targets do badly now).
- Why not 3,000 straight away: disk (~3-4 MB per episode, so ~10-12 GB for 3,000), memory (~30 million frames, ~4 GB),
  and slower test runs. Start smaller and let the **learning curve (step 17)** decide: extend towards ~3,000 only if
  performance is still improving with more episodes. Report this ("we stopped at N because the curve flattened").
- Extending is cheap: each target depends only on the seed and the episode number, so
  `run_batch.py --seed <same> --start <N> --n <more>` adds new targets without repeating old ones.
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
  | `sim_success_rate` | `compare.py --test-set` | Fraction of test targets reached within 1 cm |
  | `sim_final_err_mean`, `sim_final_err_median` | `compare.py --test-set` | Final distance to the target (median is robust to runaway runs) |
  | `sim_time_to_reach` | `compare.py --test-set` | Mean time to reach, successful runs only |

- **Wheel RMSE is always measured on the 4 wheel speeds**, even for models that predict `(v, w)`: convert their
  output back to 4 wheel speeds with `decode_outputs()` (step 10) first, so the column stays comparable.
- The new dataset (1,000-1,500 episodes after the expert fixes, possibly extended later) replaces `data_pilot`; results recorded on `data_pilot`
  are not comparable and get recomputed or deleted when we switch.

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

### Master: shared improvements
7. **Adaptive subsampling** (**done**: `afafe7a`, `bd71ddc`, see the progress log; the comparison against uniform is
   still to run) (after the split): the base/wheel data sampled densely at the start and at phase
   switches but sparsely while driving straight; the arm data sampled densely while it moves (reach) and lightly
   while it holds; a small floor rate everywhere (never zero). Per episode, keeping frame indices so past-frame
   windows stay possible. Compare against uniform subsampling.
8. **Episode cleaning:** trim the idle tail (unless step 2 removes it), filter failed episodes, sanity checks
   (NaNs, torques at the +-30 N*m limit).
9. **RANSAC / outlier preprocessing (experiment):** keep both raw and preprocessed data and compare models trained
   on each. Note: the simulator is noise-free, so we expect little difference on clean data; it becomes more
   relevant once sensor noise is added (robustness tests).
10. **`decode_outputs()` helper** in `bc_data.py`, so the policy script doesn't change when the output format does.
    (**done**: `96f3ca3`, see the progress log)
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

**Feature checks from the preprocessing slides** (small, on `make_features`):
- Arm joints without limits (`has_limits` is false) can wrap: encode them as sin/cos, like the heading.
- Add the base velocity (linear + turn rate, from consecutive poses): not an input now, unlike the arm velocities.
- Min-max instead of z-score scaling for bounded quantities (joint angles, +-30 N*m torques): quick comparison.

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
- **Model-specific ones:** history length k and frame spacing (temporal models); RNN hidden size and training
  sequence length; subsampling rates and floor (adaptive subsampling); noise level (DART data).
- Keep the search small (a few values per hyperparameter, most important first: learning rate, network size, batch);
  record each tried setting as its own model name so it lands in `results/`.

### Branches: model ideas (rough order)
12. **Gravity-residual torques:** learn torque minus the gravity torque, add it back at run time (aimed at low targets).
13. **`(v, w)` wheel outputs:** forward speed + turn rate instead of 4 wheel speeds.
14. **Temporal data** (aimed at the early stop of turning and the phase switches). Also a discussion point: why
    temporal data is needed (the expert has hidden state: navigation phase, accumulated joint targets).
    - First, **past-frame windows** into the MLP: stack k frames with a spacing, sweep both.
    - Then a **recurrent neural network (RNN)**: an LSTM or GRU (`torch.nn.LSTM` / `torch.nn.GRU`). It keeps an
      internal memory that is updated every step, so it can learn the expert's hidden state (which phase it is in)
      instead of us choosing a fixed window. Training uses short sequences cut from the episodes; in the policy
      script the memory is carried from step to step and reset at the start of each episode.
15. **Multi-model / split networks** (mirrors the expert's structure), from simple to more structured:
    - separate **base** and **arm** networks, each with its own subsampling;
    - base network also outputs an **"arrived" flag** (bool): while false the base drives and the arm holds; when
      true the base stops and the arm network takes over (imbalanced classes: use a weighted loss);
    - a model that **predicts the base's final pose** (where to park, from the target) that the base network then
      drives to;
    - **start the arm before the base arrives** (no expert change needed): train the arm network only on the reach
      phase (base parked), so the pre-arrival "arm holds" data isn't needed. At run time start the arm at t = 0 and
      give it the *predicted* final base pose instead of the current one, so it reaches as if the base had already
      parked; switch to the real base pose once the "arrived" flag is true. This goes beyond copying the expert
      (arm and base move together), so check: arm torques while the base drives (never in the training data),
      re-aiming if the base parks slightly off, and stability with a stretched arm on a moving base. If it works it
      should reach targets faster than the expert (time to reach is in `compare.py`'s table).
16. **Shallow vs deep check (once, at the end):** no full sweep. Once the final model works, train one shallower and
    one deeper variant of it (e.g. 1 and 5 layers) to show the chosen size is reasonable; report accuracy against
    parameters and inference time (also covers efficiency).

### New data (`run_batch.py` into new folders)
17. **Minimum data needed:** train on 25 / 50 / 100 / 200 / 500 / all episodes and plot test performance (learning
    curve). Also decides whether to extend the dataset towards ~3,000 episodes (see the dataset-size note after
    the expert fixes).
18. More low-height targets.
19. Noise injection (DART): noisy commands applied, clean expert commands recorded as labels. Chosen instead of
    plain input-noise augmentation, whose labels would be wrong for the perturbed states.
20. **Randomised arm starting pose:** genuinely new situations that changing the target can't produce (the cart
    start is already covered by robot-frame inputs, see 6). Randomise it and collect data for it if we want to
    claim generalisation to new initial conditions.
21. DAgger (optional): the expert relabels states the policy visits.

### Evaluation and report
22. **Compare all models:** one results table with every model (baseline, each branch, final), offline error and
    closed-loop success / final error, all on the same test set.
23. **Ablations** (done once the winning branches are combined into one final model): remove one component at a
    time (robot-frame inputs, robot-frame polar vs Cartesian, gravity residual, history window, adaptive subsampling,
    noise data, split networks),
    retrain, and report offline error + closed-loop success for each vs the full model. This shows how much each
    design choice contributes in combination (branch results show each idea alone; the ablation shows it still
    helps with everything else, and reveals ideas that overlap).
    - While combining, make each component a switch (e.g. `ROBOT_FRAME = True`, `HISTORY_FRAMES = 10`), so every
      ablation run is a one-setting change and everything else is guaranteed identical.
    - Same split, seeds and test set for all runs.
    - Budget time: e.g. 5 components x 3 seeds + the full model x 3 = ~18 trainings plus a `--test-set` run each, so
      aim to have the combined model ready several days before 6 Nov.
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
26. **Data collection rationale (report):** why targets are sampled as they are (radius 1-3 m, height 0.2-1.1 m,
    full circle, uniform over the area), why one fixed start pose, how many episodes and why.
27. **Report:** IEEE template, max 6 pages + contribution statement; code README must say what was written from
    scratch vs adapted. Due **6 Nov**. Discussion points found so far:
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
- Wheel-command safety clip in the policy script.
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

## Progress log

Commits on `master` with what they changed, newest first. Add a row with every commit. Kris's commits stay local until
his master improvements (7, 8, 10) are done, then they are pushed together and Matteo merges.

| Commit | Date | Who | Step | Pushed |
|---|---|---|---|---|
| `bd71ddc` | 28 Oct | Kris | 7 | no |
| `afafe7a` | 28 Oct | Kris | 7 | no |
| `96f3ca3` | 27 Oct | Kris | 10 | no |

**`bd71ddc` Add SUBSAMPLE switch for training frames in `train.py`**
- `SUBSAMPLE` (`None`, `"uniform"` or `"adaptive"`) and `KEEP_FRAC` (default 0.1), next to `EPOCHS` / `BATCH` / `LR`.
  Default `None` = all frames, so training is unchanged until someone switches it on.
- Validation is always subsampled **uniformly** with the same `KEEP_FRAC`, so runs with different modes are judged on
  the same frames (comparable `best_val_loss`). The test set still uses every frame.
- Checked: 2-epoch runs in all three modes train, save and test end to end; subsampled runs take ~6 s instead of ~29 s
  on `data_pilot`.

**`afafe7a` Add uniform and adaptive frame subsampling to `bc_data.py`**
- `subsample_indices(wheel_cmds, torque_arm, keep_frac, mode, floor=0.3, smooth=50, seed)` returns the sorted frame
  indices to keep from one episode. Indices (not copies), so the temporal models (14) can still look up past frames in
  the full episode.
  - `"uniform"`: evenly spaced frames.
  - `"adaptive"`: frames drawn at random, weighted by how much the expert's commands change around each frame (smoothed
    over 50 ms, so the frames around a jump count too). Chance of a frame being kept =
    `floor * uniform + (1 - floor) * change score`, so 30% of the kept frames are spread over the whole episode and no
    part is left out. Wheels and torques get half of the change score each (torques change far more often and would
    otherwise drown out the few wheel switches). Seeded per episode `(seed, episode id)`, so repeatable.
- `load_episodes(..., keep_frac=None, mode="adaptive", seed=0)`: `None` keeps every frame (existing callers unchanged).
- Why change-based and not phase rules: on the pilot data 90% of the wheel-command change happens in ~10% of the frames
  (~3 jumps per episode), but the torques change over ~44% of the frames, almost as much while driving as while
  reaching, so "sample the arm lightly while it holds" would drop useful torque data.
- Checked on 50 pilot episodes: exact 10% kept, sorted, no repeats, repeatable; frames within 50 ms of a wheel switch are
  kept ~4.5x more densely than uniform; largest gap between kept frames 128 ms median (160 ms max).
- Note: adaptive also oversamples torque chatter, e.g. the arm wiggle near joint limits; the expert fix for the wiggle
  should be in before the recollection.
- Still to do: the comparison (uniform vs adaptive at the same `KEEP_FRAC`: validation loss, per-phase offline error,
  closed-loop success).

**`96f3ca3` Add `decode_outputs()` to turn model outputs back into robot commands**
- `decode_outputs(y)` in `bc_data.py` returns `(wheel_cmds[4], torque_arm[7])` from the model output: the inverse of
  `make_labels`, for one frame or a whole episode. A new output format (e.g. `(v, w)`, step 13) only changes
  `make_labels` and `decode_outputs`.
- `robotnik_arm_policy_task.py` runs the model through it (the +-30 N*m torque clip comes after decoding).
- `train.py`'s offline test now scores the decoded commands against the expert's saved `wheel_cmds` / `torque_arm`
  instead of the training labels, so every output format is measured on the same 4 wheel speeds and 7 torques.
  Results on the current format are unchanged (checked on a test episode).
