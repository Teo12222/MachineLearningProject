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
- **The test set depends on the dataset:** `split_episodes` shuffles the episode ids found in `data_main`, so if the
  dataset changes (e.g. extended with `--start 1000` after the learning curve), the train/val/test split, and with it
  the quick 50, change too. Then **every model's results must be recomputed** (retrain + `--test-set`), as when we
  switched from `data_pilot`.
- **Uniform subsample for now:** `train.py` trains and validates on every 10th frame (`FRAME_STEP = 10`, recorded
  as the `frame_step` column), via a crude `every` argument of `load_episodes` (thinned per episode while loading,
  so the full 12 M frames never sit in memory). Placeholder until adaptive subsampling (7) replaces it. The test
  RMSE always uses every frame.
- The training seed is still fixed (`torch.manual_seed(0)`) until `--seed` (11): put the seed in the model name
  yourself (e.g. `mlp_worldframe_s0`).

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
7. **Adaptive subsampling** (after the split): the base/wheel data sampled densely at the start and at phase
   switches but sparsely while driving straight; the arm data sampled densely while it moves (reach) and lightly
   while it holds; a small floor rate everywhere (never zero). Per episode, keeping frame indices so past-frame
   windows stay possible. Compare against uniform subsampling.
8. **Episode cleaning:** trim the idle tail (unless step 2 removes it), filter failed episodes, sanity checks
   (NaNs, torques at the +-30 N*m limit).
9. **RANSAC / outlier preprocessing (experiment):** keep both raw and preprocessed data and compare models trained
   on each. Note: the simulator is noise-free, so we expect little difference on clean data; it becomes more
   relevant once sensor noise is added (robustness tests).
10. **`decode_outputs()` helper** in `bc_data.py`, so the policy script doesn't change when the output format does.
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

**Decided (28 Oct): a `tune.py` script, after the robot-frame inputs (6).** Not before: the best settings depend on
the inputs, and every branch starts from the robot-frame baseline, so that's the model whose tuning matters (tuning
the world-frame model can't fix its closed-loop failure: 0/150 at a good validation loss).
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
