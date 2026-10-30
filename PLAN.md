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
4. **Then:** multi-model / split networks (15): base and arm networks, "arrived" flag, then the phase head,
   parking-pose prediction, starting the arm early.
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
2. Temporal data (14): past-frame windows, a TCN, then the RNN (GRU).
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
14. **Temporal data** (aimed at the early stop of turning and the phase switches). Also a discussion point: why
    temporal data is needed (the expert has hidden state: navigation phase, accumulated joint targets).
    - First, **past-frame windows** into the MLP: stack k frames with a spacing, sweep both.
      - **Code done (Kris, 30 Oct, on `Kris`, not pushed yet). Still to run: the sweep.** All defaults mean no history,
        so single-frame models train and run as before.
      - `1e5703f` (`bc_data.py`): `with_history(x, frames, history_k, history_spacing)` joins each input with its past
        frames (current frame first, spacing in frames, 1 frame = 1 ms), taken from the full episode before
        subsampling, repeating the first frame where there is no past. `FrameHistory` does the same at run time.
        `load_episodes(..., history_k=0, history_spacing=1)`.
      - `3146d02` (`bc_model.py`): `history_k` / `history_spacing` saved in the checkpoint, like `gravity_residual`.
      - `222f76b` (`train.py`): `--history-k`, `--history-spacing`; also recorded in `results/<model>.csv`. The offline
        test reads them from the checkpoint.
      - `ea9beb0` (`robotnik_arm_policy_task.py`): `FrameHistory`, one per episode; does nothing for single-frame models.
      - `c9bdcd8` (`analyse_temporal.py`, new): reads a `compare.py --test-set` run and reports the turn -> drive delay
        against the expert, closest approach to the parking spot and the target, wheels at the clip. Run each model
        with its own `--out-dir`: `compare.py` deletes the episode files of the previous run in that folder.
      - Checked: k = 0 inputs identical to before on real episodes; `FrameHistory` identical to `with_history` over
        whole episodes (k = 4, d = 50; k = 2, d = 1; k = 0); a 1-epoch training with k = 4, d = 50 runs end to end (856 k
        frames, 135 inputs, 2.3 s per epoch, 0.029 ms per prediction) and both a window and a single-frame model run in
        the policy script. `analyse_temporal.py`: expert runs given as policy runs show zero delay; its drive-start
        rule (forward speed > 0.1 m/s for 50 ms) matches the logged `nav_phase` exactly on 100 episodes.
      - **Stage 1 (Kris, 30 Oct), k = 4, 3 spans, shared `data_main`, quick test, seed 0:** all with today's defaults
        (lr 3e-3, 256 x 3, early stopping, adaptive frames, no gravity residual).

        | Model | Span | Val loss | Test wheels | Test torques | Reached | Final err (median) | Within 15 cm of spot | Closest EE (median) | Turn -> drive delay | Arm 0.15 rad off | Base 0.3 m off expert |
        |---|---|---|---|---|---|---|---|---|---|---|---|
        | `mlp_robotframe_s0` | 0 | 0.00130 | 0.056 | 0.108 | 0/50 | 1.36 m | 22/50 | 0.37 m | 0.03 s | 5.3 s | 6.0 s |
        | `temporal_window_k4_d10_s0` | 40 ms | 0.00184 | 0.066 | 0.118 | 0/50 | 12.5 m | 3/50 | 0.88 m | 0.12 s | 1.9 s | 2.9 s |
        | `temporal_window_k4_d50_s0` | 200 ms | 0.00129 | 0.052 | 0.109 | 0/50 | 13.3 m | 4/50 | 0.96 m | 0.00 s | 2.0 s | 3.0 s |
        | `temporal_window_k4_d125_s0` | 500 ms | 0.00157 | 0.057 | 0.113 | 0/50 | 7.40 m | 3/50 | 0.74 m | 0.04 s | 3.0 s | 4.1 s |

        - **Windows made closed loop worse, with no offline gain** (best window = reference on validation loss).
        - **The arm fails first:** it leaves its holding pose (0.15 rad) ~1 s before the base leaves the expert's path, in
          every model, ~3 s earlier with a window. By 6 s the joints have wound up by 7-45 rad (median) with windows, 0.75
          rad in the reference (the expert stays within 0.067 rad).
        - **The late turn -> drive switch is not the failure here:** the median delay is at most 0.12 s, so the base
          starts on time and is pulled off course once the arm has gone. In 16/50 `k4_d50` runs it never drives.
        - **The reference is worse than `mlp_robotframe`** (final error 1.36 vs 0.78 m; arm deviation at 6 s 0.75 vs 0.062
          rad; wheels at the clip 80% vs 7%) despite a lower validation loss. Adaptive frames are **not** the cause, see
          below. What is left: the learning rate (3e-3 vs 1e-3) or the longer training (68 best epoch vs 20 epochs).
      - **Follow-ups (Kris, 30 Oct), same reference with one change, quick test, seed 0:**

        | Model | Change | Val loss | Test wheels | Test torques | Reached | Final err (median) | Within 15 cm of spot | Closest EE (median; best) | Arm deviation at 6 s | Base 0.3 m off expert | Wheels at clip |
        |---|---|---|---|---|---|---|---|---|---|---|---|
        | `mlp_robotframe_s0` | adaptive frames | 0.00130 | 0.056 | 0.108 | 0/50 | 1.36 m | 22/50 | 0.37 m; 33 mm | 0.75 rad | 6.0 s, never in 9 | 80% |
        | `mlp_robotframe_uniform_s0` | uniform frames | 0.00257 | 0.062 | 0.104 | 0/50 | 5.36 m | 17/50 | 0.32 m; 25 mm | 0.84 rad | 5.9 s, never in 9 | 77% |
        | `mlp_gravres_s0` | `--gravity-residual` | 0.00131 | 0.057 | 0.104 | 0/50 | 2.26 m | 23/50 | **0.16 m; 17 mm** | **0.056 rad** | **7.7 s, never in 21** | 67% |

        - **Step 7, first result: adaptive is at least as good as uniform** (validation loss half, arm no worse). One seed,
          quick test: repeat before reporting. It didn't cause the arm failure.
        - **Gravity residual fixes the early arm failure:** arm deviation at 6 s is at the expert's level (0.056 rad, the
          expert stays within 0.067), and closest EE approach halves, with almost the same offline numbers. Still 0/50.
          The median final error is higher (2.26 m) but says little here: it measures where a failed run wandered to.
        - **Window on top of the residual, `temporal_window_k4_d50_gravres_s0` (200 ms): worse than the residual alone.**
          Val loss 0.00155 (vs 0.00131), test wheels 0.056, torques 0.111, 0/50, final error 8.70 m (2.26 m), within 15 cm of
          the spot 5/50 (23/50), closest EE 0.33 m (0.16 m), arm deviation at 6 s 0.54 rad (0.056), base 0.3 m off the
          expert at 6.6 s (7.7 s), never in 3/50 (21/50). So windows lose in all five comparisons, with and without the
          residual: on the whole network they destabilise the arm. The turn -> drive delay is 0.03 s without them, so
          they have nothing to fix there. Sweep stopped here: Stages 2-3 not run. One seed, quick test.
      - **What fails after ~7 s, `mlp_gravres_s0` (saved quick-test runs, 3 plotted against the expert):
        the reach, not the navigation.** Turn, drive and the approach follow the expert's path closely. The arm then
        leaves its hold at a median of 7.6 s (expert arrival 7.9 s), but reaches far more slowly than the expert (486:
        about 3.5 s for what the expert does in about 1.5 s), gets to a median of 0.16 m at 8.9 s (best 17 mm, no run
        within 5 mm), then keeps going past the expert's final pose and winds up. Only after that does the base leave
        the spot and the wheels go to the clip (turning in place at the clip in 47% of the frames after the closest
        approach): the wheel chaos is a consequence. In 27/50 runs the base also stops short of the spot (closest
        median 0.27 m). The expert's data ends 1 s after reaching, so it shows no recovery from a missed reach.
        **So the priority moves to the arm's reach**: temporal inputs on the base can't fix it. Candidates: split
        networks (15), noise data (19), a reach-focused arm model.
      - **Old settings on the residual, `mlp_gravres_lr1e3_s0` (Kris, 30 Oct; lr 1e-3, max 20 epochs, quick
        test, seed 0): the best closed loop so far, with the worst offline numbers of the residual models.** Val loss
        0.00211 (0.00131 at lr 3e-3), test wheels 0.075 (0.057), torques 0.125 (0.104); still 0/50, but within 15 cm of the
        spot 44/50 (23/50), closest to the spot 0.04 m (0.17 m), closest EE 0.10 m (0.16 m), best 9 mm (17 mm), base 0.3 m
        off the expert in only 5/50 runs (29/50), wheels at the clip 43% (67%). Training stopped at the 20-epoch limit, on
        its best epoch, so it was still improving. Learning rate and training length changed together: which of them
        matters isn't known yet. If this holds, `train.py`'s defaults (lr 3e-3, up to 100 epochs) are worth revisiting.
      - **Result annotations (Kris, 30 Oct):** `76bba15` records `architecture` and `seed`, and leaves `history_spacing`
        empty when there is no history. `analyse_temporal.py --model X` adds the closed-loop summary to `results/X.csv` as
        `quick_*` (or `full_*`) columns, with the machine that ran it; `sim_*` stays full-test only. The four models with no
        window were renamed (`temporal_window_k0_*` -> `mlp_robotframe_s0`, `mlp_robotframe_uniform_s0`, `mlp_gravres_s0`,
        `mlp_gravres_lr1e3_s0`).
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
        arm's hidden state `q_des_arm` is clipped to within 0.15 rad of the measured angles (`max_q_lead`), so its
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
        | `mlp_gravres_s0` | 1 frame | 141,579 | 0.00131 | 0.057 / 0.104 | 8.73 | 0.013 | 12/50 | 23/50 |
        | `mlp_gravres_lr1e3_s0` | 1 frame | 141,579 | 0.00211 | 0.075 / 0.125 | 8.97 | 0.014 | 2/50 | 44/50 |
        | `tcn_L4_d10_gravres_s0` | 160 ms | 29,003 | 0.00174 | 0.059 / 0.117 | 1.25 | 0.58 | 47/50 | 4/50 |
        | `tcn_L6_d10_gravres_s0` | 640 ms | 45,515 | 0.00227 | 0.064 / 0.130 | 1.09 | 3.15 | 50/50 | 2/50 |
        | `tcn_L8_d10_gravres_s0` | 2.56 s | 62,027 | 0.00184 | 0.050 / 0.120 | 0.58 | 3.35 | 50/50 | 0/50 |

        - Arm deviation: median over episodes of the mean |q - q_expert| over the 7 joints. Max torque first hit: median
          first time a torque reaches the 30 N*m clip. Off the floor: base height moved over 0.3 m from its start.
        - **Do not read `quick_final_err_median` (30 / 371 / 1,032 m) as distances.** The wheels allow at most 10 m in
          30 s. The arm is at the torque clip almost throughout, the physics blows up, and in the worst runs the robot
          ends over 1 km below the floor. In the median run it leaves the floor at 7.2 / 1.9 / 2.4 s.
        - Not a run-time bug: `step` and `predict` agree within 5e-6 over 4,000 frames of a real episode for all three.
        - The arm fails before the base has started driving (expert drive start 1.36 s), and a longer reach fails sooner.
          Same direction as the windows (arm fails at 2-3 s). Why is not established.
        - Not comparable with the MLP on size: these TCNs have 20-45% of its parameters, and val loss is not monotonic in
          L (L6 is worse than L4 and L8), which suggests the plain deep stack trains badly (no residual connections yet).
        - **Old settings on L4, `tcn_L4_d10_gravres_lr1e3_s0` (lr 1e-3, max 20 epochs): no rescue, unlike the MLP.** Val
          0.00408, wheel / torque RMSE 0.105 / 0.166, 0/50; max torque first hit 1.89 s (L4: 1.25 s), arm dev. at 2 s 0.73
          rad, off the floor 49/50, spot within 15 cm 6/50. The arm still fails before the drive starts. Val loss was still
          falling at epoch 20, so this model is underfit; it is not a test of a well-trained TCN at lr 1e-3.
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
    - **Compare TCN vs RNN** on the same inputs, split and metrics (validation loss, test RMSE, closed loop,
      inference time per step). Report points: (a) the MLP's only temporal information is the joint velocity
      (one-step memory; the lecture's "RNNs vs MLPs"); (b) the TCN needs a fixed window, while the arrival time
      varies 3.6-12.8 s, which the RNN handles without choosing a window (the lecture's "TCN limitations"); the TCN
      is simpler to train; (c) **14 vs 15**: the same hidden phase, learned here vs built by hand in 15 (the
      forward-only latch), see the report point at the end of 15.
    - Owner: Kris (step 14); the TCN can move to Matteo if 15 finishes first (decide at the check-in).
15. **Multi-model / split networks** (mirrors the expert's structure), from simple to more structured. Matteo,
    models `split_*` (e.g. `split_arrived_s0`). Design agreed 29 Oct, not built yet:
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
        not the flag. p(arrived) and the latched flag are saved in the policy `.npz`. Open: also latch at 25 s like
        the expert's navigation timeout (then log whether the network or the timeout latched).
      - Code (the single-MLP files stay as they are): `bc_split.py` reuses `bc_model.MLP` unchanged (base
        `MLP(3, 5)`, arm `MLP(25, 7)`), saved as **one** checkpoint so `ROBOENV_POLICY` / `compare.py --model`
        work; the logit column gets `y_mean` 0 / `y_std` 1 so the output normalisation doesn't rescale it.
        `bc_data.py`: additions only (`nav_done` labels aligned with the subsampled frames, `BASE_INPUTS` /
        `ARM_INPUTS` slices). The policy script branches on the checkpoint type; the single-MLP path is unchanged.
        Open: a new `train_split.py` (recommended: `train.py`, `tune.py` and teammates' merges stay untouched, at
        the cost of ~100 duplicated lines) vs a `--split` option in `train.py`.
      - Evaluation: reference = the single MLP retrained with the current defaults (e.g. `mlp_robotframe_s0`; the
        recorded `mlp_robotframe` used lr 1e-3, a fixed 20 epochs, uniform every-10th frames and 700 episodes, so
        it isn't a fair reference). Offline: `best_val_loss`, test RMSE replayed with the predicted, latched flag
        feeding the arm net, median arrival-time error, number of latches > 0.1 s early (early is the dangerous
        direction). Closed loop (`compare.py --test-set --quick`): reached, median final error, closest to the
        parking spot and when, latch time vs the expert's arrival, closest EE error. Old reference: 0/50, 0.78 m,
        35/50 within 15 cm at ~20 s, 17 cm.
    - **v2: phase head** (after v1, so the comparison shows what it adds; added 29 Oct). The expert's phases only
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
