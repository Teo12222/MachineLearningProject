# COMP0245 — Behaviour cloning for a mobile manipulator (Robotnik base + Kinova arm)

A learned policy that drives the Robotnik base to a 3D target and brings the Kinova end effector onto it, trained by
behaviour cloning from a scripted expert in PyBullet. This README covers the most important files (section 1), which
code is ours and which comes from elsewhere (section 2), how to run the code and which outputs it writes (sections
3-7), and how the numbers in the report were obtained and how to replicate them (section 8). The method and the
results themselves are in the report.

**The models in this repository**

| model | checkpoint (`robotnik_arm_task/checkpoints/`) | what it is |
|---|---|---|
| **Final model** | `split_conc_drive_r1_s0.pt` | split policy (base network + arm network), learned from our *concurrent* expert (the arm moves to its pre-grasp pose while the base drives), DART + DAgger, plus one DAgger round that also labels the driving phase |
| Sequential comparison | `split_seqpre_ppdart_cld_r3_s0.pt` | the same recipe on the *sequential pre-shaping* expert (the arm only moves after the base has parked) |
| Camera detector: mast | `cnn_detector_mast_train.pt` | small CNN that finds the target in an RGB-D camera on a mast on the base, so the policy runs without being told the target position |
| Camera detector: palm | `cnn_detector_hand.pt` | the same detector for the other camera option we compare: a camera in the palm of the hand |

**Quick start:** set up the environment (section 3), then from `robotnik_arm_task/`:

```bash
python robotnik_arm_policy_task.py                                # the final model in the PyBullet GUI, default target
ROBOENV_TARGET=1.5,-0.8,0.6 python robotnik_arm_policy_task.py    # any target x,y,z (m)
```

**Branches:** `master` (this one) holds the code and the models above. `old-master` keeps the full working history:
every model we trained, all result files and figures, and `PLAN.md`, our lab notebook, which the "PLAN.md step N"
references in code comments point to.

The policy checkpoints are seed 0 of three training seeds, chosen on the validation set. Seeds 1 and 2 are in
`checkpoints/seeds/`; the earlier checkpoints that the training scripts start from are in `checkpoints/chain_starts/`
(see section 7).

---

## 1. Most important files

**Task and expert** (`robotnik_arm_task/`)
- `robotnik_arm_task.py` — the course's expert task script (with our fixes, section 2).
- `robotnik_arm_control.py` — the course's base and arm controllers and stop rule (with our fixes).
- `robotnik_arm_policy_task.py` — the same task driven by a trained policy instead of the expert.
- `run_batch.py` — parallel, headless data collection.

**Learning** (`robotnik_arm_task/`)
- `bc_data.py` — inputs and outputs of the networks, loading episodes, the train / validation / test split.
- `bc_model.py`, `bc_split.py` — the networks; the split policy (base network with an "arrived" flag + arm network
  with a gravity term from the robot model).
- `train.py` — training (`--split` for the split policy). `dagger.py` + `expert_relabel.py` — DAgger: runs the
  policy, relabels its states with the expert's controller offline, retrains.
- `compare.py` — closed-loop evaluation against the expert; `analyse_distances.py` — the distances table.
- Also: `tune.py` (hyperparameter search on validation), `train_gru.py` / `train_tcn.py` / `bc_gru.py` (temporal
  models we compared), `relabel_posture.py`, `analyse_temporal.py`, `plot_torques.py`, `results.py`.

**Concurrent expert and camera** (`concurrent_arm/`)
- **Concurrent expert:** `concurrent_expert.py`, `concurrent_control.py` — the base turns, then drives while the
  arm moves along a joint path to its pre-grasp pose (`ROBOENV_ARM_SEQUENTIAL=1`: only after parking).
  `concurrent_chain*.sh` and `seqpre_*.sh` train the models on it; `count_results.py` counts the results.
- **Camera:** `mast_camera.py` (RGB-D camera on a mast or in the palm, colour detector, target estimator),
  `camera_expert.py`, `camera_policy_task.py` / `camera_policy_eval.py` (the policy driven by the camera),
  `cnn_data.py` / `cnn_detector.py` / `cnn_eval.py` (the learned detector), `analyse_camera.py`,
  `check_camera_physics.py`.
- `learnability.py` — offline check of how well the arm network's inputs predict the expert's torques per phase.

**Robustness** (`robustness/`) — the run scripts and counts of section 8.3-8.4.

---

## 2. External code, and what we modified or wrote

**From the course (given):**
- `robotnik_arm_task/robotnik_arm_task.py`, `robotnik_arm_task/robotnik_arm_control.py` and
  `robotnik_arm_task/README.md` — the task, the expert's controllers and the setup note.
- RoboEnv: `simulation_and_control` (simulator interface, Pinocchio wrapper), `models/` (robot URDF) and `configs/`.

**Libraries:** PyBullet, Pinocchio, PyTorch, NumPy, Matplotlib.

**Modified course files** (`robotnik_arm_task.py`, `robotnik_arm_control.py`): the expert's stop rule (check
"reached" before "stalled", stop on reaching instead of a fixed minimum time, a shorter stall window, a 5 mm reach
tolerance), a joint-chatter fix, any start pose (`ROBOENV_START`), headless runs that save every frame to `.npz`,
DART noise on the applied torques, sensor-noise configs (`ROBOENV_CONFIG`) and an optional payload
(`ROBOENV_PAYLOAD`). Every addition is off by default.

**Adapted copies of the course's task script:** `robotnik_arm_policy_task.py` (the policy in place of the expert),
`concurrent_arm/concurrent_expert.py` (the concurrent expert), `concurrent_arm/camera_expert.py` and
`concurrent_arm/camera_policy_task.py` (with the camera).

**Written from scratch by us:** everything else — data collection, the networks and training, DART / DAgger and the
offline relabelling, evaluation and analysis, the concurrent arm path, the camera (detector, estimator, CNN) and the
robustness tests. Methods we implemented from the literature (no code taken): DAgger (Ross et al., 2011), DART
(Laskey et al., 2017), and a CenterNet-style detector head (Zhou et al., 2019).

---

## 3. Setup

The code runs in the course's **RoboEnv** environment (PyBullet, Pinocchio, NumPy, Matplotlib and the course's
`simulation_and_control` package) **plus PyTorch**. Nothing else is needed. CPU is enough for everything; only the
camera detector's training uses a GPU if one is available (CUDA or Apple MPS), and falls back to the CPU.

1. **Python environment.** Use your RoboEnv pixi environment and add PyTorch if it isn't there, e.g. from the RoboEnv
   folder: `pixi add --pypi torch`. Every command below is written as `python ...`; run it inside that environment
   (`pixi shell` in the RoboEnv folder, then `cd` here), or prefix it, e.g.
   `pixi run --manifest-path <RoboEnv>/pixi.toml python ...`.
2. **Robot model and configs.** As in the course README (`robotnik_arm_task/README.md`, step 1), copy RoboEnv's
   `models/` and `configs/` folders into `robotnik_arm_task/`:
   ```bash
   cp -r <RoboEnv>/models <RoboEnv>/configs robotnik_arm_task/
   ```
3. **Links for the concurrent and camera scripts.** The scripts in `concurrent_arm/` load the robot from `models/` and
   `configs/` in their own folder; link them to the copies made in step 2 (once per checkout):
   ```bash
   # macOS / Linux
   cd concurrent_arm && ln -s ../robotnik_arm_task/models models && ln -s ../robotnik_arm_task/configs configs && cd ..
   ```
   ```bat
   :: Windows (cmd), no admin rights needed
   cd concurrent_arm && mklink /J models ..\robotnik_arm_task\models && mklink /J configs ..\robotnik_arm_task\configs && cd ..
   ```
4. **Shell scripts** (`*.sh`) need bash: on Windows use Git Bash. They take the Python command from `PY`, e.g.
   `PY="pixi run python" bash ../robustness/robustness_runs.sh expert` (from `robotnik_arm_task/`).

Unless a command says otherwise, **run it from `robotnik_arm_task/`**.


---

## 4. Run a model

**Final model, in the PyBullet GUI** (no data needed):

```bash
cd robotnik_arm_task
python robotnik_arm_policy_task.py                                   # default target (0, 1.1, 0.5)
ROBOENV_TARGET=1.5,-0.8,0.6 python robotnik_arm_policy_task.py       # any target x,y,z (m)
ROBOENV_START=1.0,-0.5,1.57 ROBOENV_TARGET=1.5,-0.8,0.6 python robotnik_arm_policy_task.py   # base starts at x,y,yaw
```

On Windows PowerShell set a variable first, e.g. `$env:ROBOENV_TARGET = "1.5,-0.8,0.6"`, then run the script.

- **Prints** the target, the error every step, when the base's "arrived" flag latches, and the final error.
- **Writes** `policy_episode.npz`: the whole run frame by frame (1 ms per frame: time, base pose, joint angles and
  velocities, wheel commands, arm torques, end-effector position, target, stop message). The expert writes the same
  format (`episode.npz`), and every evaluation below reads these files.
- The run stops when the hand is within 5 mm of the target (then 1 s more to check it stays) or after 30 s.

**Other models:** `ROBOENV_POLICY=checkpoints/split_seqpre_ppdart_cld_r3_s0.pt python robotnik_arm_policy_task.py`.

**The expert (the course's controller, with our fixes):** `python robotnik_arm_task.py` (same variables; writes
`episode.npz`). **Our concurrent expert:** `cd concurrent_arm && python concurrent_expert.py`
(`ROBOENV_ARM_SEQUENTIAL=1` for the sequential pre-shaping variant).

**Policy finding the target with a camera** (from `concurrent_arm/`): the base first turns on the spot until the
camera sees the red target, then the final model drives from the camera's estimate. Two camera options, mast or palm
(`ROBOENV_CAMERA_PLACE`, default `mast`), each with its own detector:

```bash
cd concurrent_arm
# mast camera
ROBOENV_CAMERA_PLACE=mast ROBOENV_CNN_DETECTOR=../robotnik_arm_task/checkpoints/cnn_detector_mast_train.pt \
  ROBOENV_POLICY=../robotnik_arm_task/checkpoints/split_conc_drive_r1_s0.pt ROBOENV_DETECTOR=cnn \
  ROBOENV_TARGET=1.5,-0.8,0.6 python camera_policy_task.py
# palm camera
ROBOENV_CAMERA_PLACE=hand ROBOENV_CNN_DETECTOR=../robotnik_arm_task/checkpoints/cnn_detector_hand.pt \
  ROBOENV_POLICY=../robotnik_arm_task/checkpoints/split_conc_drive_r1_s0.pt ROBOENV_DETECTOR=cnn \
  ROBOENV_TARGET=1.5,-0.8,0.6 python camera_policy_task.py
```

`ROBOENV_DETECTOR=classical` (the default) uses the colour-threshold detector instead of the CNN.

**Expert and policy side by side:** `python compare.py --target 1.5,-0.8,0.6 --model split_conc_drive_r1_s0` runs the
expert headless and the policy in the GUI, then plots both (error, base path, end-effector path) into `compare_out/`.

**Settings by environment variable** (all off or at the course's behaviour by default): `ROBOENV_TARGET`,
`ROBOENV_START`, `ROBOENV_POLICY`, `ROBOENV_HEADLESS`, `ROBOENV_MAX_TIME`, `ROBOENV_OUT`, `ROBOENV_CONFIG`,
`ROBOENV_PAYLOAD`, `ROBOENV_NOISE` / `ROBOENV_CART_NOISE` / `ROBOENV_NOISE_FADE` (DART noise, set by `run_batch.py`),
`ROBOENV_ARM_EARLY_IK`, `ROBOENV_ARM_SEQUENTIAL`, `ROBOENV_ARM_RESOLVE`, `ROBOENV_ARM_PATH_START`,
`ROBOENV_CARTESIAN_KP`, `ROBOENV_CAMERA`, `ROBOENV_CAMERA_PLACE`, `ROBOENV_CAMERA_MOUNT`, `ROBOENV_CAMERA_REFINE`,
`ROBOENV_DETECTOR`, `ROBOENV_CNN_DETECTOR`. Each script's header explains the ones it reads.

---

## 5. Gather data quickly

Expert data is collected **in parallel and headless** with `run_batch.py`: each episode is one subprocess, so 10
workers collect 1,000 episodes in about **10-20 minutes** on a laptop (one episode is 4-10 s of wall time). Targets
are drawn from a seed (1-3 m from the start, 0.2-1.1 m high), so the same command always gives the same targets.

```bash
# a quick look: 20 episodes of the course's expert into data_try/ (about a minute)
python run_batch.py --n 20 --workers 10 --out-dir data_try
```

Each episode is `<out-dir>/ep_XXXX.npz` (format above) plus one line in `<out-dir>/index.csv` (target, outcome,
final error, simulated and wall time).

**The datasets the report's models use** (all `--seed 0`, ids 0-999, so the train / validation / test split is the
same in every folder: 700 / 150 / 150 episodes, `bc_data.split_episodes`):

| folder | what it holds | command | used for |
|---|---|---|---|
| `data_concurrent_expert` | 1,000 runs of **our concurrent expert**: the base turns to the target, then drives while the arm moves to its pre-grasp pose; after parking, the arm closes in with Cartesian control | `ROBOENV_ARM_EARLY_IK=0 python run_batch.py --n 1000 --seed 0 --workers 10 --script ../concurrent_arm/concurrent_expert.py --out-dir data_concurrent_expert` | the final model: training data, and the targets of its validation and test runs; camera runs |
| `data_sequential_expert` | 1,000 runs of the **sequential pre-shaping expert**: the same, but the arm stays still while the base drives and only moves to its pre-grasp pose after parking | `PY=python bash ../concurrent_arm/seqpre_collect.sh` (also writes its DART and low-target folders) | the sequential comparison model |
| `data_course_expert` | 1,000 runs of the **course's expert** (`robotnik_arm_task.py`, with our fixes): base drives and parks, then the arm reaches | `python run_batch.py --n 1000 --seed 0 --workers 10 --out-dir data_course_expert` | the expert column of the distances table and the sensor-noise runs (section 8) |

All three folders use the same 1,000 targets.

**To evaluate the shipped models you only need the folder whose targets the evaluation uses** (`data_concurrent_expert` for
the final model, `data_sequential_expert` for the sequential one): the split is taken from the 1,000 episode files, so collect
all 1,000. The robustness tests (start pose, distance, payload) need no dataset at all.

**Extra data for retraining only** (section 7):

```bash
S=../concurrent_arm/concurrent_expert.py; export ROBOENV_ARM_EARLY_IK=0
python run_batch.py --n 150 --seed 2 --noise 60 --workers 10 --script $S --out-dir dart/b_a60      # DART: noise on the applied arm torques, clean labels
python run_batch.py --n 150 --seed 3 --noise 60 --workers 10 --script $S --out-dir dart/b_a60b
python run_batch.py --n 60 --seed 11 --height-range 0.2,0.5 --workers 10 --script $S --out-dir data_low_b   # extra low targets
```


---

## 6. What the evaluation scripts write

| command | does | writes |
|---|---|---|
| `python compare.py --val-set --model M --data-dir D --out-dir compare_out/X` | policy M headless on the 50 fixed validation targets (all model choices were made on these) | `compare_out/X/val_set/ep_*.npz`, a per-episode table and plots; no results file |
| `python compare.py --val-full ...` (same options) | all 150 validation targets | `compare_out/X/val_set_full/` |
| `python compare.py --test-set ...` (same options) | all 150 test targets, used only for the reported numbers | `compare_out/X/test_set/`, `test_set_final_error.png`, and `results/M.csv` + `results/expert.csv` |
| `python ../concurrent_arm/count_results.py compare_out/X ... --data-dir D --set val\|val_full\|test` | the report's counts from saved runs (no simulation) | table on screen |
| `python analyse_distances.py` | ever / held within 5 mm-10 cm, by target distance and height, final model on test 150 | `results/analysis/final_distances.csv`, `final_bands.csv` |
| `python train.py ...` / `python dagger.py ...` | training (section 7) | `checkpoints/<model>.pt`, its loss curve and offline test plots (`*_loss.png`, `*_test_per_episode.png`, `*_test_per_output.png`), `results/<model>.csv`; DAgger also `dagger/<name>_rK/` (labelled runs) |

`compare.py --test-set` rewrites `results/expert.csv` with the expert of the folder it ran on; restore it with
`git checkout -- results/expert.csv`. The other files these commands write (`results/<model>.csv`, `results/analysis/`)
and the checkpoints copied back from `checkpoints/seeds/` and `checkpoints/chain_starts/` (sections 7 and 8.1) show up
as new files in `git status`: they are outputs, don't commit them.


---

## 7. Retrain the models (optional)

The shipped checkpoints are all that is needed to replicate the evaluation. Retraining runs a chain of steps, each
starting from the previous checkpoint. **Before retraining, copy the starting checkpoints back next to the others**
(the scripts look for every model in `checkpoints/`):

```bash
cp checkpoints/chain_starts/*.pt checkpoints/       # Windows: copy checkpoints\chain_starts\*.pt checkpoints\
```

**Final model, last step only** (recommended: about an hour on 10 cores). It needs `data_concurrent_expert`, the DART and
low-target folders of section 5 and the DAgger folders of the earlier rounds (`dagger/split_conc_dartdagger_bb_r1-r3`,
`..._plusCLD_r1-r3`, made by the full chain below):

```bash
python dagger.py --start split_conc_dartdagger_bb_plusCLD_r3_s0 --name split_conc_drive --rounds 1 --val-set \
  --window 25 --posture-free --low-extra 100 --data-dir data_concurrent_expert --relabel-from-final --label-drive \
  --extra-data dart/b_a60 dart/b_a60b dagger/split_conc_dartdagger_bb_r1 dagger/split_conc_dartdagger_bb_r2 \
  dagger/split_conc_dartdagger_bb_r3 data_low_b dagger/split_conc_dartdagger_bb_plusCLD_r1 \
  dagger/split_conc_dartdagger_bb_plusCLD_r2 dagger/split_conc_dartdagger_bb_plusCLD_r3
# seeds 1 and 2: the same command with --seed 1 / --seed 2 (reuses the round's labelled runs)
```

**Final model, full chain** (several hours; `PY` = your Python command):

```bash
PY=python bash ../concurrent_arm/concurrent_chain1.sh   # DART model split_conc_dart_bb_s0, then 5 DAgger rounds
PY=python bash ../concurrent_arm/concurrent_chain2.sh   # 3 rounds with whole-run, posture-free labels on round 3
PY=python bash ../concurrent_arm/concurrent_chain3.sh   # seeds 1-2 of that model
# then the last step above
```

**Sequential comparison** (needs `data_sequential_expert` and its folders from `seqpre_collect.sh`):

```bash
PY=python bash ../concurrent_arm/seqpre_chain1.sh    # DART model split_seqpre_dart_bb_s0 (only its first two commands are needed)
PY=python bash ../concurrent_arm/seqpre_chain5.sh    # post-parking DAgger rounds; round 2 kept
PY=python bash ../concurrent_arm/seqpre_chain8.sh    # 3 whole-run rounds on it: split_seqpre_ppdart_cld_r1-r3
PY=python bash ../concurrent_arm/seqpre_chain9.sh    # seeds 1-2 of r3, validation 150 and test 150
```

Every chain step skips what is already done, so an interrupted chain can be restarted.

**Camera detectors** (images rendered in the simulator from the training split's arm poses only; no validation or
test episode is used):

```bash
python ../concurrent_arm/cnn_data.py --place mast --arm-data data_concurrent_expert --arm-split train --arm-ids 0-299 --out data_camera_images_mast_train
python ../concurrent_arm/cnn_detector.py --data data_camera_images_mast_train --out checkpoints/cnn_detector_mast_train.pt --patience 40
python ../concurrent_arm/cnn_data.py --place hand --arm-data data_concurrent_expert --arm-split train --out data_camera_images_hand
python ../concurrent_arm/cnn_detector.py --data data_camera_images_hand --out checkpoints/cnn_detector_hand.pt --patience 40
python ../concurrent_arm/cnn_eval.py --model checkpoints/cnn_detector_mast_train.pt --data data_camera_images_mast_train   # held-out images
```

12,000 images take a few minutes; training takes about 40 minutes on a GPU and several hours on a CPU.


---

## 8. Replicate the report's numbers

### 8.1 The metrics

Every number is counted from the saved episode files (`ep_*.npz`, 1 ms per frame) by the scripts named below, from
the distance between the end effector and the target in each frame.

The main measure of success is getting the hand **within 1 cm** of the target and keeping it there. The metrics, in
the order the report uses them:

| metric | definition |
|---|---|
| **1. ever within 1 cm** | the error is below 1 cm in at least one frame |
| **2. held within 1 cm** | below 1 cm for **0.5 s without a break**: the hand stays on the target instead of only swinging through it (the same rule at 5 cm gives "held within 5 cm") |
| **3. within 5 mm** | the hand gets within 5 mm after the base's "arrived" flag has latched. This is the task script's stop rule (the run ends 1 s later), a stricter check than the 1 cm measures above; the scripts call it "reached" |
| **4. time to reach** | the simulated time of that 5 mm reach, mean over the runs that get within 5 mm |
| collapsed | after its closest approach, the error goes back above 20 cm |

Runs are capped at 30 s of simulated time. A run that never gets within 5 mm runs to the cap, and can still count as
within / held within 1 cm. Results are **3-seed means** (seeds
0, 1, 2 of the same recipe) unless a table says otherwise. Validation 50 = a fixed 50 of the 150 validation targets
(`bc_data.quick_val_ids`), used for every choice; **test 150 = all test targets, run only for the reported
numbers**. The "expert" rows in the robustness tables are the course's expert (`robotnik_arm_task.py`).

**Before running anything for 3 seeds**, copy seeds 1 and 2 back next to seed 0:

```bash
cp checkpoints/seeds/*.pt checkpoints/       # Windows: copy checkpoints\seeds\*.pt checkpoints\
```

**Reproducibility.** Physics and policy are deterministic on one machine, so a rerun there gives identical episodes.
Across machines (or CPU / GPU builds) floating-point differences grow over a run, so individual episodes can differ
and counts move by about the seed-to-seed spread; the conclusions don't change. Sensor-noise runs are not seeded and
differ slightly on every rerun.

### 8.2 Final model and sequential comparison: validation and test

```bash
for s in 0 1 2; do
  m=split_conc_drive_r1_s$s
  python compare.py --val-full --model $m --data-dir data_concurrent_expert --out-dir compare_out/${m}_val150
  python compare.py --test-set --model $m --data-dir data_concurrent_expert --out-dir compare_out/${m}_test
done
python ../concurrent_arm/count_results.py compare_out/split_conc_drive_r1_s{0,1,2}_val150 --data-dir data_concurrent_expert --set val_full
python ../concurrent_arm/count_results.py compare_out/split_conc_drive_r1_s{0,1,2}_test --data-dir data_concurrent_expert --set test
python analyse_distances.py                  # distances table (test 150; the expert column needs data_course_expert)
git checkout -- results/expert.csv
```

For the sequential model use `m=split_seqpre_ppdart_cld_r3_s$s` and `--data-dir data_sequential_expert`. Each `compare.py`
run of 150 episodes takes 10-20 minutes with the default workers.

What you should get (test 150, 3-seed mean; the report has the full tables):

| of 150 | ever within 1 cm | held within 1 cm | within 5 mm | time to reach |
|---|---|---|---|---|
| final model | 149.7 | 145.0 | 144.3 | 10.6 s |
| sequential comparison | 148.3 | 116.0 | 138.0 | 11.8 s |

### 8.3 Robustness: start pose, target distance, payload

One script runs all three tests for one model (seed 0) or the expert; one summary counts them, model next to expert:

```bash
PY=python bash ../robustness/robustness_runs.sh split_conc_drive_r1_s0       # ~45 min
PY=python bash ../robustness/robustness_runs.sh expert                       # the reference
python ../robustness/robustness_summary.py split_conc_drive_r1_s0
```

- **Start pose:** the base starts at (0, 0, 0), offset (2.0, 1.5, 0), rotated (0, 0, 3.0) or both (-1.5, 2.0, -2.0)
  (x m, y m, yaw rad); 50 targets per pose, all 1-3 m from the start pose. Folders `starts/<model>_<pose>`.
- **Target distance:** default start, targets 0.6-1.0 m (50) and 3.0-5.0 m (200) away, outside the training range of
  1-3 m (`robustness/out_of_range.py`). Folders `oor/<model>_<near|far>`.
- **Payload:** 0.5, 1 or 2 kg added to the hand in the simulator only (`ROBOENV_PAYLOAD`); neither the expert's nor
  the policy's model knows about it. Folders `payload/<model>_<kg>kg`.

### 8.4 Robustness: sensor noise

Gaussian noise on every sensor reading (joint angles and velocities, base position and orientation), σ = 1, 5 and
10 (mm, mrad, mrad/s), via noisy copies of the simulator config (`ROBOENV_CONFIG`); the stop rule and the saved errors still
use the true state. Validation 50, 3 seeds, targets from `data_course_expert`:

```bash
PY=python bash ../robustness/noise_runs.sh split_conc_drive_r1           # model prefix: seeds 0-2, clean + 3 levels
python ../robustness/expert_noise_runs.py compare_out/expert_clean_val    # the expert under the same noise
for n in 1 5 10; do ROBOENV_CONFIG=robotnik_arm_noise$n.json python ../robustness/expert_noise_runs.py compare_out/expert_noise${n}_val; done
python ../robustness/noise_recount.py split_conc_drive_r1
python ../robustness/noise_recount.py expert
python ../concurrent_arm/count_results.py compare_out/split_conc_drive_r1_s0_noise5_val --data-dir data_course_expert   # a second count
```

### 8.5 Camera

The final model driven by each camera option instead of the true target, validation 50 and test 150, 3 seeds:

```bash
for place in mast hand; do
  det=cnn_detector_mast_train.pt; [ $place = hand ] && det=cnn_detector_hand.pt
  for s in 0 1 2; do for set in val test; do
    ROBOENV_CAMERA_PLACE=$place ROBOENV_DETECTOR=cnn ROBOENV_CNN_DETECTOR=$PWD/checkpoints/$det \
      python ../concurrent_arm/camera_policy_eval.py --model split_conc_drive_r1_s$s --set $set \
      --data-dir data_concurrent_expert --out-dir compare_out/split_conc_drive_r1_s${s}_$place
  done; done
  python ../concurrent_arm/count_results.py compare_out/split_conc_drive_r1_s{0,1,2}_$place --data-dir data_concurrent_expert --set val
  python ../concurrent_arm/count_results.py compare_out/split_conc_drive_r1_s{0,1,2}_$place --data-dir data_concurrent_expert --set test
done
```

The episodes run inside `concurrent_arm/`, so give the detector as an absolute path (`$PWD/...`): a path that
doesn't resolve there makes every run fail quietly ("no result"). The camera expert and its analysis:
`python run_batch.py --n 300 --seed 0 --workers 10 --script ../concurrent_arm/camera_expert.py --out-dir data_camera`,
then `python ../concurrent_arm/analyse_camera.py data_camera --ref data_concurrent` (with `data_concurrent` the
concurrent expert on the same 300 targets, early IK on: as `data_concurrent_expert`'s command without `ROBOENV_ARM_EARLY_IK=0`).

