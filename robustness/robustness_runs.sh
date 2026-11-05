#!/bin/bash
# Start-pose, out-of-range and payload runs for one model (seed 0 in the report) or the expert. One job at a time;
# each test changes one thing from the default (start pose, target distance, extra mass at the hand).
#   - start pose: the base starts at default 0,0,0 / offset 2.0,1.5,0 / rotated 0,0,3.0 / both -1.5,2.0,-2.0
#     (x m, y m, yaw rad); targets from run_batch.py --seed 21 (drawn 1-3 m around the origin). For offset / both,
#     more episodes are run and robustness_summary.py keeps the 50 whose target is 1-3 m from the start pose.
#     Folders starts/<model>_<pose>.
#   - out of range: default start, targets 0.6-1.0 m (50) and 3.0-5.0 m (200), seed 31 (out_of_range.py).
#     Folders oor/<model>_<near|far>.
#   - payload: ROBOENV_PAYLOAD = 0.5 / 1 / 2 kg extra at the end-effector link, unknown to the controllers (PyBullet
#     only), the 50 default start-pose targets. 0 kg = starts/<model>_default. Folders payload/<model>_<kg>kg.
# Count them with robustness/robustness_summary.py <model>.
#
# Usage, from anywhere:
#   PY="pixi run python" bash robustness/robustness_runs.sh split_conc_drive_r1_s0      # final model
#   PY="pixi run python" bash robustness/robustness_runs.sh split_seqpre_ppdart_cld_r3_s0
#   PY="pixi run python" bash robustness/robustness_runs.sh expert                      # the expert, the reference
# Steps already done are skipped (folder with an index.csv). ~45 min per model on a 16-thread PC.
set -euo pipefail
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}
WORKERS=${WORKERS:-16}
M=${1:?usage: robustness_runs.sh <model name, e.g. split_conc_drive_r1_s0, or expert>}
cd "$(dirname "$0")/../robotnik_arm_task"

if [ "$M" = expert ]; then
  SCRIPT=robotnik_arm_task.py              # the expert used for the data (sequential: arm after parking)
  unset ROBOENV_POLICY
else
  SCRIPT=robotnik_arm_policy_task.py
  export ROBOENV_POLICY="$PWD/checkpoints/$M.pt"
  [ -f "$ROBOENV_POLICY" ] || { echo "no checkpoint $ROBOENV_POLICY"; exit 1; }
fi

for P in "default 0,0,0 50" "offset 2.0,1.5,0 155" "rotated 0,0,3.0 50" "both -1.5,2.0,-2.0 131"; do
  set -- $P
  [ -f "starts/${M}_$1/index.csv" ] || \
    $PY run_batch.py --script $SCRIPT --seed 21 --n $3 --start-pose="$2" --workers $WORKERS --out-dir "starts/${M}_$1" | tail -1
  echo "start pose $1 done $(date +%H:%M)"
done
for B in "near 50" "far 200"; do
  set -- $B
  [ -f "oor/${M}_$1/index.csv" ] || \
    $PY ../robustness/out_of_range.py --band $1 --n $2 --script $SCRIPT --workers $WORKERS --out-dir "oor/${M}_$1" | tail -1
  echo "out of range $1 done $(date +%H:%M)"
done
for kg in 0.5 1 2; do
  [ -f "payload/${M}_${kg}kg/index.csv" ] || \
    ROBOENV_PAYLOAD=$kg $PY run_batch.py --script $SCRIPT --seed 21 --n 50 --workers $WORKERS --out-dir "payload/${M}_${kg}kg" | tail -1
  echo "payload $kg kg done $(date +%H:%M)"
done
echo "robustness runs done $(date)"
