#!/bin/bash
# Sensor-noise robustness runs (PLAN.md step 12): one model, 3 seeds, val 50, clean and sensor noise sigma 1 / 5 / 10
# (mm on the base position, mrad on the joints; configs/robotnik_arm_noise<sigma>.json via ROBOENV_CONFIG).
# Each run gets its own folder: compare_out/<model>_s<seed>_<level>_val, level = clean / noise1 / noise5 / noise10.
# Count them with robustness/noise_recount.py (and concurrent_arm/route_b_eval.py --data-dir data_course_expert).
#
# Usage, from anywhere:
#   PY="pixi run python" bash robustness/noise_runs.sh split_conc_drive_r1             # final model, seeds 0 1 2
#   PY="pixi run python" bash robustness/noise_runs.sh split_seqpre_ppdart_cld_r3       # sequential pre-shaping model
#   SEEDS="0" WORKERS=4 bash robustness/noise_runs.sh <model prefix>
# The noise isn't seeded, so a rerun gives slightly different numbers. ~1 min per run on an M-series Mac (8 workers).
set -euo pipefail
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}
SEEDS=${SEEDS:-"0 1 2"}
WORKERS=${WORKERS:-8}
MODEL=${1:?usage: noise_runs.sh <model prefix, e.g. split_conc_drive_r1>}
cd "$(dirname "$0")/../robotnik_arm_task"
mkdir -p compare_out

# The noise configs are git-ignored copies of configs/robotnik_arm.json: make them once
for n in 1 5 10; do
  [ -f configs/robotnik_arm_noise$n.json ] || { $PY make_noise_configs.py; break; }
done

for s in $SEEDS; do
  m=${MODEL}_s$s
  # Clean run (the default config), so both sets of runs come from the same machine and code
  [ -d compare_out/${m}_clean_val/val_set ] || \
    $PY compare.py --val-set --model $m --workers $WORKERS --out-dir compare_out/${m}_clean_val \
      > compare_out/${m}_clean_val.log 2>&1
  echo "$m clean done $(date +%H:%M)"
  for n in 1 5 10; do
    [ -d compare_out/${m}_noise${n}_val/val_set ] || \
      ROBOENV_CONFIG=robotnik_arm_noise$n.json $PY compare.py --val-set --model $m --workers $WORKERS \
        --out-dir compare_out/${m}_noise${n}_val > compare_out/${m}_noise${n}_val.log 2>&1
    echo "$m noise $n done $(date +%H:%M)"
  done
done
echo "noise runs done $(date)"
