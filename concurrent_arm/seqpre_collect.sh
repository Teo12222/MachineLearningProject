#!/bin/bash
# Sequential pre-shaping expert data (PLAN.md step 28, 5 Nov): Route B's targets and seeds, with
# ROBOENV_ARM_SEQUENTIAL=1 (arm still while the base drives, joint path to q_final after the latch, then the
# Cartesian correction). Run from robotnik_arm_task/.
set -euo pipefail
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE ROBOENV_ARM_SEQUENTIAL=1 ROBOENV_ARM_EARLY_IK=0
PY=${PY:-python}
S=../concurrent_arm/concurrent_expert.py
$PY run_batch.py --n 1000 --seed 0 --workers 10 --script $S --out-dir data_seqpre
$PY run_batch.py --n 150 --seed 2 --noise 60 --workers 10 --script $S --out-dir dart/s_a60
$PY run_batch.py --n 150 --seed 3 --noise 60 --workers 10 --script $S --out-dir dart/s_a60b
$PY run_batch.py --n 60 --seed 11 --height-range 0.2,0.5 --workers 10 --script $S --out-dir data_low_s
echo "collect done $(date)"
