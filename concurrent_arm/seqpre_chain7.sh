#!/bin/bash
# PLAN.md step 28, 5 Nov: CLD r2 (the sequential expert's final model) on val 150 and test 150 on this machine, so
# the three designs (sequential, sequential pre-shaping, concurrent) are compared on one machine. --test-set
# rewrites results/expert.csv and CLD r2's results files (PC numbers): restored from git after each run.
set -euo pipefail
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}
for s in 0 1 2; do
  m=split_dartdagger_bb_plusCLD_r2_s$s
  [ -d compare_out/mac_${m}/val_set_full ] || $PY compare.py --val-full --model $m --out-dir compare_out/mac_$m
  if [ ! -d compare_out/mac_${m}/test_set ]; then
    $PY compare.py --test-set --model $m --out-dir compare_out/mac_$m
    git checkout -- results/expert.csv results/$m.csv
  fi
done
echo "chain 7 done $(date)"
