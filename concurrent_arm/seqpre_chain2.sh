#!/bin/bash
# Sequential pre-shaping model, part 2 (PLAN.md step 28, 5 Nov): the concurrent model's chain 2 (the final model's CLD step) on
# round R of chain 1, chosen by the rule (held within 1 cm, then ever, then reached; seed 0, val 50):
# 3 rounds, posture-free labels over the whole run, 100 extra low runs per round, data_low_s in the arm's data.
# Usage, from robotnik_arm_task/: R=3 PY=python bash ../concurrent_arm/seqpre_chain2.sh
set -euo pipefail
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}
: "${R:?set R to the chosen chain-1 round}"
EXTRA="dart/s_a60 dart/s_a60b"
for k in $(seq 1 "$R"); do EXTRA="$EXTRA dagger/split_seqpre_dartdagger_bb_r$k"; done
$PY dagger.py --start split_seqpre_dartdagger_bb_r${R}_s0 --extra-data $EXTRA data_low_s \
  --window 25 --posture-free --low-extra 100 --rounds 3 --name split_seqpre_dartdagger_bb_plusCLD --val-set \
  --data-dir data_sequential_expert --relabel-from-final --workers 11
echo "chain 2 done $(date)"
