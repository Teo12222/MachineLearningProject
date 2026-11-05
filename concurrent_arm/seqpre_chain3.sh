#!/bin/bash
# Sequential pre-shaping, part 3 (PLAN.md step 28, 5 Nov): post-park labels (dagger.py --label-postpark) on the plain
# model (round C of chain 2, seed 0): up to 3 rounds, CLD's settings, all of the plain model's data as extra data.
# Usage, from robotnik_arm_task/: R=3 C=2 PY=python bash ../concurrent_arm/seqpre_chain3.sh
set -euo pipefail
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}
: "${R:?chain-1 round}" "${C:?chain-2 round}"
EXTRA="dart/s_a60 dart/s_a60b"
for k in $(seq 1 "$R"); do EXTRA="$EXTRA dagger/split_seqpre_dartdagger_bb_r$k"; done
EXTRA="$EXTRA data_low_s"
for k in $(seq 1 "$C"); do EXTRA="$EXTRA dagger/split_seqpre_dartdagger_bb_plusCLD_r$k"; done
$PY dagger.py --start split_seqpre_dartdagger_bb_plusCLD_r${C}_s0 --extra-data $EXTRA \
  --window 25 --posture-free --low-extra 100 --rounds 3 --name split_seqpre_postpark --val-set \
  --data-dir data_sequential_expert --label-postpark --workers 11
echo "chain 3 done $(date)"
