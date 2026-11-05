#!/bin/bash
# Sequential pre-shaping, part 8 (PLAN.md step 28, 5 Nov): the final recipe's last step with matching labels on the
# best post-park model (ppdart r2, picked on val 150 too): 3 rounds, --label-postpark, 25 s window, posture-free,
# 100 extra low runs per round, data_low_s; then all 150 val episodes for each round (seed 0), for the pick.
set -euo pipefail
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}
$PY dagger.py --start split_seqpre_ppdart_r2_s0 \
  --extra-data dart/s_a60 dart/s_a60b dagger/split_seqpre_ppdart_r1 dagger/split_seqpre_ppdart_r2 data_low_s \
  --window 25 --posture-free --low-extra 100 --rounds 3 --name split_seqpre_ppdart_cld \
  --data-dir data_seqpre --label-postpark --workers 11
for r in 1 2 3; do
  m=split_seqpre_ppdart_cld_r${r}_s0
  [ -d compare_out/mac_${m}/val_set_full ] || $PY compare.py --val-full --model $m --data-dir data_seqpre --out-dir compare_out/mac_$m
done
echo "chain 8 done $(date)"
