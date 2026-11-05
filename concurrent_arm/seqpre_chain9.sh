#!/bin/bash
# Sequential pre-shaping, part 9 (PLAN.md step 28, 5 Nov): seeds 1-2 of chain 8's best round (ppdart_cld r3, held
# 1 cm on val 150, seed 0) with its round's exact options and data, then val 150 for seeds 1-2 and test 150 for
# seeds 0-2, Mac. --test-set rewrites results/expert.csv: restored from git after each run.
set -euo pipefail
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}
OPT="--split --gravity-residual --hidden 256 --layers 3 --dropout 0.0 --arm-hidden 256 --arm-layers 3 --arm-dropout 0.0
     --base-activation relu --arm-activation relu --base-output-act bounded --arm-output-act linear
     --base-from split_seqpre_ppdart_r2_s0 --data-dir data_sequential_expert --lr 0.003 --batch 2048 --weight-decay 0"
DATA="dart/s_a60 dart/s_a60b dagger/split_seqpre_ppdart_r1 dagger/split_seqpre_ppdart_r2 data_low_s
      dagger/split_seqpre_ppdart_cld_r1 dagger/split_seqpre_ppdart_cld_r2 dagger/split_seqpre_ppdart_cld_r3"
for s in 1 2; do
  m=split_seqpre_ppdart_cld_r3_s$s
  [ -f checkpoints/$m.pt ] || $PY train.py --model $m $OPT --seed $s --dagger $DATA
done
for s in 0 1 2; do
  m=split_seqpre_ppdart_cld_r3_s$s
  [ -d compare_out/mac_${m}/val_set_full ] || $PY compare.py --val-full --model $m --data-dir data_sequential_expert --out-dir compare_out/mac_$m
  if [ ! -d compare_out/mac_${m}/test_set ]; then
    $PY compare.py --test-set --model $m --data-dir data_sequential_expert --out-dir compare_out/mac_$m
    git checkout -- results/expert.csv
  fi
done
echo "chain 9 done $(date)"
