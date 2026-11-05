#!/bin/bash
# Route B, part 3 (PLAN.md step 28): seeds 1 and 2 of the chosen round (plusCLD r3: most held within 1 cm on val 50),
# same data and settings as dagger.py's round 3 (base copied from the start model), then val 50 each.
set -euo pipefail
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}
DATA="dart/b_a60 dart/b_a60b dagger/split_conc_dartdagger_bb_r1 dagger/split_conc_dartdagger_bb_r2
      dagger/split_conc_dartdagger_bb_r3 data_low_b dagger/split_conc_dartdagger_bb_plusCLD_r1
      dagger/split_conc_dartdagger_bb_plusCLD_r2 dagger/split_conc_dartdagger_bb_plusCLD_r3"
for s in 1 2; do
  m=split_conc_dartdagger_bb_plusCLD_r3_s$s
  [ -f checkpoints/$m.pt ] || $PY train.py --model $m --split --gravity-residual --hidden 256 --layers 3 --dropout 0.0 \
    --arm-hidden 256 --arm-layers 3 --arm-dropout 0.0 --base-activation relu --arm-activation relu \
    --base-output-act bounded --arm-output-act linear --seed $s --base-from split_conc_dartdagger_bb_r3_s0 \
    --data-dir data_route_b --lr 0.003 --batch 2048 --weight-decay 0 --dagger $DATA
  [ -d compare_out/$m/val_set ] || $PY compare.py --val-set --model $m --data-dir data_route_b --out-dir compare_out/$m
done
echo "chain 3 done $(date)"
