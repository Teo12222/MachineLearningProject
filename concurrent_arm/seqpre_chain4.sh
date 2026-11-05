#!/bin/bash
# Sequential pre-shaping, part 4 (PLAN.md step 28, 5 Nov): seeds 1-2 of the plain model (plusCLD r3) and of the
# post-park model (postpark r2), each with its round's exact training options and data; then all 150 val episodes
# (compare.py --val-full) and test 150 for seeds 0-2 of both and of drive-phase r1 (Matteo's final model), on this
# machine. --test-set rewrites results/expert.csv and the drive-phase models' results files (Matteo's, PC numbers):
# restored from git after each run. Run from robotnik_arm_task/.
set -euo pipefail
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}
OPT="--split --gravity-residual --hidden 256 --layers 3 --dropout 0.0 --arm-hidden 256 --arm-layers 3 --arm-dropout 0.0
     --base-activation relu --arm-activation relu --base-output-act bounded --arm-output-act linear
     --data-dir data_sequential_expert --lr 0.003 --batch 2048 --weight-decay 0"
P=dagger/split_seqpre_dartdagger_bb
PLAIN_DATA="dart/s_a60 dart/s_a60b ${P}_r1 data_low_s ${P}_plusCLD_r1 ${P}_plusCLD_r2 ${P}_plusCLD_r3"
POST_DATA="$PLAIN_DATA dagger/split_seqpre_postpark_r1 dagger/split_seqpre_postpark_r2"
for s in 1 2; do
  m=split_seqpre_dartdagger_bb_plusCLD_r3_s$s
  [ -f checkpoints/$m.pt ] || $PY train.py --model $m $OPT --seed $s --base-from split_seqpre_dartdagger_bb_r1_s0 --dagger $PLAIN_DATA
  m=split_seqpre_postpark_r2_s$s
  [ -f checkpoints/$m.pt ] || $PY train.py --model $m $OPT --seed $s --base-from split_seqpre_dartdagger_bb_plusCLD_r3_s0 --dagger $POST_DATA
done
for s in 0 1 2; do
  for m in split_seqpre_dartdagger_bb_plusCLD_r3_s$s split_seqpre_postpark_r2_s$s split_conc_drive_r1_s$s; do
    dd=data_sequential_expert; [[ $m == split_conc_* ]] && dd=data_concurrent_expert
    [ -d compare_out/mac_${m}/val_set_full ] || $PY compare.py --val-full --model $m --data-dir $dd --out-dir compare_out/mac_$m
    if [ ! -d compare_out/mac_${m}/test_set ]; then
      $PY compare.py --test-set --model $m --data-dir $dd --out-dir compare_out/mac_$m
      git checkout -- results/expert.csv
      [[ $m == split_conc_* ]] && git checkout -- results/$m.csv
    fi
  done
done
echo "chain 4 done $(date)"
