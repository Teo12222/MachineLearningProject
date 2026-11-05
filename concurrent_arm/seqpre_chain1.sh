#!/bin/bash
# Sequential pre-shaping model, part 1 (PLAN.md step 28, 5 Nov): Route B's recipe (route_b_chain1.sh) on the
# sequential pre-shaping expert's data (seqpre_collect.sh): the DART model behind the bounded base, then 5 DAgger
# rounds labelled 5 s after the latch, reach relabelled from q_final (the "plain" recipe, as Route B's).
# Run from robotnik_arm_task/; PY = the environment's python (default: python on the PATH).
set -euo pipefail
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}
DART="dart/s_a60 dart/s_a60b"
m=split_seqpre_dart_bb_s0
[ -f checkpoints/$m.pt ] || $PY train.py --model $m --split --gravity-residual --base-from split_dart_base_bound_s2 \
  --base-output-act bounded --dagger $DART --data-dir data_seqpre --seed 0
[ -d compare_out/${m}_val/val_set ] || $PY compare.py --val-set --model $m --data-dir data_seqpre --out-dir compare_out/${m}_val
$PY dagger.py --start $m --extra-data $DART --name split_seqpre_dartdagger_bb --rounds 5 --val-set \
  --data-dir data_seqpre --relabel-from-final --workers 11
echo "chain 1 done $(date)"
