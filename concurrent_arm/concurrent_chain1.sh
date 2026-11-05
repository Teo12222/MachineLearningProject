#!/bin/bash
# Concurrent model, part 1 (PLAN.md step 28): the DART model behind the bounded base, then
# 5 DAgger rounds labelled 5 s after the latch. Mirrors the final model's recipe on the concurrent expert's data
# (data_concurrent_expert, early IK off). Run from robotnik_arm_task/; logs in compare_out/concurrent/.
set -euo pipefail
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}
DART="dart/b_a60 dart/b_a60b"


# 2. DART model: bounded base copied from the final model's start, arm trained on the concurrent expert + its DART data
m=split_conc_dart_bb_s0
[ -f checkpoints/$m.pt ] || $PY train.py --model $m --split --gravity-residual --base-from split_dart_base_bound_s2 \
  --base-output-act bounded --dagger $DART --data-dir data_concurrent_expert --seed 0
[ -d compare_out/${m}_val/val_set ] || $PY compare.py --val-set --model $m --data-dir data_concurrent_expert --out-dir compare_out/${m}_val

# 3. 5 DAgger rounds behind it (5 s window, labels as the final model's rounds, reach relabelled from q_final)
$PY dagger.py --start $m --extra-data $DART --name split_conc_dartdagger_bb --rounds 5 --val-set \
  --data-dir data_concurrent_expert --relabel-from-final --workers 11
echo "chain 1 done $(date)"
