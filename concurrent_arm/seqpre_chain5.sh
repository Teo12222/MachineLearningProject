#!/bin/bash
# Sequential pre-shaping, part 5 (PLAN.md step 28, 5 Nov): post-park labels from the start, separating the recipe
# from the design. Chain 1 with --label-postpark in place of --relabel-from-final: 5 DAgger rounds on the DART
# model split_seqpre_dart_bb_s0, 5 s window after the latch (plus the frames before it, flag 0), DART data extra.
# Run from robotnik_arm_task/.
set -euo pipefail
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}
$PY dagger.py --start split_seqpre_dart_bb_s0 --extra-data dart/s_a60 dart/s_a60b --name split_seqpre_ppdart \
  --rounds 5 --val-set --data-dir data_seqpre --label-postpark --workers 11
echo "chain 5 done $(date)"
