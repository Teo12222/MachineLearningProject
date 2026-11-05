#!/bin/bash
# Route B, part 2 (PLAN.md step 28): the final model's last step (CLD) on top of the chosen DAgger round
# (r3: most held within 1 cm, then ever, on val 50): 3 rounds with posture-free labels over the whole run, 100 extra
# runs per round on low training targets and the low-target expert episodes in the arm's data. Extra data as for the
# final model: DART + the rounds up to the start model. Run from robotnik_arm_task/.
set -euo pipefail
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}
$PY dagger.py --start split_conc_dartdagger_bb_r3_s0 \
  --extra-data dart/b_a60 dart/b_a60b dagger/split_conc_dartdagger_bb_r1 dagger/split_conc_dartdagger_bb_r2 \
               dagger/split_conc_dartdagger_bb_r3 data_low_b \
  --window 25 --posture-free --low-extra 100 --rounds 3 --name split_conc_dartdagger_bb_plusCLD --val-set \
  --data-dir data_concurrent_expert --relabel-from-final --workers 11
echo "chain 2 done $(date)"
