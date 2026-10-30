#!/bin/bash
# Trial 11: start on a V100 (minimum), queue upgrades on A100/H200; handoff via scripts/gpu_lease.py.
#   bash scripts/submit_trial11.sh 0      # seed 0
#   bash scripts/submit_trial11.sh 1      # seed 1 (stability check)
SEED=${1:-0}
cd "$(dirname "$0")/.." || exit 1
[ -f data/processed/intraday_features.npz ] || { echo "missing data/processed/intraday_features.npz in $(pwd)"; exit 1; }
RUN="fillin_seed${SEED}"
PARTITION=${PARTITION:-gpu}
BASES=${BASES:-"v100-sxm2 v100-pcie"}
UPGRADES=${UPGRADES:-"h200 a100"}
for g in $BASES $UPGRADES; do
  j=$(sbatch --parsable -p "$PARTITION" --gres=gpu:$g:1 --job-name="isr_t11_${RUN}" \
      --export=ALL,SEED=$SEED scripts/trial11.sbatch) && echo "job $j ($g) for $RUN"
done
