#!/bin/bash
# Start a Trial 10 run on whatever GPU is free now, and queue upgrade jobs on better GPUs.
# When an upgrade job starts, the running job checkpoints and hands over; the new job
# resumes from the same step. Pending upgrades are cancelled when the run finishes.
#
#   bash scripts/submit_trial10.sh learned 1
#   bash scripts/submit_trial10.sh fixed 0
#
# Partition / GPU names: check with  sinfo -p gpu -o "%P %G %l"  and edit below.
MODE=${1:-learned}; HALT=${2:-1}
cd "$(dirname "$0")/.." || exit 1       # submit from this project root, regardless of where you call it
[ -f data/processed/intraday_features.npz ] || { echo "missing data/processed/intraday_features.npz in $(pwd)"; exit 1; }
RUN="${MODE}_halt${HALT}"
PARTITION=${PARTITION:-gpu}
BASES=${BASES:-"v100-sxm2 v100-pcie"}      # minimum tier: start on whichever V100 frees first
UPGRADES=${UPGRADES:-"h200 a100"}           # better GPUs to wait for, best first

for g in $BASES; do
  j=$(sbatch --parsable -p "$PARTITION" --gres=gpu:$g:1 --job-name="isr_t10_${RUN}" \
      --export=ALL,MODE=$MODE,HALT=$HALT scripts/trial10.sbatch) && echo "base job $j ($g) for $RUN"
done
for g in $UPGRADES; do
  j=$(sbatch --parsable -p "$PARTITION" --gres=gpu:$g:1 --job-name="isr_t10_${RUN}" \
      --export=ALL,MODE=$MODE,HALT=$HALT scripts/trial10.sbatch 2>/dev/null) \
    && echo "upgrade job $j (waits for $g)" || echo "skipped $g (not available on $PARTITION)"
done
echo "watch:  squeue --me -n isr_t10_${RUN}   |   tail -f outputs/trial10/slurm_isr_t10_${RUN}_*.out"
