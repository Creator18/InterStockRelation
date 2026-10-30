#!/bin/bash
# Window-oracle sweep (delta target): one model per fixed window + learned windows without span cost.
# Explorer's gpu QOS allows 8 queued / 4 running jobs per user (shared with other projects),
# so this submits ONE job per run on any GPU >= V100 (T4 node excluded); Slurm takes whichever
# frees first. Runs still checkpoint/requeue via the lease in scripts/trial10.py.
# Outputs (checkpoints, predictions) go to scratch to keep home small.
#   bash scripts/submit_trial10_windows.sh
cd "$(dirname "$0")/.." || exit 1
[ -f data/processed/intraday_features.npz ] || { echo "missing data/processed/intraday_features.npz in $(pwd)"; exit 1; }
PARTITION=${PARTITION:-gpu}
EXCLUDE=${EXCLUDE:-d1025}                       # T4 node(s): below the V100 minimum
OUTDIR=/scratch/$USER/isr-out/trial10_windows; mkdir -p "$OUTDIR"
submit() {   # TAG MODE HALT WIN SPAN
  j=$(sbatch --parsable -p "$PARTITION" --gres=gpu:1 --exclude="$EXCLUDE" --job-name="isr_t10_$1" \
      --export=ALL,TAG=$1,MODE=$2,HALT=$3,WIN=$4,SPAN=$5,TARGET=delta,ISR_OUT=$OUTDIR scripts/trial10.sbatch) \
    && echo "job $j $1"
}
for w in 3 6 12 24 75 150; do submit "fixedw${w}_delta" fixedw 0 $w 0; done
submit learned_halt1_delta_nospan learned 1 12 0
echo "outputs -> $OUTDIR"
