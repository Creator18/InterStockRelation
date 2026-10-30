#!/bin/bash
# One Trial 10 job. Submitted by scripts/submit_trial10.sh or scripts/submit_trial10_windows.sh.
# Runs on the GPU partition (never the login node). Requeues itself on time limit.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=outputs/trial10/slurm_%x_%j.out

MODE=${MODE:-learned}; HALT=${HALT:-1}; WIN=${WIN:-12}; TARGET=${TARGET:-level}; SPAN=${SPAN:-0.02}
TAG=${TAG:-${MODE}_halt${HALT}}
cd "$SLURM_SUBMIT_DIR" || exit 1          # always run inside this project folder only
mkdir -p outputs/trial10
export ISR_OUT=${ISR_OUT:-$SLURM_SUBMIT_DIR/outputs/trial10}; mkdir -p "$ISR_OUT"
source /scratch/$USER/isr-t10/bin/activate   # project-only venv on scratch
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# run python in the background so the USR1 time-limit signal reaches it via trap
python scripts/trial10.py --mode "$MODE" --halting "$HALT" --window "$WIN" --target "$TARGET" --span_cost "$SPAN" --tag "$TAG" &
PID=$!
trap "kill -USR1 $PID" USR1
wait $PID; wait $PID
