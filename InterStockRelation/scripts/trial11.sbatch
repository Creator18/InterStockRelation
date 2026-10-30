#!/bin/bash
# One Trial 11 job (same-bar fill-in attention maps). Submitted by scripts/submit_trial11.sh.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=outputs/trial11/slurm_%x_%j.out

SEED=${SEED:-0}
cd "$SLURM_SUBMIT_DIR" || exit 1          # always run inside this project folder only
mkdir -p outputs/trial11
source /scratch/$USER/isr-t10/bin/activate   # project-only venv on scratch
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/trial11.py --seed "$SEED" &
PID=$!
trap "kill -USR1 $PID" USR1
wait $PID; wait $PID
