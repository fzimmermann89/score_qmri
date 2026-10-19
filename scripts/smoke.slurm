#!/bin/bash
#SBATCH --job-name=score-recovery-smoke
#SBATCH --partition=equipment_typeG
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/recovery-smoke-%j.out
#SBATCH --error=logs/recovery-smoke-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the recovery checkout}"
source .venv/bin/activate
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
srun python -u scripts/reconstruct_smoke.py \
    --output "results/smoke-${SLURM_JOB_ID}" "$@"
