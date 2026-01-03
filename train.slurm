#!/bin/bash
#SBATCH --job-name=score-qmri-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=h100:4
#SBATCH --time=48:00:00
#SBATCH --array=1-10%1

mkdir -p logs

source .venv/bin/activate

# This ensures all tasks in the array share the same job_name and thus the same WandB run ID and checkpoint directory.
JOB_NAME="diffusion-training-${SLURM_ARRAY_JOB_ID}"

echo "Starting training for Job Name: ${JOB_NAME}"
echo "Slurm Task ID: ${SLURM_ARRAY_TASK_ID}"

srun score-qmri-train fit \
    --job_name "${JOB_NAME}" \
    --config config.yaml \
    --trainer.devices 1 \
    --trainer.accelerator gpu
