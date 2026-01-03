#!/bin/bash
#SBATCH --job-name=score-qmri-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=1-10%1

# Ensure logs directory exists
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Use the SLURM_ARRAY_JOB_ID as the base for the job name
# This ensures all tasks in the array share the same job_name and thus the same WandB run ID and checkpoint directory.
JOB_NAME="diffusion-training-${SLURM_ARRAY_JOB_ID}"

echo "Starting training for Job Name: ${JOB_NAME}"
echo "Slurm Task ID: ${SLURM_ARRAY_TASK_ID}"

# Run training
# Note: config.yaml should be present in the current directory or provided via --config
srun score-qmri-train fit \
    --job_name "${JOB_NAME}" \
    --config config.yaml \
    --trainer.devices 1 \
    --trainer.accelerator gpu
