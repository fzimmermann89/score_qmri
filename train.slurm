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

# Environment variables for stability
export WANDB_X_REQUIRE_LEGACY_SERVICE=true  # Disable Rust core for W&B
export NCCL_IB_DISABLE=1                    # Disable InfiniBand
export NCCL_P2P_DISABLE=1                   # Disable P2P

# Remove debug flags if present (implicit cleanup)
unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG

echo "Starting training for Job Name: ${JOB_NAME}"
echo "Slurm Task ID: ${SLURM_ARRAY_TASK_ID}"

srun score-qmri-train fit \
    --job_name "${JOB_NAME}" \
    --config config.yaml \
    --trainer.devices 4 \
    --trainer.accelerator gpu
