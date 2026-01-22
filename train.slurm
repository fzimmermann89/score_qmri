#!/bin/bash
#SBATCH --job-name=score
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:L40S:4
#SBATCH --partition=equipment_typeG
#SBATCH --cpus-per-task=8
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=1-2%1

JOB_NAME="score-training"

HOST="$(hostname -s)"
#export TRITON_CACHE_DIR="/scratch2/fast/zimmer08/triton/${HOST}/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR=/dev/shm/${USER}/${JOB_NAME}-triton
mkdir -p "${TRITON_CACHE_DIR}"

mkdir -p logs

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ -z "${PYTHON}" && -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python"
fi
if [[ -z "${PYTHON}" ]]; then
    echo "Error: PYTHON not set and ${SCRIPT_DIR}/.venv/bin/python not found"
    exit 1
fi

# Proxy setup
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
export PROXYCHAINS_QUIET_MODE=1
export PROXYCHAINS_CONF_FILE="/scratch/${USER}/proxychains/proxychains.conf"
export LD_PRELOAD="/scratch/${USER}/proxychains/libproxychains.so.4"

# PyTorch/CUDA config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_X_REQUIRE_LEGACY_SERVICE=true  # Rust core does not work with proxy
export NCCL_IB_DISABLE=1                    # Disable InfiniBand (unstable)
export NCCL_P2P_DISABLE=1                   # Disable P2P (unstable)

RUN_NAME="${JOB_NAME}-${SLURM_ARRAY_JOB_ID}"

echo "Starting training for Job Name: ${RUN_NAME}"
echo "Slurm Task ID: ${SLURM_ARRAY_TASK_ID}"

srun --unbuffered "${PYTHON}" -u -m score_qmri.cli fit \
    --job_name "${RUN_NAME}" \
    --config config.yaml
