#!/bin/bash
#SBATCH --job-name=score
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=4-00:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:H100:4
#SBATCH --partition=equipment_typeG
#SBATCH --cpus-per-task=16
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=1-2%1

set -euo pipefail

JOB_NAME="score-training"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

echo "=== Job Setup ==="
echo "Host: $(hostname -s)"
echo "Script dir: ${SCRIPT_DIR}"

mkdir -p logs

# Triton cache in shared memory for speed
export TRITON_CACHE_DIR="/dev/shm/${USER}/${JOB_NAME}-triton"
mkdir -p "${TRITON_CACHE_DIR}"

# Sync datasets from storage to local (skips unchanged files)
echo "=== Dataset Sync ==="
rsync -av --progress "${SCRIPT_DIR}/storage/" "${SCRIPT_DIR}/datasets/"

# Python interpreter
if [[ -z "${PYTHON:-}" && -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python"
fi
if [[ -z "${PYTHON:-}" ]]; then
    echo "Error: PYTHON not set and .venv/bin/python not found"
    exit 1
fi
echo "Python: ${PYTHON}"

# Proxy (proxychains for outbound connections)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PROXYCHAINS_QUIET_MODE=1
export PROXYCHAINS_CONF_FILE="/scratch/${USER}/proxychains/proxychains.conf"
export LD_PRELOAD="/scratch/${USER}/proxychains/libproxychains.so.4"

# PyTorch / NCCL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_X_REQUIRE_LEGACY_SERVICE=true  # Rust core incompatible with proxy
export NCCL_SOCKET_IFNAME=en

# Run
RUN_NAME="${JOB_NAME}-${SLURM_ARRAY_JOB_ID}"
echo "=== Starting Training ==="
echo "Run: ${RUN_NAME}, Array Task: ${SLURM_ARRAY_TASK_ID}"

srun --unbuffered "${PYTHON}" -u -m score_qmri.cli fit \
    --trainer.num_nodes="${SLURM_JOB_NUM_NODES}" \
    --job_name "${RUN_NAME}" \
    --config config.yaml "$@"
