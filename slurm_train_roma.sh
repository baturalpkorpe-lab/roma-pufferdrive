#!/bin/bash
#SBATCH --job-name=roma_train
#SBATCH --partition=gpu-a100
#SBATCH --account=innovation
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=roma_pufferdrive/slurm_logs/roma_%x_%j.out
#SBATCH --error=roma_pufferdrive/slurm_logs/roma_%x_%j.err

# =============================================================================
# ROMA on PufferDrive — full GPU training (DelftBlue)
#
# Usage:
#   sbatch --job-name=roma_dim1 roma_pufferdrive/slurm_train_roma.sh 1
#   sbatch --job-name=roma_dim8 roma_pufferdrive/slurm_train_roma.sh 8
#
# Arg 1: ROLE_DIM (default 1). 1 = original ROMA, 8 = proposed multidim.
# =============================================================================

set -e

ROLE_DIM=${1:-1}
TOTAL_STEPS=${2:-6000000000}

PROJECT_DIR=/scratch/$USER/pufferdrive-roma/PufferDrive
cd "$PROJECT_DIR"

mkdir -p roma_pufferdrive/slurm_logs
mkdir -p "roma_pufferdrive/checkpoints/roma_dim${ROLE_DIM}"

# --- Environment (must match interactive setup) ---
source .venv/bin/activate
export LD_LIBRARY_PATH="$(python -c 'import torch; print(torch.__path__[0])')/lib:$LD_LIBRARY_PATH"

# wandb: GPU nodes have internet and credentials are in ~/.netrc, so online
# logging works. Project is set below; entity defaults to the logged-in account.
WANDB_PROJECT=roma-pufferdrive

echo "=========================================="
echo "ROMA training | role_dim=${ROLE_DIM} | total_steps=${TOTAL_STEPS}"
echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Start: $(date)"
echo "=========================================="
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

PYTHONPATH=$PWD/roma_pufferdrive python3 roma_pufferdrive/train_roma.py \
    --role_dim "${ROLE_DIM}" \
    --num_maps 10000 \
    --total_steps "${TOTAL_STEPS}" \
    --num_agents 64 \
    --device cuda \
    --save_interval 500000000 \
    --run_eval \
    --wandb_project "${WANDB_PROJECT}" \
    --save_dir "roma_pufferdrive/checkpoints/roma_dim${ROLE_DIM}"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
