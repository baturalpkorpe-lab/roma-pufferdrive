#!/bin/bash
#SBATCH --job-name=roma_gpu_smoke
#SBATCH --partition=gpu-a100-small
#SBATCH --account=innovation
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=roma_pufferdrive/slurm_logs/roma_smoke_%j.out
#SBATCH --error=roma_pufferdrive/slurm_logs/roma_smoke_%j.err

# Quick GPU smoke test: confirms CUDA path + training loop work on a real GPU.
# ~100k steps, no eval, CSV-only logging.

set -e

PROJECT_DIR=/scratch/$USER/pufferdrive-roma/PufferDrive
cd "$PROJECT_DIR"
mkdir -p roma_pufferdrive/slurm_logs

source .venv/bin/activate
export LD_LIBRARY_PATH="$(python -c 'import torch; print(torch.__path__[0])')/lib:$LD_LIBRARY_PATH"

echo "Node: $(hostname) | start: $(date)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

PYTHONPATH=$PWD/roma_pufferdrive python3 roma_pufferdrive/train_roma.py \
    --role_dim 1 \
    --num_maps 100 \
    --total_steps 100000 \
    --num_agents 64 \
    --device cuda \
    --log_interval 10000 \
    --save_interval 1000000 \
    --save_dir roma_pufferdrive/checkpoints/roma_gpu_smoke

echo "Done: $(date)"
