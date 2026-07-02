#!/bin/bash
#SBATCH --job-name=bench_env_sps
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/bench_env_sps_%j.out
#SBATCH --error=/scratch/e452103/logs/bench_env_sps_%j.err

# Benchmarks raw env stepping on the same node type as training.
# Compare "agent steps/s" across stacks / after rebuilding the C binding.

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh

# ── Stack selection: baseline (default) or main ──────────────────────────────
conda activate roma
export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
CODE_DIR=$HOME/roma_pufferdrive
# main stack instead:
#   conda activate /scratch/e452103/envs/roma_guid
#   export PYTHONPATH=$HOME/roma_main:/scratch/e452103/PufferDrive_guid
#   CODE_DIR=$HOME/roma_main

mkdir -p /scratch/e452103/logs
cd "$CODE_DIR"

python bench_env_sps.py \
    --data_dir /scratch/e452103/PufferDrive/pufferlib/resources/drive/binaries/training \
    --num_agents 3072 \
    --num_maps 10000 \
    --steps 2000
