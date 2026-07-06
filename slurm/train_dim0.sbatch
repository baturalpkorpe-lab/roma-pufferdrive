#!/bin/bash
#SBATCH --job-name=roma_dim0
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/roma_dim0_%j.out
#SBATCH --error=/scratch/e452103/logs/roma_dim0_%j.err

# dim-0 NO-ROLE ablation (branch baseline_role_0_dim) from its OWN clone at
# ~/roma_dim0. Option A (true removal): the role encoder and MI/diversity
# losses are gone; same perception encoders + policy GRU (input 128 instead
# of 128+role_dim). This is the "same architecture without the ROMA role
# machinery" baseline for the paper -- report param counts alongside dim-3/4/8.
#
# One-time setup before first submit:
#   git clone -b baseline_role_0_dim \
#       https://github.com/baturalpkorpe-lab/roma-pufferdrive.git ~/roma_dim0
#
# total_steps=3B: matches dim-3/4/8 for the four-way WOSAC table. Mid-run
# WOSAC every 500M steps, lite (10 batches); post-training WOSAC full (100
# batches). --role_episodes 0 (there is no role to analyse anyway).

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_dim0:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
# export WANDB_API_KEY=...   # only if not already set in ~/.bashrc

mkdir -p /scratch/e452103/logs
# Run from the PufferDrive dir: the C binding loads
# pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

python $HOME/roma_dim0/roma_pufferdrive/train_roma.py \
    --role_dim           0 \
    --num_maps           10000 \
    --total_steps        3000000000 \
    --num_agents         3072 \
    --device             cuda \
    --run_eval \
    --role_episodes      0 \
    --wosac_periodic     1 \
    --wosac_interval     500000000 \
    --wosac_eval_maps    10000 \
    --wosac_eval_rollouts 32 \
    --wosac_eval_batches 10 \
    --wosac_rollouts     32 \
    --wosac_num_maps     10000 \
    --wosac_max_batches  100 \
    --data_dir           /scratch/e452103/PufferDrive/pufferlib/resources/drive/binaries/training \
    --save_dir           /scratch/e452103/checkpoints/roma_baseline_dim0 \
    --save_interval      500000000 \
    --wandb_project      roma-pufferdrive \
    --wandb_entity       s-rahmani-tu-delft \
    --seed               0
