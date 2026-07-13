#!/bin/bash
#SBATCH --job-name=roma_dim1
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/roma_dim1_%j.out
#SBATCH --error=/scratch/e452103/logs/roma_dim1_%j.err

# dim-1 baseline run (branch baseline_role_1_dim) from its OWN clone at
# ~/roma_dim1, so ~/roma_pufferdrive can stay on the baseline branch.
#
# One-time setup before first submit:
#   git clone -b baseline_role_1_dim \
#       https://github.com/baturalpkorpe-lab/roma-pufferdrive.git ~/roma_dim1
#
# Why dim-1: the role space at dim-2/4/8 is consistently ~1-D-dominant (one
# PC carries the large majority of variance/behavior). This run tests the
# original ROMA scalar role directly -- does a single dim reproduce the same
# assertiveness axis and hold WOSAC parity?
#
# total_steps=3B: matches dim-0/2/3/4/8 for an apples-to-apples WOSAC table.
# Mid-run WOSAC every 500M steps, lite (10 batches). --role_episodes 0 skips
# the old role suite (role analysis is done separately via
# role_direct_cluster.py / role_paired_sweep.py on the final checkpoint).
#
# NO --run_eval: the in-process post-training eval SEGFAULTS reproducibly
# (signal 11, C binding -- see baseline_role_2_dim / eval_wosac.sbatch commit
# history). The final checkpoint is saved BEFORE the eval, so nothing is
# lost. Run the full 100-batch WOSAC + role health in a FRESH process:
#   sbatch --export=ALL,CKPT=/scratch/e452103/checkpoints/roma_baseline_dim1/roma_dim1_final.pt,ROLE_DIM=1,OUT=/scratch/e452103/eval_results/dim1 \
#       ~/roma_pufferdrive/slurm/eval_wosac.sbatch

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_dim1:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1   # stream prints to the .out log immediately
# export WANDB_API_KEY=...   # only if not already set in ~/.bashrc

mkdir -p /scratch/e452103/logs
# Run from the PufferDrive dir: the C binding loads
# pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

python $HOME/roma_dim1/roma_pufferdrive/train_roma.py \
    --role_dim           1 \
    --num_maps           10000 \
    --total_steps        3000000000 \
    --num_agents         3072 \
    --device             cuda \
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
    --save_dir           /scratch/e452103/checkpoints/roma_baseline_dim1 \
    --save_interval      500000000 \
    --wandb_project      roma-pufferdrive \
    --wandb_entity       s-rahmani-tu-delft \
    --seed               0
