#!/bin/bash
#SBATCH --job-name=roma_baseline
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/roma_baseline_%j.out
#SBATCH --error=/scratch/e452103/logs/roma_baseline_%j.err

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load miniconda3

# Activate your conda env (adjust name if different)
source activate roma

export PYTHONPATH=/scratch/e452103/PufferDrive/roma_pufferdrive
export WANDB_API_KEY=your_key_here   # or set in ~/.bashrc

mkdir -p /scratch/e452103/logs
cd /scratch/e452103/PufferDrive

# ── Training ──────────────────────────────────────────────────────────────────
# NOTE: run from the baseline branch checkout:
#   git checkout baseline
#   git pull origin baseline
#
# BEFORE RUNNING: apply the 3 fixes listed below to train_roma.py
#   Fix 1: remove "prep_human_data": False  from env_cfg dict
#   Fix 2: remove kl_weight=0.0             from RomaAuxLoss(...)
#   Fix 3: in run_evaluation(), change global_step=args.total_steps
#           to just not pass it (or pass None) — avoids wandb step overshoot

python roma_pufferdrive/train_roma.py \
    --role_dim           8 \
    --num_maps           10000 \
    --total_steps        2500000000 \
    --num_agents         3072 \
    --device             cuda \
    --run_eval \
    --wosac_interval     500000000 \
    --wosac_eval_maps    10000 \
    --wosac_eval_rollouts 32 \
    --wosac_eval_batches 20 \
    --data_dir           data/processed/training \
    --save_dir           /scratch/e452103/checkpoints/roma_baseline \
    --save_interval      500000000 \
    --wandb_project      roma-pufferdrive \
    --wandb_entity       baturalpkorpe \
    --wandb_run_name     roma_baseline_2.5B_s0 \
    --seed               0
