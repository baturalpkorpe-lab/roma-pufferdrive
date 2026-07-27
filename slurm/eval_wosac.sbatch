#!/bin/bash
#SBATCH --job-name=eval_wosac
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/eval_wosac_%j.out
#SBATCH --error=/scratch/e452103/logs/eval_wosac_%j.err

# Standalone post-training eval: role health check (dim correlations + PCA)
# then FULL WOSAC (100 batches x 32 rollouts). Exists because the 24h training
# jobs reliably hit the wall clock right at the post-training eval phase --
# the final checkpoint is already saved, so the eval runs separately here.
#
# Parametrized via environment variables (defaults = dim-2):
#   sbatch slurm/eval_wosac.sbatch                          # dim-2
#   sbatch --export=ALL,CKPT=/path/x_final.pt,ROLE_DIM=3,OUT=/scratch/e452103/eval_results/dim3 \
#       slurm/eval_wosac.sbatch

CKPT=${CKPT:-/scratch/e452103/checkpoints/roma_baseline_dim2/roma_dim2_final.pt}
ROLE_DIM=${ROLE_DIM:-2}
OUT=${OUT:-/scratch/e452103/eval_results/dim2}
# eval_roma falls back to the CHECKPOINT STEM for the wandb run name, and every
# per-cluster checkpoint is called roma_dim<D>_final.pt -- so without this every
# cluster eval shows up in wandb as the same indistinguishable run. Default to
# the output folder name, which is already unique per eval.
RUN_NAME=${RUN_NAME:-$(basename "$OUT")}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
# eval_roma has no --wandb_entity flag; it uses the runner's default entity.
export WANDB_ENTITY=s-rahmani-tu-delft

mkdir -p /scratch/e452103/logs "$OUT"
# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

python $HOME/roma_pufferdrive/eval_roma.py \
    --checkpoint        "$CKPT" \
    --role_dim          "$ROLE_DIM" \
    --num_agents        3072 \
    --device            cuda \
    --map_dir           pufferlib/resources/drive/binaries/training \
    --wosac_rollouts    32 \
    --wosac_num_maps    10000 \
    --wosac_max_batches 100 \
    --role_episodes     10 \
    --output_dir        "$OUT" \
    --wandb --wandb_run_name "$RUN_NAME"

echo "Done -> $OUT"
