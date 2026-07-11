#!/bin/bash
#SBATCH --job-name=role_paired
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/role_paired_%j.out
#SBATCH --error=/scratch/e452103/logs/role_paired_%j.err

# PAIRED forced-role dose-response along PC1/PC2. Same map pool per episode
# serves EVERY condition; all stats are within-(map,vehicle) paired deltas,
# so map noise cancels. Headline = alpha +2 vs -2 on the same maps.
# Forcing = natural-offset: z_focal(t) = z_natural(t) + alpha*sigma*PC_d.
#
# Parametrized via environment variables (defaults = dim-4):
#   sbatch slurm/role_paired_sweep.sbatch
#   sbatch --export=ALL,CKPT=/scratch/e452103/checkpoints/roma_baseline_dim2/roma_dim2_final.pt,OUT=/scratch/e452103/role_paired/dim2 \
#       slurm/role_paired_sweep.sbatch

CKPT=${CKPT:-/scratch/e452103/checkpoints/roma_baseline_dim4/roma_dim4_final.pt}
OUT=${OUT:-/scratch/e452103/role_paired/dim4}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
CODE_DIR=$HOME/roma_pufferdrive

mkdir -p /scratch/e452103/logs
# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

python "$CODE_DIR/role_paired_sweep.py" \
    --checkpoint "$CKPT" \
    --clusters   /scratch/e452103/map_atlas/k4/map_clusters.csv \
    --data_dir   pufferlib/resources/drive/binaries/training \
    --out_dir    "$OUT" \
    --n_axes 2 --alphas="-2,-1,0,1,2" --warmup_episodes 3 --episodes 6

echo "Done -> $OUT"
