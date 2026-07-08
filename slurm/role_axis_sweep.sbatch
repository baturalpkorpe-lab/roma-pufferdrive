#!/bin/bash
#SBATCH --job-name=role_axis
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/role_axis_%j.out
#SBATCH --error=/scratch/e452103/logs/role_axis_%j.err

# Phase 2 (causal): forced-role dose-response along PC1/PC2 for the dim-4 model.
# Warmup -> PCA axes; then force one focal per scene along each axis at alpha in
# {-2,-1,0,1,2} sigma, others natural, same scene/seed -> behavior vs alpha.

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

python "$CODE_DIR/role_axis_sweep.py" \
    --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim4/roma_dim4_final.pt \
    --clusters   /scratch/e452103/map_atlas/k4/map_clusters.csv \
    --data_dir   pufferlib/resources/drive/binaries/training \
    --out_dir    /scratch/e452103/role_axis/dim4 \
    --n_axes 2 --alphas "-2,-1,0,1,2" --warmup_episodes 3 --episodes 4

echo "Done -> /scratch/e452103/role_axis/dim4/role_axis_*.png"
