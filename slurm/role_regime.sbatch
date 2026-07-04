#!/bin/bash
#SBATCH --job-name=role_regime
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/role_regime_%j.out
#SBATCH --error=/scratch/e452103/logs/role_regime_%j.err

# Phase C: role vs GT-referenced behavior deltas, stratified by map regime.
# Runs BOTH 3B checkpoints (dim-8 and dim-4) so the comparison uses the
# identical pipeline. Requires Phase A+B outputs (map_clusters.csv).

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1

CODE_DIR=$HOME/roma_pufferdrive
CLUSTERS=/scratch/e452103/map_atlas/map_clusters.csv
DATA_DIR=pufferlib/resources/drive/binaries/training

mkdir -p /scratch/e452103/logs
# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

echo "=== Phase C on dim-8 checkpoint ==="
python "$CODE_DIR/role_regime_analysis.py" \
    --checkpoint /scratch/e452103/checkpoints/roma_baseline/roma_dim8_step3000238080.pt \
    --clusters   "$CLUSTERS" \
    --data_dir   "$DATA_DIR" \
    --out_dir    /scratch/e452103/role_regime/dim8 \
    --episodes   20

echo "=== Phase C on dim-4 checkpoint ==="
python "$CODE_DIR/role_regime_analysis.py" \
    --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim4/roma_dim4_step3000238080.pt \
    --clusters   "$CLUSTERS" \
    --data_dir   "$DATA_DIR" \
    --out_dir    /scratch/e452103/role_regime/dim4 \
    --episodes   20

echo "Results: /scratch/e452103/role_regime/{dim8,dim4}/phaseC_*.png"
