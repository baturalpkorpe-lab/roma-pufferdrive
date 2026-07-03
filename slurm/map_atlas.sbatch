#!/bin/bash
#SBATCH --job-name=map_atlas
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=01:30:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/map_atlas_%j.out
#SBATCH --error=/scratch/e452103/logs/map_atlas_%j.err

# Map-regime analysis, Phases A+B:
#   A) map_features.py -- one feature vector per scenario (GT behavior +
#      road geometry), no policy, reset-only. -> map_features.csv
#   B) map_atlas.py    -- k-means regimes + atlas figures. -> atlas_*.png
# No GPU needed.

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
CODE_DIR=$HOME/roma_pufferdrive
OUT_DIR=/scratch/e452103/map_atlas

mkdir -p /scratch/e452103/logs "$OUT_DIR"
# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

python "$CODE_DIR/map_features.py" \
    --data_dir pufferlib/resources/drive/binaries/training \
    --out_dir  "$OUT_DIR" \
    --num_agents 3072 \
    --num_maps 10000 \
    --max_batches 100 \
    --target_unique 9500

python "$CODE_DIR/map_atlas.py" --in_dir "$OUT_DIR"

echo "Atlas in: $OUT_DIR"
