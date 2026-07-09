#!/bin/bash
#SBATCH --job-name=traj_atlas
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=02:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/traj_atlas_%j.out
#SBATCH --error=/scratch/e452103/logs/traj_atlas_%j.err

# Trajectory-level stratification, Phases A+B:
#   A) trajectory_features.py -- one row per (scenario, vehicle) from the
#      HUMAN GT trajectory: distance, mean/max/min speed, net_turn, with
#      step-level physical plausibility masking (replaces map-level junk
#      quarantine). Reset-only, no policy, no GPU.
#   B) trajectory_atlas.py -- K-means task types + atlas figures.
# Auto-K by silhouette here; recluster with a fixed K in seconds via:
#   python trajectory_atlas.py --in_dir $OUT --out_dir $OUT/k<K> --k <K>

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
CODE_DIR=$HOME/roma_pufferdrive
OUT_DIR=/scratch/e452103/traj_atlas

mkdir -p /scratch/e452103/logs "$OUT_DIR"
# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

python "$CODE_DIR/trajectory_features.py" \
    --data_dir pufferlib/resources/drive/binaries/training \
    --out_dir  "$OUT_DIR" \
    --num_agents 3072 \
    --num_maps 10000 \
    --max_batches 100 \
    --target_scenarios 9500

python "$CODE_DIR/trajectory_atlas.py" --in_dir "$OUT_DIR"

echo "Atlas in: $OUT_DIR"
