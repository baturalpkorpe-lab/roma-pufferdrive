#!/bin/bash
#SBATCH --job-name=role_direct
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/role_direct_%j.out
#SBATCH --error=/scratch/e452103/logs/role_direct_%j.err

# Role-VARIABLE clustering (K-means on z, per map regime), profiled by BOTH raw
# kinematics and GT-deltas + an eta^2 separability panel. Uses the dim-4 final
# checkpoint and the NEW K=4 atlas (junk already removed, so no cluster to
# exclude). Requires Phase A+B outputs (map_clusters.csv).

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1

CODE_DIR=$HOME/roma_pufferdrive
# NOTE: confirm this path matches where your K=4 atlas wrote map_clusters.csv.
CLUSTERS=/scratch/e452103/map_atlas/k4/map_clusters.csv
DATA_DIR=pufferlib/resources/drive/binaries/training

mkdir -p /scratch/e452103/logs
# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

python "$CODE_DIR/role_direct_cluster.py" \
    --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim4/roma_dim4_final.pt \
    --clusters   "$CLUSTERS" \
    --data_dir   "$DATA_DIR" \
    --out_dir    /scratch/e452103/role_direct/dim4 \
    --episodes   20 \
    --kmeans_k   3

echo "Results: /scratch/e452103/role_direct/dim4/role_direct_*.png + *.csv"
