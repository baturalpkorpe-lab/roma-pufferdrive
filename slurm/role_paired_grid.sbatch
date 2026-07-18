#!/bin/bash
#SBATCH --job-name=role_grid_paired
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/role_grid_paired_%j.out
#SBATCH --error=/scratch/e452103/logs/role_grid_paired_%j.err

# Statistically-solid PC1xPC2 interaction (paired grid sweep). Defaults to the
# cluster0 (fast) checkpoint on its OWN maps -- the run whose render grid
# showed the speed interaction. Override CLUSTER/CKPT/AXES for another cluster.
#
#   sbatch --export=ALL,CLUSTER=0 slurm/role_paired_grid.sbatch
#   sbatch --export=ALL,CLUSTER=0,ALPHAS="-2,-1,0,1,2" slurm/role_paired_grid.sbatch  # 5x5, smoother

CLUSTER=${CLUSTER:-0}
ROLE_DIM=${ROLE_DIM:-4}
CKPT=${CKPT:-/scratch/e452103/checkpoints/roma_cluster${CLUSTER}_dim${ROLE_DIM}/roma_dim${ROLE_DIM}_final.pt}
AXES=${AXES:-/scratch/e452103/role_paired/cluster${CLUSTER}/role_paired_axes.csv}
DATA_DIR=${DATA_DIR:-/scratch/e452103/cluster_maps/cluster${CLUSTER}}
MAP_CLUSTERS=${MAP_CLUSTERS:-/scratch/e452103/map_atlas/k4/map_clusters.csv}
OUT=${OUT:-/scratch/e452103/role_paired/cluster${CLUSTER}_grid}
ALPHAS=${ALPHAS:--2,0,2}
EPISODES=${EPISODES:-8}
METRIC=${METRIC:-speed_mean}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
CODE_DIR=$HOME/roma_pufferdrive

mkdir -p /scratch/e452103/logs "$OUT"
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
[ -f "$AXES" ] || { echo "ERROR: axes not found: $AXES (run the sweep first)"; exit 1; }
# per-cluster dirs have fewer maps than the 10k default -> cap num_maps
NUM_MAPS=$(ls "$DATA_DIR"/map_*.bin 2>/dev/null | wc -l)
[ "$NUM_MAPS" -gt 0 ] || { echo "ERROR: no maps in $DATA_DIR"; exit 1; }
echo "cluster=$CLUSTER ckpt=$CKPT maps=$DATA_DIR ($NUM_MAPS) alphas=$ALPHAS"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

python "$CODE_DIR/role_paired_grid.py" \
    --checkpoint "$CKPT" \
    --axes_csv   "$AXES" \
    --clusters   "$MAP_CLUSTERS" \
    --data_dir   "$DATA_DIR" \
    --num_maps   "$NUM_MAPS" \
    --out_dir    "$OUT" \
    --alphas     "$ALPHAS" \
    --episodes   "$EPISODES" \
    --metric     "$METRIC"

echo "Done -> $OUT"
