#!/bin/bash
#SBATCH --job-name=roma_cl0
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/roma_cluster_%j.out
#SBATCH --error=/scratch/e452103/logs/roma_cluster_%j.err

# Per-cluster ROMA training (dim=4): the policy controls ONLY the vehicles of
# one stop_frac trajectory cluster; every other object replays its GT
# trajectory (mark_as_expert baked into the per-cluster binaries by
# make_cluster_maps.py -- run slurm/make_cluster_maps.sbatch first).
#
# Cluster names: 0=mid-speed  1=fast  2=stop&go  3=turning
#
# The cluster<K>-* branches pin CLUSTER; on baseline pass it explicitly:
#   sbatch --export=ALL,CLUSTER=1 slurm/train_cluster.sbatch
#
# No --run_eval and --wosac_periodic 0: repeated Drive() instantiations in one
# long process segfault the C binding (~8th instantiation). Run WOSAC
# standalone afterwards via slurm/eval_wosac.sbatch (fresh process).

CLUSTER=${CLUSTER:-0}
[ -n "$CLUSTER" ] || { echo "ERROR: set CLUSTER=0..3 (or submit from a cluster<K> branch)"; exit 1; }

ROLE_DIM=${ROLE_DIM:-4}
MAP_DIR=${MAP_DIR:-/scratch/e452103/cluster_maps/cluster${CLUSTER}}
SAVE_DIR=${SAVE_DIR:-/scratch/e452103/checkpoints/roma_cluster${CLUSTER}_dim${ROLE_DIM}}
TOTAL_STEPS=${TOTAL_STEPS:-2000000000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500000000}
SEED=${SEED:-0}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

# CODE_DIR defaults to where you ran sbatch from, so each per-cluster clone
# (~/roma_cluster<K>) runs its own checkout. Override with CODE_DIR=... if
# submitting from elsewhere.
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
export PYTHONPATH=$CODE_DIR:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1

mkdir -p /scratch/e452103/logs "$SAVE_DIR"

NUM_MAPS=$(ls "$MAP_DIR"/map_*.bin 2>/dev/null | wc -l)
[ "$NUM_MAPS" -gt 0 ] || { echo "ERROR: no maps in $MAP_DIR -- run slurm/make_cluster_maps.sbatch first"; exit 1; }
echo "cluster=$CLUSTER  role_dim=$ROLE_DIM  maps=$NUM_MAPS  code=$CODE_DIR"

# Optional wandb: sbatch --export=ALL,CLUSTER=1,WANDB_PROJECT=roma-pufferdrive,WANDB_ENTITY=<team>
WANDB_ARGS=""
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARGS="--wandb_project $WANDB_PROJECT"
    [ -n "$WANDB_ENTITY" ] && WANDB_ARGS="$WANDB_ARGS --wandb_entity $WANDB_ENTITY"
fi

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

python "$CODE_DIR/roma_pufferdrive/train_roma.py" \
    --role_dim      "$ROLE_DIM" \
    --data_dir      "$MAP_DIR" \
    --num_maps      "$NUM_MAPS" \
    --num_agents    3072 \
    --device        cuda \
    --total_steps   "$TOTAL_STEPS" \
    --save_interval "$SAVE_INTERVAL" \
    --save_dir      "$SAVE_DIR" \
    --seed          "$SEED" \
    --wosac_periodic 0 \
    $WANDB_ARGS

echo "Done -> $SAVE_DIR"
