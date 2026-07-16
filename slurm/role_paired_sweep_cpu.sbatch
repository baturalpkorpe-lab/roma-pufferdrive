#!/bin/bash
#SBATCH --job-name=role_paired_cpu
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=08:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/role_paired_cpu_%j.out
#SBATCH --error=/scratch/e452103/logs/role_paired_cpu_%j.err

# CPU-only variant of role_paired_sweep.sbatch: role_paired_sweep.py already
# auto-falls-back to CPU when no GPU is visible, so this just runs it on the
# `compute` partition to skip the gpu-a100 Priority backlog. Slower per-step
# (no GPU for the policy forward pass) but the env sim itself is CPU-bound
# either way, so wall-time cost is smaller than it looks.
#
# Points --data_dir at the CLUSTER-SPECIFIC map set (not the full training
# pool), matching what the checkpoint was actually trained on. Defaults to an
# in-progress checkpoint (--ckpt not required to be *_final.pt).
#
# CAVEAT: --clusters (map-regime stratification) still points at the ORIGINAL
# full-pool map_clusters.csv; since cluster_maps/cluster<K> renumbers map ids,
# that regime breakdown may not resolve correctly against the renumbered
# binaries. The core PC1/PC2 dose-response numbers do not depend on it -- only
# the map-regime sub-grouping might come out empty/wrong. Trajectory-type
# stratification (role_paired_by_traj.py) is unaffected either way.
#
#   sbatch --export=ALL,CLUSTER=1 slurm/role_paired_sweep_cpu.sbatch
#   sbatch --export=ALL,CLUSTER=1,CKPT=/path/to/specific_step.pt slurm/role_paired_sweep_cpu.sbatch

CLUSTER=${CLUSTER:-}
[ -n "$CLUSTER" ] || { echo "ERROR: set CLUSTER=0..3"; exit 1; }

MAP_DIR=${MAP_DIR:-/scratch/e452103/cluster_maps/cluster${CLUSTER}}
CKPT=${CKPT:-$(ls -t /scratch/e452103/checkpoints/roma_cluster${CLUSTER}_dim4/*.pt 2>/dev/null | head -1)}
OUT=${OUT:-/scratch/e452103/role_paired/cluster${CLUSTER}_cpu}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
CODE_DIR=$HOME/roma_pufferdrive

mkdir -p /scratch/e452103/logs "$OUT"

[ -n "$CKPT" ] || { echo "ERROR: no checkpoint found in /scratch/e452103/checkpoints/roma_cluster${CLUSTER}_dim4/ -- pass CKPT= explicitly (training may not have saved one yet)"; exit 1; }
echo "cluster=$CLUSTER  ckpt=$CKPT  map_dir=$MAP_DIR"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

python "$CODE_DIR/role_paired_sweep.py" \
    --checkpoint "$CKPT" \
    --clusters   /scratch/e452103/map_atlas/k4/map_clusters.csv \
    --data_dir   "$MAP_DIR" \
    --out_dir    "$OUT" \
    --n_axes 2 --alphas="-2,-1,0,1,2" --warmup_episodes 3 --episodes 6

echo "Done -> $OUT"
