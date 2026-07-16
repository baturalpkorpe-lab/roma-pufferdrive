#!/bin/bash
#SBATCH --job-name=nat_extremes
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=03:45:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/nat_extremes_%j.out
#SBATCH --error=/scratch/e452103/logs/nat_extremes_%j.err

# OBSERVATIONAL natural-contrast: per trajectory cluster, find the agents
# whose NATURAL (unforced) roles sit at the two PC1 poles and render how
# they drive -- no forcing anywhere. Complements the causal paired sweep.
# Axes come from the SAME checkpoint's role_paired_axes.csv so PC1 here is
# the identical direction used in every other dim-2 figure.
#
#   sbatch slurm/role_natural_extremes.sbatch                    # dim-2 final
#   sbatch --export=ALL,CKPT=...,AXES=...,OUT=... slurm/role_natural_extremes.sbatch

CKPT=${CKPT:-/scratch/e452103/checkpoints/roma_baseline_dim2/roma_dim2_final.pt}
AXES=${AXES:-/scratch/e452103/role_paired/dim2/role_paired_axes.csv}
TRAJ=${TRAJ:-/scratch/e452103/traj_atlas/trajectory_clusters_stopfrac.csv}
OUT=${OUT:-/scratch/e452103/role_natural/dim2}
CLUSTERS=${CLUSTERS:-0,1,2,3}
PER_SIDE=${PER_SIDE:-1}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
CODE_DIR=$HOME/roma_pufferdrive

mkdir -p /scratch/e452103/logs "$OUT"
[ -f "$AXES" ] || { echo "ERROR: axes csv not found: $AXES (run role_paired_sweep first)"; exit 1; }
[ -f "$TRAJ" ] || { echo "ERROR: clusters csv not found: $TRAJ"; exit 1; }

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

python "$CODE_DIR/role_natural_extremes.py" \
    --checkpoint    "$CKPT" \
    --axes_csv      "$AXES" \
    --traj_clusters "$TRAJ" \
    --data_dir      pufferlib/resources/drive/binaries/training \
    --out_dir       "$OUT" \
    --clusters      "$CLUSTERS" \
    --per_side      "$PER_SIDE" \
    --warmup_episodes 4

echo "Done -> $OUT"
