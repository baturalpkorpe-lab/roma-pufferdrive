#!/bin/bash
#SBATCH --job-name=render_cmp
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=08:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/render_cmp_%j.out
#SBATCH --error=/scratch/e452103/logs/render_cmp_%j.err

# Same-scene dim0-vs-role comparison videos: for each K=4 trajectory type,
# 2 core (non-edge) scenes; each rolled 6x = dim0 + role ckpt at
# alpha=-2,-1,0,1,2 along PC1 (natural-offset). Background = GT replay.
# 4 types x 2 scenes x 6 = 48 videos + 8 overlays + metric CSVs.
#
# Parametrized (defaults = dim-2 vs dim-0):
#   sbatch slurm/render_role_compare.sbatch
#   sbatch --export=ALL,CKPT=...dim4/roma_dim4_final.pt,AXES=...role_paired/dim4/role_paired_axes.csv,OUT=.../role_compare_dim4 \
#       slurm/render_role_compare.sbatch

CKPT=${CKPT:-/scratch/e452103/checkpoints/roma_baseline_dim2/roma_dim2_final.pt}
CKPT0=${CKPT0:-/scratch/e452103/checkpoints/roma_baseline_dim0/roma_dim0_final.pt}
AXES=${AXES:-/scratch/e452103/role_paired/dim2/role_paired_axes.csv}
TRAJ=${TRAJ:-/scratch/e452103/traj_atlas/k4/trajectory_clusters.csv}
OUT=${OUT:-/scratch/e452103/renders/role_compare_dim2}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
CODE_DIR=$HOME/roma_pufferdrive

mkdir -p /scratch/e452103/logs "$OUT"
# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

AXES_ARG=""
if [ -f "$AXES" ]; then AXES_ARG="--axes_csv $AXES"; fi

python "$CODE_DIR/render_role_compare.py" \
    --checkpoint      "$CKPT" \
    --checkpoint_dim0 "$CKPT0" \
    $AXES_ARG \
    --traj_clusters   "$TRAJ" \
    --data_dir        pufferlib/resources/drive/binaries/training \
    --out_dir         "$OUT" \
    --axis PC1 --alphas="-2,-1,0,1,2" --scenes_per_type 2

echo "Done -> $OUT"
