#!/bin/bash
#SBATCH --job-name=render_grid
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=16:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/render_grid_%j.out
#SBATCH --error=/scratch/e452103/logs/render_grid_%j.err

# OVERNIGHT render batch: for every trajectory type of the stop_frac K-means
# clustering, --scenes_per_type scenes; per scene 13 rollouts = natural +
# PC1 fan (-2..2) + PC2 fan (-2..2) + 4 PC1xPC2 corners. ~156 videos +
# 3 overlays/scene + metrics CSVs + grid_summary.csv.
#
# IMPORTANT: TRAJ points at a frozen COPY of the stop_frac clustering, because
# re-running trajectory_atlas.sbatch overwrites trajectory_clusters.csv in the
# atlas root. The copy is created below if missing.
#
#   sbatch slurm/render_role_grid.sbatch                       # dim-2
#   sbatch --export=ALL,CKPT=...dim4_final.pt,AXES=...dim4/role_paired_axes.csv,OUT=.../role_grid_dim4 \
#       slurm/render_role_grid.sbatch

CKPT=${CKPT:-/scratch/e452103/checkpoints/roma_baseline_dim2/roma_dim2_final.pt}
AXES=${AXES:-/scratch/e452103/role_paired/dim2/role_paired_axes.csv}
TRAJ=${TRAJ:-/scratch/e452103/traj_atlas/trajectory_clusters_stopfrac.csv}
OUT=${OUT:-/scratch/e452103/renders/role_grid_dim2}
SCENES=${SCENES:-3}
NAMES=${NAMES:-"0:fast,1:stop&go,2:mid-speed,3:turning"}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
CODE_DIR=$HOME/roma_pufferdrive

mkdir -p /scratch/e452103/logs "$OUT"

# freeze the stop_frac clustering so later atlas reruns can't change our strata
if [ ! -f "$TRAJ" ]; then
    cp /scratch/e452103/traj_atlas/trajectory_clusters.csv "$TRAJ"
    echo "frozen strata copy -> $TRAJ"
fi

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

python "$CODE_DIR/render_role_grid.py" \
    --checkpoint    "$CKPT" \
    --axes_csv      "$AXES" \
    --traj_clusters "$TRAJ" \
    --data_dir      pufferlib/resources/drive/binaries/training \
    --out_dir       "$OUT" \
    --alphas="-2,-1,0,1,2" --grid 1 \
    --scenes_per_type "$SCENES" \
    --type_names    "$NAMES"

echo "Done -> $OUT"
