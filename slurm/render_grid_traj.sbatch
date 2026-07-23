#!/bin/bash
#SBATCH --job-name=grid_traj
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=16:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=grid_traj_%j.out
#SBATCH --error=grid_traj_%j.err

# The trajectory-cluster VIDEO grid: render_role_grid picks --scenes_per_type
# scenes PER TRAJECTORY CLUSTER (keyed by the (scenario, vehicle) trajectory
# clustering) and renders each across the PC1xPC2 forcing grid, so the videos
# SHOW how each trajectory type (mid-speed / fast / stop&go / turning) responds
# to role forcing. CPU (compute partition) on purpose -- headless matplotlib,
# no GPU, so it does not eat the supervisor's GPU allocation for 16h.
#
# Reads the axes from role_paired_axes.csv (from nodiv_analysis.sbatch) so the
# forcing directions match the statistics. Usually launched chained afterok
# via slurm/run_nodiv_analysis.sh; standalone:
#   sbatch --export=ALL,CKPT=/scratch/$USER/checkpoints/roma_nodiv_dim4/roma_dim4_final.pt,\
#     AXES=/scratch/$USER/analysis/nodiv_dim4/paired/role_paired_axes.csv,\
#     OUT=/scratch/$USER/analysis/nodiv_dim4/renders/role_grid \
#     slurm/render_grid_traj.sbatch

set -u

ROLE_DIM=${ROLE_DIM:-4}
TAG=${TAG:-nodiv_dim4}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-$PUFFER_DIR/pufferlib/resources/drive/binaries/training}

CKPT=${CKPT:-$SCRATCH_ROOT/checkpoints/roma_nodiv_dim4/roma_dim${ROLE_DIM}_final.pt}
AXES=${AXES:-$SCRATCH_ROOT/analysis/$TAG/paired/role_paired_axes.csv}
OUT=${OUT:-$SCRATCH_ROOT/analysis/$TAG/renders/role_grid}
TRAJ=${TRAJ:-$SCRATCH_ROOT/traj_atlas/trajectory_clusters_stopfrac.csv}
TYPE_NAMES=${TYPE_NAMES:-0:mid-speed,1:fast,2:stop&go,3:turning}
ALPHAS=${ALPHAS:--2,-1,0,1,2}
SCENES_PER_TYPE=${SCENES_PER_TYPE:-6}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1

fail=0
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; fail=1; }
[ -f "$TRAJ" ] || { echo "ERROR: trajectory clustering not found: $TRAJ"; fail=1; }
[ -f "$AXES" ] || { echo "ERROR: axes CSV not found: $AXES (run nodiv_analysis.sbatch first)"; fail=1; }
[ -d "$DATA_DIR" ] || { echo "ERROR: DATA_DIR not found: $DATA_DIR"; fail=1; }
[ "$fail" -eq 0 ] || { echo "[grid_traj] preflight failed."; exit 1; }

mkdir -p "$OUT"
echo "[grid_traj] ckpt=$CKPT"
echo "[grid_traj] axes=$AXES traj=$TRAJ scenes_per_type=$SCENES_PER_TYPE out=$OUT"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

python "$CODE_DIR/render_role_grid.py" \
    --checkpoint     "$CKPT" \
    --axes_csv       "$AXES" \
    --traj_clusters  "$TRAJ" \
    --data_dir       "$DATA_DIR" \
    --out_dir        "$OUT" \
    --alphas="$ALPHAS" --grid 1 \
    --scenes_per_type "$SCENES_PER_TYPE" \
    --type_names     "$TYPE_NAMES" \
  || echo "[grid_traj] WARNING: render exited nonzero -- zipping whatever exists"

# --- Collect this job's log + zip the WHOLE analysis folder ----------------
# ROOT is the single folder the analysis job + this render both write into, so
# one zip is the complete deliverable to send.
ROOT=${ROOT:-$SCRATCH_ROOT/analysis/$TAG}
mkdir -p "$ROOT/logs"
cp -f "$SLURM_SUBMIT_DIR/grid_traj_${SLURM_JOB_ID}.out" "$ROOT/logs/" 2>/dev/null || true
cp -f "$SLURM_SUBMIT_DIR/grid_traj_${SLURM_JOB_ID}.err" "$ROOT/logs/" 2>/dev/null || true

# Prefer zip; fall back to tar.gz if the node has no zip binary. Either way it
# is ONE archive of the whole folder for the supervisor to send.
base=$(basename "$ROOT"); parent=$(dirname "$ROOT")
if command -v zip >/dev/null 2>&1; then
    ARCHIVE="$SCRATCH_ROOT/analysis/${TAG}.zip"
    echo "[grid_traj] zipping $ROOT -> $ARCHIVE  (videos dominate the size)"
    ( cd "$parent" && rm -f "$ARCHIVE" && zip -r -q "$ARCHIVE" "$base" )
else
    ARCHIVE="$SCRATCH_ROOT/analysis/${TAG}.tar.gz"
    echo "[grid_traj] no zip binary -- using tar: $ROOT -> $ARCHIVE"
    ( cd "$parent" && tar -czf "$ARCHIVE" "$base" )
fi \
  && echo "[grid_traj] SEND THIS ONE FILE: $ARCHIVE ($(du -h "$ARCHIVE" 2>/dev/null | cut -f1))" \
  || echo "[grid_traj] archive failed -- the folder is still complete at $ROOT"

echo "[grid_traj] done -> $OUT"
