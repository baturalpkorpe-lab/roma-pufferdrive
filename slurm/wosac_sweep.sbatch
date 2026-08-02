#!/bin/bash
#SBATCH --job-name=wosac_sw
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=14:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=wosac_sw_%j.out
#SBATCH --error=wosac_sw_%j.err

# WOSAC + safety metrics across a role forcing sweep, on the FULL map set.
# Same machinery as wosac_cluster.sbatch (role_cluster_wosac.py), but pointed
# at the whole training pool rather than one cluster's maps.
#
#   sbatch --export=ALL,CKPT=/scratch/$USER/checkpoints/roma_x/roma_dim1_final.pt,\
# AXES=/scratch/$USER/analysis/x/paired/role_paired_axes.csv,ROLE_DIM=1,TAG=x \
#       slurm/wosac_sweep.sbatch
#
# Per alpha you get, in wosac_by_alpha.csv:
#   realism_meta_score, kinematic/interactive/map_based metrics, ade, min_ade,
#   the 8 WOSAC likelihoods, score, collision_rate, offroad_rate, goal_rate.
#
# CONTROL MODE. Defaults to control_wosac (100% of agents policy-driven),
# because that IS the Waymo Open Sim Agents protocol and it is how every
# full-dataset number in this project was measured -- including the ~0.69
# baseline these results get compared against. role_cluster_wosac.py aborts
# after batch 1 at >=99% control unless told otherwise, so ALLOW_FULL_CONTROL
# is on by default HERE and only here: on cluster maps that abort is the point,
# on the full pool it is the intended configuration.
# Set CONTROL_MODE=control_agents to measure the cluster-style population
# instead (GT-replay traffic, mark_as_expert respected).
#
# WALL CLOCK: one pass per alpha, so 5 alphas = 5x a normal WOSAC run.
# 40 batches x 5 conditions is ~10h; 14h wall leaves margin.

set -u

CKPT=${CKPT:-}
AXES=${AXES:-}
[ -n "$CKPT" ] || { echo "ERROR: set CKPT=<checkpoint.pt>"; exit 1; }
[ -n "$AXES" ] || { echo "ERROR: set AXES=<role_paired_axes.csv>"; exit 1; }

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-pufferlib/resources/drive/binaries/training}   # rel. to PUFFER_DIR

ROLE_DIM=${ROLE_DIM:-4}
TAG=${TAG:-$(basename "$(dirname "$CKPT")")}
OUT=${OUT:-$SCRATCH_ROOT/wosac_sweep/$TAG}
ALPHAS=${ALPHAS:--1,-0.5,0,0.5,1}
AXIS=${AXIS:-PC1}
BATCHES=${BATCHES:-40}
ROLLOUTS=${ROLLOUTS:-32}
NUM_MAPS=${NUM_MAPS:-10000}
CONTROL_MODE=${CONTROL_MODE:-control_wosac}
ALLOW_FULL_CONTROL=${ALLOW_FULL_CONTROL:-1}

set +u   # conda compiler activation scripts are not set -u safe
module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
set -u

export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1

fail=0
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; fail=1; }
[ -f "$AXES" ] || { echo "ERROR: axes csv not found: $AXES"; fail=1; }
[ -d "$PUFFER_DIR" ] || { echo "ERROR: PUFFER_DIR not found: $PUFFER_DIR"; fail=1; }
[ -f "$CODE_DIR/role_cluster_wosac.py" ] || {
    echo "ERROR: role_cluster_wosac.py not under CODE_DIR=$CODE_DIR"; fail=1; }
[ "$fail" -eq 0 ] || { echo "[wosac_sw] preflight failed."; exit 1; }

# The axes CSV must actually contain the axis being swept -- a dim-1 checkpoint
# has PC1 only, so asking for PC2 here is a silent misconfiguration otherwise.
grep -q "^${AXIS}," "$AXES" || {
    echo "ERROR: '$AXIS' not in $AXES"
    echo "       axes present: $(cut -d, -f1 "$AXES" | tail -n +2 | tr '\n' ' ')"
    exit 1; }

AFC_ARG=""
case "$ALLOW_FULL_CONTROL" in 1|true|yes) AFC_ARG="--allow_full_control" ;; esac

mkdir -p "$OUT"
echo "[wosac_sw] ckpt=$CKPT"
echo "[wosac_sw] axes=$AXES axis=$AXIS role_dim=$ROLE_DIM"
echo "[wosac_sw] alphas=$ALPHAS batches=$BATCHES rollouts=$ROLLOUTS"
echo "[wosac_sw] maps=$DATA_DIR control_mode=$CONTROL_MODE"
echo "[wosac_sw] out=$OUT"
echo "[wosac_sw] branch=$(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?) commit=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo ?)"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

python "$CODE_DIR/role_cluster_wosac.py" \
    --checkpoint  "$CKPT" \
    --axes_csv    "$AXES" \
    --axis        "$AXIS" \
    --alphas="$ALPHAS" \
    --role_dim    "$ROLE_DIM" \
    --data_dir    "$DATA_DIR" \
    --out_dir     "$OUT" \
    --num_maps    "$NUM_MAPS" \
    --wosac_rollouts    "$ROLLOUTS" \
    --wosac_max_batches "$BATCHES" \
    --control_mode "$CONTROL_MODE" \
    $AFC_ARG \
  || { echo "[wosac_sw] FAILED (python exit $?) -- see the traceback above"; exit 1; }

# Never end on a bare echo: that is what made earlier jobs report COMPLETED 0:0
# after writing nothing.
[ -s "$OUT/wosac_by_alpha.csv" ] || {
    echo "[wosac_sw] FAILED: python exited 0 but wrote no wosac_by_alpha.csv"
    exit 1; }

echo
echo "[wosac_sw] done -> $OUT/wosac_by_alpha.csv"
