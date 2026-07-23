#!/bin/bash
#SBATCH --job-name=nodiv_an
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=nodiv_an_%j.out
#SBATCH --error=nodiv_an_%j.err

# ===========================================================================
# ONE-SHOT downstream analysis for a finished checkpoint (default: the
# diversity-off dim-4 run). Runs the whole role-analysis + render pipeline as
# a chain of SEPARATE python processes so nobody has to submit them one by one.
#
# WOSAC is NOT here on purpose -- it is already covered by slurm/eval_nodiv
# (role health check + full 100-batch WOSAC). This job is everything else.
#
# Each step is its own `python x.py` process, which is what keeps the C binding
# from segfaulting (that only happens on repeated Drive() inits WITHIN one
# process). Order is SCIENCE FIRST, RENDERS LAST: if the wall clock cuts the
# job off, every CSV/stat/plot is already written and only videos are lost.
#
# Pipeline:
#   1. role_paired_sweep      PC1/PC2 paired dose-response  -> agent/deltas/
#                             tests/warmup/axes CSVs + PC plots        [GPU]
#   2. role_scene_consistency per-scene role consistency (ICC-style)   [cpu]
#   3. role_paired_by_traj    dose-response split by trajectory type   [cpu]
#   4. role_space_map         the 2-D PCA role-space map               [cpu]
#   5. role_scene_icc         scene-vs-agent variance decomposition    [GPU]
#   6. render_role_alpha      the alpha-sweep videos + role heatmaps   [cpu]
#   7. render_role_grid       PC1xPC2 render grid  -- OPTIONAL, ~16h,  [cpu]
#                             off unless RENDER_GRID=1
#
# Usage on the account that HOLDS the checkpoint (defaults resolve to
# /scratch/$USER, i.e. the supervisor's scratch when he submits it):
#   sbatch slurm/nodiv_analysis.sbatch
# Override anything:
#   sbatch --export=ALL,CKPT=/scratch/$USER/checkpoints/roma_nodiv_dim4/roma_dim4_final.pt,\
#     TAG=nodiv_dim4,RENDER_GRID=1 slurm/nodiv_analysis.sbatch

set -u

# ---- Identity -------------------------------------------------------------
ROLE_DIM=${ROLE_DIM:-4}
TAG=${TAG:-nodiv_dim4}

# ---- Paths (default to the submitting account's own scratch) --------------
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-$PUFFER_DIR/pufferlib/resources/drive/binaries/training}
CKPT=${CKPT:-$SCRATCH_ROOT/checkpoints/roma_nodiv_dim4/roma_dim${ROLE_DIM}_final.pt}
OUT_ROOT=${OUT_ROOT:-$SCRATCH_ROOT/analysis/$TAG}

# Data artifacts from the map/trajectory clustering. These were built once on
# the ORIGINAL analysis account; the nodiv run trained on the SAME Waymo
# binaries, so the same files apply unchanged. They are small -- copy them to
# this account's scratch if missing (the preflight prints the exact paths).
CLUSTERS=${CLUSTERS:-$SCRATCH_ROOT/map_atlas/k4/map_clusters.csv}
TRAJ=${TRAJ:-$SCRATCH_ROOT/traj_atlas/trajectory_clusters_stopfrac.csv}

# ---- Knobs ----------------------------------------------------------------
ALPHAS=${ALPHAS:--2,-1,0,1,2}
WARMUP_EP=${WARMUP_EP:-3}
SWEEP_EP=${SWEEP_EP:-6}
ICC_EP=${ICC_EP:-3}
RENDER_GRID=${RENDER_GRID:-0}
SCENES_PER_TYPE=${SCENES_PER_TYPE:-40}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1

# ---- Preflight: fail on HARD deps before the GPU is wasted ----------------
fail=0
[ -f "$CKPT" ]      || { echo "ERROR: checkpoint not found: $CKPT"; fail=1; }
[ -d "$PUFFER_DIR" ]|| { echo "ERROR: PUFFER_DIR not found: $PUFFER_DIR"; fail=1; }
[ -d "$DATA_DIR" ]  || { echo "ERROR: DATA_DIR not found: $DATA_DIR"; fail=1; }
[ -f "$CODE_DIR/role_paired_sweep.py" ] || {
    echo "ERROR: analysis code not under CODE_DIR=$CODE_DIR (wrong submit dir / branch?)"; fail=1; }
if [ ! -f "$CLUSTERS" ]; then
    echo "ERROR: map clusters CSV not found: $CLUSTERS"
    echo "       This is map_atlas/k4/map_clusters.csv from the ORIGINAL"
    echo "       analysis account. Copy it to THIS account's scratch (small file),"
    echo "       or point CLUSTERS= at it. It is required by the sweep + renders."
    fail=1
fi
[ "$fail" -eq 0 ] || { echo "[nodiv_an] preflight failed -- nothing submitted to the GPU."; exit 1; }

HAVE_TRAJ=1
[ -f "$TRAJ" ] || { HAVE_TRAJ=0
    echo "[nodiv_an] NOTE: trajectory clusters not found ($TRAJ) -- steps 3 and"
    echo "           the optional grid (7) will be SKIPPED. Everything else runs."; }

PAIRED="$OUT_ROOT/paired"
ICC_OUT="$OUT_ROOT/icc"
ALPHA_OUT="$OUT_ROOT/renders/role_alpha"
GRID_OUT="$OUT_ROOT/renders/role_grid"
mkdir -p "$PAIRED" "$ICC_OUT" "$ALPHA_OUT"

echo "[nodiv_an] tag=$TAG role_dim=$ROLE_DIM"
echo "[nodiv_an] ckpt=$CKPT"
echo "[nodiv_an] out=$OUT_ROOT  clusters=$CLUSTERS  traj=$TRAJ(have=$HAVE_TRAJ)"
echo "[nodiv_an] branch=$(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?) commit=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo ?)"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

step() { echo; echo "==================== $* ===================="; }

# --- 1. Paired PC1/PC2 dose-response (the core statistics + the axes) ------
step "1/7 role_paired_sweep [GPU]"
python "$CODE_DIR/role_paired_sweep.py" \
    --checkpoint "$CKPT" \
    --clusters   "$CLUSTERS" \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$PAIRED" \
    --n_axes 2 --alphas="$ALPHAS" \
    --warmup_episodes "$WARMUP_EP" --episodes "$SWEEP_EP" \
  || echo "[nodiv_an] WARNING: sweep exited nonzero -- dependent steps may be partial"

AGENT="$PAIRED/role_paired_agent.csv"
AXES="$PAIRED/role_paired_axes.csv"
WARMUP="$PAIRED/role_paired_warmup.csv"

# --- 2. Per-scene role consistency -----------------------------------------
step "2/7 role_scene_consistency [cpu]"
if [ -f "$AGENT" ]; then
    TRAJ_ARG=""; [ "$HAVE_TRAJ" -eq 1 ] && TRAJ_ARG="--traj_clusters $TRAJ"
    python "$CODE_DIR/role_scene_consistency.py" \
        --agent_csv "$AGENT" --out_dir "$PAIRED" $TRAJ_ARG \
      || echo "[nodiv_an] WARNING: consistency failed"
else
    echo "[nodiv_an] skip: $AGENT missing (sweep did not produce it)"
fi

# --- 3. Dose-response split by trajectory type -----------------------------
step "3/7 role_paired_by_traj [cpu]"
if [ -f "$AGENT" ] && [ "$HAVE_TRAJ" -eq 1 ]; then
    python "$CODE_DIR/role_paired_by_traj.py" \
        --in_dir "$PAIRED" --traj_clusters "$TRAJ" \
      || echo "[nodiv_an] WARNING: by_traj failed"
else
    echo "[nodiv_an] skip: needs the agent CSV and the trajectory clusters"
fi

# --- 4. Role-space PCA map -------------------------------------------------
step "4/7 role_space_map [cpu]"
if [ -f "$WARMUP" ]; then
    python "$CODE_DIR/role_space_map.py" \
        --agent_csv "$WARMUP" --out "$PAIRED/role_space_map.png" \
      || echo "[nodiv_an] WARNING: role_space_map failed"
else
    echo "[nodiv_an] skip: $WARMUP missing"
fi

# --- 5. Scene-vs-agent ICC (is the role the map or the driver?) ------------
step "5/7 role_scene_icc [GPU]"
python "$CODE_DIR/role_scene_icc.py" \
    --checkpoint "$CKPT" \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$ICC_OUT" \
    --episodes   "$ICC_EP" \
  || echo "[nodiv_an] WARNING: role_scene_icc failed"

# --- 6. Alpha-sweep videos + role heatmaps ---------------------------------
step "6/7 render_role_alpha [cpu]"
AXES_ARG=""; [ -f "$AXES" ] && AXES_ARG="--axes_csv $AXES"
python "$CODE_DIR/render_role_alpha.py" \
    --checkpoint "$CKPT" \
    --clusters   "$CLUSTERS" \
    $AXES_ARG \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$ALPHA_OUT" \
    --axis PC1 --alphas="$ALPHAS" --maps_per_regime 1 \
  || echo "[nodiv_an] WARNING: render_role_alpha failed"

# --- 7. OPTIONAL PC1xPC2 render grid (~16h -- off by default) --------------
step "7/7 render_role_grid [cpu, optional]"
if [ "$RENDER_GRID" -eq 1 ] && [ "$HAVE_TRAJ" -eq 1 ] && [ -f "$AXES" ]; then
    mkdir -p "$GRID_OUT"
    python "$CODE_DIR/render_role_grid.py" \
        --checkpoint "$CKPT" \
        --axes_csv   "$AXES" \
        --traj_clusters "$TRAJ" \
        --data_dir   "$DATA_DIR" \
        --out_dir    "$GRID_OUT" \
        --alphas="$ALPHAS" --grid 1 --scenes_per_type "$SCENES_PER_TYPE" \
      || echo "[nodiv_an] WARNING: render_role_grid failed"
else
    echo "[nodiv_an] skip: set RENDER_GRID=1 (and provide TRAJ + axes) to enable"
    echo "           the ~16h grid render -- better run standalone if you want it."
fi

echo
echo "[nodiv_an] DONE -> $OUT_ROOT"
echo "  paired/  role_paired_{agent,deltas,tests,warmup,axes}.csv + PC plots + consistency + role_space_map.png"
echo "  icc/     role_scene_icc.csv + .png"
echo "  renders/role_alpha/  alpha-sweep mp4s + role heatmaps"
