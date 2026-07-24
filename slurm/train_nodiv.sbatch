#!/bin/bash
#SBATCH --job-name=roma_nodiv
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=roma_nodiv_%j.out
#SBATCH --error=roma_nodiv_%j.err

# ===========================================================================
# DIVERSITY-LOSS ABLATION: --div_weight 0, for role_dim 1 AND role_dim 4.
# ===========================================================================
#
# Why: the dim-1 diversity term is -(std/scale), a coefficient of variation,
# and for a fixed scale a CV is maximised by a TWO-POINT distribution -- so
# the dim-1 objective actively rewards bimodality, while the role_dim>=2 term
# (mean pairwise cosine similarity of unit-normalised role means) is
# magnitude-invariant and has no such preference. The observed dim-1 collapse
# to a binary switch (Sarle BC=0.99) vs the dim-2/4 continuum (BC=0.54) is
# therefore confounded: it may be the scalar bottleneck, or it may be the
# objective geometry. Setting div_weight=0 in BOTH arms removes the objective
# as an explanation -- if dim-1 still collapses, the bottleneck is the cause.
#
# Runs from the BASELINE branch for both dims on purpose. The old dim-1 and
# dim-4 control runs lived on separate branches whose code differs only in
# argparse defaults (dim-1) and in the dim-1 aux path + the role_dim==0
# ablation hook (dim-4) -- neither is reachable in a dim-4 run, so baseline
# is behaviourally identical to both controls while keeping the two ablation
# arms bit-identical to each other.
#
# Usage (see the block of vars below; every one is overridable):
#   sbatch --export=ALL,ROLE_DIM=1 slurm/train_nodiv.sbatch
#   sbatch --export=ALL,ROLE_DIM=4 slurm/train_nodiv.sbatch
#
# AUTO-RESUME: if SAVE_DIR already holds roma_dim<D>_step*.pt checkpoints,
# the highest-step one is resumed automatically. 3B steps does NOT fit in one
# 24h job, so submit a second job with --dependency=afterany:<jobid> and it
# picks up where the wall clock cut the first one off. Adam moments were
# never checkpointed, so each restart has a brief warm-restart transient.
#
# NO --run_eval: the in-process post-training eval segfaults (signal 11 in
# the C binding -- repeated Drive() instantiations in one process die around
# the ~8th). The final checkpoint is written BEFORE that point, so nothing is
# lost; the full 100-batch WOSAC + role health check runs in a FRESH process
# via slurm/eval_wosac.sbatch. Splitting 3B across two jobs also keeps each
# process at ~3 periodic Drive() instantiations instead of 6, well clear of
# the segfault threshold.

set -u

ROLE_DIM=${ROLE_DIM:-}
case "$ROLE_DIM" in
    1|4) ;;
    *) echo "ERROR: set ROLE_DIM=1 or ROLE_DIM=4  (sbatch --export=ALL,ROLE_DIM=1 ...)"; exit 1 ;;
esac

# ---- Paths. Defaults assume the caller's own scratch; override if sharing. --
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-$PUFFER_DIR/pufferlib/resources/drive/binaries/training}
SAVE_DIR=${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/roma_nodiv_dim${ROLE_DIM}}

# ---- Run config ------------------------------------------------------------
DIV_WEIGHT=${DIV_WEIGHT:-0}                  # <-- the ablation
MI_WEIGHT=${MI_WEIGHT:-1.0}                  # MI loss stays ON
TOTAL_STEPS=${TOTAL_STEPS:-3000000000}       # matches the div-on controls
SAVE_INTERVAL=${SAVE_INTERVAL:-500000000}
NUM_MAPS=${NUM_MAPS:-10000}
NUM_AGENTS=${NUM_AGENTS:-3072}
SEED=${SEED:-0}
WANDB_PROJECT=${WANDB_PROJECT:-roma-pufferdrive}
WANDB_ENTITY=${WANDB_ENTITY:-s-rahmani-tu-delft}
WANDB_NAME=${WANDB_NAME:-roma_dim${ROLE_DIM}_nodiv_maps${NUM_MAPS}_${SEED}}

set +u   # conda compiler activation scripts (gcc_linux-64) are not set -u safe: SYS_SYSROOT/CONDA_BACKUP_CXX unbound
module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
set -u

export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1   # stream prints to the .out log immediately

# ---- Preflight: fail loudly here rather than 30s into a GPU allocation -----
[ -d "$PUFFER_DIR" ] || { echo "ERROR: PUFFER_DIR not found: $PUFFER_DIR"; exit 1; }
[ -d "$DATA_DIR" ]   || { echo "ERROR: DATA_DIR not found: $DATA_DIR"; exit 1; }
[ -f "$CODE_DIR/roma_pufferdrive/train_roma.py" ] || {
    echo "ERROR: train_roma.py not under CODE_DIR: $CODE_DIR"; exit 1; }
mkdir -p "$SAVE_DIR"

# ---- Auto-resume from the highest-step checkpoint in SAVE_DIR --------------
# Step checkpoints carry aux_loss_state (the MI/div critics); final.pt does
# not, so it is deliberately excluded from this search.
RESUME_ARGS=""
LATEST=$(ls "$SAVE_DIR"/roma_dim${ROLE_DIM}_step*.pt 2>/dev/null \
         | sed 's/.*_step\([0-9]*\)\.pt/\1 &/' | sort -k1,1n | tail -1 | cut -d' ' -f2-)
if [ -n "$LATEST" ]; then
    # Already finished? Exit 0, not 1. These jobs are submitted as a chain of
    # 3 (--dependency=afterany) because 3B steps does not fit in one 24h
    # allocation, and the eval job hangs off the last one with afterok. If the
    # run completed early, train_roma.py would exit non-zero on
    # "--total_steps <= resumed step" and take the eval down with it.
    LATEST_STEP=$(echo "$LATEST" | sed 's/.*_step\([0-9]*\)\.pt/\1/')
    if [ -f "$SAVE_DIR/roma_dim${ROLE_DIM}_final.pt" ] || [ "$LATEST_STEP" -ge "$TOTAL_STEPS" ]; then
        echo "[nodiv] already at step $LATEST_STEP / $TOTAL_STEPS -- nothing to do"
        exit 0
    fi
    echo "[nodiv] resuming from $LATEST (step $LATEST_STEP)"
    RESUME_ARGS="--resume $LATEST"
else
    echo "[nodiv] no existing checkpoint in $SAVE_DIR -- fresh run"
fi

echo "[nodiv] role_dim=$ROLE_DIM div_weight=$DIV_WEIGHT total_steps=$TOTAL_STEPS"
echo "[nodiv] code=$CODE_DIR  save=$SAVE_DIR"
echo "[nodiv] commit=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# The C binding loads pufferlib/config/ocean/drive.ini via a RELATIVE path,
# so the process must run from the PufferDrive root.
cd "$PUFFER_DIR"

python "$CODE_DIR/roma_pufferdrive/train_roma.py" \
    --role_dim            "$ROLE_DIM" \
    --div_weight          "$DIV_WEIGHT" \
    --mi_weight           "$MI_WEIGHT" \
    --num_maps            "$NUM_MAPS" \
    --num_agents          "$NUM_AGENTS" \
    --total_steps         "$TOTAL_STEPS" \
    --device              cuda \
    --data_dir            "$DATA_DIR" \
    --save_dir            "$SAVE_DIR" \
    --save_interval       "$SAVE_INTERVAL" \
    --seed                "$SEED" \
    --role_episodes       0 \
    --wosac_periodic      1 \
    --wosac_interval      500000000 \
    --wosac_eval_maps     10000 \
    --wosac_eval_rollouts 32 \
    --wosac_eval_batches  10 \
    --wandb_project       "$WANDB_PROJECT" \
    --wandb_entity        "$WANDB_ENTITY" \
    --wandb_name          "$WANDB_NAME" \
    $RESUME_ARGS

echo "[nodiv] done -> $SAVE_DIR"
