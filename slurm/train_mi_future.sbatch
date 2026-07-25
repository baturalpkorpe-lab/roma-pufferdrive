#!/bin/bash
#SBATCH --job-name=roma_mifut
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=roma_mifut_%j.out
#SBATCH --error=roma_mifut_%j.err

# FUTURE-prediction MI training (mi-future branch, dim=4): z_t must predict the
# behaviour summary of the NEXT 8 env embeddings (t+1..t+8) instead of the past
# window -- R3DM-inspired; the role becomes an INTENT signal rather than a
# recap of what already happened / what the scene looks like. Full 10k map set,
# standard self-play (NOT per-cluster maps).
#
# Runs on the account that submits it (defaults resolve to /scratch/$USER), so
# it works unchanged on any account -- just clone the mi-future branch and:
#   sbatch slurm/train_mi_future.sbatch
#   sbatch --export=ALL,MI_HORIZON=16 slurm/train_mi_future.sbatch
#
# 3B steps. Mid-run WOSAC every 500M (lite, 10 batches). The FULL 100-batch
# WOSAC + role analysis runs as a SEPARATE job (eval_mi_future.sbatch, chained
# afterok via run_mi_future.sh) -- NOT --run_eval: the in-process post-training
# eval plus 5 periodic Drive() instantiations would cross the ~8th-Drive C
# binding segfault. 5 periodic evals alone (the terminal 3B tick is skipped in
# train_roma.py) stay under that.
#
# Submitted as ONE job: 3B fits the 24h allocation. AUTO-RESUME is here as a
# RECOVERY path, not a chain -- if SAVE_DIR already holds roma_dim<D>_step*.pt
# the highest-step one is resumed, so a run the wall clock did cut short only
# needs a resubmit. Adam moments are not checkpointed, so a restart has a brief
# warm-restart transient; a run that finished exits 0 immediately.
#
# DIVERSITY ABLATION: DIV_WEIGHT=0 turns the cosine diversity loss off (MI loss
# stays on). It gets its own SAVE_DIR suffix and wandb name -- sharing a
# SAVE_DIR with the div-on run would have auto-resume continue the WRONG
# experiment, and sharing a wandb name makes the two indistinguishable in the
# sidebar. At the default 0.1 both the path and the name are unchanged, so the
# div-on run stays bit-identical to what this file did before.
#   DIV_WEIGHT=0 bash slurm/run_mi_future.sh   # div_weight 0, whole chain
#   sbatch --export=ALL,DIV_WEIGHT=0 slurm/train_mi_future.sbatch

ROLE_DIM=${ROLE_DIM:-4}
MI_HORIZON=${MI_HORIZON:-8}

# ---- The ablation ---------------------------------------------------------
DIV_WEIGHT=${DIV_WEIGHT:-0.1}                # 0 = diversity loss OFF
MI_WEIGHT=${MI_WEIGHT:-1.0}                  # MI loss stays ON either way
case "$DIV_WEIGHT" in 0|0.|0.0|0.00) DIV_TAG=_nodiv ;; *) DIV_TAG= ;; esac

# ---- Paths: default to the submitting account's own scratch ---------------
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-$PUFFER_DIR/pufferlib/resources/drive/binaries/training}
SAVE_DIR=${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/roma_mifuture_dim${ROLE_DIM}_H${MI_HORIZON}${DIV_TAG}}

TOTAL_STEPS=${TOTAL_STEPS:-3000000000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500000000}
SEED=${SEED:-0}
NUM_MAPS=${NUM_MAPS:-10000}
# Mid-run WOSAC every 500M (lite: 10 batches). The terminal 3B tick is skipped
# in train_roma.py (redundant with the full 100-batch eval that follows).
WOSAC_INTERVAL=${WOSAC_INTERVAL:-500000000}
WOSAC_EVAL_BATCHES=${WOSAC_EVAL_BATCHES:-10}
WANDB_PROJECT=${WANDB_PROJECT:-roma-pufferdrive}
WANDB_ENTITY=${WANDB_ENTITY:-s-rahmani-tu-delft}
WANDB_NAME=${WANDB_NAME:-roma_mifutH${MI_HORIZON}_dim${ROLE_DIM}${DIV_TAG}_maps${NUM_MAPS}_${SEED}}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1

# ---- Preflight: fail loudly here, not 30s into a GPU allocation ------------
[ -d "$PUFFER_DIR" ] || { echo "ERROR: PUFFER_DIR not found: $PUFFER_DIR"; exit 1; }
[ -d "$DATA_DIR" ]   || { echo "ERROR: DATA_DIR not found: $DATA_DIR"; exit 1; }
[ -f "$CODE_DIR/roma_pufferdrive/train_roma.py" ] || {
    echo "ERROR: train_roma.py not under CODE_DIR: $CODE_DIR"; exit 1; }
# --mi_horizon only exists on the mi-future branch -- fail here rather than as
# an argparse error after the GPU is allocated.
grep -q -- "--mi_horizon" "$CODE_DIR/roma_pufferdrive/train_roma.py" || {
    echo "ERROR: this train_roma.py has no --mi_horizon -- wrong branch?"
    echo "       clone the mi-future branch and submit from that checkout."; exit 1; }
# --resume/--wandb_name were backported from baseline; a stale mi-future clone
# predates them and would die on an argparse error after the GPU is allocated.
grep -q -- "--wandb_name" "$CODE_DIR/roma_pufferdrive/train_roma.py" || {
    echo "ERROR: this train_roma.py has no --resume/--wandb_name -- stale clone?"
    echo "       git -C $CODE_DIR pull  (mi-future)"; exit 1; }
mkdir -p "$SAVE_DIR"

# ---- Auto-resume from the highest-step checkpoint in SAVE_DIR --------------
# Step checkpoints carry aux_loss_state (the MI/div critics); final.pt does
# not, so it is excluded from the search but does mark completion.
RESUME_ARGS=""
LATEST=$(ls "$SAVE_DIR"/roma_dim${ROLE_DIM}_step*.pt 2>/dev/null \
         | sed 's/.*_step\([0-9]*\)\.pt/\1 &/' | sort -k1,1n | tail -1 | cut -d' ' -f2-)
if [ -n "$LATEST" ]; then
    # Already finished? Exit 0, not 1 -- the eval hangs off this job with
    # afterok, so a non-zero exit here would take the eval down with it.
    LATEST_STEP=$(echo "$LATEST" | sed 's/.*_step\([0-9]*\)\.pt/\1/')
    if [ -f "$SAVE_DIR/roma_dim${ROLE_DIM}_final.pt" ] || [ "$LATEST_STEP" -ge "$TOTAL_STEPS" ]; then
        echo "[mifut] already at step $LATEST_STEP / $TOTAL_STEPS -- nothing to do"
        exit 0
    fi
    echo "[mifut] resuming from $LATEST (step $LATEST_STEP)"
    RESUME_ARGS="--resume $LATEST"
else
    echo "[mifut] no existing checkpoint in $SAVE_DIR -- fresh run"
fi

echo "[mifut] role_dim=$ROLE_DIM  H=$MI_HORIZON  maps=$NUM_MAPS  total_steps=$TOTAL_STEPS"
echo "[mifut] div_weight=$DIV_WEIGHT  mi_weight=$MI_WEIGHT  wandb_name=$WANDB_NAME"
echo "[mifut] code=$CODE_DIR  save=$SAVE_DIR"
echo "[mifut] branch=$(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?) commit=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo ?)"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

python "$CODE_DIR/roma_pufferdrive/train_roma.py" \
    --role_dim      "$ROLE_DIM" \
    --mi_horizon    "$MI_HORIZON" \
    --div_weight    "$DIV_WEIGHT" \
    --mi_weight     "$MI_WEIGHT" \
    --data_dir      "$DATA_DIR" \
    --num_maps      "$NUM_MAPS" \
    --num_agents    3072 \
    --device        cuda \
    --total_steps   "$TOTAL_STEPS" \
    --save_interval "$SAVE_INTERVAL" \
    --save_dir      "$SAVE_DIR" \
    --seed          "$SEED" \
    --wosac_periodic 1 \
    --wosac_interval      "$WOSAC_INTERVAL" \
    --wosac_eval_maps     10000 \
    --wosac_eval_rollouts 32 \
    --wosac_eval_batches  "$WOSAC_EVAL_BATCHES" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity  "$WANDB_ENTITY" \
    --wandb_name    "$WANDB_NAME" \
    $RESUME_ARGS \
  || { echo "[mifut] FAILED (python exit $?)"; exit 1; }

echo "[mifut] done -> $SAVE_DIR"
