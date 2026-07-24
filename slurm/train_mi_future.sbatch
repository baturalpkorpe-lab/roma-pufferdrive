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
# No --run_eval, --wosac_periodic 0: repeated Drive() instantiations in one
# long process segfault the C binding. WOSAC runs standalone afterwards.
#
# 2B steps, single job, no auto-resume (this branch's train_roma.py has no
# --resume). A dim-4 run reaches ~2B in one 24h allocation; if the wall clock
# cuts it short the last checkpoint (every 100M) is still saved, but a rerun
# starts fresh -- so keep TOTAL_STEPS at a value that fits 24h.

ROLE_DIM=${ROLE_DIM:-4}
MI_HORIZON=${MI_HORIZON:-8}

# ---- Paths: default to the submitting account's own scratch ---------------
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-$PUFFER_DIR/pufferlib/resources/drive/binaries/training}
SAVE_DIR=${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/roma_mifuture_dim${ROLE_DIM}_H${MI_HORIZON}}

TOTAL_STEPS=${TOTAL_STEPS:-2000000000}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000000}
SEED=${SEED:-0}
NUM_MAPS=${NUM_MAPS:-10000}
WANDB_PROJECT=${WANDB_PROJECT:-roma-pufferdrive}
WANDB_ENTITY=${WANDB_ENTITY:-s-rahmani-tu-delft}

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
mkdir -p "$SAVE_DIR"

echo "[mifut] role_dim=$ROLE_DIM  H=$MI_HORIZON  maps=$NUM_MAPS  total_steps=$TOTAL_STEPS"
echo "[mifut] code=$CODE_DIR  save=$SAVE_DIR"
echo "[mifut] branch=$(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?) commit=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo ?)"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

python "$CODE_DIR/roma_pufferdrive/train_roma.py" \
    --role_dim      "$ROLE_DIM" \
    --mi_horizon    "$MI_HORIZON" \
    --data_dir      "$DATA_DIR" \
    --num_maps      "$NUM_MAPS" \
    --num_agents    3072 \
    --device        cuda \
    --total_steps   "$TOTAL_STEPS" \
    --save_interval "$SAVE_INTERVAL" \
    --save_dir      "$SAVE_DIR" \
    --seed          "$SEED" \
    --wosac_periodic 0 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity  "$WANDB_ENTITY" \
  || { echo "[mifut] FAILED (python exit $?)"; exit 1; }

echo "[mifut] done -> $SAVE_DIR"
