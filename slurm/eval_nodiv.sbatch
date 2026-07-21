#!/bin/bash
#SBATCH --job-name=eval_nodiv
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=eval_nodiv_%j.out
#SBATCH --error=eval_nodiv_%j.err

# Post-training eval for the div_weight=0 ablation: role health check
# (per-dim behaviour correlations + PCA of the role space, --role_episodes)
# followed by the FULL WOSAC realism eval (100 batches x 32 rollouts).
#
# Separate job, fresh process, on purpose: the in-process post-training eval
# inside train_roma.py segfaults in the C binding (repeated Drive()
# instantiations). Chain it behind training with --dependency=afterok.
#
#   sbatch --dependency=afterok:<train_jobid> --export=ALL,ROLE_DIM=1 \
#       slurm/eval_nodiv.sbatch
#
# CKPT defaults to roma_dim<D>_final.pt in SAVE_DIR, falling back to the
# highest-step checkpoint (which is what exists if training was still
# resuming toward TOTAL_STEPS).

set -u

ROLE_DIM=${ROLE_DIM:-}
case "$ROLE_DIM" in
    1|4) ;;
    *) echo "ERROR: set ROLE_DIM=1 or ROLE_DIM=4"; exit 1 ;;
esac

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
SAVE_DIR=${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/roma_nodiv_dim${ROLE_DIM}}
OUT=${OUT:-$SCRATCH_ROOT/eval_results/nodiv_dim${ROLE_DIM}}
MAP_DIR=${MAP_DIR:-pufferlib/resources/drive/binaries/training}   # relative to PUFFER_DIR
ROLE_EPISODES=${ROLE_EPISODES:-10}
WOSAC_MAX_BATCHES=${WOSAC_MAX_BATCHES:-100}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1
# eval_roma.py has no --wandb_entity flag; it reads the env var.
export WANDB_ENTITY=${WANDB_ENTITY:-s-rahmani-tu-delft}

# ---- Pick the checkpoint --------------------------------------------------
CKPT=${CKPT:-}
if [ -z "$CKPT" ]; then
    if [ -f "$SAVE_DIR/roma_dim${ROLE_DIM}_final.pt" ]; then
        CKPT="$SAVE_DIR/roma_dim${ROLE_DIM}_final.pt"
    else
        CKPT=$(ls "$SAVE_DIR"/roma_dim${ROLE_DIM}_step*.pt 2>/dev/null \
               | sed 's/.*_step\([0-9]*\)\.pt/\1 &/' | sort -k1,1n | tail -1 | cut -d' ' -f2-)
    fi
fi
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "ERROR: no checkpoint found in $SAVE_DIR"; exit 1; }

mkdir -p "$OUT"
echo "[eval_nodiv] role_dim=$ROLE_DIM ckpt=$CKPT out=$OUT"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

python "$CODE_DIR/eval_roma.py" \
    --checkpoint        "$CKPT" \
    --role_dim          "$ROLE_DIM" \
    --num_agents        3072 \
    --device            cuda \
    --map_dir           "$MAP_DIR" \
    --wosac_rollouts    32 \
    --wosac_num_maps    10000 \
    --wosac_max_batches "$WOSAC_MAX_BATCHES" \
    --role_episodes     "$ROLE_EPISODES" \
    --output_dir        "$OUT" \
    --wandb \
    --wandb_run_name    "eval_dim${ROLE_DIM}_nodiv"

echo "[eval_nodiv] done -> $OUT"
