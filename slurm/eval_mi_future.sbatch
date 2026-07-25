#!/bin/bash
#SBATCH --job-name=eval_mifua
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=eval_mifua_%j.out
#SBATCH --error=eval_mifua_%j.err

# Post-training eval for the future-MI run, in a FRESH process: small role
# analysis (--role_episodes: per-dim behaviour correlations + role-space PCA)
# then the FULL WOSAC realism eval (100 batches x 32 rollouts). Separate job on
# purpose -- doing this in the training process, on top of the 5 periodic
# Drive() instantiations, crosses the ~8th-Drive C-binding segfault. Chain it:
#   sbatch --dependency=afterok:<train_jobid> slurm/eval_mi_future.sbatch
#
# CKPT defaults to roma_dim<D>_final.pt in SAVE_DIR, falling back to the
# highest-step checkpoint (what exists if the wall clock cut training short).

ROLE_DIM=${ROLE_DIM:-4}
MI_HORIZON=${MI_HORIZON:-8}
# Mirrors train_mi_future.sbatch: the arch (role view + MI target) and the
# DIV_WEIGHT ablation both live in SAVE_DIR/OUT, so no config ever scores
# another config's checkpoint or overwrites its results. eval_roma.py rebuilds
# the role encoder at the trained width from the args dict saved in the
# checkpoint, so nothing arch-specific needs passing here.
ROLE_ROAD_DIM=${ROLE_ROAD_DIM:-16}
ROLE_PARTNER_DIM=${ROLE_PARTNER_DIM:-64}
MI_TARGET=${MI_TARGET:-ego_partner}
case "$MI_TARGET" in ego_partner) MI_TAG=ep ;; *) MI_TAG=$MI_TARGET ;; esac
ARCH_TAG=_r${ROLE_ROAD_DIM}p${ROLE_PARTNER_DIM}_${MI_TAG}
DIV_WEIGHT=${DIV_WEIGHT:-0.1}
case "$DIV_WEIGHT" in 0|0.|0.0|0.00) DIV_TAG=_nodiv ;; *) DIV_TAG= ;; esac
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
SAVE_DIR=${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/roma_mifutagent_dim${ROLE_DIM}_H${MI_HORIZON}${ARCH_TAG}${DIV_TAG}}
OUT=${OUT:-$SCRATCH_ROOT/eval_results/mifutagent_dim${ROLE_DIM}_H${MI_HORIZON}${ARCH_TAG}${DIV_TAG}}
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
[ -d "$PUFFER_DIR" ] || { echo "ERROR: PUFFER_DIR not found: $PUFFER_DIR"; exit 1; }

mkdir -p "$OUT"
echo "[eval_mifut] role_dim=$ROLE_DIM ckpt=$CKPT out=$OUT"

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
    --wandb_run_name    "eval_mifutagentH${MI_HORIZON}_dim${ROLE_DIM}${ARCH_TAG}${DIV_TAG}" \
  || { echo "[eval_mifut] FAILED (python exit $?)"; exit 1; }

echo "[eval_mifut] done -> $OUT"
