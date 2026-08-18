#!/bin/bash
#SBATCH --job-name=roma_bc
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/roma_bc_%j.out
#SBATCH --error=/scratch/e452103/logs/roma_bc_%j.err

# Train tau, the frozen human reference policy for the BC anchor.
#
#   sbatch --export=ALL,DATA=/scratch/$USER/bc_probe_tight/bc_dataset.npz \
#          slurm/train_bc.sbatch
#
# GPU because it is 289k x 1121 through the ego/partner/road encoders for 20
# epochs. On a login-node CPU that is tens of minutes and against DelftBlue
# policy; on an A100 it is a few minutes.
#
# MEMORY, not compute, is the thing to watch: train_bc.py loads the whole npz
# into host RAM as float32. 289k x 1121 x 4 B ~ 1.3 GB, plus the decompressed
# copy numpy makes while reading. 16 x 4G = 64 G leaves plenty of head room.
# Note gpu-a100 caps mem-per-cpu at 8000 MB, so --mem-per-cpu=8G is rejected
# outright; 16 x 4G is the combination train_cluster.sbatch already uses.
# Raise --episodes in verify_bc_data.py far enough and host RAM is what breaks
# first, not the GPU.
#
# One hour is generous: --patience 5 usually stops it well before 20 epochs.

set -u

DATA=${DATA:-/scratch/$USER/bc_probe_tight/bc_dataset.npz}
OUT=${OUT:-/scratch/$USER/checkpoints/tau/tau.pt}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-4096}
LR=${LR:-3e-4}

[ -f "$DATA" ] || { echo "ERROR: no dataset at $DATA -- run verify_bc_data.py first"; exit 1; }

module load miniconda3
# conda's activation scripts reference unset vars; set -u kills the job here.
set +u
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
set -u

CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
export PYTHONPATH=$CODE_DIR:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1

mkdir -p /scratch/e452103/logs "$(dirname "$OUT")"

echo "[bc] data    : $DATA"
echo "[bc] out     : $OUT"
echo "[bc] node    : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python "$CODE_DIR/train_bc.py" \
    --data       "$DATA" \
    --out        "$OUT" \
    --epochs     "$EPOCHS" \
    --batch_size "$BATCH" \
    --lr         "$LR" \
    --device     cuda

echo "[bc] done. Held-out accuracy is the number that matters -- HR-PPO's BC"
echo "[bc] reference reached 0.97-0.99. Below ~0.90 tau is not a usable"
echo "[bc] anchor and the KL term would pull the policy toward noise."
