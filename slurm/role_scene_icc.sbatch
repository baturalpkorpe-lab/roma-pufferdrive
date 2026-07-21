#!/bin/bash
#SBATCH --job-name=role_icc
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=role_icc_%j.out
#SBATCH --error=role_icc_%j.err

# Scene-vs-agent variance decomposition of the role (role_scene_icc.py).
# Cheap: 3 natural rollouts, no forcing, no sweep.
#
#   sbatch --export=ALL,CKPT=/scratch/$USER/checkpoints/roma_baseline_dim4/roma_dim4_final.pt \
#       slurm/role_scene_icc.sbatch
#
# Run it on the CURRENT architecture first -- that number is the baseline the
# role-agent-centric changes have to beat.

set -u

CKPT=${CKPT:-}
[ -n "$CKPT" ] || { echo "ERROR: set CKPT=<checkpoint.pt>"; exit 1; }

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-pufferlib/resources/drive/binaries/training}   # rel. to PUFFER_DIR
OUT=${OUT:-$SCRATCH_ROOT/role_icc/$(basename "$(dirname "$CKPT")")}
EPISODES=${EPISODES:-3}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1

[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
mkdir -p "$OUT"
echo "[icc] ckpt=$CKPT out=$OUT"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

python "$CODE_DIR/role_scene_icc.py" \
    --checkpoint "$CKPT" \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$OUT" \
    --episodes   "$EPISODES"

echo "[icc] done -> $OUT"
