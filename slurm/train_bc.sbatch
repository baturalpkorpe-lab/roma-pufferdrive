#!/bin/bash
#SBATCH --job-name=roma_bc
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
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
# SIZED SMALL ON PURPOSE, hence gpu-a100-small (one node, gpu020, 4h cap).
# Host RAM peaks near 3 GB:
# the npz obs array is 289k x 1121 x 4 B ~ 1.3 GB, torch.from_numpy SHARES
# that buffer rather than copying it, the train/val split is a view, and each
# minibatch is an 18 MB gather. The model is ~100k params, so GPU memory is
# nowhere near a MIG slice's limit either. 2 x 4G = 8 G is still 2.7x head
# room. Raise --episodes in verify_bc_data.py far enough and host RAM is what
# breaks first, not the GPU; if this ever OOMs, raise --mem-per-cpu, not CPUs.
#
# PARTITION CAPS differ, are not guessable, and each one only tells you about
# itself. Measured on DelftBlue, the hard way:
#   gpu-a100-small   max 2 CPUs-per-task, 4h, ONE node (gpu020)
#   gpu-a100         mem-per-cpu caps at 8000 MB (so 8G is rejected), 2d, 9 nodes
#   compute-p1       mem-per-cpu caps at 3968 MB
# Check before changing:
#   scontrol show partition gpu-a100-small | grep -iE "maxmem|maxtime|maxnodes"
# sbatch CLI flags override the directives above, so you can retarget without
# editing this file:
#   sbatch --partition=gpu-a100 --time=01:00:00 --export=ALL,DATA=... slurm/train_bc.sbatch
#
# Half an hour is generous: --patience 5 usually stops it well before 20
# epochs, and on a MIG slice the epochs are still seconds, not minutes.

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
