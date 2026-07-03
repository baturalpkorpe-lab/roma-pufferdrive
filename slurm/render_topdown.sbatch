#!/bin/bash
#SBATCH --job-name=render_topdown
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M   # compute-p1 caps at 3968 MB/CPU
#SBATCH --time=01:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/render_topdown_%j.out
#SBATCH --error=/scratch/e452103/logs/render_topdown_%j.err

# Headless top-down MP4/GIF render — no GPU, no X server, no GLX needed.
# One-time for MP4 output (otherwise falls back to GIF automatically):
#   conda activate roma && conda install -y -c conda-forge ffmpeg

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh

# ── Stack selection: baseline (default) or main ──────────────────────────────
conda activate roma
export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
CODE_DIR=$HOME/roma_pufferdrive
PUFFER_DIR=/scratch/e452103/PufferDrive
CKPT=/scratch/e452103/checkpoints/roma_baseline/roma_dim8_step3000238080.pt
# main stack instead:
#   conda activate /scratch/e452103/envs/roma_guid
#   export PYTHONPATH=$HOME/roma_main:/scratch/e452103/PufferDrive_guid
#   CODE_DIR=$HOME/roma_main
#   PUFFER_DIR=/scratch/e452103/PufferDrive_guid
#   CKPT=/scratch/e452103/checkpoints/roma_main/<checkpoint>.pt

DATA_DIR=/scratch/e452103/PufferDrive/pufferlib/resources/drive/binaries/training
OUT_DIR=/scratch/e452103/renders/topdown_$(date +%m%d_%H%M)

mkdir -p /scratch/e452103/logs "$OUT_DIR"
# Run from the PufferDrive dir like training does: the C binding loads
# pufferlib/config/ocean/drive.ini via a RELATIVE path, so any other cwd
# floods the log with "Error while loading ..." and uses compiled defaults.
cd "$PUFFER_DIR"

# Free rollouts: watch the policy drive 4 scenarios
python "$CODE_DIR/render_topdown.py" \
    --checkpoint "$CKPT" \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$OUT_DIR" \
    --n_maps     4 \
    --mode       free

# Forced role sweep on the same 4 maps: role causality, visually
python "$CODE_DIR/render_topdown.py" \
    --checkpoint "$CKPT" \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$OUT_DIR" \
    --n_maps     4 \
    --mode       sweep

echo "Videos in: $OUT_DIR"
