#!/bin/bash
#SBATCH --job-name=role_icc_cpu
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=01:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=role_icc_%j.out
#SBATCH --error=role_icc_%j.err

# CPU-only variant of role_scene_icc.sbatch: role_scene_icc.py already
# auto-falls-back to CPU when no GPU is visible (args.device="cuda" ->
# torch.cuda.is_available() is False -> "cpu"), so this just runs the SAME
# analysis on the `compute` partition and skips the gpu-a100 Priority backlog.
# The ICC is 3 natural rollouts with no forcing and no sweep -- the env sim is
# CPU-bound either way, so only the policy forward pass loses the GPU.
#
# Same output filename pattern as the GPU version (role_icc_%j.out), so the
# `grep -c "ICC = " role_icc_*.out` line-count check still works.
#
# REPEATS: pass a distinct OUT per run. The default OUT is derived from the
# checkpoint's folder, so submitting this twice with no OUT would overwrite
# role_scene_icc.csv and you would lose the first measurement.
#
#   for i in 1 2 3; do
#     sbatch --export=ALL,CKPT=/scratch/$USER/checkpoints/<run>/roma_dim4_final.pt,\
# OUT=/scratch/$USER/role_icc/<tag>_rep$i slurm/role_scene_icc_cpu.sbatch
#   done
#
# A 27s job asking for 1h of wall clock still queues against that 1h. Override
# with --time=00:15:00 to land in a backfill slot sooner.

set -u

CKPT=${CKPT:-}
[ -n "$CKPT" ] || { echo "ERROR: set CKPT=<checkpoint.pt>"; exit 1; }

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-pufferlib/resources/drive/binaries/training}   # rel. to PUFFER_DIR
OUT=${OUT:-$SCRATCH_ROOT/role_icc/$(basename "$(dirname "$CKPT")")}
EPISODES=${EPISODES:-3}

set +u   # conda compiler activation scripts (gcc_linux-64) are not set -u safe: SYS_SYSROOT/CONDA_BACKUP_CXX unbound
module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
set -u

export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1
# torch would otherwise spawn one thread per visible core and thrash against
# the env sim; match what SLURM actually allocated.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
mkdir -p "$OUT"
echo "[icc] ckpt=$CKPT out=$OUT (CPU / compute partition)"
echo "[icc] branch=$(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?) commit=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo ?)"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

python "$CODE_DIR/role_scene_icc.py" \
    --checkpoint "$CKPT" \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$OUT" \
    --episodes   "$EPISODES" \
    --device     cpu   || { echo "[icc] FAILED (python exit $?) -- see the traceback above"; exit 1; }

# Never end on a bare echo: jobs 10515777 / 10519983 / 10520165 all reported
# COMPLETED 0:0 after dying in ~30s because of that, and the pre-registered
# metric was silently missing.
[ -s "$OUT/role_scene_icc.csv" ] || {
    echo "[icc] FAILED: python exited 0 but wrote no role_scene_icc.csv in $OUT"
    exit 1; }

echo "[icc] done -> $OUT"
grep "role overall ICC" "$SLURM_SUBMIT_DIR/role_icc_${SLURM_JOB_ID}.out" 2>/dev/null || true
