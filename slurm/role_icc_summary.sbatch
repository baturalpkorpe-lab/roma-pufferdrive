#!/bin/bash
#SBATCH --job-name=icc_sum
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=00:10:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=icc_sum_%j.out
#SBATCH --error=icc_sum_%j.err

# Collapse N scene-ICC repeats into one mean +- sd row (role_icc_summary.py).
# Pure pandas over the repeats' CSVs -- no GPU, no env, no Drive() init.
# Normally chained afterok behind the repeats by run_pairing_icc3.sh.
#
#   sbatch --export=ALL,TAG=mifuture_dim4,DIRS="/scratch/$USER/role_icc/a /scratch/$USER/role_icc/b" \
#       slurm/role_icc_summary.sbatch

set -u

TAG=${TAG:-}
DIRS=${DIRS:-}
[ -n "$TAG" ]  || { echo "ERROR: set TAG=<arm name>"; exit 1; }
[ -n "$DIRS" ] || { echo "ERROR: set DIRS=\"<dir> <dir> ...\""; exit 1; }

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
OUT=${OUT:-$SCRATCH_ROOT/role_icc/${TAG}_summary.csv}
PER_FEATURE=${PER_FEATURE:-$SCRATCH_ROOT/role_icc/${TAG}_per_feature.csv}
# One cross-arm table every run appends to, so a new arm lands next to the
# existing ones instead of in its own file nobody compares.
APPEND=${APPEND:-$SCRATCH_ROOT/role_icc/ALL_ARMS.csv}

set +u   # conda compiler activation scripts are not set -u safe
module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
set -u

export PYTHONPATH=$CODE_DIR
export PYTHONUNBUFFERED=1

# $DIRS is intentionally unquoted: it is a space-separated list for --dirs.
python "$CODE_DIR/role_icc_summary.py" \
    --tag "$TAG" --dirs $DIRS \
    --out "$OUT" --per_feature "$PER_FEATURE" --append "$APPEND" \
  || { echo "[icc_sum] FAILED (python exit $?)"; exit 1; }

[ -s "$OUT" ] || { echo "[icc_sum] FAILED: no summary written to $OUT"; exit 1; }
echo "[icc_sum] done -> $OUT"
