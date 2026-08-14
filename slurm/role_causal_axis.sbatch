#!/bin/bash
#SBATCH --job-name=causal_ax
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=causal_ax_%j.out
#SBATCH --error=causal_ax_%j.err

# Which direction in role space actually CONTROLS behaviour (role_causal_axis.py),
# plus the policy's input-weight share for the role.
#
#   sbatch --export=ALL,CKPT=/scratch/$USER/checkpoints/roma_x/roma_dim4_final.pt,TAG=x \
#       slurm/role_causal_axis.sbatch
#
# NOT a login-node job: warmup + episodes x 2 rollouts of 91 steps x 3072
# agents (14 rollouts at the defaults) -- an order more work than the scene
# ICC. Two hours is generous; it normally finishes well inside that.
#
# The policy-sensitivity readout at the top of the log needs no rollouts and
# prints first, so a job that dies later still tells you whether the policy
# reads the role at all.

set -u

CKPT=${CKPT:-}
[ -n "$CKPT" ] || { echo "ERROR: set CKPT=<checkpoint.pt>"; exit 1; }

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-pufferlib/resources/drive/binaries/training}   # rel. to PUFFER_DIR
TAG=${TAG:-$(basename "$(dirname "$CKPT")")}
OUT=${OUT:-$SCRATCH_ROOT/analysis/$TAG/causal_axis}
EPISODES=${EPISODES:-6}
WARMUP=${WARMUP:-2}
ALPHA_MAX=${ALPHA_MAX:-2.0}

set +u   # conda compiler activation scripts are not set -u safe
module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
set -u

export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1

fail=0
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; fail=1; }
[ -d "$PUFFER_DIR" ] || { echo "ERROR: PUFFER_DIR not found: $PUFFER_DIR"; fail=1; }
[ -f "$CODE_DIR/role_causal_axis.py" ] || {
    echo "ERROR: role_causal_axis.py not under CODE_DIR=$CODE_DIR"
    echo "       git fetch origin baseline && git checkout origin/baseline -- role_causal_axis.py"
    fail=1; }
[ "$fail" -eq 0 ] || { echo "[causal_ax] preflight failed."; exit 1; }

mkdir -p "$OUT"
echo "[causal_ax] ckpt=$CKPT"
echo "[causal_ax] out=$OUT episodes=$EPISODES warmup=$WARMUP alpha_max=$ALPHA_MAX"
echo "[causal_ax] branch=$(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?) commit=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo ?)"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

python "$CODE_DIR/role_causal_axis.py" \
    --checkpoint "$CKPT" \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$OUT" \
    --episodes   "$EPISODES" \
    --warmup_episodes "$WARMUP" \
    --alpha_max  "$ALPHA_MAX" \
  || { echo "[causal_ax] FAILED (python exit $?) -- see the traceback above"; exit 1; }

# Never end on a bare echo -- that is what made earlier jobs report
# COMPLETED 0:0 after writing nothing.
[ -s "$OUT/role_causal_axis.csv" ] || {
    echo "[causal_ax] FAILED: python exited 0 but wrote no role_causal_axis.csv"
    exit 1; }

echo "[causal_ax] done -> $OUT"
grep -A3 "POLICY SENSITIVITY" "$SLURM_SUBMIT_DIR/causal_ax_${SLURM_JOB_ID}.out" 2>/dev/null || true
