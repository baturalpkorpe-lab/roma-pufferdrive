#!/bin/bash
#SBATCH --job-name=regime_roll
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=04:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=regime_roll_%j.out
#SBATCH --error=regime_roll_%j.err

# The POLICY side of the regime analysis: roll out, measure the three style
# parameters through the same conflict_metrics functions the GT pass used, keep
# the role per step.
#
# CPU by design. The env sim is CPU-bound and the policy forward pass is small,
# so this costs effectively nothing against fairshare -- unlike the trainings it
# is meant to be compared against.
#
#   sbatch --export=ALL,CKPT=...,AXES=...,OUT=... slurm/regime_rollout.sbatch
#
# ALPHAS=0 is the natural (unforced) reference arm and needs no AXES.
# Any other alpha shifts each agent's OWN role by alpha*sigma along AXIS,
# matching role_paired_sweep, so results stay comparable with
# role_paired_tests.csv.
#
# SUBMIT FROM THE CLONE THAT MATCHES THE CHECKPOINT. CODE_DIR defaults to the
# submit dir: a FiLM checkpoint analysed through a non-FiLM clone builds a
# policy without the FiLM modules and load_state_dict raises on the extra keys.

set -u

CKPT=${CKPT:-}
[ -n "$CKPT" ] || { echo "ERROR: set CKPT=<checkpoint.pt>"; exit 1; }

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
ZONES=${ZONES:-$SCRATCH_ROOT/regimes/junction_control_audit.csv}
OUT=${OUT:-$SCRATCH_ROOT/regimes_rollout/$(basename "$(dirname "$CKPT")")}
EPISODES=${EPISODES:-6}
MAP_POOL=${MAP_POOL:-1000}
# SLURM --export splits on commas, so a comma list of alphas is shredded into
# separate variables and you silently get a one-point "sweep". Semicolons
# survive; regime_rollout accepts either.   ALPHAS='-2;-1;0;1;2'
ALPHAS=${ALPHAS:-0}
AXIS=${AXIS:-PC1}
AXES=${AXES:-}
DATA_DIR=${DATA_DIR:-pufferlib/resources/drive/binaries/training}
EXTRA=${EXTRA:-}

# conda's gcc activation scripts are not set -u safe on e452103
set +u
module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
set -u

export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

[ -f "$CKPT" ]  || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
[ -f "$ZONES" ] || { echo "ERROR: zones csv not found: $ZONES"; exit 1; }
[ -f "$CODE_DIR/regime_rollout.py" ] || {
    echo "ERROR: no regime_rollout.py in $CODE_DIR -- fetch it:"
    echo "  git -C $CODE_DIR fetch -q origin baseline && \\"
    echo "  git -C $CODE_DIR checkout origin/baseline -- regime_rollout.py conflict_metrics.py"
    exit 1; }
# AXES is now OPTIONAL. Without it regime_rollout derives the axis from THIS
# checkpoint's own natural rollouts, which is what you want: role scales do not
# transfer between runs. A stale axes file declared sigma=729 against a measured
# 1.25, so alpha=-2 set the role ~1200 sigma out of distribution and the job ran
# three hours producing a plausible-looking table. If AXES is given,
# regime_rollout scale-checks it and aborts on a mismatch.
[ -z "$AXES" ] || [ -f "$AXES" ] || { echo "ERROR: axes csv not found: $AXES"; exit 1; }

mkdir -p "$OUT"
echo "[roll] ckpt=$CKPT"
echo "[roll] code=$CODE_DIR branch=$(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?)"
echo "[roll] alphas=$ALPHAS axis=$AXIS episodes=$EPISODES map_pool=$MAP_POOL"
echo "[roll] out=$OUT"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

AXES_ARG=""
[ -n "$AXES" ] && AXES_ARG="--axes_csv $AXES"

python "$CODE_DIR/regime_rollout.py" \
    --checkpoint "$CKPT" \
    --zones_csv  "$ZONES" \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$OUT" \
    --episodes   "$EPISODES" \
    --map_pool   "$MAP_POOL" \
    --alphas     "$ALPHAS" \
    --axis       "$AXIS" \
    --device     cpu \
    $AXES_ARG $EXTRA \
  || { echo "[roll] FAILED (python exit $?)"; exit 1; }

echo "[roll] done -> $OUT"
ls -la "$OUT"
