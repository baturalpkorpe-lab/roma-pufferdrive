#!/bin/bash
# Run the REAL training loop, tiny, on CPU, before spending a GPU allocation.
#
#   bash slurm/smoke_test.sh                 # default: FiLM + compliance
#   ROLE_FILM=0 bash slurm/smoke_test.sh     # plain branch
#
# WHY THIS EXISTS: two bugs on this branch (step_idx, then T_steps) were plain
# NameError/UnboundLocalError in the update path -- invisible to unit tests of
# the maths, fatal 5 seconds into a 24h job, and only reachable by actually
# executing a rollout AND an update. This does exactly that at a size that runs
# on a login node in under a minute.
#
# It is a CRASH test, not a correctness test: it proves the code path executes,
# not that the model learns. Correctness lives in the unit checks.

set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
CODE_DIR=${CODE_DIR:-$(cd "$HERE/.." && pwd)}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
DATA_DIR=${DATA_DIR:-pufferlib/resources/drive/binaries/training}
OUT=${OUT:-${TMPDIR:-/tmp}/roma_smoke_$$}

ROLE_DIM=${ROLE_DIM:-4}
ROLE_FILM=${ROLE_FILM:-1}
COMPLIANCE_WEIGHT=${COMPLIANCE_WEIGHT:-0.05}
PERTURB_FRAC=${PERTURB_FRAC:-0.15}

# Small enough to finish fast, big enough to complete SEVERAL rollout+update
# cycles -- one full cycle is the point, since that is where both bugs lived.
AGENTS=${AGENTS:-16}
ROLLOUT=${ROLLOUT:-32}
MAPS=${MAPS:-100}
STEPS=${STEPS:-4096}          # = 8 rollouts at 16 agents x 32 steps

case "$ROLE_FILM" in 1|true|yes) FILM_ARG="--role_film" ;; *) FILM_ARG="" ;; esac

set +u
module purge 2>/dev/null || true
module load miniconda3 2>/dev/null || true
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh 2>/dev/null || true
conda activate roma 2>/dev/null || true
set -u

export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1

[ -d "$PUFFER_DIR" ] || { echo "ERROR: PUFFER_DIR not found: $PUFFER_DIR"; exit 1; }
mkdir -p "$OUT"
cd "$PUFFER_DIR"

echo "[smoke] code=$CODE_DIR"
echo "[smoke] branch=$(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?) commit=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo ?)"
echo "[smoke] film=${FILM_ARG:-off} compliance=$COMPLIANCE_WEIGHT perturb=$PERTURB_FRAC"
echo "[smoke] $AGENTS agents x $ROLLOUT rollout_steps, $STEPS total steps (CPU)"
echo

python "$CODE_DIR/roma_pufferdrive/train_roma.py" \
    --role_dim      "$ROLE_DIM" \
    --num_agents    "$AGENTS" \
    --num_maps      "$MAPS" \
    --rollout_steps "$ROLLOUT" \
    --total_steps   "$STEPS" \
    --num_minibatch 2 \
    --device        cpu \
    --no_amp \
    --data_dir      "$DATA_DIR" \
    --save_dir      "$OUT" \
    --save_interval "$STEPS" \
    $FILM_ARG \
    --compliance_weight "$COMPLIANCE_WEIGHT" \
    --perturb_frac      "$PERTURB_FRAC" \
  || { echo; echo "[smoke] *** FAILED *** -- do NOT submit to the GPU."; exit 1; }

echo
echo "[smoke] PASSED -- rollout + PPO update executed cleanly. Safe to submit."
rm -rf "$OUT"
