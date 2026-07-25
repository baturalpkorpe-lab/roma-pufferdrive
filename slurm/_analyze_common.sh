# Shared preflight for the analyze_*.sh launchers. Not runnable on its own --
# each launcher sets RUN_DIR / TAG / SKIP_ICC and sources this.
#
# Job #1 of every launcher: refuse to submit anything unless the run's
# final checkpoint actually exists, and say EXACTLY why when it does not --
# "training still going (last checkpoint = step N)" is a different situation
# from "no such run on this account", and neither should cost a queue slot to
# find out.

set -eu

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
ROLE_DIM=${ROLE_DIM:-4}
CKPT_DIR="$SCRATCH_ROOT/checkpoints/$RUN_DIR"
CKPT="$CKPT_DIR/roma_dim${ROLE_DIM}_final.pt"

# WAIT_FOR=<training jobid>: queue everything NOW with an afterok
# dependency so it fires the moment that training finishes. The checkpoint
# does not exist yet by definition, so the existence check is skipped --
# role_analysis.sbatch re-checks it at run time, when it must be there.
if [ -n "${WAIT_FOR:-}" ] && [ ! -f "$CKPT" ]; then
    echo "[analyze] $TAG"
    echo "[analyze] ckpt (expected) = $CKPT"
    echo "[analyze] queued with --dependency=afterok:$WAIT_FOR"
elif [ ! -f "$CKPT" ]; then
    echo "ERROR: final checkpoint not there yet: $CKPT"
    STEPS=$(ls "$CKPT_DIR"/roma_dim${ROLE_DIM}_step*.pt 2>/dev/null \
            | sed 's/.*_step\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
    if [ -n "${STEPS:-}" ]; then
        echo "       Training is STILL RUNNING or was cut short: latest step"
        echo "       checkpoint = ${STEPS}. Wait for the training job to finish"
        echo "       (squeue --me), then run this launcher again."
    elif [ -d "$CKPT_DIR" ]; then
        echo "       $CKPT_DIR exists but holds no checkpoints at all:"
        ls -la "$CKPT_DIR" || true
    else
        echo "       No such run directory. Checkpoint dirs on this account:"
        ls -d "$SCRATCH_ROOT"/checkpoints/roma_* 2>/dev/null || echo "       (none)"
    fi
    exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
echo "[analyze] $TAG"
echo "[analyze] ckpt = $CKPT"
CKPT="$CKPT" TAG="$TAG" SKIP_ICC="${SKIP_ICC:-0}" WAIT_FOR="${WAIT_FOR:-}" \
SKIP_RENDER="${SKIP_RENDER:-0}" SCRATCH_ROOT="$SCRATCH_ROOT" \
    bash "$HERE/run_role_analysis.sh"
