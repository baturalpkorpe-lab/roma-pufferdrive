#!/bin/bash
# One command for the FULL role analysis of any finished checkpoint.
#
#   CKPT=/scratch/$USER/checkpoints/roma_agentrole_dim4_r16p64_ep/roma_dim4_final.pt \
#   TAG=agentrole_dim4 bash slurm/run_role_analysis.sh
#
#   # the control it has to be read against:
#   CKPT=/scratch/$USER/checkpoints/roma_baseline_dim4/roma_dim4_final.pt \
#   TAG=baseline_dim4 bash slurm/run_role_analysis.sh
#
# Submits THREE jobs:
#   1. role_analysis.sbatch    [GPU]  paired PC1/PC2 dose-response with the
#                                     TAIL + BRAKING-EVENT metrics, split by
#                                     trajectory cluster, + scene consistency,
#                                     role-space PCA map, feature correlations
#   2. render_grid_traj.sbatch [CPU]  afterok - the PC1xPC2 forcing VIDEO grid,
#                                     scenes_per_type per trajectory cluster.
#                                     compute partition on purpose, so 16h of
#                                     rendering does not sit on a GPU.
#   3. role_scene_icc.sbatch   [GPU]  scene ICC: what fraction of role variance
#                                     is explained by WHICH SCENE the agent is
#                                     in. The pre-registered map-independence
#                                     metric. Cheap (3 rollouts, no sweep), so
#                                     it runs in parallel, not chained.
#
# Everything defaults to /scratch/$USER and reads the code from the submitting
# clone, so this runs unchanged on any account -- clone the branch, then run it.
#
# SET SKIP_RENDER=1 to skip the video grid (statistics only, much faster).

set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
ROLE_DIM=${ROLE_DIM:-4}
CKPT=${CKPT:-}
TAG=${TAG:-}

if [ -z "$CKPT" ] || [ -z "$TAG" ]; then
    echo "usage: CKPT=<checkpoint.pt> TAG=<short-name> bash slurm/run_role_analysis.sh"
    echo
    echo "  CKPT  the .pt to analyse (roma_dim<D>_final.pt)"
    echo "  TAG   names the output folder: \$SCRATCH_ROOT/analysis/<TAG>"
    exit 1
fi
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }

ROOT=$SCRATCH_ROOT/analysis/$TAG
E="ALL,TAG=$TAG,CKPT=$CKPT,ROLE_DIM=$ROLE_DIM,SCRATCH_ROOT=$SCRATCH_ROOT"
for v in TRAJ PUFFER_DIR DATA_DIR ALPHAS SCENES_PER_TYPE TYPE_NAMES \
         WARMUP_EP SWEEP_EP; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && E="$E,$v=$val"
done

A=$(sbatch --parsable --export="$E" "$HERE/role_analysis.sbatch")
echo "analysis  (GPU) : $A   -> $ROOT/paired/"

if [ "${SKIP_RENDER:-0}" != "1" ]; then
    # Pass AXES/OUT explicitly rather than relying on default-path guessing --
    # the render must force the SAME directions the statistics used.
    AXES="$ROOT/paired/role_paired_axes.csv"
    ROUT="$ROOT/renders/role_grid"
    B=$(sbatch --parsable --dependency=afterok:$A \
        --export="$E,AXES=$AXES,OUT=$ROUT,ROOT=$ROOT" \
        "$HERE/render_grid_traj.sbatch")
    echo "videos    (CPU) : $B   afterok:$A -> $ROUT/  (+ zips the whole folder)"
else
    echo "videos          : skipped (SKIP_RENDER=1)"
fi

if [ "${SKIP_ICC:-0}" != "1" ]; then
    I=$(sbatch --parsable \
        --export="$E,OUT=$SCRATCH_ROOT/role_icc/$TAG" \
        "$HERE/role_scene_icc.sbatch")
    echo "scene ICC (GPU) : $I   -> $SCRATCH_ROOT/role_icc/$TAG/"
else
    # For runs whose training chain ALREADY submitted a scene-ICC job
    # (run_agent_role.sh / the combined run_mi_future.sh chain it afterok) --
    # resubmitting it here would burn a GPU on a duplicate.
    echo "scene ICC       : skipped (SKIP_ICC=1 -- already chained to the training job)"
fi

echo
echo "checkpoint -> $CKPT"
echo "results    -> $ROOT/   (README.txt inside explains every file)"
echo
echo "Read in this order:"
echo "  1. the accel_mask_frac panel in role_paired_PC1.png -- if it is NOT"
echo "     flat across alpha, the tail metrics are truncated differently per"
echo "     condition and are not comparable. Everything else depends on this."
echo "  2. role_paired_tests.csv  -- alpha +2 vs -2 on the SAME (map,vehicle)"
echo "  3. role_paired_traj_PC1.png -- the same split by trajectory cluster"
echo "  4. role_icc/$TAG/ -- compare against the baseline_dim4 ICC"
