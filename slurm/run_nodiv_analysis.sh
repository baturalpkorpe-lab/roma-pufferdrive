#!/bin/bash
# One command for the whole trajectory-cluster analysis of the nodiv checkpoint.
# Submits the GPU analysis (paired sweep + by-trajectory dose-response +
# consistency + role-space map + feature corr), then the trajectory-cluster
# VIDEO grid as a chained CPU job (afterok) so it does not eat a GPU allocation.
#
#   bash slurm/run_nodiv_analysis.sh
#
# Everything defaults to /scratch/$USER; export CKPT / TRAJ / SCRATCH_ROOT /
# PUFFER_DIR / TAG to override. Both jobs share the same exports.

set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:-nodiv_dim4}
# nodiv default; role_analysis.sbatch itself requires CKPT.
CKPT=${CKPT:-$SCRATCH_ROOT/checkpoints/roma_nodiv_dim4/roma_dim${ROLE_DIM:-4}_final.pt}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}

E="ALL,TAG=$TAG,SCRATCH_ROOT=$SCRATCH_ROOT"
for v in CKPT TRAJ PUFFER_DIR DATA_DIR ROLE_DIM ALPHAS SCENES_PER_TYPE TYPE_NAMES; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && E="$E,$v=$val"
done

A=$(sbatch --parsable --export="$E" "$HERE/role_analysis.sbatch")
echo "analysis (GPU) : $A"
# AXES lands in the analysis output; pass it explicitly so the render does not
# depend on default-path guessing.
AXES="$SCRATCH_ROOT/analysis/$TAG/paired/role_paired_axes.csv"
OUT="$SCRATCH_ROOT/analysis/$TAG/renders/role_grid"
B=$(sbatch --parsable --dependency=afterok:$A \
    --export="$E,AXES=$AXES,OUT=$OUT" "$HERE/render_grid_traj.sbatch")
echo "grid render (CPU, afterok:$A) : $B"
echo
echo "results -> $SCRATCH_ROOT/analysis/$TAG/"
