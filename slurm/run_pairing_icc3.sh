#!/bin/bash
# Role PAIRING + a ROBUST (repeated) scene ICC for one finished checkpoint.
# NO VIDEO RENDER -- statistics only.
#
#   CKPT=/scratch/someone/checkpoints/roma_x/roma_dim4_final.pt \
#   TAG=x_dim4 bash slurm/run_pairing_icc3.sh
#
# Submits REPEATS+2 jobs:
#   1. role_analysis.sbatch    [GPU]  the paired PC1/PC2 dose-response, per
#                                     trajectory cluster, + scene consistency,
#                                     role-space map, feature correlations.
#                                     Writes analysis/<TAG>_paired.tgz -- the
#                                     one file to send when only the numbers
#                                     are wanted.
#   2. role_scene_icc.sbatch   [GPU]  x REPEATS, each to its OWN out dir.
#   3. role_icc_summary.sbatch [cpu]  afterok on ALL repeats -> mean, sd, sem,
#                                     the behaviour median and the speed_mean
#                                     reference, appended to ALL_ARMS.csv.
#
# WHY REPEATS: one scene-ICC run is not a result. Measured run-to-run sd is
# 0.005-0.021 depending on the arm, and arms differ by as little as 0.01, so a
# single value cannot rank two configurations. Each repeat draws a different map
# pool; nothing else varies.
#
# The ICC jobs run on the GPU here rather than the CPU variant on purpose: this
# launcher already takes a GPU for the pairing sweep, the ICC is ~30s of it, and
# a mixed GPU/CPU submission would sit in two different queues. Use
# slurm/role_scene_icc_cpu.sbatch when ICC is all that is wanted.
#
# CROSS-ACCOUNT: CKPT may live on another user's scratch (readable), while
# everything written goes to THIS account's $SCRATCH_ROOT. PUFFER_DIR must
# exist on the account that runs this.

set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
CKPT=${CKPT:-}
TAG=${TAG:-}
ROLE_DIM=${ROLE_DIM:-4}
REPEATS=${REPEATS:-3}

if [ -z "$CKPT" ] || [ -z "$TAG" ]; then
    echo "usage: CKPT=<checkpoint.pt> TAG=<short-name> bash slurm/run_pairing_icc3.sh"
    echo
    echo "  CKPT     the .pt to analyse (may be on another account's scratch)"
    echo "  TAG      names the outputs: \$SCRATCH_ROOT/analysis/<TAG>/ and"
    echo "           \$SCRATCH_ROOT/role_icc/<TAG>_rep<N>/"
    echo "  REPEATS  scene-ICC repeats (default 3)"
    echo "  ROLE_DIM role dimensionality of the checkpoint (default 4)"
    exit 1
fi

[ -f "$CKPT" ] || { echo "ERROR: checkpoint not readable: $CKPT"; exit 1; }
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
[ -d "$PUFFER_DIR" ] || {
    echo "ERROR: PUFFER_DIR not found: $PUFFER_DIR"
    echo "       The analysis needs a PufferDrive build on THIS account."
    echo "       Set PUFFER_DIR=<path> if it lives elsewhere."; exit 1; }

E="ALL,TAG=$TAG,CKPT=$CKPT,ROLE_DIM=$ROLE_DIM,SCRATCH_ROOT=$SCRATCH_ROOT,PUFFER_DIR=$PUFFER_DIR"
for v in TRAJ DATA_DIR ALPHAS TYPE_NAMES WARMUP_EP SWEEP_EP; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && E="$E,$v=$val"
done

echo "[pairing+icc] tag=$TAG role_dim=$ROLE_DIM repeats=$REPEATS"
echo "[pairing+icc] ckpt=$CKPT"
echo

A=$(sbatch --parsable --export="$E" "$HERE/role_analysis.sbatch")
echo "role pairing (GPU) : $A   -> $SCRATCH_ROOT/analysis/$TAG/paired/"

DEPS=""
DIRS=""
i=1
while [ "$i" -le "$REPEATS" ]; do
    O="$SCRATCH_ROOT/role_icc/${TAG}_rep${i}"
    J=$(sbatch --parsable --export="$E,OUT=$O" "$HERE/role_scene_icc.sbatch")
    echo "scene ICC $i/$REPEATS (GPU): $J   -> $O/"
    DEPS="$DEPS:$J"
    DIRS="$DIRS $O"
    i=$((i + 1))
done

S=$(sbatch --parsable --dependency=afterok${DEPS} \
    --export="$E,DIRS=$DIRS" "$HERE/role_icc_summary.sbatch")
echo "ICC summary  (cpu) : $S   afterok${DEPS}"
echo

cat <<EOF
Outputs
  $SCRATCH_ROOT/analysis/${TAG}_paired.tgz        <- SEND THIS: pairing stats + figures
  $SCRATCH_ROOT/role_icc/${TAG}_summary.csv       role ICC mean / sd / sem
  $SCRATCH_ROOT/role_icc/${TAG}_per_feature.csv   per-quantity mean / sd
  $SCRATCH_ROOT/role_icc/ALL_ARMS.csv             every arm measured so far

Read in this order
  1. the accel_mask_frac panel in role_paired_PC1.png -- if it is NOT flat
     across alpha, the tail metrics are truncated differently per condition
     and are not comparable. Everything else depends on this.
  2. role_icc/${TAG}_summary.csv -- role ICC as mean +- sd, never one value
  3. role_paired_tests.csv -- alpha +2 vs -2 on the SAME (map, vehicle)
  4. role_paired_PC1.png / PC2.png -- the dose-response

The summary prints role - speed_mean. speed_mean ICC is the high-reliability
behaviour reference; the behaviour MEDIAN is dragged down by event metrics
measured from a handful of events per agent, so it flatters the role.
EOF
