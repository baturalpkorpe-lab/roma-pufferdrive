#!/bin/bash
# One-command launcher for the div_weight=0 ablation.
#
#   bash slurm/run_nodiv.sh 1     # role_dim 1, div_weight 0
#   bash slurm/run_nodiv.sh 4     # role_dim 4, div_weight 0
#
# Submits the whole chain and returns: three training jobs (3B steps does not
# fit in one 24h allocation, so each picks up from the previous one's last
# checkpoint; extras exit immediately once the run is complete) followed by the
# full post-training eval -- role health check + WOSAC at 100 batches.
#
# Everything defaults to the caller's own scratch (/scratch/$USER/PufferDrive,
# /scratch/$USER/checkpoints/...). Override by exporting SCRATCH_ROOT or
# PUFFER_DIR before running, nothing else needs touching.

set -eu

D=${1:-}
case "$D" in
    1|4) ;;
    *) echo "usage: bash slurm/run_nodiv.sh <1|4>"; exit 1 ;;
esac

HERE=$(cd "$(dirname "$0")" && pwd)
E="ALL,ROLE_DIM=$D${SCRATCH_ROOT+,SCRATCH_ROOT=$SCRATCH_ROOT}${PUFFER_DIR+,PUFFER_DIR=$PUFFER_DIR}"

J=$(sbatch --parsable --export="$E" "$HERE/train_nodiv.sbatch")
echo "dim-$D train 1/3 : $J"
for i in 2 3; do
    J=$(sbatch --parsable --dependency=afterany:$J --export="$E" "$HERE/train_nodiv.sbatch")
    echo "dim-$D train $i/3 : $J"
done
E_JOB=$(sbatch --parsable --dependency=afterok:$J --export="$E" "$HERE/eval_nodiv.sbatch")
echo "dim-$D eval      : $E_JOB"
