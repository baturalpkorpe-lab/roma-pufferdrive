#!/bin/bash
# Submit the per-cluster WOSAC + PC1 sweep for every trajectory cluster.
#
#   bash slurm/run_wosac_clusters.sh            # clusters 0,1,2,3
#   bash slurm/run_wosac_clusters.sh 2,3        # a subset
#
# One independent job per cluster (no dependencies -- they share nothing).
# Override CKPT_ROOT / MAP_ROOT / AXES_ROOT / BATCHES / CONTROL_MODE by
# exporting them first; per-cluster paths are derived from the roots.
#
# Before the first run, confirm the axes CSVs exist -- each cluster needs its
# OWN role_paired_axes.csv (PC1 direction + sigma are cluster-specific, so
# borrowing another cluster's axis silently sweeps the wrong direction):
#   ls /scratch/$USER/role_paired/cluster*/role_paired_axes.csv

set -eu

CLUSTERS=${1:-0,1,2,3}
HERE=$(cd "$(dirname "$0")" && pwd)
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
ROLE_DIM=${ROLE_DIM:-4}

for K in $(echo "$CLUSTERS" | tr ',' ' '); do
    E="ALL,CLUSTER=$K,ROLE_DIM=$ROLE_DIM,SCRATCH_ROOT=$SCRATCH_ROOT"
    for v in MAP_DIR CKPT AXES OUT ALPHAS AXIS BATCHES ROLLOUTS CONTROL_MODE PUFFER_DIR; do
        eval "val=\${$v:-}"
        [ -n "$val" ] && E="$E,$v=$val"
    done
    J=$(sbatch --parsable --export="$E" "$HERE/wosac_cluster.sbatch")
    echo "cluster $K : $J"
done
echo
echo "results -> $SCRATCH_ROOT/wosac_cluster/cluster<K>/wosac_by_alpha.csv"
