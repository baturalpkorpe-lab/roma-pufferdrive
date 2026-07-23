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
# The per-cluster axes file and checkpoint are resolved below and echoed with
# each job id -- check that line before walking away. The sbatch hard-fails if
# either path is missing rather than falling back to something plausible.

set -eu

CLUSTERS=${1:-0,1,2,3}
HERE=$(cd "$(dirname "$0")" && pwd)
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
ROLE_DIM=${ROLE_DIM:-4}

# Per-cluster axes + checkpoint. Hardcoded because the directory names do not
# follow one pattern and the wrong axes file is a SILENT failure: PC1's
# direction and sigma are cluster-specific, so borrowing another cluster's
# axis sweeps a direction that means nothing for this policy and every number
# downstream is quietly wrong. cluster3 has no plain dir at all -- it is
# _3B/_2B/_*_only -- and its final checkpoint carries a _3B suffix.
axes_for() {
    case "$1" in
        3) echo "$SCRATCH_ROOT/role_paired/cluster3_3B/role_paired_axes.csv" ;;
        *) echo "$SCRATCH_ROOT/role_paired/cluster$1/role_paired_axes.csv" ;;
    esac
}
ckpt_for() {
    case "$1" in
        3) echo "$SCRATCH_ROOT/checkpoints/roma_cluster3_dim${ROLE_DIM}/roma_dim${ROLE_DIM}_final_3B.pt" ;;
        *) echo "$SCRATCH_ROOT/checkpoints/roma_cluster$1_dim${ROLE_DIM}/roma_dim${ROLE_DIM}_final.pt" ;;
    esac
}

for K in $(echo "$CLUSTERS" | tr ',' ' '); do
    E="ALL,CLUSTER=$K,ROLE_DIM=$ROLE_DIM,SCRATCH_ROOT=$SCRATCH_ROOT"
    E="$E,AXES=${AXES:-$(axes_for "$K")}"
    E="$E,CKPT=${CKPT:-$(ckpt_for "$K")}"
    for v in MAP_DIR OUT ALPHAS AXIS BATCHES ROLLOUTS CONTROL_MODE PUFFER_DIR; do
        eval "val=\${$v:-}"
        [ -n "$val" ] && E="$E,$v=$val"
    done
    J=$(sbatch --parsable --export="$E" "$HERE/wosac_cluster.sbatch")
    # Print the PARENT DIR of each path, not just the basename: every cluster's
    # checkpoint is named roma_dim<D>_final.pt and only the directory
    # distinguishes them, so a basename-only echo looks like all four jobs
    # share one checkpoint.
    a="${AXES:-$(axes_for "$K")}"; c="${CKPT:-$(ckpt_for "$K")}"
    echo "cluster $K : $J"
    echo "    axes  $(basename "$(dirname "$a")")/$(basename "$a")"
    echo "    ckpt  $(basename "$(dirname "$c")")/$(basename "$c")"
done
echo
echo "results -> $SCRATCH_ROOT/wosac_cluster/cluster<K>/wosac_by_alpha.csv"
