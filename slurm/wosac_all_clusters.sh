#!/bin/bash
# Submit the per-cluster WOSAC PC1 sweep for ALL FOUR clusters in one command.
#
#   bash slurm/wosac_all_clusters.sh
#   BATCHES=20 bash slurm/wosac_all_clusters.sh      # faster, fewer scenarios
#   CLUSTERS="0 3" bash slurm/wosac_all_clusters.sh  # just these two
#
# Each cluster is one wosac_cluster.sbatch job: 40 map batches x 5 alphas x 32
# rollouts, ~10h, GPU. All four run in parallel if the queue allows.
#
# WHY THIS EXISTS: the four jobs have to be submitted with IDENTICAL settings
# or the cross-cluster comparison is not a comparison. Doing it by hand is how
# one cluster ends up on a different batch count or control mode than the rest.
#
# PREFLIGHT PER CLUSTER (all inside the sbatch, before the GPU does anything):
#   - checkpoint / cluster maps / axes CSV exist
#   - the cluster maps actually contain mark_as_expert agents. Without them
#     control_agents has nothing to exclude, the whole scene goes under policy
#     control, and the per-cluster eval is meaningless. That is the failure the
#     earlier cluster0 run hit -- it reported control_agents and still measured
#     controlled_frac = 1.0, then ran for 17h.
# And role_cluster_wosac.py now ABORTS after batch 1 if it measures ~100%
# control, so the worst case costs ~2 minutes instead of the wall clock.

set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
CLUSTERS=${CLUSTERS:-0 1 2 3}
BATCHES=${BATCHES:-40}
ROLLOUTS=${ROLLOUTS:-32}
ALPHAS=${ALPHAS:--1,-0.5,0,0.5,1}
AXIS=${AXIS:-PC1}
ROLE_DIM=${ROLE_DIM:-4}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}

echo "[all_clusters] clusters='$CLUSTERS' batches=$BATCHES rollouts=$ROLLOUTS"
echo "[all_clusters] axis=$AXIS alphas=$ALPHAS role_dim=$ROLE_DIM"
echo

ids=""
for c in $CLUSTERS; do
    CKPT="$SCRATCH_ROOT/checkpoints/roma_cluster${c}_dim${ROLE_DIM}/roma_dim${ROLE_DIM}_final.pt"
    MAPS="$SCRATCH_ROOT/cluster_maps/cluster${c}"
    AXES="$SCRATCH_ROOT/role_paired/cluster${c}/role_paired_axes.csv"
    miss=""
    [ -f "$CKPT" ] || miss="$miss ckpt"
    [ -d "$MAPS" ] || miss="$miss maps"
    [ -f "$AXES" ] || miss="$miss axes"
    if [ -n "$miss" ]; then
        echo "cluster $c: SKIPPED -- missing:$miss"
        echo "    ckpt = $CKPT"
        echo "    maps = $MAPS"
        echo "    axes = $AXES"
        continue
    fi
    j=$(sbatch --parsable \
        --export="ALL,CLUSTER=$c,BATCHES=$BATCHES,ROLLOUTS=$ROLLOUTS,ALPHAS=$ALPHAS,AXIS=$AXIS,ROLE_DIM=$ROLE_DIM,SCRATCH_ROOT=$SCRATCH_ROOT" \
        "$HERE/wosac_cluster.sbatch")
    ids="$ids $j"
    echo "cluster $c: job $j -> $SCRATCH_ROOT/wosac_cluster/cluster${c}/"
done

echo
[ -n "$ids" ] || { echo "[all_clusters] nothing submitted."; exit 1; }
echo "[all_clusters] submitted:$ids"
echo
echo "Watch the expert check and the batch-1 abort first -- both fire early:"
echo "  grep -H 'expert check\|CONTROLLED\|ABORT' wosac_cl*.out"
echo
echo "When they finish, compare all four:"
echo "  head -1 $SCRATCH_ROOT/wosac_cluster/cluster0/wosac_by_alpha.csv"
echo "  for c in $CLUSTERS; do echo \"-- cluster \$c\"; cat $SCRATCH_ROOT/wosac_cluster/cluster\$c/wosac_by_alpha.csv; done"
