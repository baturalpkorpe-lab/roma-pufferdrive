#!/bin/bash
#SBATCH --job-name=wosac_mix
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=wosac_mix_%j.out
#SBATCH --error=wosac_mix_%j.err

# Idea 2: MIXTURE-of-policies WOSAC, swept along each policy's PC1. Every
# VEHICLE is driven by the policy trained on ITS trajectory cluster (peds/
# cyclists stay on the GT log via control_vehicles); the rare unlabelled
# vehicle uses a fallback policy. Evaluated at 5 forcing levels along PC1
# (alpha in -1,-0.5,0,0.5,1 x sigma, per-cluster) -> WOSAC + collision/offroad/
# goal per alpha.
#
# Runs on the account that HOLDS the per-cluster checkpoints (defaults ->
# /scratch/$USER). Partitioned by policy so each condition costs like one
# normal WOSAC; 5 conditions (forced ones 2x) -> BATCHES=70 fits 24h.
#
#   sbatch slurm/wosac_mixture.sbatch
#   sbatch --export=ALL,FALLBACK_CKPT=/scratch/$USER/checkpoints/roma_baseline_dim4/roma_dim4_final.pt \
#       slurm/wosac_mixture.sbatch      # full-dataset policy as the fallback

set -u

ROLE_DIM=${ROLE_DIM:-4}
N_CLUSTERS=${N_CLUSTERS:-4}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-$PUFFER_DIR/pufferlib/resources/drive/binaries/training}

# Build the explicit per-cluster checkpoint list. cluster3 finished at 3B and
# is stored as roma_dim4_final_3B.pt (matching the per-cluster WOSAC run);
# clusters 0-2 use roma_dim4_final.pt. Override CKPTS= to change entirely.
ckpt_for() {
    local k=$1
    case "$k" in
        3) echo "$SCRATCH_ROOT/checkpoints/roma_cluster3_dim${ROLE_DIM}/roma_dim${ROLE_DIM}_final_3B.pt" ;;
        *) echo "$SCRATCH_ROOT/checkpoints/roma_cluster${k}_dim${ROLE_DIM}/roma_dim${ROLE_DIM}_final.pt" ;;
    esac
}
if [ -z "${CKPTS:-}" ]; then
    parts=()
    for k in $(seq 0 $((N_CLUSTERS-1))); do parts+=("$(ckpt_for "$k")"); done
    CKPTS=$(IFS=,; echo "${parts[*]}")
fi

# Per-cluster PC1 axes (role_paired_axes.csv). cluster3's sweep lives in
# role_paired/cluster3_3B (matching its 3B checkpoint); 0-2 in role_paired/
# cluster<K>. Override AXES= to change entirely.
axes_for() {
    local k=$1
    case "$k" in
        3) echo "$SCRATCH_ROOT/role_paired/cluster3_3B/role_paired_axes.csv" ;;
        *) echo "$SCRATCH_ROOT/role_paired/cluster${k}/role_paired_axes.csv" ;;
    esac
}
if [ -z "${AXES:-}" ]; then
    aparts=()
    for k in $(seq 0 $((N_CLUSTERS-1))); do aparts+=("$(axes_for "$k")"); done
    AXES=$(IFS=,; echo "${aparts[*]}")
fi
TRAJ=${TRAJ:-$SCRATCH_ROOT/traj_atlas/trajectory_clusters_stopfrac.csv}
OUT=${OUT:-$SCRATCH_ROOT/wosac_mixture}

FALLBACK_CKPT=${FALLBACK_CKPT:-}
FALLBACK_CLUSTER=${FALLBACK_CLUSTER:-0}
BATCHES=${BATCHES:-70}
ROLLOUTS=${ROLLOUTS:-32}

set +u   # conda compiler activation scripts (gcc_linux-64) are not set -u safe: SYS_SYSROOT/CONDA_BACKUP_CXX unbound
module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
set -u
export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1

# ---- Preflight ------------------------------------------------------------
fail=0
[ -d "$DATA_DIR" ] || { echo "ERROR: DATA_DIR not found: $DATA_DIR"; fail=1; }
[ -f "$TRAJ" ] || { echo "ERROR: trajectory clustering not found: $TRAJ"; fail=1; }
IFS=',' read -ra CKPT_ARR <<< "$CKPTS"
IFS=',' read -ra AXES_ARR <<< "$AXES"
for k in "${!CKPT_ARR[@]}"; do
    [ -f "${CKPT_ARR[$k]}" ] || { echo "ERROR: cluster-$k checkpoint not found: ${CKPT_ARR[$k]}"; fail=1; }
done
for k in "${!AXES_ARR[@]}"; do
    [ -f "${AXES_ARR[$k]}" ] || { echo "ERROR: cluster-$k axes csv not found: ${AXES_ARR[$k]}"; fail=1; }
done
[ -z "$FALLBACK_CKPT" ] || [ -f "$FALLBACK_CKPT" ] || {
    echo "ERROR: FALLBACK_CKPT set but not found: $FALLBACK_CKPT"; fail=1; }
[ "$fail" -eq 0 ] || { echo "[wosac_mix] preflight failed."; exit 1; }

mkdir -p "$OUT"
echo "[wosac_mix] clusters=$N_CLUSTERS  batches=$BATCHES  out=$OUT"
for k in "${!CKPT_ARR[@]}"; do
    echo "[wosac_mix]   cluster $k -> $(basename "$(dirname "${CKPT_ARR[$k]}")")/$(basename "${CKPT_ARR[$k]}")  axes $(basename "$(dirname "${AXES_ARR[$k]}")")"
done

FB_ARG=""
if [ -n "$FALLBACK_CKPT" ]; then
    FB_ARG="--fallback_ckpt $FALLBACK_CKPT"
else
    FB_ARG="--fallback_cluster $FALLBACK_CLUSTER"
fi

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

python "$CODE_DIR/role_mixture_wosac.py" \
    --ckpts             "$CKPTS" \
    --axes              "$AXES" \
    --n_clusters        "$N_CLUSTERS" \
    --role_dim          "$ROLE_DIM" \
    --alphas="${ALPHAS:--1,-0.5,0,0.5,1}" \
    --control_mode      "${CONTROL_MODE:-control_vehicles}" \
    --traj_clusters     "$TRAJ" \
    --data_dir          "$DATA_DIR" \
    --out_dir           "$OUT" \
    --wosac_rollouts    "$ROLLOUTS" \
    --wosac_max_batches "$BATCHES" \
    $FB_ARG || { echo "[wosac_mix] FAILED (python exit $?)"; exit 1; }

echo "[wosac_mix] done -> $OUT"
