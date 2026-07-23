#!/bin/bash
#SBATCH --job-name=wosac_mix
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=wosac_mix_%j.out
#SBATCH --error=wosac_mix_%j.err

# Idea 2: MIXTURE-of-policies WOSAC. The full WOSAC population is controlled;
# each vehicle is driven by the policy trained on ITS trajectory cluster, with
# unlabelled/non-vehicle agents on a fallback policy. Standard WOSAC otherwise.
#
# Runs on the account that HOLDS the per-cluster checkpoints (defaults ->
# /scratch/$USER). Cost is ~4x a normal WOSAC (4 policy forwards/step), so
# BATCHES defaults to 50 -- raise if the wall clock allows.
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
TRAJ=${TRAJ:-$SCRATCH_ROOT/traj_atlas/trajectory_clusters_stopfrac.csv}
OUT=${OUT:-$SCRATCH_ROOT/wosac_mixture}

FALLBACK_CKPT=${FALLBACK_CKPT:-}
FALLBACK_CLUSTER=${FALLBACK_CLUSTER:-0}
MIN_MARGIN=${MIN_MARGIN:-0}
BATCHES=${BATCHES:-50}
ROLLOUTS=${ROLLOUTS:-32}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1

# ---- Preflight ------------------------------------------------------------
fail=0
[ -d "$DATA_DIR" ] || { echo "ERROR: DATA_DIR not found: $DATA_DIR"; fail=1; }
[ -f "$TRAJ" ] || { echo "ERROR: trajectory clustering not found: $TRAJ"; fail=1; }
IFS=',' read -ra CKPT_ARR <<< "$CKPTS"
for k in "${!CKPT_ARR[@]}"; do
    [ -f "${CKPT_ARR[$k]}" ] || { echo "ERROR: cluster-$k checkpoint not found: ${CKPT_ARR[$k]}"; fail=1; }
done
[ -z "$FALLBACK_CKPT" ] || [ -f "$FALLBACK_CKPT" ] || {
    echo "ERROR: FALLBACK_CKPT set but not found: $FALLBACK_CKPT"; fail=1; }
[ "$fail" -eq 0 ] || { echo "[wosac_mix] preflight failed."; exit 1; }

mkdir -p "$OUT"
echo "[wosac_mix] clusters=$N_CLUSTERS"
for k in "${!CKPT_ARR[@]}"; do echo "[wosac_mix]   cluster $k -> ${CKPT_ARR[$k]}"; done
echo "[wosac_mix] traj=$TRAJ  out=$OUT  batches=$BATCHES"

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
    --n_clusters        "$N_CLUSTERS" \
    --role_dim          "$ROLE_DIM" \
    --traj_clusters     "$TRAJ" \
    --data_dir          "$DATA_DIR" \
    --out_dir           "$OUT" \
    --wosac_rollouts    "$ROLLOUTS" \
    --wosac_max_batches "$BATCHES" \
    --min_margin        "$MIN_MARGIN" \
    $FB_ARG

echo "[wosac_mix] done -> $OUT"
