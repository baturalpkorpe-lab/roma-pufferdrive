#!/bin/bash
#SBATCH --job-name=roma_agentrole
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=roma_agentrole_%j.out
#SBATCH --error=roma_agentrole_%j.err

# ===========================================================================
# AGENT-CENTRIC ROLE: rebalanced role input + road-free MI target.
# ===========================================================================
#
# The point: get ONE policy over the full dataset whose role is a property of
# the driver rather than of the map, so the four per-cluster policies are no
# longer needed. Two changes, both flag-gated (see the role-agent-centric
# commit for the full argument):
#
#   --role_road_dim 16     road stays in the role encoder's input -- an agent
#   --role_partner_dim 64  should still hold back on a curve -- but stops
#                          being half of its bandwidth. The POLICY GRU keeps
#                          all 64 road dims regardless, so nothing that keeps
#                          the car inside the boundaries is touched.
#   --mi_target ego_partner
#                          drops road from the MI TARGET. mi_weight=1.0 vs
#                          div_weight=0.1 makes MI the largest gradient on the
#                          role encoder, and its default target is 50% road
#                          geometry, so the cheapest way to lower that MSE is
#                          to encode the map. The role may still SEE the road;
#                          it just stops being paid to recite it.
#
# Everything else matches the existing dim-4 baseline (roma_baseline_dim4) so
# the comparison is clean: same 3B steps, same div_weight 0.1, same mi_weight,
# same map pool, same seed. The ONLY differences are the three flags above.
#
# Usage:
#   sbatch --export=ALL,ROLE_DIM=4 slurm/train_agent_role.sbatch
# or, for the whole chain in one line:
#   bash slurm/run_agent_role.sh
#
# AUTO-RESUME + the 3-job chain + no --run_eval: same reasoning as
# train_nodiv.sbatch. 3B does not fit one 24h allocation; the in-process
# post-training eval segfaults in the C binding, so the full 100-batch WOSAC
# and the role health check run separately via slurm/eval_final.sbatch.

set -u

ROLE_DIM=${ROLE_DIM:-4}

# ---- Paths (default to the caller's own scratch) --------------------------
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-$PUFFER_DIR/pufferlib/resources/drive/binaries/training}

# ---- The architecture change ----------------------------------------------
ROLE_ROAD_DIM=${ROLE_ROAD_DIM:-16}
ROLE_PARTNER_DIM=${ROLE_PARTNER_DIM:-64}
MI_TARGET=${MI_TARGET:-ego_partner}

# ---- Everything below matches the dim-4 baseline --------------------------
DIV_WEIGHT=${DIV_WEIGHT:-0.1}
MI_WEIGHT=${MI_WEIGHT:-1.0}
TOTAL_STEPS=${TOTAL_STEPS:-3000000000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500000000}
NUM_MAPS=${NUM_MAPS:-10000}
NUM_AGENTS=${NUM_AGENTS:-3072}
SEED=${SEED:-0}

# The config goes in the path. Auto-resume matches checkpoints by filename, and
# a different role_road_dim/partner_dim builds a different-shaped role encoder,
# so sharing one SAVE_DIR across configs would have job 2 of a chain try to
# resume from weights that cannot load.
case "$MI_TARGET" in ego_partner) MI_TAG=ep ;; *) MI_TAG=$MI_TARGET ;; esac
SAVE_DIR=${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/roma_agentrole_dim${ROLE_DIM}_r${ROLE_ROAD_DIM}p${ROLE_PARTNER_DIM}_${MI_TAG}}
WANDB_PROJECT=${WANDB_PROJECT:-roma-pufferdrive}
WANDB_ENTITY=${WANDB_ENTITY:-s-rahmani-tu-delft}
WANDB_NAME=${WANDB_NAME:-roma_dim${ROLE_DIM}_agentrole_road${ROLE_ROAD_DIM}p${ROLE_PARTNER_DIM}_${MI_TARGET}_${SEED}}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1

# ---- Preflight ------------------------------------------------------------
[ -d "$PUFFER_DIR" ] || { echo "ERROR: PUFFER_DIR not found: $PUFFER_DIR"; exit 1; }
[ -d "$DATA_DIR" ]   || { echo "ERROR: DATA_DIR not found: $DATA_DIR"; exit 1; }
[ -f "$CODE_DIR/roma_pufferdrive/train_roma.py" ] || {
    echo "ERROR: train_roma.py not under CODE_DIR: $CODE_DIR"; exit 1; }
# The three flags only exist on the role-agent-centric branch. Failing here
# beats discovering it as an argparse error after the GPU is allocated.
grep -q -- "--mi_target" "$CODE_DIR/roma_pufferdrive/train_roma.py" || {
    echo "ERROR: this train_roma.py has no --mi_target -- wrong branch?"
    echo "       git checkout role-agent-centric && git pull"; exit 1; }
mkdir -p "$SAVE_DIR"

# ---- Auto-resume from the highest-step checkpoint -------------------------
# Step checkpoints carry aux_loss_state (the MI/div critics); final.pt does
# not, so it is excluded from the search but does mark completion.
RESUME_ARGS=""
LATEST=$(ls "$SAVE_DIR"/roma_dim${ROLE_DIM}_step*.pt 2>/dev/null \
         | sed 's/.*_step\([0-9]*\)\.pt/\1 &/' | sort -k1,1n | tail -1 | cut -d' ' -f2-)
if [ -n "$LATEST" ]; then
    LATEST_STEP=$(echo "$LATEST" | sed 's/.*_step\([0-9]*\)\.pt/\1/')
    if [ -f "$SAVE_DIR/roma_dim${ROLE_DIM}_final.pt" ] || [ "$LATEST_STEP" -ge "$TOTAL_STEPS" ]; then
        echo "[agentrole] already at step $LATEST_STEP / $TOTAL_STEPS -- nothing to do"
        exit 0
    fi
    echo "[agentrole] resuming from $LATEST (step $LATEST_STEP)"
    RESUME_ARGS="--resume $LATEST"
else
    echo "[agentrole] no existing checkpoint in $SAVE_DIR -- fresh run"
fi

echo "[agentrole] role_dim=$ROLE_DIM road=$ROLE_ROAD_DIM partner=$ROLE_PARTNER_DIM mi_target=$MI_TARGET"
echo "[agentrole] div_weight=$DIV_WEIGHT total_steps=$TOTAL_STEPS save=$SAVE_DIR"
echo "[agentrole] branch=$(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?)"
echo "[agentrole] commit=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo ?)"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

python "$CODE_DIR/roma_pufferdrive/train_roma.py" \
    --role_dim            "$ROLE_DIM" \
    --role_road_dim       "$ROLE_ROAD_DIM" \
    --role_partner_dim    "$ROLE_PARTNER_DIM" \
    --mi_target           "$MI_TARGET" \
    --div_weight          "$DIV_WEIGHT" \
    --mi_weight           "$MI_WEIGHT" \
    --num_maps            "$NUM_MAPS" \
    --num_agents          "$NUM_AGENTS" \
    --total_steps         "$TOTAL_STEPS" \
    --device              cuda \
    --data_dir            "$DATA_DIR" \
    --save_dir            "$SAVE_DIR" \
    --save_interval       "$SAVE_INTERVAL" \
    --seed                "$SEED" \
    --role_episodes       0 \
    --wosac_periodic      1 \
    --wosac_interval      500000000 \
    --wosac_eval_maps     10000 \
    --wosac_eval_rollouts 32 \
    --wosac_eval_batches  10 \
    --wandb_project       "$WANDB_PROJECT" \
    --wandb_entity        "$WANDB_ENTITY" \
    --wandb_name          "$WANDB_NAME" \
    $RESUME_ARGS

echo "[agentrole] done -> $SAVE_DIR"
