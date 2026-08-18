#!/bin/bash
#SBATCH --job-name=roma_anchor
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/roma_anchor_%j.out
#SBATCH --error=/scratch/e452103/logs/roma_anchor_%j.err

# ===========================================================================
# HR-PPO BC ANCHOR fine-tune: (1-lambda)*L_ppo + lambda*KL(tau||pi)
# ===========================================================================
#
#   sbatch --export=ALL,SRC=/scratch/$USER/checkpoints/<collstop_dir>,ROLE_DIM=1 \
#          slurm/train_anchor.sbatch
#
#   # sweep lambda, one job each -- the frontier is the point
#   for L in 0.02 0.05 0.1; do
#     sbatch --export=ALL,SRC=...,ROLE_DIM=1,REG_WEIGHT=$L slurm/train_anchor.sbatch
#   done
#
# WHAT IS BEING TESTED
# tau is frozen and queried on the states the POLICY visits, so this is an
# anchor and not a BC loss: a BC loss lives on dataset states, open-loop, and
# has nothing to say once the policy drifts off the logs. tau carries +1.43
# nats of conditional information (14.9 effective actions -> 3.6, top-5 0.962),
# so KL(tau||pi) is pulling pi toward state-appropriate human behaviour rather
# than toward a fixed action histogram.
#
# WHY FINE-TUNE AND NOT A FRESH RUN
# Same reason the compliance fine-tune worked: collstop already has a role wire
# worth defending. A fresh run would have the anchor fighting the policy before
# the policy had learned to use z at all.
#
# WHY LAMBDA IS FIXED, NOT ANNEALED
# nocturne_lab's reg_ppo.py multiplies reg_weight by _current_progress_remaining,
# decaying the anchor to nothing. Right for a goal-reaching result, wrong here:
# accel_ff has resisted every reward change and needs the anchor to PERSIST.
# Pass REG_ANNEAL=1 to reproduce their behaviour for comparison.
#
# THE PRE-REGISTERED QUESTION -- write the answer down before reading it
# An unconditioned anchor pulls every alpha toward one reference, so the
# expectation is: accel_ff improves, and the dial NARROWS. role_on_human.py
# predicted exactly this. If the dial cost is small, smoothness is nearly free.
# If it collapses, that IS the empirical case for the role-conditioned anchor
# tau(a|o,z) built from the human style labels already in regimes_gt.csv.
# Either outcome is a result. A run that is never swept is not.

set -u

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
SRC=${SRC:-}
TAU=${TAU:-$SCRATCH_ROOT/checkpoints/tau/tau.pt}
ROLE_DIM=${ROLE_DIM:-1}
REG_WEIGHT=${REG_WEIGHT:-0.05}
REG_ANNEAL=${REG_ANNEAL:-0}
BC_INIT=${BC_INIT:-0}
FT_STEPS=${FT_STEPS:-500000000}
SEED=${SEED:-0}
NUM_MAPS=${NUM_MAPS:-10000}
NUM_AGENTS=${NUM_AGENTS:-3072}
DIV_WEIGHT=${DIV_WEIGHT:-0}
MI_WEIGHT=${MI_WEIGHT:-1.0}
WANDB_PROJECT=${WANDB_PROJECT:-roma-pufferdrive}

[ -n "$SRC" ] || { echo "ERROR: set SRC=<checkpoint dir to fine-tune from>"; exit 1; }
[ -f "$TAU" ] || { echo "ERROR: no tau at $TAU -- run slurm/train_bc.sbatch first"; exit 1; }

module load miniconda3
# conda's activation scripts touch unset vars; set -u kills the job here.
set +u
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
set -u

CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
PUFFER_DIR=${PUFFER_DIR:-/scratch/e452103/PufferDrive}
DATA_DIR=${DATA_DIR:-$PUFFER_DIR/pufferlib/resources/drive/binaries/training}
export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1

TAG=anchor$(echo "$REG_WEIGHT" | tr -d '.')
[ "$REG_ANNEAL" = "1" ] && TAG=${TAG}_anneal
[ "$BC_INIT" = "1" ]    && TAG=${TAG}_bcinit
SAVE_DIR=${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/roma_dim${ROLE_DIM}_${TAG}}
WANDB_NAME=${WANDB_NAME:-roma_dim${ROLE_DIM}_${TAG}}

# ---- Seed from the highest STEP checkpoint, never final.pt -----------------
# Only step checkpoints carry aux_loss_state (the MI/div critics). Seeding from
# final.pt would hand the run a randomly initialised MI decoder.
SEED_CKPT=$(ls "$SRC"/roma_dim${ROLE_DIM}_step*.pt 2>/dev/null \
            | sed 's/.*_step\([0-9]*\)\.pt/\1 &/' | sort -k1,1n | tail -1 | cut -d' ' -f2-)
[ -n "$SEED_CKPT" ] || {
    echo "ERROR: no roma_dim${ROLE_DIM}_step*.pt in $SRC"
    echo "       final.pt has no aux_loss_state and must not be used as a seed."
    exit 1; }
SEED_STEP=$(echo "$SEED_CKPT" | sed 's/.*_step\([0-9]*\)\.pt/\1/')
TOTAL_STEPS=$((SEED_STEP + FT_STEPS))
SAVE_INTERVAL=$((FT_STEPS / 2))   # a mid-point checkpoint, to judge early

mkdir -p /scratch/e452103/logs "$SAVE_DIR"

# ---- Auto-resume inside SAVE_DIR, so a requeue continues ------------------
RESUME=$(ls "$SAVE_DIR"/roma_dim${ROLE_DIM}_step*.pt 2>/dev/null \
         | sed 's/.*_step\([0-9]*\)\.pt/\1 &/' | sort -k1,1n | tail -1 | cut -d' ' -f2-)
if [ -n "$RESUME" ]; then
    RESUME_STEP=$(echo "$RESUME" | sed 's/.*_step\([0-9]*\)\.pt/\1/')
    if [ "$RESUME_STEP" -ge "$TOTAL_STEPS" ]; then
        echo "[anchor] already at $RESUME_STEP / $TOTAL_STEPS -- nothing to do"; exit 0
    fi
    echo "[anchor] resuming this fine-tune from $RESUME (step $RESUME_STEP)"
else
    RESUME=$SEED_CKPT
    echo "[anchor] seeding from $SEED_CKPT (step $SEED_STEP)"
fi

ANNEAL_ARG=""; [ "$REG_ANNEAL" = "1" ] && ANNEAL_ARG="--reg_anneal"
BCINIT_ARG="";  [ "$BC_INIT"    = "1" ] && BCINIT_ARG="--bc_init"

echo "[anchor] tau         : $TAU"
echo "[anchor] reg_weight  : $REG_WEIGHT${ANNEAL_ARG:+ (annealed)}"
echo "[anchor] steps       : $SEED_STEP -> $TOTAL_STEPS (+${FT_STEPS})"
echo "[anchor] save_dir    : $SAVE_DIR"
echo "[anchor] commit      : $(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# The C binding loads pufferlib/config/ocean/drive.ini by RELATIVE path, so the
# process has to run from the PufferDrive root.
cd "$PUFFER_DIR"

python "$CODE_DIR/roma_pufferdrive/train_roma.py" \
    --resume        "$RESUME" \
    --bc_anchor     "$TAU" \
    --reg_weight    "$REG_WEIGHT" \
    $ANNEAL_ARG $BCINIT_ARG \
    --role_dim      "$ROLE_DIM" \
    --div_weight    "$DIV_WEIGHT" \
    --mi_weight     "$MI_WEIGHT" \
    --num_maps      "$NUM_MAPS" \
    --num_agents    "$NUM_AGENTS" \
    --total_steps   "$TOTAL_STEPS" \
    --device        cuda \
    --data_dir      "$DATA_DIR" \
    --save_dir      "$SAVE_DIR" \
    --save_interval "$SAVE_INTERVAL" \
    --seed          "$SEED" \
    --role_episodes 0 \
    --wosac_periodic 0 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name    "$WANDB_NAME"

cat <<MSG

VERIFY IN THE FIRST MINUTE (in the .out banner):
    [ROMA] BC anchor    : $TAU (val_acc ...)
    [ROMA] reg_weight   : $REG_WEIGHT (fixed)
    [ROMA] Resumed policy+aux from ... @ step $SEED_STEP
and then a NON-ZERO kl_loss on the first log line. kl_loss=0.0000 means the
anchor is not attached and this run is just a plain fine-tune -- cancel it.

GO / NO-GO AT THE MID-POINT ($((SEED_STEP + FT_STEPS/2))):
    sbatch --export=ALL,SWEEP=1,\\
CKPT=$SAVE_DIR/roma_dim${ROLE_DIM}_step$((SEED_STEP + FT_STEPS/2)).pt,\\
OUT=$SCRATCH_ROOT/regimes_rollout/${TAG}_mid slurm/regime_rollout.sbatch

    python ft_check.py --rollout_dir $SCRATCH_ROOT/regimes_rollout/${TAG}_mid \\
        --gt_regimes $SCRATCH_ROOT/regimes/regimes_gt.csv

READ TWO NUMBERS, and note that they can move independently:
  accel_ff   3.716 now, 2.512 human. This is what the anchor is FOR -- it has
             resisted every reward change and is role-independent.
  dial range collstop is 1.110 s. A narrower dial is the EXPECTED cost of an
             unconditioned anchor, not a bug. Below ~0.5 s the trade has gone
             too far: drop lambda, or go to the role-conditioned tau(a|o,z).
MSG
