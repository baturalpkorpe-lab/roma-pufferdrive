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
SEED_CKPT=${SEED_CKPT:-}     # name a checkpoint outright; overrides SRC
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
WANDB_ENTITY=${WANDB_ENTITY:-s-rahmani-tu-delft}

[ -n "$SRC$SEED_CKPT" ] || {
    echo "ERROR: set SEED_CKPT=<checkpoint.pt> (preferred) or SRC=<dir>"; exit 1; }
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

# ---- Pick the seed ---------------------------------------------------------
# SEED_CKPT names one outright and is PREFERRED. The SRC-directory fallback
# picks the HIGHEST STEP, which is not the same as the BEST checkpoint and has
# already bitten once: pointed at collstop it silently chose 5.0B, past that
# arm's peak, when the dial and PET were measured at 4.0B.
# Either way: never final.pt. Only step checkpoints carry aux_loss_state (the
# MI/div critics); final.pt would hand the run a random MI decoder.
if [ -n "$SEED_CKPT" ]; then
    [ -f "$SEED_CKPT" ] || { echo "ERROR: no such checkpoint: $SEED_CKPT"; exit 1; }
    case "$SEED_CKPT" in
        *final.pt) echo "ERROR: final.pt has no aux_loss_state -- use a step checkpoint"; exit 1 ;;
    esac
else
    SEED_CKPT=$(ls "$SRC"/roma_dim${ROLE_DIM}_step*.pt 2>/dev/null \
                | sed 's/.*_step\([0-9]*\)\.pt/\1 &/' | sort -k1,1n | tail -1 | cut -d' ' -f2-)
    [ -n "$SEED_CKPT" ] || {
        echo "ERROR: no roma_dim${ROLE_DIM}_step*.pt in $SRC"
        echo "       final.pt has no aux_loss_state and must not be used as a seed."
        exit 1; }
    echo "[anchor] WARNING: picked the HIGHEST step in $SRC, which is not"
    echo "[anchor] necessarily the best. Pass SEED_CKPT=<file.pt> to be explicit."
fi
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

# ---- Derive the role geometry FROM THE CHECKPOINT --------------------------
# The H8_r16p64 arms were trained with role_road_dim=16, role_partner_dim=64.
# Omitting them silently rebuilds the role encoder at the defaults (128-dim
# input instead of 112) and collapses role_partner_proj/role_road_proj to
# identity -- nn.Linear(...) if dim != out else None -- so they carry no
# parameters and the checkpoint's copies read as "unexpected keys". That is
# exactly how 10667202 died. Read the shapes rather than trust a directory
# name.
GEOM=$(python - "$SEED_CKPT" <<'PYGEOM'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["policy_state"]
pp, rp = sd.get("role_partner_proj.weight"), sd.get("role_road_proj.weight")
fc = sd.get("role_encoder.fc_obs.0.weight")
p = pp.shape[0] if pp is not None else 32          # None => identity, i.e. p_out
r = rp.shape[0] if rp is not None else (fc.shape[1] - 32 - p if fc is not None else 64)
print(p, r)
PYGEOM
) || { echo "ERROR: could not read role geometry from $SEED_CKPT"; exit 1; }
ROLE_PARTNER_DIM=${ROLE_PARTNER_DIM:-$(echo "$GEOM" | cut -d' ' -f1)}
ROLE_ROAD_DIM=${ROLE_ROAD_DIM:-$(echo "$GEOM" | cut -d' ' -f2)}

# Pass them only if this clone's train_roma.py understands them: baseline has
# no such flags, the agent-centric branches do.
GEOM_ARGS=""
if python "$CODE_DIR/roma_pufferdrive/train_roma.py" --help 2>/dev/null | grep -q -- --role_partner_dim; then
    GEOM_ARGS="--role_partner_dim $ROLE_PARTNER_DIM --role_road_dim $ROLE_ROAD_DIM"
    echo "[anchor] role geometry: partner=$ROLE_PARTNER_DIM road=$ROLE_ROAD_DIM (read from checkpoint)"
else
    echo "[anchor] this train_roma.py has no --role_partner_dim, so baseline geometry is assumed."
    echo "[anchor] The checkpoint wants partner=$ROLE_PARTNER_DIM road=$ROLE_ROAD_DIM; if that is"
    echo "[anchor] not 32/64 it will NOT load here. Submit from the mi-future-agentcentric clone."
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
    $ANNEAL_ARG $BCINIT_ARG $GEOM_ARGS \
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
    --wandb_entity  "$WANDB_ENTITY" \
    --wandb_name    "$WANDB_NAME"
RC=$?

# Without this the script sails past a dead python into its trailer and the
# job still reports RUNNING/COMPLETED. That is how an architecture mismatch
# looked healthy for nine minutes: state_dict load raised, python exited, and
# sacct showed RUNNING.
if [ "$RC" -ne 0 ]; then
    echo "[anchor] TRAINING FAILED (exit $RC) -- see the .err file."
    echo "[anchor] If it is a state_dict mismatch, CODE_DIR's policy.py does"
    echo "[anchor] not match the seed checkpoint's architecture. baseline has"
    echo "[anchor] neither role_partner_proj nor role_road_proj;"
    echo "[anchor] mi-future-agentcentric has both; ...-film adds film on top."
    echo "[anchor] Submit from the clone whose policy.py matches, and retrain"
    echo "[anchor] tau there too -- tau's key set is clone-specific."
    exit "$RC"
fi

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
