#!/bin/bash
#SBATCH --job-name=roma_mifua
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=roma_mifua_%j.out
#SBATCH --error=roma_mifua_%j.err

# FUTURE-MI x AGENT-CENTRIC training (mi-future-agentcentric branch, dim=4):
# the two role fixes COMBINED, each attacking a different leg of the
# map-determined-role problem.
#
#   WHEN the MI target is (mi-future): z_t must predict the behaviour summary
#   of the NEXT 8 env embeddings (t+1..t+8) instead of the past window --
#   R3DM-inspired; the role becomes an INTENT signal rather than a recap.
#
#   WHAT the role sees / is paid for (role-agent-centric):
#     ROLE_ROAD_DIM=16 ROLE_PARTNER_DIM=64 -- road shrinks from 64/128 to
#       16/112 of the role encoder's input, partner doubles; the POLICY GRU
#       keeps the full 128 either way, so road-keeping is untouched.
#     MI_TARGET=ego_partner -- road is dropped from the MI TARGET: the role
#       must predict upcoming EGO+PARTNER dynamics, not upcoming road
#       geometry (which is a near-deterministic function of the scene and
#       was the cheapest thing to encode).
#
# Full 10k map set, standard self-play (NOT per-cluster maps). Runs on the
# account that submits it (defaults resolve to /scratch/$USER):
#   sbatch slurm/train_mi_future.sbatch
#   sbatch --export=ALL,MI_HORIZON=16 slurm/train_mi_future.sbatch
#   sbatch --export=ALL,MI_TARGET=full,ROLE_ROAD_DIM=64,ROLE_PARTNER_DIM=32 \
#       slurm/train_mi_future.sbatch     # plain mi-future, for comparison
#
# 3B steps. Mid-run WOSAC every 500M (lite, 10 batches). The FULL 100-batch
# WOSAC + role analysis runs as a SEPARATE job (eval_mi_future.sbatch, chained
# afterok via run_mi_future.sh) -- NOT --run_eval: the in-process post-training
# eval plus 5 periodic Drive() instantiations would cross the ~8th-Drive C
# binding segfault. 5 periodic evals alone (the terminal 3B tick is skipped in
# train_roma.py) stay under that.
#
# Submitted as ONE job: 3B fits the 24h allocation. AUTO-RESUME is here as a
# RECOVERY path, not a chain -- if SAVE_DIR already holds roma_dim<D>_step*.pt
# the highest-step one is resumed, so a run the wall clock did cut short only
# needs a resubmit. Adam moments are not checkpointed, so a restart has a brief
# warm-restart transient; a run that finished exits 0 immediately.
#
# DIVERSITY ABLATION: DIV_WEIGHT=0 turns the cosine diversity loss off (MI loss
# stays on). It gets its own SAVE_DIR suffix and wandb name -- sharing a
# SAVE_DIR with the div-on run would have auto-resume continue the WRONG
# experiment, and sharing a wandb name makes the two indistinguishable in the
# sidebar. At the default 0.1 both the path and the name are unchanged, so the
# div-on run stays bit-identical to what this file did before.
#   DIV_WEIGHT=0 bash slurm/run_mi_future.sh   # div_weight 0, whole chain
#   sbatch --export=ALL,DIV_WEIGHT=0 slurm/train_mi_future.sbatch

ROLE_DIM=${ROLE_DIM:-4}
MI_HORIZON=${MI_HORIZON:-8}

# ---- The agent-centric role view (this branch's defaults) ------------------
ROLE_ROAD_DIM=${ROLE_ROAD_DIM:-16}
ROLE_PARTNER_DIM=${ROLE_PARTNER_DIM:-64}
MI_TARGET=${MI_TARGET:-ego_partner}
case "$MI_TARGET" in ego_partner) MI_TAG=ep ;; *) MI_TAG=$MI_TARGET ;; esac
# FiLM adds modules to the policy, so a FiLM and a non-FiLM run MUST NOT share
# a SAVE_DIR -- resuming across them would try to load a state dict with
# missing/unexpected keys. It goes in the arch tag for the same reason the role
# view does.
# Compliance = reward the policy for behaving the way its own role predicts.
# Perturbation without compliance teaches the policy to IGNORE the role, so the
# two are tagged together and guarded below.
COMPLIANCE_WEIGHT=${COMPLIANCE_WEIGHT:-0}
PERTURB_FRAC=${PERTURB_FRAC:-0}
PERTURB_ALPHA=${PERTURB_ALPHA:-2.0}
COMPLIANCE_PERTURBED_ONLY=${COMPLIANCE_PERTURBED_ONLY:-1}
case "$COMPLIANCE_WEIGHT" in 0|0.|0.0|0.00) COMPLY_TAG= ;; *) COMPLY_TAG=_comply ;; esac
ROLE_FILM=${ROLE_FILM:-0}
case "$ROLE_FILM" in 1|true|yes) FILM_ARG="--role_film"; FILM_TAG=_film ;;
                     *)          FILM_ARG=""          ; FILM_TAG=      ;; esac
ARCH_TAG=_r${ROLE_ROAD_DIM}p${ROLE_PARTNER_DIM}_${MI_TAG}${FILM_TAG}${COMPLY_TAG}

# ---- The ablation ---------------------------------------------------------
DIV_WEIGHT=${DIV_WEIGHT:-0.1}                # 0 = diversity loss OFF
MI_WEIGHT=${MI_WEIGHT:-1.0}                  # MI loss stays ON either way
case "$DIV_WEIGHT" in 0|0.|0.0|0.00) DIV_TAG=_nodiv ;; *) DIV_TAG= ;; esac

# ---- Paths: default to the submitting account's own scratch ---------------
# Both the arch and the ablation go in the path: auto-resume matches
# checkpoints by filename, and a different role view builds a different-shaped
# role encoder, so two configs sharing one SAVE_DIR would resume from weights
# that cannot load (or silently continue the wrong experiment).
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-$PUFFER_DIR/pufferlib/resources/drive/binaries/training}
SAVE_DIR=${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/roma_mifutagent_dim${ROLE_DIM}_H${MI_HORIZON}${ARCH_TAG}${DIV_TAG}}

TOTAL_STEPS=${TOTAL_STEPS:-3000000000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500000000}
SEED=${SEED:-0}
NUM_MAPS=${NUM_MAPS:-10000}
# Mid-run WOSAC every 500M (lite: 10 batches). The terminal 3B tick is skipped
# in train_roma.py (redundant with the full 100-batch eval that follows).
WOSAC_INTERVAL=${WOSAC_INTERVAL:-500000000}
WOSAC_EVAL_BATCHES=${WOSAC_EVAL_BATCHES:-10}
WANDB_PROJECT=${WANDB_PROJECT:-roma-pufferdrive}
WANDB_ENTITY=${WANDB_ENTITY:-s-rahmani-tu-delft}
WANDB_NAME=${WANDB_NAME:-roma_mifutagentH${MI_HORIZON}_dim${ROLE_DIM}${ARCH_TAG}${DIV_TAG}_maps${NUM_MAPS}_${SEED}}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1

# ---- Preflight: fail loudly here, not 30s into a GPU allocation ------------
[ -d "$PUFFER_DIR" ] || { echo "ERROR: PUFFER_DIR not found: $PUFFER_DIR"; exit 1; }
[ -d "$DATA_DIR" ]   || { echo "ERROR: DATA_DIR not found: $DATA_DIR"; exit 1; }
[ -f "$CODE_DIR/roma_pufferdrive/train_roma.py" ] || {
    echo "ERROR: train_roma.py not under CODE_DIR: $CODE_DIR"; exit 1; }
# --mi_horizon only exists on the mi-future branch -- fail here rather than as
# an argparse error after the GPU is allocated.
grep -q -- "--mi_horizon" "$CODE_DIR/roma_pufferdrive/train_roma.py" || {
    echo "ERROR: this train_roma.py has no --mi_horizon -- wrong branch?"
    echo "       clone the mi-future branch and submit from that checkout."; exit 1; }
# --resume/--wandb_name were backported from baseline; a stale mi-future clone
# predates them and would die on an argparse error after the GPU is allocated.
grep -q -- "--wandb_name" "$CODE_DIR/roma_pufferdrive/train_roma.py" || {
    echo "ERROR: this train_roma.py has no --resume/--wandb_name -- stale clone?"
    echo "       git -C $CODE_DIR pull  (mi-future-agentcentric)"; exit 1; }
# --mi_target only exists on the mi-future-agentcentric branch: a plain
# mi-future clone would run FUTURE-only and silently skip the agent-centric
# half of this experiment.
# Guard the combination that silently trains the opposite of what is wanted.
case "$PERTURB_FRAC" in 0|0.|0.0|0.00) ;; *)
    case "$COMPLIANCE_WEIGHT" in 0|0.|0.0|0.00)
        echo "ERROR: PERTURB_FRAC=$PERTURB_FRAC with COMPLIANCE_WEIGHT=0."
        echo "       Perturbing the role with nothing rewarding compliance"
        echo "       teaches the policy to IGNORE the role -- the opposite of"
        echo "       the intent. Set COMPLIANCE_WEIGHT>0 or PERTURB_FRAC=0."
        exit 1 ;;
    esac ;;
esac
# And the mirror image, which the guard above did NOT cover. train_roma.py
# applies the perturbed-only mask as
#     if args.compliance_perturbed_only and use_perturb:
# so with PERTURB_FRAC=0 the mask never engages and compliance silently grades
# EVERY agent -- the exact "farm compliance by driving predictably" failure the
# --compliance_perturbed_only help text exists to prevent.
case "$COMPLIANCE_WEIGHT" in 0|0.|0.0|0.00) ;; *)
    case "$PERTURB_FRAC" in 0|0.|0.0|0.00)
        if [ "$COMPLIANCE_PERTURBED_ONLY" = "1" ]; then
            echo "ERROR: COMPLIANCE_WEIGHT=$COMPLIANCE_WEIGHT with"
            echo "       PERTURB_FRAC=0 and COMPLIANCE_PERTURBED_ONLY=1."
            echo "       The perturbed-only mask only engages when perturbation"
            echo "       is on, so compliance would grade every agent and the"
            echo "       policy can farm it by behaving predictably."
            echo "       Set PERTURB_FRAC>0, or COMPLIANCE_PERTURBED_ONLY=0 to"
            echo "       deliberately grade all agents."
            exit 1
        fi ;;
    esac ;;
esac
if [ -n "$COMPLY_TAG" ]; then
    grep -q -- "--compliance_weight" "$CODE_DIR/roma_pufferdrive/train_roma.py" || {
        echo "ERROR: COMPLIANCE_WEIGHT set but this train_roma.py has no"
        echo "       --compliance_weight -- wrong branch?"; exit 1; }
fi
if [ -n "$FILM_ARG" ]; then
    grep -q -- "--role_film" "$CODE_DIR/roma_pufferdrive/train_roma.py" || {
        echo "ERROR: ROLE_FILM=1 but this train_roma.py has no --role_film."
        echo "       FiLM lives on mi-future-agentcentric-film -- wrong branch?"
        exit 1; }
fi
grep -q -- "--mi_target" "$CODE_DIR/roma_pufferdrive/train_roma.py" || {
    echo "ERROR: this train_roma.py has no --mi_target -- wrong branch?"
    echo "       clone mi-future-agentcentric and submit from that checkout."; exit 1; }
mkdir -p "$SAVE_DIR"

# ---- Auto-resume from the highest-step checkpoint in SAVE_DIR --------------
# Step checkpoints carry aux_loss_state (the MI/div critics); final.pt does
# not, so it is excluded from the search but does mark completion.
RESUME_ARGS=""
LATEST=$(ls "$SAVE_DIR"/roma_dim${ROLE_DIM}_step*.pt 2>/dev/null \
         | sed 's/.*_step\([0-9]*\)\.pt/\1 &/' | sort -k1,1n | tail -1 | cut -d' ' -f2-)
if [ -n "$LATEST" ]; then
    # Already finished? Exit 0, not 1 -- the eval hangs off this job with
    # afterok, so a non-zero exit here would take the eval down with it.
    LATEST_STEP=$(echo "$LATEST" | sed 's/.*_step\([0-9]*\)\.pt/\1/')
    if [ -f "$SAVE_DIR/roma_dim${ROLE_DIM}_final.pt" ] || [ "$LATEST_STEP" -ge "$TOTAL_STEPS" ]; then
        echo "[mifut] already at step $LATEST_STEP / $TOTAL_STEPS -- nothing to do"
        exit 0
    fi
    echo "[mifut] resuming from $LATEST (step $LATEST_STEP)"
    RESUME_ARGS="--resume $LATEST"
else
    echo "[mifut] no existing checkpoint in $SAVE_DIR -- fresh run"
fi

echo "[mifut] role_dim=$ROLE_DIM  H=$MI_HORIZON  maps=$NUM_MAPS  total_steps=$TOTAL_STEPS"
echo "[mifut] role view: road=$ROLE_ROAD_DIM partner=$ROLE_PARTNER_DIM  mi_target=$MI_TARGET"
# These echoes, the run-name tags and the branch guards all read the same
# variables -- but for one run they ALL said "film + comply ON" while the
# training used argparse defaults, because the variables were never added to the
# python invocation below. Everything that reports config here is downstream of
# the same values; only the actual command line decides. If a flag is added
# above, it MUST also be added to the python call, and the wandb config dump is
# the only report that proves it.
echo "[mifut] FiLM conditioning: ${FILM_ARG:-off}"
echo "[mifut] compliance=$COMPLIANCE_WEIGHT perturb_frac=$PERTURB_FRAC alpha=$PERTURB_ALPHA"
echo "[mifut] compliance_perturbed_only=$COMPLIANCE_PERTURBED_ONLY"
echo "[mifut] div_weight=$DIV_WEIGHT  mi_weight=$MI_WEIGHT  wandb_name=$WANDB_NAME"
echo "[mifut] code=$CODE_DIR  save=$SAVE_DIR"
echo "[mifut] branch=$(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?) commit=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo ?)"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

python "$CODE_DIR/roma_pufferdrive/train_roma.py" \
    --role_dim      "$ROLE_DIM" \
    --mi_horizon    "$MI_HORIZON" \
    --role_road_dim    "$ROLE_ROAD_DIM" \
    --role_partner_dim "$ROLE_PARTNER_DIM" \
    --mi_target        "$MI_TARGET" \
    --div_weight    "$DIV_WEIGHT" \
    --mi_weight     "$MI_WEIGHT" \
    --compliance_weight "$COMPLIANCE_WEIGHT" \
    --perturb_frac      "$PERTURB_FRAC" \
    --perturb_alpha     "$PERTURB_ALPHA" \
    --compliance_perturbed_only "$COMPLIANCE_PERTURBED_ONLY" \
    $FILM_ARG \
    --data_dir      "$DATA_DIR" \
    --num_maps      "$NUM_MAPS" \
    --num_agents    3072 \
    --device        cuda \
    --total_steps   "$TOTAL_STEPS" \
    --save_interval "$SAVE_INTERVAL" \
    --save_dir      "$SAVE_DIR" \
    --seed          "$SEED" \
    --wosac_periodic 1 \
    --wosac_interval      "$WOSAC_INTERVAL" \
    --wosac_eval_maps     10000 \
    --wosac_eval_rollouts 32 \
    --wosac_eval_batches  "$WOSAC_EVAL_BATCHES" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity  "$WANDB_ENTITY" \
    --wandb_name    "$WANDB_NAME" \
    $RESUME_ARGS \
  || { echo "[mifut] FAILED (python exit $?)"; exit 1; }

echo "[mifut] done -> $SAVE_DIR"
