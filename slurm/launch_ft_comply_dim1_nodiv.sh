#!/bin/bash
# FINE-TUNE ep_nodiv with perturbation + a compliance weight that is actually
# non-zero, role encoder and MI decoder FROZEN.
#
#   bash slurm/launch_ft_comply_dim1_nodiv.sh
#   COMPLIANCE_WEIGHT=5.0 bash slurm/launch_ft_comply_dim1_nodiv.sh   # stronger
#
# WHY 2.5 AND NOT 0.05
# The compliance error is the same quantity as the MI loss (see the
# compliance-reward block in train_roma.py), and mi_loss converged to ~1e-3 on
# both dim-1 mifix arms. So the penalty those runs actually applied was
#     0.05 * 1e-3 * 91 steps = 0.005 per episode
# against a mean_return of ~1.05 -- 0.4%. The --perturb_frac help text warns
# that perturbation WITHOUT compliance teaches the policy to ignore the role;
# at that magnitude the pairing was nominal and the warning applied anyway.
# 2.5 puts compliance at roughly a quarter of the task return:
#     w = 0.25 * 1.05 / (1e-3 * 91) ~= 2.9
#
# WHY A FINE-TUNE AND NOT A FRESH RUN
# Both previous compliance+perturbation arms were trained from scratch, so the
# policy never learned to use the role before the perturbation started
# corrupting the channel. ep_nodiv ends with a role-input column norm of 13.0
# and a headway dial spanning 2.01 -> 0.86 s across +-2 sigma. There is a wire
# to defend rather than one to grow through noise.
#
# WHY FROZEN
# On the two perturbed arms the encoder drifted onto the map (scene ICC 0.170
# -> 0.434) once the policy stopped reading z. Freezing the encoder pins ICC at
# ep_nodiv's 0.170 and freezes the MI decoder so the compliance target cannot
# move to meet the policy. The experiment then has exactly one free variable:
# can a real compliance signal keep the dial alive under perturbation?
set -eu

for v in ROLE_FILM COMPLIANCE_WEIGHT PERTURB_FRAC PERTURB_ALPHA \
         COMPLIANCE_PERTURBED_ONLY MI_EXCLUDE_PERTURBED DIV_WEIGHT MI_WEIGHT \
         MI_TARGET ROLE_ROAD_DIM ROLE_PARTNER_DIM SEED WANDB_NAME \
         INIT_FROM FREEZE_ROLE_ENCODER TOTAL_STEPS SAVE_INTERVAL; do
    case "$v" in
        COMPLIANCE_WEIGHT) : ;;          # honour an override from the caller
        *) unset "$v" 2>/dev/null || true ;;
    esac
done

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
SRC=${SRC:-$SCRATCH_ROOT/checkpoints/roma_mifutagent_dim1_H8_r16p64_ep_nodiv}
FT_STEPS=${FT_STEPS:-500000000}          # how much fine-tuning on top of the seed

# Seed from the highest STEP checkpoint, never final.pt: only step checkpoints
# carry aux_loss_state, and a randomly initialised MI decoder would make the
# compliance target meaningless -- doubly so once it is frozen.
INIT_FROM=$(ls "$SRC"/roma_dim1_step*.pt 2>/dev/null \
            | sed 's/.*_step\([0-9]*\)\.pt/\1 &/' | sort -k1,1n | tail -1 \
            | cut -d' ' -f2-)
[ -n "$INIT_FROM" ] || {
    echo "ERROR: no roma_dim1_step*.pt in $SRC"
    echo "       ep_nodiv must have a STEP checkpoint (final.pt has no"
    echo "       aux_loss_state). Override the source with SRC=<dir>."
    exit 1; }
SEED_STEP=$(echo "$INIT_FROM" | sed 's/.*_step\([0-9]*\)\.pt/\1/')

export INIT_FROM
export TOTAL_STEPS=$((SEED_STEP + FT_STEPS))
export SAVE_INTERVAL=$((FT_STEPS / 2))   # a mid-point checkpoint to judge early
export FREEZE_ROLE_ENCODER=1

export ROLE_FILM=0
export COMPLIANCE_WEIGHT=${COMPLIANCE_WEIGHT:-2.5}
export PERTURB_FRAC=0.15
export PERTURB_ALPHA=2.0
export COMPLIANCE_PERTURBED_ONLY=1
export MI_EXCLUDE_PERTURBED=1
export DIV_WEIGHT=0

# MUST be explicit. run_mi_future.sh derives SAVE_DIR from the flag tags, and
# with FiLM off + compliance on + mifix on those tags resolve to
# ..._ep_comply_mifix_nodiv -- the FAILED arm's directory. The auto-resume
# would then load its weights and this would silently continue that run.
TAG=ft_comply$(echo "$COMPLIANCE_WEIGHT" | tr -d '.')
export SAVE_DIR=$SCRATCH_ROOT/checkpoints/roma_mifutagent_dim1_H8_r16p64_${TAG}_nodiv
export WANDB_NAME=roma_mifutagent_dim1_${TAG}_nodiv

HERE=$(cd "$(dirname "$0")" && pwd)
echo "[arm] FINE-TUNE from  : $INIT_FROM"
echo "[arm] steps           : $SEED_STEP -> $TOTAL_STEPS (+${FT_STEPS})"
echo "[arm] compliance      : $COMPLIANCE_WEIGHT   (the failed arms used 0.05)"
echo "[arm] perturb_frac    : 0.15   alpha +-2 sigma, uniform"
echo "[arm] encoder+decoder : FROZEN"
echo "[arm] save_dir        : $SAVE_DIR"
echo
bash "$HERE/run_mi_future.sh" 1 8

cat <<MSG

VERIFY IN THE FIRST MINUTE (the .out banner):
    [mifut] FINE-TUNE seed: .../roma_dim1_step${SEED_STEP}.pt
    [mifut] freeze_role_encoder=1 init_from=...
    [ROMA] FROZEN role encoder + MI decoder (... params)
    [ROMA] Resumed policy+aux from ... @ step ${SEED_STEP}
If "no existing checkpoint -- fresh run" appears instead, INIT_FROM did not
reach the sbatch and this is NOT a fine-tune. Cancel it.

GO / NO-GO AT THE MID-POINT ($((SEED_STEP + FT_STEPS/2))):
    sbatch --export=ALL,SWEEP=1,\\
CKPT=$SAVE_DIR/roma_dim1_step$((SEED_STEP + FT_STEPS/2)).pt,\\
OUT=$SCRATCH_ROOT/regimes_rollout/${TAG}_mid slurm/regime_rollout.sbatch

    python ft_check.py --rollout_dir $SCRATCH_ROOT/regimes_rollout/${TAG}_mid \\
        --gt_regimes $SCRATCH_ROOT/regimes/regimes_gt.csv

KILL IT if the headway dial range is below 0.5 s. Reference points:
    ep_nodiv (no perturbation)      1.150 s
    comply 0.05 + perturb (arm A)   0.274 s
    + FiLM (arm B)                  0.010 s
Both previous arms ran the full 3B before anyone knew. Do not repeat that.

The chained scene-ICC job is redundant here -- ICC is pinned by the freeze --
so it is safe to scancel it if the GPU queue matters.
MSG
