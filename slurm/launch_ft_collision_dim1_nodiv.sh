#!/bin/bash
# Fine-tune the compliance-2.5 checkpoint with collisions made COSTLY.
#
#   bash slurm/launch_ft_collision_dim1_nodiv.sh
#   REWARD_VEHICLE_COLLISION=-2.0 bash slurm/launch_ft_collision_dim1_nodiv.sh
#
# THE PROBLEM THIS TARGETS
# Measured on WOSAC, 6997 scenarios: the policy collides 2.9x more than humans
# (0.109 vs 0.037) while matching them offroad (0.153 vs 0.144). And the role
# cannot fix it -- four arms with completely different role behaviour all
# landed at WOSAC meta 0.686-0.695 and kinematic 0.296-0.304. Realism is
# orthogonal to the role machinery.
#
# WHY THE REWARD IS THE CAUSE
# drive.ini pays +1.0 for a goal and +0.25 for every respawn goal after it,
# charges -0.5 for a collision, and sets collision_behavior=0 (Ignore) so the
# episode continues regardless. Two extra respawn goals therefore pay for one
# collision. Nothing anywhere penalises following too closely. The agent is
# doing what it was paid to do.
#
# collision_behavior=1 (Stop) makes a collision forfeit the REST OF THE
# EPISODE. That is a far bigger change in incentive than any tweak to -0.5,
# which the policy can still trade against.
#
# WHY START FROM THE COMPLIANCE-2.5 CHECKPOINT
# It is a strict superset of ep_nodiv: identical realism (WOSAC 0.6863 vs
# 0.6869), identical ICC (0.181 vs 0.175), identical role wire (12.56 vs
# 13.00) -- PLUS a trained off-manifold response. ep_nodiv at alpha=-2 crawls
# at 1.8 m/s and stops following anyone (n_fol 653); this one drives at 8.6 m/s
# and stays in traffic (n_fol 1578). That training is expensive to redo and
# free to carry forward.
#
# PERTURBATION AND COMPLIANCE STAY ON. Nothing maintains the off-manifold
# response without them, and it would decay over 500M steps.
#
# NOT FROZEN this time. The compliance target is the MI decoder, and it encodes
# what each role meant under the OLD aggressive policy. This run is meant to
# change that behaviour substantially, so holding the target fixed would ask
# the policy to become less aggressive while matching a description of how it
# used to drive. ICC may move as a result -- that is a measurement, and the
# 3-rep ICC protocol will catch it.
set -eu

for v in ROLE_FILM COMPLIANCE_WEIGHT PERTURB_FRAC PERTURB_ALPHA \
         COMPLIANCE_PERTURBED_ONLY MI_EXCLUDE_PERTURBED DIV_WEIGHT MI_WEIGHT \
         MI_TARGET ROLE_ROAD_DIM ROLE_PARTNER_DIM SEED WANDB_NAME \
         INIT_FROM FREEZE_ROLE_ENCODER TOTAL_STEPS SAVE_INTERVAL \
         OFFROAD_BEHAVIOR GOAL_SPEED REWARD_OFFROAD_COLLISION; do
    case "$v" in
        COMPLIANCE_WEIGHT) : ;;
        *) unset "$v" 2>/dev/null || true ;;
    esac
done

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
SRC=${SRC:-$SCRATCH_ROOT/checkpoints/roma_mifutagent_dim1_H8_r16p64_ft_comply25_nodiv}
FT_STEPS=${FT_STEPS:-1500000000}

INIT_FROM=$(ls "$SRC"/roma_dim1_step*.pt 2>/dev/null \
            | sed 's/.*_step\([0-9]*\)\.pt/\1 &/' | sort -k1,1n | tail -1 \
            | cut -d' ' -f2-)
[ -n "$INIT_FROM" ] || {
    echo "ERROR: no roma_dim1_step*.pt in $SRC"; exit 1; }
SEED_STEP=$(echo "$INIT_FROM" | sed 's/.*_step\([0-9]*\)\.pt/\1/')

export INIT_FROM
export TOTAL_STEPS=$((SEED_STEP + FT_STEPS))
# 250M between checkpoints: six decision points instead of two. A reward change
# needs watching, unlike the compliance fine-tune which was a refinement.
export SAVE_INTERVAL=${SAVE_INTERVAL:-250000000}
export FREEZE_ROLE_ENCODER=0

# THE CHANGE. Everything else below is carried over unchanged from the
# compliance-2.5 run so this is a single-variable experiment.
export COLLISION_BEHAVIOR=${COLLISION_BEHAVIOR:-1}

export ROLE_FILM=0
export COMPLIANCE_WEIGHT=${COMPLIANCE_WEIGHT:-2.5}
export PERTURB_FRAC=0.15
export PERTURB_ALPHA=2.0
export COMPLIANCE_PERTURBED_ONLY=1
export MI_EXCLUDE_PERTURBED=1
export DIV_WEIGHT=0

# MUST be explicit -- run_mi_future.sh derives SAVE_DIR from the flag tags,
# which resolve to the FAILED arm's directory, and the auto-resume would load
# its weights.
export SAVE_DIR=$SCRATCH_ROOT/checkpoints/roma_mifutagent_dim1_H8_r16p64_ft_collstop_nodiv
export WANDB_NAME=roma_mifutagent_dim1_ft_collstop_nodiv

HERE=$(cd "$(dirname "$0")" && pwd)
echo "[arm] FINE-TUNE from  : $INIT_FROM"
echo "[arm] steps           : $SEED_STEP -> $TOTAL_STEPS (+${FT_STEPS}), save every $SAVE_INTERVAL"
echo "[arm] collision_behav : $COLLISION_BEHAVIOR   (drive.ini has 0 = Ignore)"
echo "[arm] compliance      : $COMPLIANCE_WEIGHT   perturb 0.15   encoder NOT frozen"
echo "[arm] save_dir        : $SAVE_DIR"
echo
bash "$HERE/run_mi_future.sh" 1 8

cat <<MSG

VERIFY IN THE FIRST MINUTE:
    [mifut] env overrides: --collision_behavior 1
    [ROMA] env override: collision_behavior = 0 -> 1
    [mifut] FINE-TUNE seed: .../roma_dim1_step${SEED_STEP}.pt
If the env-override lines are missing, drive.ini is still in charge and this
run is a duplicate of the last one. Cancel it.

EXPECT A DIP FIRST. The critic was trained where a collision cost 0.5 and the
episode continued; it now forfeits the episode. Every value estimate is wrong
for a while and --resume restarts Adam. Do not judge before 100M steps.

AT EVERY CHECKPOINT (one CPU job, ~20 min, then one command):
    CK=\$(ls $SAVE_DIR/roma_dim1_step*.pt | tail -1)
    sbatch --export=ALL,SWEEP=1,CKPT=\$CK,\\
OUT=$SCRATCH_ROOT/regimes_rollout/collstop_\$(basename \$CK .pt) \\
        slurm/regime_rollout.sbatch

    python ft_check.py \\
        --rollout_dir $SCRATCH_ROOT/regimes_rollout/collstop_<step> \\
        --baseline_dir $SCRATCH_ROOT/regimes_rollout/ft_comply25_final_rep2 \\
        --gt_regimes $SCRATCH_ROOT/regimes/regimes_gt.csv \\
        --gt_conflicts $SCRATCH_ROOT/regimes/conflicts_gt.csv
MSG
