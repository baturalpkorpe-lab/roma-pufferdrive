#!/bin/bash
# FROM SCRATCH, ep_nodiv architecture, with the published reward magnitudes.
# No compliance, no perturbation, no FiLM, no diversity loss.
#
#   bash slurm/launch_paperreward_dim1_nodiv.sh
#   COLLISION_BEHAVIOR=1 bash slurm/launch_paperreward_dim1_nodiv.sh   # + Stop
#
# THE CHANGE
#   reward_vehicle_collision  -0.5 -> -1.0
#   reward_offroad_collision  -0.5 -> -1.0
# Everything else is ep_nodiv's configuration, unchanged.
#
# WHY THESE VALUES
# Not tuned. "Human-like autonomy emerges from self-play and a pinch of human
# data" (Emerge Lab, 2026) uses +1 for reaching the goal and -1 for a collision
# or off-road event, on PufferDrive 2.0 -- this simulator. drive.ini charges
# -0.5. Matching a published reference is defensible in a way that sweeping to
# -2 and keeping whatever worked is not.
#
# collision_behavior is left at drive.ini's 0 (Ignore) ON PURPOSE. The paper
# states its reward magnitudes but not its termination behaviour, and Stop is
# being tested separately by launch_ft_collision_dim1_nodiv.sh. Combining both
# would confound them. COLLISION_BEHAVIOR=1 is available if you want them
# together anyway.
#
# WHY NO COMPLIANCE OR PERTURBATION
# This run is about the REWARD. Arm A -- from scratch, perturbation, compliance
# at 0.4% of the return -- ended with ICC 0.434 and a dead role wire (0.44
# against ep_nodiv's 13.00), so putting untested role machinery on top of an
# untested reward would leave two candidate causes for any failure.
#
# The off-manifold response can be added afterwards and cheaply: the
# compliance-2.5 fine-tune bought it in 500M steps and under three GPU hours,
# and preserved realism, ICC and the wire while doing it. Train clean first,
# perturb second.
#
# WHAT TO COMPARE IT AGAINST
# ep_nodiv, same architecture and step budget, old reward:
#   WOSAC meta 0.6869 | kinematic 0.3020 | collisions 0.109 vs human 0.037
#   speed_ff 15.9 | headway 1.02 | all-way minTTC 1.73 (all at alpha=0)
# The target is not a higher WOSAC score. It is that the CAUTIOUS END of the
# role range reaches human values -- headway ~2.06 s, all-way minTTC ~6.3 s,
# speed_ff ~10.5 -- while the agent STAYS IN TRAFFIC. ep_nodiv reached low
# speeds at alpha=-2 only by crawling at 1.8 m/s until it stopped following
# anyone (n_fol 653 against 2254 at alpha=0). That does not count.
set -eu

for v in ROLE_FILM COMPLIANCE_WEIGHT PERTURB_FRAC PERTURB_ALPHA \
         COMPLIANCE_PERTURBED_ONLY MI_EXCLUDE_PERTURBED DIV_WEIGHT MI_WEIGHT \
         MI_TARGET ROLE_ROAD_DIM ROLE_PARTNER_DIM SEED SAVE_DIR WANDB_NAME \
         INIT_FROM FREEZE_ROLE_ENCODER TOTAL_STEPS SAVE_INTERVAL \
         OFFROAD_BEHAVIOR GOAL_SPEED; do
    unset "$v" 2>/dev/null || true
done

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}

# The reward change.
export REWARD_VEHICLE_COLLISION=${REWARD_VEHICLE_COLLISION:--1.0}
export REWARD_OFFROAD_COLLISION=${REWARD_OFFROAD_COLLISION:--1.0}
[ -n "${COLLISION_BEHAVIOR:-}" ] && export COLLISION_BEHAVIOR

# ep_nodiv's configuration, untouched.
export ROLE_FILM=0
export COMPLIANCE_WEIGHT=0
export PERTURB_FRAC=0
export DIV_WEIGHT=0
export TOTAL_STEPS=3000000000       # match ep_nodiv so the comparison is like-for-like

TAG=paperrew$([ -n "${COLLISION_BEHAVIOR:-}" ] && echo "_collstop" || echo "")
export SAVE_DIR=$SCRATCH_ROOT/checkpoints/roma_mifutagent_dim1_H8_r16p64_${TAG}_nodiv
export WANDB_NAME=roma_mifutagent_dim1_${TAG}_nodiv

HERE=$(cd "$(dirname "$0")" && pwd)
echo "[arm] FROM SCRATCH, ep_nodiv architecture"
echo "[arm] collision reward : $REWARD_VEHICLE_COLLISION  (drive.ini has -0.5)"
echo "[arm] offroad reward   : $REWARD_OFFROAD_COLLISION  (drive.ini has -0.5)"
echo "[arm] collision_behav  : ${COLLISION_BEHAVIOR:-drive.ini default (0 = Ignore)}"
echo "[arm] compliance=0  perturb=0  film=0  div=0   total_steps=$TOTAL_STEPS"
echo "[arm] save_dir         : $SAVE_DIR"
echo
bash "$HERE/run_mi_future.sh" 1 8

cat <<MSG

VERIFY IN THE FIRST MINUTE:
    [mifut] env overrides: --reward_vehicle_collision -1.0 --reward_offroad_collision -1.0
    [ROMA] env override: reward_vehicle_collision = -0.5 -> -1.0
    [ROMA] env override: reward_offroad_collision = -0.5 -> -1.0
    [mifut] no existing checkpoint in ... -- fresh run
If the override lines are missing, drive.ini is still in charge and this is a
duplicate of ep_nodiv. Cancel it.

MEAN RETURN WILL SIT LOWER THAN ep_nodiv'S 1.048 and that is not a fault --
the same collisions now cost twice as much. Judge on the checkpoint sweeps,
not on the return.

AT EACH 500M CHECKPOINT:
    CK=\$(ls $SAVE_DIR/roma_dim1_step*.pt | tail -1)
    sbatch --export=ALL,SWEEP=1,CKPT=\$CK,\\
OUT=$SCRATCH_ROOT/regimes_rollout/paperrew_\$(basename \$CK .pt) \\
        slurm/regime_rollout.sbatch

    python ft_check.py \\
        --rollout_dir  $SCRATCH_ROOT/regimes_rollout/paperrew_<step> \\
        --baseline_dir $SCRATCH_ROOT/regimes_rollout/dim1_v4 \\
        --gt_regimes   $SCRATCH_ROOT/regimes/regimes_gt.csv \\
        --gt_conflicts $SCRATCH_ROOT/regimes/conflicts_gt.csv

--baseline_dir is ep_nodiv's own sweep, so section 4 reads "is this closer to
human than the old reward was" directly.
MSG
