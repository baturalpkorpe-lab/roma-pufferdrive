#!/bin/bash
# Fine-tune the collision-cost checkpoint AFTER stop signs are added to the
# observation. 1B extra steps.
#
#   bash slurm/launch_ft_signs_dim1_nodiv.sh
#
# REQUIRES A REBUILT ENV. This launcher changes nothing about the observation
# itself -- that is a drive.h change plus a rebuild of the C binding. If the
# env has not been rebuilt, this run is a duplicate of the one it seeds from
# and the preflight below will not catch it, because obs_dim is IDENTICAL
# either way. Verify by hand before submitting (see VERIFY at the end).
#
# WHY obs_dim DOES NOT CHANGE, which is what makes this a fine-tune
# The road observation is a fixed 128 slots x 7 features, and the 7th feature
# is already the type: obs[idx+6] = entity->type - 4.0f, giving lane 0, line 1,
# edge 2. A stop sign is type 7, so it arrives as 3.0 in a feature that already
# exists. Nothing about the observation shape moves, so the checkpoint loads.
#
# WHAT THE AGENT HAS TO LEARN, AND THE HONEST RISK
# The type feature is ORDINAL, and this policy has only ever seen 0, 1, 2 in
# it. A sign at 3.0 will initially read as "more edge-like than a road edge" --
# i.e. as a wall. Slowing near a perceived wall is not the worst possible prior
# for a stop sign, but expect a transient, and expect it to take real steps to
# separate "sign" from "edge".
#
# There is no reward for stopping at a sign. The only mechanism is
#     see sign -> slow -> fewer collisions -> higher return
# and it only pays if the sign carries information the agent does not already
# have from partner observations. If every nearby vehicle is visible regardless
# of occlusion, the sign's value is mostly the right-of-way CONVENTION (who
# yields at an all-way stop), not detection. That is still real but weaker.
#
# THE MEASUREMENT THIS EXISTS FOR
# Junction metrics split by right-of-way class, before and after. all_way_stop
# should improve; signalised_likely CANNOT, because no traffic-light state
# exists anywhere in the env. That asymmetry is what makes the result causal
# rather than just a better number -- a global improvement would point at
# something else having changed.
#
# Baseline to beat, collision-cost run at alpha=0, pooled over three sweeps:
#   all_way   minTTC ~2.85 vs human 6.324   PET ~2.1 vs 3.5   MRD ~0.78 vs 0.192
#   speed_zone ~7.7 vs human 5.108, and barely responding to the role
set -eu

for v in ROLE_FILM COMPLIANCE_WEIGHT PERTURB_FRAC PERTURB_ALPHA \
         COMPLIANCE_PERTURBED_ONLY MI_EXCLUDE_PERTURBED DIV_WEIGHT MI_WEIGHT \
         MI_TARGET ROLE_ROAD_DIM ROLE_PARTNER_DIM SEED WANDB_NAME \
         FREEZE_ROLE_ENCODER TOTAL_STEPS SAVE_INTERVAL \
         OFFROAD_BEHAVIOR GOAL_SPEED REWARD_OFFROAD_COLLISION \
         REWARD_VEHICLE_COLLISION; do
    case "$v" in
        COMPLIANCE_WEIGHT) : ;;
        *) unset "$v" 2>/dev/null || true ;;
    esac
done

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
SRC=${SRC:-$SCRATCH_ROOT/checkpoints/roma_mifutagent_dim1_H8_r16p64_ft_collstop_nodiv}
FT_STEPS=${FT_STEPS:-1000000000}

# INIT_FROM=<checkpoint> seeds from a SPECIFIC step rather than the last one.
# The final checkpoint is not automatically the best: on the collision-cost run,
# 4.0B beat 5.0B on both dial ranges and on every conflict level, and accel_ff
# degraded monotonically 3.620 -> 4.009 across the run.
INIT_FROM=${INIT_FROM:-$(ls "$SRC"/roma_dim1_step*.pt 2>/dev/null \
            | sed 's/.*_step\([0-9]*\)\.pt/\1 &/' | sort -k1,1n | tail -1 \
            | cut -d' ' -f2-)}
[ -n "$INIT_FROM" ] || { echo "ERROR: no roma_dim1_step*.pt in $SRC"; exit 1; }
SEED_STEP=$(echo "$INIT_FROM" | sed 's/.*_step\([0-9]*\)\.pt/\1/')

export INIT_FROM
export TOTAL_STEPS=$((SEED_STEP + FT_STEPS))
export SAVE_INTERVAL=${SAVE_INTERVAL:-250000000}
export FREEZE_ROLE_ENCODER=0

# Carried over unchanged from the collision-cost run, so the observation is
# the only variable.
export COLLISION_BEHAVIOR=${COLLISION_BEHAVIOR:-1}
export ROLE_FILM=0
export COMPLIANCE_WEIGHT=${COMPLIANCE_WEIGHT:-2.5}
export PERTURB_FRAC=0.15
export PERTURB_ALPHA=2.0
export COMPLIANCE_PERTURBED_ONLY=1
export MI_EXCLUDE_PERTURBED=1
export DIV_WEIGHT=0

# TAG keeps two seeds apart. Sharing a SAVE_DIR would let the auto-resume load
# the other seed's weights and silently continue the wrong experiment.
TAG=${TAG:-signs}
export SAVE_DIR=$SCRATCH_ROOT/checkpoints/roma_mifutagent_dim1_H8_r16p64_ft_${TAG}_nodiv
export WANDB_NAME=roma_mifutagent_dim1_ft_${TAG}_nodiv

HERE=$(cd "$(dirname "$0")" && pwd)
echo "[arm] FINE-TUNE from  : $INIT_FROM"
echo "[arm] steps           : $SEED_STEP -> $TOTAL_STEPS (+${FT_STEPS})"
echo "[arm] collision_behav : $COLLISION_BEHAVIOR   compliance $COMPLIANCE_WEIGHT   perturb 0.15"
echo "[arm] save_dir        : $SAVE_DIR"
echo
bash "$HERE/run_mi_future.sh" 1 8

cat <<MSG

VERIFY THE ENV WAS ACTUALLY REBUILT -- obs_dim is identical either way, so
nothing downstream will notice if it was not:

    grep -n "type < 8" \$PUFFER_DIR/pufferlib/ocean/drive/drive.h
      expect THREE hits (map bounds x2, grid insertion)

    python3 -c "
import numpy as np, sys
sys.path.insert(0,'\$PUFFER_DIR')
from pufferlib.ocean.drive.drive import Drive
e=Drive(num_maps=20,num_agents=64,map_dir='pufferlib/resources/drive/binaries/training')
o,_=e.reset(); o=np.asarray(o)
ego=8; road=o[:, ego+7*63:].reshape(len(o),128,7)
t=road[...,6]
print('road type values seen:', sorted(set(np.round(t[np.abs(road).sum(-1)>0],2).tolist()))[:10])
"
      BEFORE the change you see {0.0, 1.0, 2.0}. AFTER you must also see 3.0.
      If 3.0 never appears, signs are still not reaching the observation and
      this run is a duplicate -- cancel it.

GO / NO-GO at each 250M checkpoint, and the comparison that matters is the
right-of-way SPLIT, not the pooled number:

    cd \$HOME/roma_film
    CK=\$(ls $SAVE_DIR/roma_dim1_step*.pt | tail -1)
    sbatch --export=ALL,SWEEP=1,CKPT=\$CK,\\
OUT=$SCRATCH_ROOT/regimes_rollout/signs_\$(basename \$CK .pt) \\
        slurm/regime_rollout.sbatch

    python alpha_table.py --rollout_dir <that dir> \\
        --gt_regimes   $SCRATCH_ROOT/regimes/regimes_gt.csv \\
        --gt_conflicts $SCRATCH_ROOT/regimes/conflicts_gt.csv

    all_way_stop improving while signalised_likely does not is the result.
    Both improving equally means something other than the signs did it.
MSG
