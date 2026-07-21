#!/bin/bash
# One-command launcher for the agent-centric role run.
#
#   bash slurm/run_agent_role.sh          # dim 4, road 16, partner 64, MI ego_partner
#   bash slurm/run_agent_role.sh 4        # explicit role_dim
#
# Submits ONE training job (3B steps in the 24h allocation), then the full
# post-training eval -- role health check + WOSAC at 100 batches -- and the
# scene-ICC diagnostic, both chained afterok so the number this change is
# aimed at lands without a second submission.
#
# Deliberately not a resume chain: checkpoints are written every 500M steps,
# so a job that dies at 1.29B would resume from 1.0B and burn 290M steps over
# again. If the wall clock does cut it short, just run this script again --
# the sbatch auto-resumes from the last checkpoint in SAVE_DIR.
#
# Override any of ROLE_ROAD_DIM / ROLE_PARTNER_DIM / MI_TARGET / SEED /
# SCRATCH_ROOT / PUFFER_DIR by exporting them first.

set -eu

D=${1:-4}
HERE=$(cd "$(dirname "$0")" && pwd)

TAG=agentrole
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}

# Keep the config in the checkpoint path: a different road/partner width builds
# a different-shaped role encoder, so two configs sharing one SAVE_DIR would
# have the chain's job 2 try to resume from weights that cannot load.
ROAD=${ROLE_ROAD_DIM:-16}
PARTNER=${ROLE_PARTNER_DIM:-64}
case "${MI_TARGET:-ego_partner}" in ego_partner) MI_TAG=ep ;; *) MI_TAG=${MI_TARGET} ;; esac
RUN=dim${D}_r${ROAD}p${PARTNER}_${MI_TAG}
SAVE_DIR=${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/roma_${TAG}_${RUN}}

E="ALL,ROLE_DIM=$D,SAVE_DIR=$SAVE_DIR,TAG=$TAG"
for v in ROLE_ROAD_DIM ROLE_PARTNER_DIM MI_TARGET SEED SCRATCH_ROOT PUFFER_DIR DIV_WEIGHT; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && E="$E,$v=$val"
done

J=$(sbatch --parsable --export="$E" "$HERE/train_agent_role.sbatch")
echo "train     : $J"
EV=$(sbatch --parsable --dependency=afterok:$J --export="$E" "$HERE/eval_final.sbatch")
echo "eval      : $EV"
IC=$(sbatch --parsable --dependency=afterok:$J \
     --export="$E,CKPT=$SAVE_DIR/roma_dim${D}_final.pt,OUT=$SCRATCH_ROOT/role_icc/${TAG}_${RUN}" \
     "$HERE/role_scene_icc.sbatch")
echo "scene ICC : $IC"
echo
echo "checkpoints -> $SAVE_DIR"
