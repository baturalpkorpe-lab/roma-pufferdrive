#!/bin/bash
# One-command launcher for the agent-centric role run.
#
#   bash slurm/run_agent_role.sh                # dim 4, road 16, partner 64, MI ego_partner
#   bash slurm/run_agent_role.sh 4              # explicit role_dim
#   DIV_WEIGHT=0 bash slurm/run_agent_role.sh   # the no-diversity arm
#
# Submits ONE training job (3B steps, 24h wall), then the full post-training
# eval -- role health check + WOSAC at 100 batches -- and the scene-ICC
# diagnostic, both chained afterok so the number this change is aimed at lands
# without a second submission.
#
# Deliberately not a resume chain: checkpoints are written every 500M steps, so
# a job resuming one that died at 1.29B would restart from 1.0B and burn 290M
# steps over again. If the wall clock does cut it short, just run this script
# again -- the sbatch auto-resumes from the last checkpoint in SAVE_DIR.
#
# Every run config -- road/partner width, MI target, div_weight -- goes in the
# checkpoint path and the wandb name. Two configs sharing a SAVE_DIR would have
# a later link resume from the WRONG experiment (and, for a different
# road/partner width, from weights that cannot even load).
#
# Override any of ROLE_ROAD_DIM / ROLE_PARTNER_DIM / MI_TARGET / DIV_WEIGHT /
# SEED / SCRATCH_ROOT / PUFFER_DIR by exporting them first.

set -eu

D=${1:-4}
HERE=$(cd "$(dirname "$0")" && pwd)

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}

ROAD=${ROLE_ROAD_DIM:-16}
PARTNER=${ROLE_PARTNER_DIM:-64}
case "${MI_TARGET:-ego_partner}" in ego_partner) MI_TAG=ep ;; *) MI_TAG=${MI_TARGET} ;; esac
case "${DIV_WEIGHT:-0.1}" in 0|0.|0.0|0.00) DIV_TAG=_nodiv ;; *) DIV_TAG= ;; esac

TAG=agentrole${DIV_TAG}
RUN=dim${D}_r${ROAD}p${PARTNER}_${MI_TAG}${DIV_TAG}
SAVE_DIR=${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/roma_agentrole_${RUN}}
# eval_final.sbatch's default OUT is keyed on TAG+dim only, so every config
# would land in the same directory. Pass it explicitly, keyed on RUN.
EVAL_OUT=${EVAL_OUT:-$SCRATCH_ROOT/eval_results/agentrole_${RUN}}

E="ALL,ROLE_DIM=$D,SAVE_DIR=$SAVE_DIR,TAG=$TAG"
for v in ROLE_ROAD_DIM ROLE_PARTNER_DIM MI_TARGET SEED SCRATCH_ROOT PUFFER_DIR DIV_WEIGHT TOTAL_STEPS; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && E="$E,$v=$val"
done

J=$(sbatch --parsable --export="$E" "$HERE/train_agent_role.sbatch")
echo "train       : $J  (3B, 500M periodic)"
EV=$(sbatch --parsable --dependency=afterok:$J --export="$E,OUT=$EVAL_OUT" \
     "$HERE/eval_final.sbatch")
echo "eval        : $EV  (role analysis + 100-batch WOSAC, afterok:$J)"
IC=$(sbatch --parsable --dependency=afterok:$J \
     --export="$E,CKPT=$SAVE_DIR/roma_dim${D}_final.pt,OUT=$SCRATCH_ROOT/role_icc/agentrole_${RUN}" \
     "$HERE/role_scene_icc.sbatch")
echo "scene ICC   : $IC"
echo
echo "div_weight  -> ${DIV_WEIGHT:-0.1}"
echo "checkpoints -> $SAVE_DIR"
echo "eval out    -> $EVAL_OUT"
