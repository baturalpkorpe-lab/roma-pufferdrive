#!/bin/bash
# One command for the FUTURE-MI x AGENT-CENTRIC run: training to 3B (mid-run
# WOSAC every 500M), then role analysis + the full 100-batch WOSAC, then the
# scene-ICC diagnostic -- both chained afterok.
#
#   bash slurm/run_mi_future.sh                 # dim 4, H 8, r16 p64 ego_partner
#   bash slurm/run_mi_future.sh 4 16            # role_dim 4, mi_horizon 16
#   DIV_WEIGHT=0 bash slurm/run_mi_future.sh    # the no-diversity arm
#
# THREE jobs: 3B training fits the 24h wall; eval and ICC hang off it afterok.
# Not a resume chain -- checkpoints land every 500M, so a link resuming a job
# that died at 1.29B would restart from 1.0B and redo 290M steps.
# train_mi_future.sbatch still auto-resumes as a RECOVERY path: if the wall
# clock does cut it short, just run this script again and it continues from
# the last checkpoint.
#
# Scene ICC is the pre-registered map-independence metric: the fraction of
# role variance explained by WHICH SCENE the agent is in. The agent-centric
# changes exist to push it DOWN; compare against the plain dim-4 baseline ICC.
#
# All jobs default to /scratch/$USER; export SCRATCH_ROOT / PUFFER_DIR /
# ROLE_ROAD_DIM / ROLE_PARTNER_DIM / MI_TARGET / etc. to override. Submit from
# the mi-future-agentcentric clone root.

set -e

HERE=$(cd "$(dirname "$0")" && pwd)
D=${1:-4}
H=${2:-8}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}

# Keep the arch AND the ablation in the path: a different role view builds a
# different-shaped role encoder, so two configs sharing one SAVE_DIR would
# resume from weights that cannot load (or silently continue the wrong
# experiment).
ROAD=${ROLE_ROAD_DIM:-16}
PARTNER=${ROLE_PARTNER_DIM:-64}
case "${MI_TARGET:-ego_partner}" in ego_partner) MI_TAG=ep ;; *) MI_TAG=${MI_TARGET} ;; esac
# Must match train_mi_future.sbatch: FiLM changes the policy's state dict, so
# it belongs in the path that decides which checkpoints a run resumes from.
case "${ROLE_FILM:-0}" in 1|true|yes) FILM_TAG=_film ;; *) FILM_TAG= ;; esac
case "${COMPLIANCE_WEIGHT:-0}" in 0|0.|0.0|0.00) COMPLY_TAG= ;; *) COMPLY_TAG=_comply ;; esac
# Must match train_mi_future.sbatch exactly. The MI-decoder fix changes what the
# run IS, so it belongs in the path that decides which checkpoints a run resumes
# from -- a fixed run must never auto-resume from a broken one's weights.
case "${MI_EXCLUDE_PERTURBED:-0}" in 1|true|yes) MIFIX_TAG=_mifix ;; *) MIFIX_TAG= ;; esac
ARCH_TAG=_r${ROAD}p${PARTNER}_${MI_TAG}${FILM_TAG}${COMPLY_TAG}${MIFIX_TAG}
DIV_WEIGHT=${DIV_WEIGHT:-0.1}
case "$DIV_WEIGHT" in 0|0.|0.0|0.00) DIV_TAG=_nodiv ;; *) DIV_TAG= ;; esac
RUN=dim${D}_H${H}${ARCH_TAG}${DIV_TAG}
SAVE_DIR=${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/roma_mifutagent_${RUN}}

E="ALL,ROLE_DIM=$D,MI_HORIZON=$H,DIV_WEIGHT=$DIV_WEIGHT,SAVE_DIR=$SAVE_DIR"
for v in ROLE_ROAD_DIM ROLE_PARTNER_DIM MI_TARGET MI_WEIGHT SEED SCRATCH_ROOT \
         ROLE_FILM COMPLIANCE_WEIGHT PERTURB_FRAC PERTURB_ALPHA \
         COMPLIANCE_PERTURBED_ONLY MI_EXCLUDE_PERTURBED PUFFER_DIR TOTAL_STEPS WANDB_PROJECT WANDB_ENTITY WANDB_NAME; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && E="$E,$v=$val"
done

J=$(sbatch --parsable --export="$E" "$HERE/train_mi_future.sbatch")
echo "train (3B, 500M periodic)              : $J"
EV=$(sbatch --parsable --dependency=afterok:$J --export="$E" "$HERE/eval_mi_future.sbatch")
echo "eval (role analysis + 100-batch WOSAC) : $EV  (afterok:$J)"
IC=$(sbatch --parsable --dependency=afterok:$J \
     --export="$E,CKPT=$SAVE_DIR/roma_dim${D}_final.pt,OUT=$SCRATCH_ROOT/role_icc/mifutagent_${RUN}" \
     "$HERE/role_scene_icc.sbatch")
echo "scene ICC (map-independence metric)    : $IC  (afterok:$J)"
# ANALYZE=<analyze_*.sh>: queue the paired role analysis + video grid on the
# same afterok, so the whole chain is submitted in one go instead of having to
# come back once training lands. _analyze_common.sh skips its checkpoint-exists
# check when WAIT_FOR is set, because the checkpoint cannot exist yet.
if [ -n "${ANALYZE:-}" ]; then
    echo
    WAIT_FOR=$J ROLE_DIM=$D SCRATCH_ROOT=$SCRATCH_ROOT         bash "$HERE/$ANALYZE"
fi
echo
echo "arch        -> road=$ROAD partner=$PARTNER mi_target=${MI_TARGET:-ego_partner} film=${ROLE_FILM:-0}"
echo "div_weight  -> $DIV_WEIGHT"
echo "checkpoints -> $SAVE_DIR"
