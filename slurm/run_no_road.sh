#!/bin/bash
# One command for the NO-ROAD role run: road never reaches the role encoder,
# only the policy GRU. Everything downstream is chained, so this is submit-and-
# forget.
#
#   bash slurm/run_no_road.sh
#   DIV_WEIGHT=0 bash slurm/run_no_road.sh      # the no-diversity twin
#
# Submits FIVE jobs:
#   1 train              [GPU] 3B steps, div loss ON, mid-run WOSAC every 500M
#   2 eval_final         [GPU] afterok  full 100-batch WOSAC + role health
#   3 role_analysis      [GPU] afterok  FIXED role pairing (tail/braking-event
#                                      metrics, mask audit, trajectory-cluster
#                                      stratification)
#   4 render_grid_traj   [CPU] afterok(3)  PC1xPC2 forcing videos + zip
#   5 role_scene_icc     [GPU] afterok  scene ICC, the map-independence number
#                                      this branch exists to move
#
# 3, 4 and 5 come from run_role_analysis.sh via WAIT_FOR, so the pairing
# pipeline here is exactly the one used for every other checkpoint -- no
# second copy to drift.
#
# Defaults to /scratch/$USER; export SCRATCH_ROOT / PUFFER_DIR / SEED etc. to
# override. Submit from the role-no-road clone root.

set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
D=${1:-4}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}

ROAD=${ROLE_ROAD_DIM:-0}
PARTNER=${ROLE_PARTNER_DIM:-64}
case "${MI_TARGET:-ego_partner}" in ego_partner) MI_TAG=ep ;; *) MI_TAG=${MI_TARGET} ;; esac
DIV_WEIGHT=${DIV_WEIGHT:-0.1}
case "$DIV_WEIGHT" in 0|0.|0.0|0.00) DIV_TAG=_nodiv ;; *) DIV_TAG= ;; esac

RUN=dim${D}_r${ROAD}p${PARTNER}_${MI_TAG}${DIV_TAG}
TAG=noroad_$RUN
SAVE_DIR=${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/roma_noroad_${RUN}}
CKPT=$SAVE_DIR/roma_dim${D}_final.pt

E="ALL,ROLE_DIM=$D,SAVE_DIR=$SAVE_DIR,TAG=$TAG,DIV_WEIGHT=$DIV_WEIGHT"
E="$E,ROLE_ROAD_DIM=$ROAD,ROLE_PARTNER_DIM=$PARTNER,MI_TARGET=${MI_TARGET:-ego_partner}"
for v in SEED SCRATCH_ROOT PUFFER_DIR TOTAL_STEPS MI_WEIGHT; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && E="$E,$v=$val"
done

J=$(sbatch --parsable --export="$E" "$HERE/train_agent_role.sbatch")
echo "1 train            : $J   (3B, div_weight=$DIV_WEIGHT, road NOT in role)"

EV=$(sbatch --parsable --dependency=afterok:$J \
     --export="$E,OUT=$SCRATCH_ROOT/eval_results/$TAG" "$HERE/eval_final.sbatch")
echo "2 eval_final       : $EV  afterok:$J  (100-batch WOSAC + role health)"

# 3/4/5 -- reuse the shared pairing pipeline rather than duplicating it here.
echo
echo "-- role pairing + videos + scene ICC (via run_role_analysis.sh) --"
CKPT="$CKPT" TAG="$TAG" WAIT_FOR="$J" SCRATCH_ROOT="$SCRATCH_ROOT" \
    bash "$HERE/run_role_analysis.sh"

echo
echo "road in role : NO  (role encoder = 32 ego + 64 partner = 96 dims)"
echo "road in policy: YES (policy GRU keeps the full 128-dim env embedding)"
echo "mi_target    : ${MI_TARGET:-ego_partner}  (road not in the MI target either)"
echo "checkpoints  -> $SAVE_DIR"
echo "results      -> $SCRATCH_ROOT/analysis/$TAG/  +  eval_results/$TAG/"
echo "scene ICC    -> $SCRATCH_ROOT/role_icc/$TAG/   compare vs baseline 0.398"
