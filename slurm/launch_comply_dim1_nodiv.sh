#!/bin/bash
# Compliance + perturbation, NO FiLM, role_dim=1, div loss OFF.
#
#   bash slurm/launch_comply_dim1_nodiv.sh
#
# NOT an sbatch: submits training, eval+WOSAC, scene ICC and the paired role
# analysis + video grid, all chained afterok on the training.
#
# No new branch is needed for this. Compliance and perturbation are plain
# arguments on mi-future-agentcentric-film and are independent of --role_film;
# only the configuration differs. A branch would add one more clone to keep in
# sync, and stale clones have been the single most expensive failure mode here.
#
# Every forwarded variable is unset first: train_mi_future.sbatch is submitted
# with --export=ALL and inherits the login shell, so a leftover ROLE_FILM=1 from
# another launch would silently turn this into the arm it is meant to contrast.
set -eu

for v in ROLE_FILM COMPLIANCE_WEIGHT PERTURB_FRAC PERTURB_ALPHA \
         COMPLIANCE_PERTURBED_ONLY DIV_WEIGHT MI_WEIGHT MI_TARGET \
         ROLE_ROAD_DIM ROLE_PARTNER_DIM SEED SAVE_DIR WANDB_NAME; do
    unset "$v" 2>/dev/null || true
done

export ROLE_FILM=0
export COMPLIANCE_WEIGHT=0.05
export PERTURB_FRAC=0.15
export PERTURB_ALPHA=2.0
export COMPLIANCE_PERTURBED_ONLY=1
export DIV_WEIGHT=0

HERE=$(cd "$(dirname "$0")" && pwd)
echo "[arm] FiLM=0 compliance=0.05 perturb=0.15 div=0  role_dim=1 H=8"
echo "[arm] -> roma_mifutagent_dim1_H8_r16p64_ep_comply_nodiv"
echo
ANALYZE=analyze_mifutcomply_dim1_nodiv.sh bash "$HERE/run_mi_future.sh" 1 8

cat <<'MSG'

VERIFY IN THE FIRST MINUTE (wandb config, not the job log):
    role_film         false
    compliance_weight 0.05
    perturb_frac      0.15
    div_weight        0
    role_dim          1

AFTERWARDS, for the ICC that goes in the table (3 repeats, CPU, no GPU cost):
  for i in 1 2 3; do
    sbatch --time=00:40:00 --export=ALL,\
CKPT=$SCRATCH_ROOT/checkpoints/roma_mifutagent_dim1_H8_r16p64_ep_comply_nodiv/roma_dim1_final.pt,\
OUT=$SCRATCH_ROOT/role_icc/mifutcomply_dim1_nodiv_rep$i slurm/role_scene_icc_cpu.sbatch
  done
MSG
