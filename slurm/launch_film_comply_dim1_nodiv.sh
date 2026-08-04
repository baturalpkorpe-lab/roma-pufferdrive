#!/bin/bash
# ARM 2 of the dim-1 FiLM pair: FiLM ON + compliance + perturbation, div OFF.
#
#   bash slurm/launch_film_comply_dim1_nodiv.sh
#
# Differs from launch_film_dim1_nodiv.sh ONLY in COMPLIANCE_WEIGHT and
# PERTURB_FRAC, so the pair isolates compliance with FiLM held constant.
# Weights match the dim-4 comply arm so the two dims stay comparable.
#
# NOT an sbatch -- it submits four chained jobs. See the sibling launcher for
# why every forwarded variable is unset first.
set -eu

for v in ROLE_FILM COMPLIANCE_WEIGHT PERTURB_FRAC PERTURB_ALPHA \
         COMPLIANCE_PERTURBED_ONLY DIV_WEIGHT MI_WEIGHT MI_TARGET \
         ROLE_ROAD_DIM ROLE_PARTNER_DIM SEED SAVE_DIR WANDB_NAME; do
    unset "$v" 2>/dev/null || true
done

export ROLE_FILM=1
export COMPLIANCE_WEIGHT=0.05
export PERTURB_FRAC=0.15
export PERTURB_ALPHA=2.0
export COMPLIANCE_PERTURBED_ONLY=1
export DIV_WEIGHT=0

HERE=$(cd "$(dirname "$0")" && pwd)
echo "[arm] FiLM=1 compliance=0.05 perturb=0.15 alpha=2.0 div=0  role_dim=1 H=8"
echo "[arm] -> roma_mifutagent_dim1_H8_r16p64_ep_film_comply_nodiv"
echo
ANALYZE=analyze_mifutfilm_comply_dim1_nodiv.sh bash "$HERE/run_mi_future.sh" 1 8

cat <<'MSG'

VERIFY IN THE FIRST MINUTE (wandb config, not the job log):
    role_film         true
    compliance_weight 0.05
    perturb_frac      0.15
    div_weight        0
    role_dim          1
MSG
