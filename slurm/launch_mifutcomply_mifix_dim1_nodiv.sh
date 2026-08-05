#!/bin/bash
# No FiLM. compliance + perturbation, MI-decoder fix, dim-1, div OFF.
#
#   bash slurm/launch_mifutcomply_mifix_dim1_nodiv.sh
#
# Submits training, eval+WOSAC, scene ICC and the paired role analysis, chained.
# The _mifix tag keeps this out of the broken run's SAVE_DIR, so it cannot
# auto-resume weights trained under the co-adapting decoder.
set -eu

for v in ROLE_FILM COMPLIANCE_WEIGHT PERTURB_FRAC PERTURB_ALPHA          COMPLIANCE_PERTURBED_ONLY MI_EXCLUDE_PERTURBED DIV_WEIGHT MI_WEIGHT          MI_TARGET ROLE_ROAD_DIM ROLE_PARTNER_DIM SEED SAVE_DIR WANDB_NAME; do
    unset "$v" 2>/dev/null || true
done

export ROLE_FILM=0
export COMPLIANCE_WEIGHT=0.05
export PERTURB_FRAC=0.15
export PERTURB_ALPHA=2.0
export COMPLIANCE_PERTURBED_ONLY=1
export MI_EXCLUDE_PERTURBED=1
export DIV_WEIGHT=0

HERE=$(cd "$(dirname "$0")" && pwd)
echo "[arm] FiLM=0 compliance=0.05 perturb=0.15 mi_exclude_perturbed=1 div=0  role_dim=1"
echo "[arm] -> roma_mifutagent_dim1_H8_r16p64_ep_comply_mifix_nodiv"
echo
ANALYZE=analyze_mifutcomply_mifix_dim1_nodiv.sh bash "$HERE/run_mi_future.sh" 1 8

cat <<'MSG'

VERIFY IN THE FIRST MINUTE (wandb config):
    role_film              see above
    compliance_weight      0.05
    perturb_frac           0.15
    mi_exclude_perturbed   1      <- the fix; 0 means you got the broken version
    role_dim               1
MSG
