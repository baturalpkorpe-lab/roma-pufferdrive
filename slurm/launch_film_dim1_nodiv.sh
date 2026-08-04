#!/bin/bash
# ARM 1 of the dim-1 FiLM pair: FiLM ON, compliance OFF, div loss OFF.
#
#   bash slurm/launch_film_dim1_nodiv.sh
#
# NOT an sbatch. It submits FOUR jobs -- training, eval+WOSAC, scene ICC, and
# the paired role analysis + video grid -- chained afterok on the training, so
# the whole arm goes in with one command and nothing needs revisiting.
#
# Every variable run_mi_future.sh forwards is UNSET first and then set here.
# train_mi_future.sbatch is submitted with --export=ALL, so it inherits the
# login shell: a stray COMPLIANCE_WEIGHT or PERTURB_FRAC left over from another
# launch would silently change this arm into a different experiment.
set -eu

for v in ROLE_FILM COMPLIANCE_WEIGHT PERTURB_FRAC PERTURB_ALPHA \
         COMPLIANCE_PERTURBED_ONLY DIV_WEIGHT MI_WEIGHT MI_TARGET \
         ROLE_ROAD_DIM ROLE_PARTNER_DIM SEED SAVE_DIR WANDB_NAME; do
    unset "$v" 2>/dev/null || true
done

export ROLE_FILM=1
export COMPLIANCE_WEIGHT=0
export PERTURB_FRAC=0
export DIV_WEIGHT=0

HERE=$(cd "$(dirname "$0")" && pwd)
echo "[arm] FiLM=1 compliance=0 perturb=0 div=0  role_dim=1 H=8"
echo "[arm] -> roma_mifutagent_dim1_H8_r16p64_ep_film_nodiv"
echo
ANALYZE=analyze_mifutfilm_dim1_nodiv.sh bash "$HERE/run_mi_future.sh" 1 8

cat <<'MSG'

VERIFY IN THE FIRST MINUTE (wandb config, not the job log):
    role_film         true
    compliance_weight 0
    perturb_frac      0
    div_weight        0
    role_dim          1
The job log's "[mifut] FiLM conditioning:" line reads the shell variable, which
was already correct on the two runs that trained without FiLM. Only the wandb
config reflects what argparse actually received.
MSG
