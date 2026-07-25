#!/bin/bash
# One command for the future-MI run: training to 3B (mid-run WOSAC every 500M),
# then role analysis + the full 100-batch WOSAC chained afterok.
#
#   bash slurm/run_mi_future.sh                 # dim 4, H 8, div_weight 0.1
#   bash slurm/run_mi_future.sh 4 16            # role_dim 4, mi_horizon 16
#   DIV_WEIGHT=0 bash slurm/run_mi_future.sh    # the no-diversity ablation
#
# TWO jobs: 3B training fits the 24h wall, then the eval afterok. Not a resume
# chain -- checkpoints land every 500M, so a link resuming a job that died at
# 1.29B would restart from 1.0B and redo 290M steps. train_mi_future.sbatch
# still auto-resumes as a RECOVERY path: if the wall clock does cut it short,
# just run this script again and it continues from the last checkpoint.
#
# Both jobs default to /scratch/$USER; export SCRATCH_ROOT / PUFFER_DIR / etc.
# to override. Submit from the mi-future clone root.

set -e

HERE=$(cd "$(dirname "$0")" && pwd)
D=${1:-4}
H=${2:-8}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}

# Keep the ablation in the path: sharing a SAVE_DIR with the div-on run would
# have auto-resume silently continue the WRONG experiment.
DIV_WEIGHT=${DIV_WEIGHT:-0.1}
case "$DIV_WEIGHT" in 0|0.|0.0|0.00) DIV_TAG=_nodiv ;; *) DIV_TAG= ;; esac
SAVE_DIR=${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/roma_mifuture_dim${D}_H${H}${DIV_TAG}}

E="ALL,ROLE_DIM=$D,MI_HORIZON=$H,DIV_WEIGHT=$DIV_WEIGHT,SAVE_DIR=$SAVE_DIR"
for v in MI_WEIGHT SEED SCRATCH_ROOT PUFFER_DIR TOTAL_STEPS WANDB_PROJECT WANDB_ENTITY WANDB_NAME; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && E="$E,$v=$val"
done

J=$(sbatch --parsable --export="$E" "$HERE/train_mi_future.sbatch")
echo "train (3B, 500M periodic)              : $J"
EV=$(sbatch --parsable --dependency=afterok:$J --export="$E" "$HERE/eval_mi_future.sbatch")
echo "eval (role analysis + 100-batch WOSAC) : $EV  (afterok:$J)"
echo
echo "div_weight  -> $DIV_WEIGHT"
echo "checkpoints -> $SAVE_DIR"
