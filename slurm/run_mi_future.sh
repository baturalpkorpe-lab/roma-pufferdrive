#!/bin/bash
# One command for the future-MI run: training (3B, mid-run WOSAC every 500M),
# then the full 100-batch WOSAC + role analysis chained afterok.
#
#   bash slurm/run_mi_future.sh          # dim 4, H 8
#   bash slurm/run_mi_future.sh 4 16     # role_dim 4, mi_horizon 16
#
# Both jobs default to /scratch/$USER; export SCRATCH_ROOT / PUFFER_DIR / etc.
# to override. Submit from the mi-future clone root.

set -e

HERE=$(cd "$(dirname "$0")" && pwd)
D=${1:-4}
H=${2:-8}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
SAVE_DIR=${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/roma_mifuture_dim${D}_H${H}}

E="ALL,ROLE_DIM=$D,MI_HORIZON=$H,SAVE_DIR=$SAVE_DIR"
for v in SCRATCH_ROOT PUFFER_DIR TOTAL_STEPS WANDB_PROJECT WANDB_ENTITY; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && E="$E,$v=$val"
done

J=$(sbatch --parsable --export="$E" "$HERE/train_mi_future.sbatch")
echo "train (3B, 500M periodic) : $J"
EV=$(sbatch --parsable --dependency=afterok:$J --export="$E" "$HERE/eval_mi_future.sbatch")
echo "eval (100-batch + role)   : $EV  (afterok:$J)"
echo
echo "checkpoints -> $SAVE_DIR"
