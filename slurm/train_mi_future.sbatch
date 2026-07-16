#!/bin/bash
#SBATCH --job-name=roma_mifut
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/roma_mifut_%j.out
#SBATCH --error=/scratch/e452103/logs/roma_mifut_%j.err

# FUTURE-prediction MI training (mi-future branch, dim=4): z_t must predict
# the behaviour summary of the NEXT 8 env embeddings (t+1..t+8) instead of
# the past window -- R3DM-inspired; the role becomes an intent signal rather
# than a recap. Full 10k map set, standard self-play (NOT per-cluster maps).
#
#   sbatch slurm/train_mi_future.sbatch
#   sbatch --export=ALL,MI_HORIZON=16,WANDB_PROJECT=roma-pufferdrive,WANDB_ENTITY=<team> \
#       slurm/train_mi_future.sbatch
#
# No --run_eval and --wosac_periodic 0: repeated Drive() instantiations in one
# long process segfault the C binding. WOSAC runs standalone afterwards.

ROLE_DIM=${ROLE_DIM:-4}
MI_HORIZON=${MI_HORIZON:-8}
SAVE_DIR=${SAVE_DIR:-/scratch/e452103/checkpoints/roma_mifuture_dim${ROLE_DIM}}
TOTAL_STEPS=${TOTAL_STEPS:-2000000000}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000000}
SEED=${SEED:-0}
NUM_MAPS=${NUM_MAPS:-10000}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

# Submit from the mi-future clone root (~/roma_mifuture) so this checkout's
# train_roma.py (future-MI) is the one that runs.
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
export PYTHONPATH=$CODE_DIR:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1

mkdir -p /scratch/e452103/logs "$SAVE_DIR"
echo "mi-future  role_dim=$ROLE_DIM  H=$MI_HORIZON  maps=$NUM_MAPS  code=$CODE_DIR"

# Optional wandb passthrough
WANDB_ARGS=""
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARGS="--wandb_project $WANDB_PROJECT"
    [ -n "$WANDB_ENTITY" ] && WANDB_ARGS="$WANDB_ARGS --wandb_entity $WANDB_ENTITY"
fi

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

python "$CODE_DIR/roma_pufferdrive/train_roma.py" \
    --role_dim      "$ROLE_DIM" \
    --mi_horizon    "$MI_HORIZON" \
    --data_dir      pufferlib/resources/drive/binaries/training \
    --num_maps      "$NUM_MAPS" \
    --num_agents    3072 \
    --device        cuda \
    --total_steps   "$TOTAL_STEPS" \
    --save_interval "$SAVE_INTERVAL" \
    --save_dir      "$SAVE_DIR" \
    --seed          "$SEED" \
    --wosac_periodic 0 \
    $WANDB_ARGS

echo "Done -> $SAVE_DIR"
