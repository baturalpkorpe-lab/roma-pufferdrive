#!/bin/bash
#SBATCH --job-name=roma_main_guid
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/roma_main_guid_%j.out
#SBATCH --error=/scratch/e452103/logs/roma_main_guid_%j.err

# Main-branch (guidance) training run:
#   - all guidance REWARDS active (use_guided_autonomy + speed/heading
#     penalties; guidance OBSERVATIONS stay off so obs_dim matches the
#     baseline architecture)
#   - kl_weight 0 (no KL regularizer on the role encoder)
#   - mid-run WOSAC every 500M steps at 10 batches
#   - post-training: role analysis (10 episodes) THEN full WOSAC (100 batches)
#
# PREREQUISITE: rebuild the guid stack's C binding with
#   -O3 -mavx2 -mfma -fopenmp (same SPS fix as the baseline stack) and
#   verify with bench_env_sps.sbatch BEFORE launching -- on the old -O2
#   build, 3B steps do not fit in 24h.
#
# CONFIRM before first submit:
#   - guidance_speed_weight / guidance_heading_weight values (1.0 is a
#     placeholder -- set the values agreed with your supervisor)
#   - DATA_DIR exists:  ls $DATA_DIR | head

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate /scratch/e452103/envs/roma_guid

export PYTHONPATH=$HOME/roma_main:/scratch/e452103/PufferDrive_guid
export PYTHONUNBUFFERED=1
# WANDB_API_KEY comes from ~/.bashrc at submit time

DATA_DIR=/scratch/e452103/PufferDrive_guid/pufferlib/resources/drive/binaries/training

mkdir -p /scratch/e452103/logs
# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive_guid

# Pre-flight: fail in 1s if the map binaries are missing, instead of crashing
# inside Drive() and leaving a hung wandb process holding the GPU. The guid
# stack needs its OWN binaries -- its binary format segfaults on the baseline
# ones -- generated via process_all_maps into $DATA_DIR.
if [ ! -f "$DATA_DIR/map_000.bin" ]; then
    echo "FATAL: no map binaries at $DATA_DIR -- generate them first." >&2
    exit 1
fi

python $HOME/roma_main/roma_pufferdrive/train_roma.py \
    --role_dim                8 \
    --num_maps                10000 \
    --total_steps             3000000000 \
    --num_agents              3072 \
    --device                  cuda \
    --kl_weight               0 \
    --use_guided_autonomy     1 \
    --guidance_speed_weight   1.0 \
    --guidance_heading_weight 1.0 \
    --waypoint_reach_threshold 2.0 \
    --use_guidance_observations 0 \
    --run_eval \
    --role_episodes           10 \
    --wosac_periodic          1 \
    --wosac_interval          500000000 \
    --wosac_eval_maps         10000 \
    --wosac_eval_rollouts     32 \
    --wosac_eval_batches      10 \
    --wosac_rollouts          32 \
    --wosac_num_maps          10000 \
    --wosac_max_batches       100 \
    --data_dir                "$DATA_DIR" \
    --save_dir                /scratch/e452103/checkpoints/roma_main_guidance \
    --save_interval           500000000 \
    --wandb_project           roma-pufferdrive \
    --wandb_entity            s-rahmani-tu-delft \
    --seed                    0
