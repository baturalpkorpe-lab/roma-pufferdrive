#!/bin/bash
#SBATCH --job-name=role_videos
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=05:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/role_videos_%j.out
#SBATCH --error=/scratch/e452103/logs/role_videos_%j.err

# Controlled role-condition videos for the dim-4 checkpoint:
# 4 regimes x 2 maps x 3 forced cluster-mean roles = 24 MP4s + 8 overlay PNGs.
# Same scene + same action noise per map; only the focal agent's role differs.

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
CODE_DIR=$HOME/roma_pufferdrive

mkdir -p /scratch/e452103/logs
# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

# 3072 agents over 128 maps (~24/scene) so the focal vehicle is reliably
# dealt into the allocation. --max_resamples 20 so rare regimes (parking,
# congested) get enough distinct scenes; --max_relocate 150 as insurance.
python "$CODE_DIR/render_role_conditions.py" \
    --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim4/roma_dim4_step3000238080.pt \
    --clusters   /scratch/e452103/map_atlas/map_clusters.csv \
    --agent_data /scratch/e452103/role_regime/dim4/phaseC_agent_data.csv \
    --data_dir   pufferlib/resources/drive/binaries/training \
    --out_dir    /scratch/e452103/renders/role_conditions_dim4 \
    --map_pool     128 \
    --total_agents 3072 \
    --max_resamples 20 \
    --max_relocate 150

echo "Videos in: /scratch/e452103/renders/role_conditions_dim4"
