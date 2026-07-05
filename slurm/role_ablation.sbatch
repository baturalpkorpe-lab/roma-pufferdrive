#!/bin/bash
#SBATCH --job-name=role_ablation
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/role_ablation_%j.out
#SBATCH --error=/scratch/e452103/logs/role_ablation_%j.err

# The "is the role useful?" experiment: WOSAC + heterogeneity-calibration
# under natural / shuffled / collapsed roles, on both 3B checkpoints.
# ~15 batches x 3 conditions x 2 checkpoints -> expect 5-8h total.

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

echo "=== role ablation: dim-8 checkpoint ==="
python "$CODE_DIR/role_ablation_eval.py" \
    --checkpoint /scratch/e452103/checkpoints/roma_baseline/roma_dim8_step3000238080.pt \
    --agent_data /scratch/e452103/role_regime/dim8/phaseC_agent_data.csv \
    --data_dir   pufferlib/resources/drive/binaries/training \
    --out_dir    /scratch/e452103/role_ablation/dim8 \
    --batches    15

echo "=== role ablation: dim-4 checkpoint ==="
python "$CODE_DIR/role_ablation_eval.py" \
    --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim4/roma_dim4_step3000238080.pt \
    --agent_data /scratch/e452103/role_regime/dim4/phaseC_agent_data.csv \
    --data_dir   pufferlib/resources/drive/binaries/training \
    --out_dir    /scratch/e452103/role_ablation/dim4 \
    --batches    15

echo "Results: /scratch/e452103/role_ablation/{dim8,dim4}/"
