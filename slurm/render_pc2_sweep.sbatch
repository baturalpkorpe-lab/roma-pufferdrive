#!/bin/bash
#SBATCH --job-name=pc2_sweep
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=03:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/pc2_sweep_%j.out
#SBATCH --error=/scratch/e452103/logs/pc2_sweep_%j.err

# PC2 sweep on 3 specific scenes from the dim-2 role_compare run: hold PC1
# fixed, sweep PC2 = -2..+2 (5 videos + overlay + metrics each). Answers
# "can PC2 tune the beginner-driver steering seen at PC1=-2?".
#
# BG=gt (default, matches the compare videos) or BG=sim (draw context cars at
# their ACTUAL simulated positions -- use to check the 'clone/ghost').
#   sbatch slurm/render_pc2_sweep.sbatch
#   sbatch --export=ALL,BG=sim slurm/render_pc2_sweep.sbatch

CKPT=${CKPT:-/scratch/e452103/checkpoints/roma_baseline_dim2/roma_dim2_final.pt}
AXES=${AXES:-/scratch/e452103/role_paired/dim2/role_paired_axes.csv}
# IMPORTANT: the SAME clusters CSV the compare run used (original k4, pre-stop_frac)
TRAJ=${TRAJ:-/scratch/e452103/traj_atlas/k4/trajectory_clusters.csv}
BG=${BG:-gt}
BASE=${BASE:-/scratch/e452103/renders/pc2_sweep_dim2}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
CODE_DIR=$HOME/roma_pufferdrive

mkdir -p /scratch/e452103/logs
cd /scratch/e452103/PufferDrive

run () {   # $1=sid_prefix  $2=traj_type  $3=pc1_alpha  $4=tag
    python "$CODE_DIR/render_pc2_sweep.py" \
        --checkpoint    "$CKPT" \
        --axes_csv      "$AXES" \
        --traj_clusters "$TRAJ" \
        --data_dir      pufferlib/resources/drive/binaries/training \
        --sid_prefix    "$1" --traj_type "$2" --pc1_alpha "$3" \
        --alphas="-2,-1,0,1,2" --background "$BG" \
        --out_dir       "$BASE/$4_bg$BG"
}

run 1b22f29f4c 2 -2  t2_1b22f29f4c_pc1m2
run d3b03a63b9 3 -2  t3_d3b03a63b9_pc1m2
run d3b03a63b9 3  2  t3_d3b03a63b9_pc1p2

echo "Done -> $BASE"
