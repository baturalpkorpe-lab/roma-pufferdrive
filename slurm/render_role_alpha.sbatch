#!/bin/bash
#SBATCH --job-name=render_alpha
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=05:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/render_alpha_%j.out
#SBATCH --error=/scratch/e452103/logs/render_alpha_%j.err

# Same-scene alpha-sweep videos: per regime, one core scene rolled at
# alpha=-2,-1,0,1,2 along PC1 (focal forced, others natural, GT background).
# 4 regimes x 5 alphas = 20 videos + 4 overlays + per-scene metric CSVs.
# Axes are read from role_paired_axes.csv so videos match the statistics --
# run role_paired_sweep first (or drop --axes_csv to let it warm up its own).
#
# Defaults render the BASELINE dim4 checkpoint on the map_atlas/k4 regimes.
# For a PER-CLUSTER-trained checkpoint, override CKPT/AXES/DATA_DIR/OUT (the
# --clusters regime atlas stays map_atlas/k4 -- it's keyed by scenario_id,
# which survives the per-cluster binary rewrite):
#   sbatch --export=ALL,CKPT=/scratch/e452103/checkpoints/roma_cluster3_dim4/roma_dim4_final_2B.pt,\
#     AXES=/scratch/e452103/role_paired/cluster3_2B/role_paired_axes.csv,\
#     DATA_DIR=/scratch/e452103/cluster_maps/cluster3,\
#     OUT=/scratch/e452103/renders/role_alpha_cluster3_2B \
#     slurm/render_role_alpha.sbatch

CKPT=${CKPT:-/scratch/e452103/checkpoints/roma_baseline_dim4/roma_dim4_final.pt}
CLUSTERS=${CLUSTERS:-/scratch/e452103/map_atlas/k4/map_clusters.csv}
DATA_DIR=${DATA_DIR:-pufferlib/resources/drive/binaries/training}
OUT=${OUT:-/scratch/e452103/renders/role_alpha_dim4}
AXES=${AXES:-/scratch/e452103/role_paired/dim4/role_paired_axes.csv}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

# CODE_DIR defaults to the SUBMITTING clone, not to $HOME/roma_pufferdrive.
# That path is on role-agent-centric despite its name, so hardcoding it ran
# another branch's policy classes against whatever checkpoint was passed --
# a FiLM or agent-centric checkpoint then dies in load_state_dict on the extra
# keys. Same fix already applied to role_natural_extremes.sbatch and
# eval_wosac.sbatch; this one was missed.
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
export PYTHONPATH=$CODE_DIR:${PUFFER_DIR:-/scratch/$USER/PufferDrive}
export PYTHONUNBUFFERED=1

mkdir -p /scratch/e452103/logs "$OUT"
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
echo "ckpt=$CKPT  clusters=$CLUSTERS  data_dir=$DATA_DIR  out=$OUT"
# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

# --axes_csv is only passed if the paired sweep (new enough to save axes)
# produced it; otherwise the renderer's own warmup estimates PC1 (85% of role
# variance -- the estimated direction is essentially identical).
AXES_ARG=""
if [ -f "$AXES" ]; then AXES_ARG="--axes_csv $AXES"; fi

python "$CODE_DIR/render_role_alpha.py" \
    --checkpoint "$CKPT" \
    --clusters   "$CLUSTERS" \
    $AXES_ARG \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$OUT" \
    --axis PC1 --alphas="-2,-1,0,1,2" --maps_per_regime 1

echo "Done -> $OUT/"
