#!/bin/bash
#SBATCH --job-name=cmaps
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=02:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/cmaps_%j.out
#SBATCH --error=/scratch/e452103/logs/cmaps_%j.err

# Build the 4 per-cluster training map sets (stop_frac K-means strata):
# per cluster K, every scene with >=1 cluster-K vehicle is re-emitted with
# ONLY those vehicles controllable; all other objects mark_as_expert=1
# (GT replay background). See make_cluster_maps.py.
#
# JSON_DIR must be the processed Waymo scenario JSONs the training binaries
# were built from. NOTE the download nests the folder twice:
#   data/processed/training/data/processed/training  <- the JSONs live here
# Verify with: ls "$JSON_DIR"/*.json | head
#
#   sbatch slurm/make_cluster_maps.sbatch
#   sbatch --export=ALL,JSON_DIR=/path/to/jsons slurm/make_cluster_maps.sbatch

CSV=${CSV:-/scratch/e452103/traj_atlas/trajectory_clusters_stopfrac.csv}
JSON_DIR=${JSON_DIR:-/scratch/e452103/PufferDrive/data/processed/training/data/processed/training}
OUT_ROOT=${OUT_ROOT:-/scratch/e452103/cluster_maps}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
CODE_DIR=$HOME/roma_pufferdrive

mkdir -p /scratch/e452103/logs

[ -f "$CSV" ] || { echo "ERROR: clusters csv not found: $CSV (freeze it first, see render_role_grid.sbatch)"; exit 1; }
ls "$JSON_DIR"/*.json >/dev/null 2>&1 || { echo "ERROR: no JSONs in $JSON_DIR -- check the nested download path"; exit 1; }

python "$CODE_DIR/make_cluster_maps.py" \
    --clusters_csv "$CSV" \
    --json_dir     "$JSON_DIR" \
    --out_root     "$OUT_ROOT" \
    --clusters 0,1,2,3 \
    --workers  16

echo "Done -> $OUT_ROOT"
