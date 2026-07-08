#!/bin/bash
#SBATCH --job-name=map_cluster
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=00:20:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/map_cluster_%j.out
#SBATCH --error=/scratch/e452103/logs/map_cluster_%j.err

# Re-cluster ONLY (Phase B) on the existing map_features.csv -- no env, no GPU,
# no re-featurization. Uses the trimmed 5-feature set (n_vehicles, speed_p85,
# speed_std_across, stop_frac, curviness); junk maps (turn_rate > 0.5 rad/s) are
# dropped BEFORE clustering. Runs K=3 and K=5 into separate subdirs so their
# atlas_*.png + map_clusters.csv don't overwrite each other.

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
CODE_DIR=$HOME/roma_pufferdrive
IN_DIR=/scratch/e452103/map_atlas    # holds map_features.csv + geometry_cache.pkl

mkdir -p /scratch/e452103/logs

for K in 3 5; do
    echo "=== clustering K=$K ==="
    python "$CODE_DIR/map_atlas.py" \
        --in_dir  "$IN_DIR" \
        --out_dir "$IN_DIR/k$K" \
        --k "$K"
done

echo "Done. Atlases in: $IN_DIR/k3 and $IN_DIR/k5"
