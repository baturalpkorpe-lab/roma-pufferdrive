#!/bin/bash
#SBATCH --job-name=render_gT
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=03:45:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/render_gT_%j.out
#SBATCH --error=/scratch/e452103/logs/render_gT_%j.err

# ONE-TYPE slice of render_role_grid.sbatch: run 4x (TYPE=0..3) instead of
# one 16h job, so each piece fits DelftBlue's <4h fast-track queue tier
# instead of sitting at general Priority. Filters the frozen strata CSV down
# to a single cluster (render_role_grid.py has no --types flag, so this pre-
# filters the CSV instead of touching the script) and writes into its own
# out_dir subfolder so the 4 runs never collide.
#
# Also usable for a PER-CLUSTER-trained checkpoint: point CKPT/AXES/DATA_DIR
# at that cluster's own checkpoint/axes/maps (cluster_maps/cluster<K>) so the
# rendered scenes only ever control genuinely-that-cluster vehicles -- e.g.
# the stop&go-trained policy's own PC1=turning / PC2=braking-throttle effect.
#
#   sbatch --export=ALL,TYPE=0 slurm/render_role_grid_type.sbatch
#   sbatch --export=ALL,TYPE=1 slurm/render_role_grid_type.sbatch
#   sbatch --export=ALL,TYPE=1,DATA_DIR=/scratch/e452103/cluster_maps/cluster1,\
#CKPT=/scratch/e452103/checkpoints/roma_cluster1_dim4/roma_dim4_final.pt,\
#AXES=/scratch/e452103/role_paired/cluster1/role_paired_axes.csv,\
#OUT_ROOT=/scratch/e452103/renders/role_grid_cluster1 slurm/render_role_grid_type.sbatch
#   ... run one at a time, or queue all 4 -- they write disjoint out_dirs.

TYPE=${TYPE:-}
[ -n "$TYPE" ] || { echo "ERROR: set TYPE=0..3"; exit 1; }

CKPT=${CKPT:-/scratch/e452103/checkpoints/roma_baseline_dim2/roma_dim2_final.pt}
AXES=${AXES:-/scratch/e452103/role_paired/dim2/role_paired_axes.csv}
TRAJ=${TRAJ:-/scratch/e452103/traj_atlas/trajectory_clusters_stopfrac.csv}
DATA_DIR=${DATA_DIR:-pufferlib/resources/drive/binaries/training}
OUT_ROOT=${OUT_ROOT:-/scratch/e452103/renders/role_grid_dim2}
OUT=${OUT:-$OUT_ROOT/type${TYPE}}
SCENES=${SCENES:-3}
NAMES=${NAMES:-"0:mid-speed,1:fast,2:stop&go,3:turning"}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
CODE_DIR=$HOME/roma_pufferdrive

mkdir -p /scratch/e452103/logs "$OUT"

# freeze the stop_frac clustering so later atlas reruns can't change our strata
if [ ! -f "$TRAJ" ]; then
    cp /scratch/e452103/traj_atlas/trajectory_clusters.csv "$TRAJ"
    echo "frozen strata copy -> $TRAJ"
fi

# filter to just this cluster -- render_role_grid.py takes whatever clusters
# are present in --traj_clusters, so a 1-cluster CSV = a 1-type render.
TRAJ_TYPE="$OUT/traj_clusters_type${TYPE}.csv"
python3 -c "
import pandas as pd, sys
df = pd.read_csv('$TRAJ')
df = df[df['cluster'] == $TYPE]
if len(df) == 0:
    sys.exit(f'ERROR: no rows with cluster == $TYPE in $TRAJ')
df.to_csv('$TRAJ_TYPE', index=False)
print(f'[split] cluster $TYPE: {len(df)} labeled trajectories -> $TRAJ_TYPE')
"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

python "$CODE_DIR/render_role_grid.py" \
    --checkpoint    "$CKPT" \
    --axes_csv      "$AXES" \
    --traj_clusters "$TRAJ_TYPE" \
    --data_dir      "$DATA_DIR" \
    --out_dir       "$OUT" \
    --alphas="-2,-1,0,1,2" --grid 1 \
    --scenes_per_type "$SCENES" \
    --type_names    "$NAMES"

echo "Done -> $OUT"
