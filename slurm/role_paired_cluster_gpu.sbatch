#!/bin/bash
#SBATCH --job-name=role_cl_gpu
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/role_cl_gpu_%j.out
#SBATCH --error=/scratch/e452103/logs/role_cl_gpu_%j.err

# FULL role analysis of a PER-CLUSTER-trained checkpoint, on GPU (much faster
# than the login-node CPU reduced version). Runs the paired forced-role sweep
# on the cluster's OWN map set (--data_dir = cluster_maps/cluster<K>, the maps
# it trained on) + the whole cheap downstream chain in one job:
#   1. role_paired_sweep.py      dose-response PNGs + agent/warmup/axes CSVs
#   2. role_paired_by_traj.py    re-stratified by trajectory type
#   3. role_scene_consistency.py per-scene consistency + monotone_rate
#   4. role_space_map.py         2-D role map (speed-colored, PC1/PC2 dial)
#   5. role_feature_corr.py      role-role + role-vs-feature correlations
#   6. bimodality coefficient    is this cluster's role continuous or collapsed?
#
#   sbatch --export=ALL,CLUSTER=1 slurm/role_paired_cluster_gpu.sbatch
#   sbatch --export=ALL,CLUSTER=0,EPISODES=12 slurm/role_paired_cluster_gpu.sbatch
#
# --clusters is the MAP-regime atlas (keyed by real scenario_id, preserved
# through the per-cluster binary rewrite, so it still resolves). trajectory
# stratification will be ~all one cluster here (that IS the point -- e.g.
# cluster1 = stop&go, which the all-data sweep was blind to at n=4).

CLUSTER=${CLUSTER:-}
[ -n "$CLUSTER" ] || { echo "ERROR: set CLUSTER=0..3"; exit 1; }

ROLE_DIM=${ROLE_DIM:-4}
MAP_DIR=${MAP_DIR:-/scratch/e452103/cluster_maps/cluster${CLUSTER}}
CKPT=${CKPT:-/scratch/e452103/checkpoints/roma_cluster${CLUSTER}_dim${ROLE_DIM}/roma_dim${ROLE_DIM}_final.pt}
OUT=${OUT:-/scratch/e452103/role_paired/cluster${CLUSTER}}
MAP_CLUSTERS=${MAP_CLUSTERS:-/scratch/e452103/map_atlas/k4/map_clusters.csv}
TRAJ_CLUSTERS=${TRAJ_CLUSTERS:-/scratch/e452103/traj_atlas/trajectory_clusters.csv}
EPISODES=${EPISODES:-10}
WARMUP=${WARMUP:-3}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma

export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
CODE_DIR=$HOME/roma_pufferdrive

mkdir -p /scratch/e452103/logs "$OUT"
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
# The per-cluster dir has FEWER maps than the 10k default; role_paired_sweep.py
# passes --num_maps straight into Drive(), which rejects num_maps > available.
# Count the cluster's actual maps and cap num_maps to it.
NUM_MAPS=$(ls "$MAP_DIR"/map_*.bin 2>/dev/null | wc -l)
[ "$NUM_MAPS" -gt 0 ] || { echo "ERROR: no maps in $MAP_DIR"; exit 1; }
echo "cluster=$CLUSTER  role_dim=$ROLE_DIM  ckpt=$CKPT  maps=$MAP_DIR ($NUM_MAPS)  episodes=$EPISODES"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd /scratch/e452103/PufferDrive

# -- 1. paired forced-role sweep (the GPU part) -------------------------------
python "$CODE_DIR/role_paired_sweep.py" \
    --checkpoint "$CKPT" \
    --clusters   "$MAP_CLUSTERS" \
    --data_dir   "$MAP_DIR" \
    --num_maps   "$NUM_MAPS" \
    --out_dir    "$OUT" \
    --n_axes 2 --alphas="-2,-1,0,1,2" \
    --warmup_episodes "$WARMUP" --episodes "$EPISODES" || exit 1

# -- 2-6. cheap downstream (login-node-class, but run here so it's one job) ----
python "$CODE_DIR/role_paired_by_traj.py" \
    --in_dir "$OUT" --traj_clusters "$TRAJ_CLUSTERS" || true

python "$CODE_DIR/role_scene_consistency.py" \
    --agent_csv "$OUT/role_paired_agent.csv" \
    --traj_clusters "$TRAJ_CLUSTERS" --out_dir "$OUT" || true

python "$CODE_DIR/role_space_map.py" \
    --agent_csv "$OUT/role_paired_warmup.csv" \
    --out "$OUT/role_space_map.png" || true

python "$CODE_DIR/role_feature_corr.py" \
    --agent_csv "$OUT/role_paired_warmup.csv" --out_dir "$OUT" || true

# bimodality: is this cluster's role a continuum or a collapsed binary?
python - "$OUT/role_paired_warmup.csv" <<'PYEOF' || true
import sys, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1])
def bc(z):
    z=np.asarray(z,float); z=z[np.isfinite(z)]; n=len(z)
    m,sd=z.mean(),z.std(); g=((z-m)**3).mean()/sd**3; kex=((z-m)**4).mean()/sd**4-3
    return (g**2+1)/(kex+3*(n-1)**2/((n-2)*(n-3)))
for c in [c for c in df.columns if c.startswith('role_') or c in ('pc1','pc2')]:
    v=bc(df[c]); print(f'[bimodality] {c}: BC={v:.3f}'
          f'{"  <== BIMODAL/collapsed" if v>0.555 else "  (continuous-ish)"}')
PYEOF

echo "Done -> $OUT"
