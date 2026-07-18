#!/bin/bash
#SBATCH --job-name=cl_post
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=00:30:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=/scratch/e452103/logs/cl_post_%j.out
#SBATCH --error=/scratch/e452103/logs/cl_post_%j.err

# Post-processing for a per-cluster sweep (run AFTER role_paired_cluster_gpu):
# filter the agent CSV to the cluster's OWN focals (kills the select_focals
# leakage -- the ~40% fast interlopers), then per-scene consistency on the
# clean set + the role-space map. Chain it:
#   JID=$(sbatch --parsable --export=ALL,CLUSTER=2 slurm/role_paired_cluster_gpu.sbatch)
#   sbatch --dependency=afterok:$JID --export=ALL,CLUSTER=2 slurm/cluster_post.sbatch

CLUSTER=${CLUSTER:-}
[ -n "$CLUSTER" ] || { echo "ERROR: set CLUSTER=0..3"; exit 1; }
IN=${IN:-/scratch/e452103/role_paired/cluster${CLUSTER}}
OUT=${OUT:-/scratch/e452103/role_paired/cluster${CLUSTER}_only}
TRAJ=${TRAJ:-/scratch/e452103/traj_atlas/trajectory_clusters_stopfrac.csv}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
export PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive
export PYTHONUNBUFFERED=1
CODE_DIR=$HOME/roma_pufferdrive

[ -f "$IN/role_paired_agent.csv" ] || { echo "ERROR: $IN/role_paired_agent.csv missing (sweep not done?)"; exit 1; }
mkdir -p "$OUT"

python - "$IN" "$OUT" "$TRAJ" "$CLUSTER" <<'PYEOF'
import sys, shutil
import pandas as pd
d, out, traj, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
df = pd.read_csv(f"{d}/role_paired_agent.csv")
tc = pd.read_csv(traj).rename(columns={"scenario_id": "sid", "vehicle_id": "vid"})
m = df.merge(tc[["sid", "vid", "cluster"]], on=["sid", "vid"], how="inner")
kept = m[m["cluster"] == k].drop(columns="cluster")
kept.to_csv(f"{out}/role_paired_agent.csv", index=False)
for f in ["role_paired_warmup.csv", "role_paired_axes.csv"]:
    shutil.copy(f"{d}/{f}", f"{out}/{f}")
print(f"cluster-{k}-only focal rows: {len(kept)}/{len(df)}")
PYEOF

python "$CODE_DIR/role_scene_consistency.py" \
    --agent_csv "$OUT/role_paired_agent.csv" --out_dir "$OUT"

# Regenerate the by-trajectory-type dose-response plot too: the copy already
# in $IN was built by role_paired_cluster_gpu.sbatch straight from the RAW
# (unfiltered) agent CSV, so it still shows all 4 traj types (the
# select_focals leakage). On the FILTERED CSV every row already has
# traj_cluster==CLUSTER, so this collapses to the one genuine line.
python "$CODE_DIR/role_paired_by_traj.py" \
    --in_dir "$OUT" --traj_clusters "$TRAJ"

python "$CODE_DIR/role_space_map.py" \
    --agent_csv "$IN/role_paired_warmup.csv" \
    --out "$IN/role_space_map.png"

echo "Done -> $OUT (consistency + clean single-type role_paired_traj_PC1/2.png) + $IN/role_space_map.png"
