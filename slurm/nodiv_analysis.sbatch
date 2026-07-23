#!/bin/bash
#SBATCH --job-name=nodiv_an
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=nodiv_an_%j.out
#SBATCH --error=nodiv_an_%j.err

# ===========================================================================
# TRAJECTORY-CLUSTER role analysis for a finished checkpoint (default: the
# diversity-off dim-4 run). EVERYTHING here is stratified by TRAJECTORY
# CLUSTER (trajectory_clusters_stopfrac.csv: mid-speed / fast / stop&go /
# turning). Map-regime clustering is NOT used anywhere.
#
# The GPU part. The trajectory-cluster VIDEO grid is a separate CPU job --
# submit both with one command via slurm/run_nodiv_analysis.sh, or chain the
# render yourself with slurm/render_grid_traj.sbatch afterok.
#
# WOSAC is elsewhere (eval_nodiv). Each step is its own python process so the
# C binding cannot segfault on repeated Drive() inits.
#
# Pipeline (all trajectory-cluster based):
#   0. build a scenario->trajectory-cluster scaffold so the sweep's --clusters
#      is driven by the TRAJECTORY clustering, not map regimes
#   1. role_paired_sweep       PC1/PC2 paired dose-response + axes       [GPU]
#   2. role_paired_by_traj     dose-response SPLIT BY TRAJECTORY CLUSTER [cpu]
#   3. role_scene_consistency  per-scene consistency, per traj cluster   [cpu]
#   4. role_space_map          2-D PCA role-space map                    [cpu]
#   5. role_feature_corr       role-role + role-vs-behaviour correlations[cpu]
#
# Usage on the account that HOLDS the checkpoint (defaults -> /scratch/$USER):
#   sbatch slurm/nodiv_analysis.sbatch
#   sbatch --export=ALL,CKPT=/scratch/$USER/checkpoints/roma_nodiv_dim4/roma_dim4_final.pt \
#       slurm/nodiv_analysis.sbatch

set -u

ROLE_DIM=${ROLE_DIM:-4}
TAG=${TAG:-nodiv_dim4}

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-$PUFFER_DIR/pufferlib/resources/drive/binaries/training}
CKPT=${CKPT:-$SCRATCH_ROOT/checkpoints/roma_nodiv_dim4/roma_dim${ROLE_DIM}_final.pt}
OUT_ROOT=${OUT_ROOT:-$SCRATCH_ROOT/analysis/$TAG}

# THE clustering. Everything is keyed to this. Frozen stop_frac trajectory
# K-means: scenario_id, vehicle_id -> cluster. Built once on the original
# analysis account; the nodiv run trained on the SAME Waymo binaries, so it
# applies unchanged -- copy it to this account's scratch if missing.
# TRAJ: prefer this account's own scratch copy; otherwise fall back to the copy
# committed in the repo under data/. trajectory_clusters_stopfrac.csv is a DATA
# artifact of the trajectory-clustering pipeline -- it cannot be regenerated
# without the raw Waymo feature cache, so it ships with the repo and an account
# that never generated it still finds it after `git pull`.
if [ -z "${TRAJ:-}" ]; then
    if [ -f "$SCRATCH_ROOT/traj_atlas/trajectory_clusters_stopfrac.csv" ]; then
        TRAJ="$SCRATCH_ROOT/traj_atlas/trajectory_clusters_stopfrac.csv"
    else
        TRAJ="$CODE_DIR/data/trajectory_clusters_stopfrac.csv"
    fi
fi
# Verified against the K=4 centroid plot: 0=mid-speed 1=fast 2=stop&go 3=turning
TYPE_NAMES=${TYPE_NAMES:-0:mid-speed,1:fast,2:stop&go,3:turning}

ALPHAS=${ALPHAS:--2,-1,0,1,2}
WARMUP_EP=${WARMUP_EP:-3}
SWEEP_EP=${SWEEP_EP:-6}

module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1

# ---- Preflight ------------------------------------------------------------
fail=0
[ -f "$CKPT" ]      || { echo "ERROR: checkpoint not found: $CKPT"; fail=1; }
[ -d "$PUFFER_DIR" ]|| { echo "ERROR: PUFFER_DIR not found: $PUFFER_DIR"; fail=1; }
[ -d "$DATA_DIR" ]  || { echo "ERROR: DATA_DIR not found: $DATA_DIR"; fail=1; }
[ -f "$CODE_DIR/role_paired_sweep.py" ] || {
    echo "ERROR: analysis code not under CODE_DIR=$CODE_DIR (wrong submit dir/branch?)"; fail=1; }
if [ ! -f "$TRAJ" ]; then
    echo "ERROR: trajectory clustering not found: $TRAJ"
    echo "       Expected data/trajectory_clusters_stopfrac.csv in the repo"
    echo "       (ships via git pull) or a copy at"
    echo "       \$SCRATCH_ROOT/traj_atlas/trajectory_clusters_stopfrac.csv."
    echo "       If it is committed, 'git pull' fixes this. Everything in this"
    echo "       pipeline is keyed to it."
    fail=1
fi
[ "$fail" -eq 0 ] || { echo "[nodiv_an] preflight failed -- nothing submitted to the GPU."; exit 1; }

PAIRED="$OUT_ROOT/paired"
mkdir -p "$PAIRED"
SCAFFOLD="$PAIRED/scenario_traj_scaffold.csv"

echo "[nodiv_an] tag=$TAG role_dim=$ROLE_DIM"
echo "[nodiv_an] ckpt=$CKPT"
echo "[nodiv_an] out=$OUT_ROOT  traj_clusters=$TRAJ"
echo "[nodiv_an] branch=$(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?) commit=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo ?)"

cd "$PUFFER_DIR"
step() { echo; echo "==================== $* ===================="; }

# --- 0. scenario -> trajectory-cluster scaffold ----------------------------
# role_paired_sweep --clusters is keyed by scenario_id (one label per scene),
# but the trajectory clustering is per (scenario, vehicle). Collapse it to the
# scene's dominant trajectory cluster so the sweep is driven by the TRAJECTORY
# clustering rather than map regimes. This scaffold only decides scene
# eligibility + a coarse per-scene tag; role_paired_by_traj (step 2) reapplies
# the exact per-focal trajectory label from the full (sid,vid) clustering.
step "0/5 build scenario->trajectory scaffold"
python - "$TRAJ" "$SCAFFOLD" <<'PYEOF' || { echo "[nodiv_an] scaffold build failed"; exit 1; }
import sys, pandas as pd
tc = pd.read_csv(sys.argv[1])
tc.columns = [c.strip() for c in tc.columns]
scen = (tc.groupby("scenario_id")["cluster"]
          .agg(lambda s: s.value_counts().idxmax()).reset_index())
scen.to_csv(sys.argv[2], index=False)
print(f"[scaffold] {len(scen)} scenarios from {len(tc)} labeled trajectories; "
      f"cluster mix {scen['cluster'].value_counts().to_dict()}")
PYEOF

# --- 1. Paired PC1/PC2 dose-response (core stats + axes) -------------------
step "1/5 role_paired_sweep [GPU]"
python "$CODE_DIR/role_paired_sweep.py" \
    --checkpoint "$CKPT" \
    --clusters   "$SCAFFOLD" \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$PAIRED" \
    --n_axes 2 --alphas="$ALPHAS" \
    --regime_names "$TYPE_NAMES" \
    --warmup_episodes "$WARMUP_EP" --episodes "$SWEEP_EP" \
  || echo "[nodiv_an] WARNING: sweep exited nonzero -- dependent steps may be partial"

AGENT="$PAIRED/role_paired_agent.csv"
WARMUP="$PAIRED/role_paired_warmup.csv"

# --- 2. Dose-response SPLIT BY TRAJECTORY CLUSTER --------------------------
step "2/5 role_paired_by_traj [cpu]"
if [ -f "$AGENT" ]; then
    python "$CODE_DIR/role_paired_by_traj.py" \
        --in_dir "$PAIRED" --traj_clusters "$TRAJ" --cluster_names "$TYPE_NAMES" \
      || echo "[nodiv_an] WARNING: by_traj failed"
else
    echo "[nodiv_an] skip: $AGENT missing (sweep did not produce it)"
fi

# --- 3. Per-scene consistency, per trajectory cluster ----------------------
step "3/5 role_scene_consistency [cpu]"
if [ -f "$AGENT" ]; then
    python "$CODE_DIR/role_scene_consistency.py" \
        --agent_csv "$AGENT" --traj_clusters "$TRAJ" \
        --cluster_names "$TYPE_NAMES" --out_dir "$PAIRED" \
      || echo "[nodiv_an] WARNING: consistency failed"
else
    echo "[nodiv_an] skip: $AGENT missing"
fi

# --- 4. Role-space PCA map -------------------------------------------------
step "4/5 role_space_map [cpu]"
if [ -f "$WARMUP" ]; then
    python "$CODE_DIR/role_space_map.py" \
        --agent_csv "$WARMUP" --out "$PAIRED/role_space_map.png" \
      || echo "[nodiv_an] WARNING: role_space_map failed"
else
    echo "[nodiv_an] skip: $WARMUP missing"
fi

# --- 5. Role-role + role-vs-behaviour correlations -------------------------
step "5/5 role_feature_corr [cpu]"
if [ -f "$WARMUP" ]; then
    python "$CODE_DIR/role_feature_corr.py" \
        --agent_csv "$WARMUP" --out_dir "$PAIRED" \
      || echo "[nodiv_an] WARNING: role_feature_corr failed"
else
    echo "[nodiv_an] skip: $WARMUP missing"
fi

# --- Collect logs + write the folder index ---------------------------------
# Everything lands under ONE root ($OUT_ROOT) so it can be zipped and sent as a
# single archive. The chained render job (render_grid_traj) drops its videos in
# $OUT_ROOT/renders/role_grid and makes the final zip.
mkdir -p "$OUT_ROOT/logs"
cp -f "$SLURM_SUBMIT_DIR/nodiv_an_${SLURM_JOB_ID}.out" "$OUT_ROOT/logs/" 2>/dev/null || true
cp -f "$SLURM_SUBMIT_DIR/nodiv_an_${SLURM_JOB_ID}.err" "$OUT_ROOT/logs/" 2>/dev/null || true

cat > "$OUT_ROOT/README.txt" <<READMEEOF
Trajectory-cluster role analysis -- $TAG
checkpoint: $CKPT
trajectory clustering: $TRAJ  (0=mid-speed 1=fast 2=stop&go 3=turning)
generated: $(date)

EVERYTHING here is stratified by TRAJECTORY CLUSTER. No map-regime clustering.

paired/                            role-pairing analysis
  role_paired_agent.csv            one row per (episode, forced-role condition,
                                   scene, focal vehicle): the alpha condition +
                                   every measured behaviour metric. The raw table.
  role_paired_deltas.csv           paired deltas vs alpha=0 (same map+vehicle),
                                   per forcing direction x alpha.
  role_paired_tests.csv            HEADLINE stats: alpha +2 vs -2 same-map paired
                                   difference -- mean, sem, t, n -- per direction
                                   x metric. "does forcing the role move behaviour?"
  role_paired_PC1.png              dose-response of the paired deltas along PC1
  role_paired_PC2.png              same along PC2 (0 at alpha=0 by construction)
  role_paired_traj_PC1.png         *** THE distinction figure ***: the PC1
  role_paired_traj_PC2.png         dose-response SPLIT BY TRAJECTORY CLUSTER --
                                   one line per type. Shows the roles pull each
                                   trajectory type differently.
  role_paired_traj_deltas.csv      the numbers behind those two figures
  role_paired_traj_tests.csv       per-trajectory-cluster significance
  role_scene_consistency.csv       within-scene role consistency, per trajectory
                                   cluster (does a scene's agents get consistent
                                   roles?)
  role_paired_warmup.csv           natural (UNFORCED) roles + behaviour per agent
                                   -- the observational baseline
  role_paired_observational.png    natural role-projection vs behaviour scatter
  role_paired_axes.csv             the PC1/PC2 forcing directions + sigma (also
                                   the input to the video grid)
  role_space_map.png               2-D PCA of natural roles, coloured by speed,
                                   with the PC1/PC2 dials + alpha tick marks
  role_role_corr.png               role dims against each other (redundancy)
  role_feature_corr.png            role dims against raw behaviour features
  scenario_traj_scaffold.csv       internal: scene -> dominant trajectory cluster
                                   (drives sweep scene-eligibility; not a result)

renders/role_grid/                 trajectory-cluster VIDEOS (from the CPU job)
  *.mp4                            scenes_per_type scenes PER TRAJECTORY CLUSTER,
                                   each rendered across the PC1xPC2 forcing grid --
                                   watch how mid-speed/fast/stop&go/turning each
                                   respond to role forcing.
  grid_summary.csv                 per-cell metric summary for the grid

logs/                              the SLURM .out/.err of both jobs

CAVEAT on the paired STATS (not the videos): the sweep picks one focal per
scene = the farthest-driving vehicle, which skews toward fast/long trajectories.
So the role_paired_traj_* lines are solid for mid-speed/fast but thin (small n)
for stop&go/turning. The VIDEO grid does NOT have this bias -- it samples each
trajectory cluster evenly. For statistically strong per-type dose-response, use
the per-cluster-trained models instead.
READMEEOF

echo
echo "[nodiv_an] DONE -> $OUT_ROOT  (see README.txt for the file guide)"
echo "[nodiv_an] the chained render job adds renders/role_grid/ and zips the folder"
