#!/bin/bash
#SBATCH --job-name=wosac_cl
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --time=14:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=wosac_cl%j.out
#SBATCH --error=wosac_cl%j.err

# WOSAC + safety metrics for ONE per-cluster policy across the PC1 sweep
# (role_cluster_wosac.py). One job per cluster.
#
#   sbatch --export=ALL,CLUSTER=3 slurm/wosac_cluster.sbatch
#
# WALL CLOCK: cost is 5x a normal WOSAC run because every map batch is rolled
# out once per alpha. A 100-batch eval runs ~5h, so 40 batches x 5 conditions
# is ~10h and fits 14h with margin. If it does hit the wall clock the metrics
# written so far are still aggregated per batch, so a short run is not a lost
# run -- it just has fewer scenarios. (Was 70 batches / 23h; 40 keeps the
# turnaround under half a day and queues faster for a shorter --time.)
#
# CONTROL MODE IS NOT THE DEFAULT ONE, on purpose -- see the CONTROL_MODE
# comment below. drive.ini's eval.wosac_control_mode is control_wosac, which
# ignores mark_as_expert entirely and would control every agent in the scene.

set -u

CLUSTER=${CLUSTER:-}
[ -n "$CLUSTER" ] || { echo "ERROR: set CLUSTER=0..3"; exit 1; }

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}

ROLE_DIM=${ROLE_DIM:-4}
MAP_DIR=${MAP_DIR:-$SCRATCH_ROOT/cluster_maps/cluster${CLUSTER}}
CKPT=${CKPT:-$SCRATCH_ROOT/checkpoints/roma_cluster${CLUSTER}_dim${ROLE_DIM}/roma_dim${ROLE_DIM}_final.pt}
AXES=${AXES:-$SCRATCH_ROOT/role_paired/cluster${CLUSTER}/role_paired_axes.csv}
OUT=${OUT:-$SCRATCH_ROOT/wosac_cluster/cluster${CLUSTER}}

ALPHAS=${ALPHAS:--1,-0.5,0,0.5,1}
AXIS=${AXIS:-PC1}
BATCHES=${BATCHES:-40}
ROLLOUTS=${ROLLOUTS:-32}
# MUST be control_agents, not drive.ini's eval default of control_wosac.
# drive.h should_control_agent(): the CONTROL_WOSAC branch returns early on
# "valid types only, ignore expert flag and goal distance", so it controls
# EVERY vehicle/pedestrian/cyclist and never consults mark_as_expert -- which
# would put 100% of the scene under policy control and destroy the per-cluster
# evaluation. control_agents falls through to the default branch, which does
# check mark_as_expert, and is the mode training used.
CONTROL_MODE=${CONTROL_MODE:-control_agents}

set +u   # conda compiler activation scripts (gcc_linux-64) are not set -u safe: SYS_SYSROOT/CONDA_BACKUP_CXX unbound
module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
set -u

export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1

# ---- Preflight ------------------------------------------------------------
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
[ -d "$MAP_DIR" ] || { echo "ERROR: cluster maps not found: $MAP_DIR"; exit 1; }
[ -f "$AXES" ] || {
    echo "ERROR: axes csv not found: $AXES"
    echo "       This is role_paired_axes.csv from role_paired_sweep.py for"
    echo "       cluster $CLUSTER -- it supplies the PC1 direction and sigma."
    echo "       Point AXES= at the right per-cluster sweep output."
    exit 1; }
NMAPS=$(ls "$MAP_DIR"/map_*.bin 2>/dev/null | wc -l)
[ "$NMAPS" -gt 0 ] || { echo "ERROR: no map_*.bin in $MAP_DIR"; exit 1; }

# control_agents excludes agents flagged mark_as_expert. If the cluster maps
# carry no expert flags there is nothing to exclude, the whole scene ends up
# under policy control, and the per-cluster evaluation is meaningless -- which
# is exactly what happened before (controlled_frac=1.0 with control_agents
# passed). Cost: reading one map header. Worth it against 10h of GPU.
if [ "${ALLOW_FULL_CONTROL:-0}" != "1" ]; then
python - "$MAP_DIR" "$CODE_DIR" <<'PYEOF' || exit 1
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from map_binary import read_map_binary
fp = sorted(Path(sys.argv[1]).glob("map_*.bin"))[0]
objs = read_map_binary(fp)["objects"]
n_exp = sum(bool(o["mark_as_expert"]) for o in objs)
print(f"[wosac_cl] expert check on {fp.name}: {n_exp}/{len(objs)} marked expert")
if n_exp == 0:
    print("[wosac_cl] ERROR: no mark_as_expert agents in these maps.")
    print("           control_agents has nothing to exclude, so 100% of the")
    print("           scene would be policy-controlled and this would NOT be a")
    print("           per-cluster eval. Regenerate the cluster maps with")
    print("           make_cluster_maps.py, or set ALLOW_FULL_CONTROL=1 if a")
    print("           whole-scene eval is genuinely what you want.")
    raise SystemExit(1)
PYEOF
else
    echo "[wosac_cl] ALLOW_FULL_CONTROL=1 -- skipping the expert check"
fi

mkdir -p "$OUT"
echo "[wosac_cl] cluster=$CLUSTER maps=$NMAPS ckpt=$CKPT"
echo "[wosac_cl] axes=$AXES alphas=$ALPHAS batches=$BATCHES"

CM_ARG=""
[ -n "$CONTROL_MODE" ] && CM_ARG="--control_mode $CONTROL_MODE"
# Threaded through so the python-side batch-1 abort matches this job's intent.
AFC_ARG=""
[ "${ALLOW_FULL_CONTROL:-0}" = "1" ] && AFC_ARG="--allow_full_control"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

python "$CODE_DIR/role_cluster_wosac.py" \
    --checkpoint        "$CKPT" \
    --data_dir          "$MAP_DIR" \
    --axes_csv          "$AXES" \
    --axis              "$AXIS" \
    --alphas="$ALPHAS" \
    --role_dim          "$ROLE_DIM" \
    --out_dir           "$OUT" \
    --num_maps          "$NMAPS" \
    --wosac_rollouts    "$ROLLOUTS" \
    --wosac_max_batches "$BATCHES" \
    $CM_ARG $AFC_ARG || { echo "[wosac_cl] FAILED (python exit $?)"; exit 1; }

echo "[wosac_cl] done -> $OUT"
