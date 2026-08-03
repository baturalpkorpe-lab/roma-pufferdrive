#!/bin/bash
#SBATCH --job-name=regime_audit
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=02:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=regime_audit_%j.out
#SBATCH --error=regime_audit_%j.err

# Right-of-way audit + regime/conflict extraction over the GT map pool.
#
# NO GPU. Reads map binaries and does trajectory geometry in numpy, so it costs
# effectively nothing against fairshare (currently 0.0123 -- one 24h gpu-a100
# job is 28.7M billing units, this is noise).
#
# Two stages, both pure CPU:
#   1. junction_control_audit.py  -- how many junctions of each right-of-way
#      class exist, and how much traffic uses them. This is the GO/NO-GO gate:
#      Rahmani et al. got ~2,600 conflicts out of 1,570 hours, so the strict
#      all-way-stop filter may not have the supply to support per-cluster work
#      here. Decide on the printed number, not in advance.
#   2. regime_extract.py -- conflicts (with PET / minTTC / MRD / time
#      advantage) plus per-trajectory regime fractions and style parameters
#      (desired headway, jam spacing, free-flow speed ratio). GT only: this is
#      the human reference distribution every rollout number gets compared to.
#
# LOGIN NODE: for a look at the numbers before committing a job, both scripts
# run fine on a login node at --limit 300 (about a minute each). Use this
# sbatch for the full pool.
#
#   sbatch slurm/regime_audit.sbatch
#   sbatch --export=ALL,LIMIT=1000 slurm/regime_audit.sbatch      # subset
#
# PREFLIGHT (the missing-file failure mode that has bitten this project five
# times): analysis code lands on `baseline` and does NOT propagate between
# clones. From the clone you are submitting FROM:
#
#   git -C . fetch -q origin baseline && git -C . checkout origin/baseline -- \
#       conflict_metrics.py junction_control_audit.py regime_extract.py \
#       test_regimes.py slurm/regime_audit.sbatch

set -u

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-$PUFFER_DIR/pufferlib/resources/drive/binaries/training}
OUT=${OUT:-$SCRATCH_ROOT/regimes}
LIMIT=${LIMIT:-0}
# Sensitivity knob for the merge-vs-queue test. One lane width is
# ~3.2-3.7 m, so 3.5 is the permissive end and 5.0 the strict end.
MIN_OFFSET=${MIN_OFFSET:-5.0}

# conda's gcc_linux-64 activation scripts are not set -u safe
# (SYS_SYSROOT / CONDA_BACKUP_CXX unbound) and will kill the job on e452103.
set +u
module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
set -u

export PYTHONPATH=$CODE_DIR
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

for f in conflict_metrics.py junction_control_audit.py regime_extract.py \
         gap_acceptance.py intersection_features.py map_binary.py; do
    [ -f "$CODE_DIR/$f" ] || { echo "ERROR: missing $CODE_DIR/$f -- see the"\
        "PREFLIGHT block at the top of this file"; exit 1; }
done
[ -d "$DATA_DIR" ] || { echo "ERROR: no data_dir $DATA_DIR"; exit 1; }

mkdir -p "$OUT"
echo "[regime] data=$DATA_DIR out=$OUT limit=$LIMIT min_approach_offset=$MIN_OFFSET"
echo "[regime] branch=$(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?)"\
     "commit=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo ?)"
echo "[regime] maps available: $(ls "$DATA_DIR"/map_*.bin 2>/dev/null | wc -l)"

# The geometry self-test is the cheap version of the two crashes this week:
# both were scope errors in code that unit tests never entered.
echo; echo "=== 0/2  geometry self-test ==="
python "$CODE_DIR/test_regimes.py" || { echo "ERROR: self-test failed, not"\
    "running the audit"; exit 1; }

echo; echo "=== 1/2  junction right-of-way audit ==="
python "$CODE_DIR/junction_control_audit.py" \
    --data_dir "$DATA_DIR" \
    --out "$OUT/junction_control_audit.csv" \
    --limit "$LIMIT"

echo; echo "=== 2/2  regime + conflict extraction ==="
python "$CODE_DIR/regime_extract.py" \
    --data_dir "$DATA_DIR" \
    --out_conflicts "$OUT/conflicts_gt.csv" \
    --out_traj      "$OUT/regimes_gt.csv" \
    --min_approach_offset "$MIN_OFFSET" \
    --limit "$LIMIT"

echo; echo "[regime] done -> $OUT"
ls -la "$OUT"
