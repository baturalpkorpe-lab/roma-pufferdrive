#!/bin/bash
#SBATCH --job-name=regime_vid
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=03:00:00
#SBATCH --account=research-ceg-tp
#SBATCH --output=regime_vid_%j.out
#SBATCH --error=regime_vid_%j.err

# Videos of a policy rollout with every vehicle coloured by its CURRENT regime.
#
# This exists because the --wrap form does not work: --wrap inherits the
# submitting shell and therefore only finds numpy if conda happened to be active
# when you typed sbatch. Submitted from a fresh login shell it dies with
# "No module named 'numpy'" after the job has already been scheduled. Activate
# explicitly, like every other sbatch here.
#
#   sbatch --export=ALL,CKPT=...,ZONES=...,OUT=... slurm/regime_video.sbatch
#
# SUBMIT FROM THE CLONE THAT MATCHES THE CHECKPOINT. CODE_DIR defaults to the
# submit dir. NOTE the render_topdown guard below: baseline's load_policy does
# NOT read role_road_dim / role_partner_dim / role_film, so fetching that one
# file from baseline into a film clone makes an r16p64 checkpoint fail with a
# state-dict size mismatch (64x112 vs 64x128). Analysis scripts that merely CALL
# load_policy are safe to fetch; render_topdown.py itself is not.

set -u

CKPT=${CKPT:-}
[ -n "$CKPT" ] || { echo "ERROR: set CKPT=<checkpoint.pt>"; exit 1; }

SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PUFFER_DIR=${PUFFER_DIR:-$SCRATCH_ROOT/PufferDrive}
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
ZONES=${ZONES:-$SCRATCH_ROOT/regimes/junction_control_audit.csv}
OUT=${OUT:-$SCRATCH_ROOT/regime_videos}
N_VIDEOS=${N_VIDEOS:-10}
MAP_POOL=${MAP_POOL:-64}
TOTAL_AGENTS=${TOTAL_AGENTS:-512}
MIN_VEHICLES=${MIN_VEHICLES:-4}
MIN_SWITCHES=${MIN_SWITCHES:-3}
DATA_DIR=${DATA_DIR:-pufferlib/resources/drive/binaries/training}

# conda's gcc activation scripts are not set -u safe on e452103
set +u
module purge
module load miniconda3
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate roma
set -u

export PYTHONPATH=$CODE_DIR:$PUFFER_DIR
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

python -c "import numpy, torch, matplotlib" || {
    echo "ERROR: conda env 'roma' is not usable in this job"; exit 1; }

[ -f "$CKPT" ]  || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
[ -f "$ZONES" ] || { echo "ERROR: zones csv not found: $ZONES"; exit 1; }

# Every local module the renderer needs, directly or transitively.
for m in render_regime_video render_regime_strip conflict_metrics \
         junction_control_audit intersection_features map_binary \
         render_topdown role_regime_analysis traj_kinematics; do
    [ -f "$CODE_DIR/$m.py" ] || {
        echo "ERROR: missing $CODE_DIR/$m.py"
        echo "  git -C $CODE_DIR fetch -q origin baseline && \\"
        echo "  git -C $CODE_DIR checkout origin/baseline -- $m.py"
        echo "  (but NEVER fetch render_topdown.py from baseline into a film clone)"
        exit 1; }
done
grep -q "role_road_dim" "$CODE_DIR/render_topdown.py" || {
    echo "ERROR: $CODE_DIR/render_topdown.py has no role_road_dim handling."
    echo "  It is baseline's copy, which cannot load an r16p64 checkpoint."
    echo "  git -C $CODE_DIR checkout origin/mi-future-agentcentric-film -- render_topdown.py"
    exit 1; }

mkdir -p "$OUT"
echo "[vid] ckpt=$CKPT"
echo "[vid] code=$CODE_DIR branch=$(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?)"
echo "[vid] out=$OUT  n_videos=$N_VIDEOS  map_pool=$MAP_POOL"
echo "[vid] ffmpeg: $(python -c 'from matplotlib import animation; print(animation.FFMpegWriter.isAvailable())')"

# C binding loads pufferlib/config/ocean/drive.ini via a relative path.
cd "$PUFFER_DIR"

python "$CODE_DIR/render_regime_video.py" \
    --checkpoint    "$CKPT" \
    --zones_csv     "$ZONES" \
    --data_dir      "$DATA_DIR" \
    --out_dir       "$OUT" \
    --n_videos      "$N_VIDEOS" \
    --map_pool      "$MAP_POOL" \
    --total_agents  "$TOTAL_AGENTS" \
    --min_vehicles  "$MIN_VEHICLES" \
    --min_switches  "$MIN_SWITCHES" \
    --device        cpu \
  || { echo "[vid] FAILED (python exit $?)"; exit 1; }

echo "[vid] done -> $OUT"
ls -la "$OUT"
