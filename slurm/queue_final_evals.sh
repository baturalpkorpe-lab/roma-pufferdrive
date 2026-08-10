#!/bin/bash
# Queue the whole end-of-run evaluation NOW, to fire when training finishes.
# Nothing runs, and nothing is charged, until the training job leaves the queue.
#
#   bash slurm/queue_final_evals.sh                 # auto-detect the training job
#   bash slurm/queue_final_evals.sh 10620123        # or name it
#   RUN=roma_..._paperrew_nodiv TAG=paperrew_final bash slurm/queue_final_evals.sh
#
# Submits, all with --dependency=afterany on the training job:
#   REPS sweeps  (regime_rollout, ~20 min each on CPU)
#   REPS scene-ICC runs (~3 min each on CPU)
#
# WHY THREE OF EACH AND NOT ONE
# One sweep cannot resolve this. ep_nodiv rolled out four times ON IDENTICAL
# WEIGHTS gave headway ranges of 0.477 / 0.492 / 0.544 / 1.150 -- a factor of
# 2.4 -- and a single low draw at 4.5B had me believing the run was regressing
# when the five-checkpoint series showed it was not. ICC bounces too: 0.164 /
# 0.174 / 0.220 / 0.157 / 0.198 across consecutive checkpoints of one run.
# Quote a mean and a spread, never a single sweep.
#
# WHY afterany AND NOT afterok
# The final checkpoint is written before the job exits, and a training job that
# hits its wall clock still leaves a usable roma_dim<D>_final.pt. afterok would
# discard the evaluation in exactly the case where you most want it.
#
# WHY final.pt AND NOT THE LAST STEP CHECKPOINT
# Its name is fixed, so it can be referenced at SUBMIT time -- step checkpoints
# overshoot the round number (4750835712, not 4750000000) and cannot be named
# in advance. final.pt carries no aux_loss_state, which neither of these jobs
# needs: both only load policy_state.
set -eu

# set -u kills the script on an unset USER, which happens in stripped
# environments and inside some job steps.
USER=${USER:-$(whoami)}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
RUN=${RUN:-roma_mifutagent_dim1_H8_r16p64_ft_collstop_nodiv}
TAG=${TAG:-collstop_final}
ROLE_DIM=${ROLE_DIM:-1}
REPS=${REPS:-3}

SAVE_DIR=$SCRATCH_ROOT/checkpoints/$RUN
CK=$SAVE_DIR/roma_dim${ROLE_DIM}_final.pt
HERE=$(cd "$(dirname "$0")" && pwd)

JOB=${1:-}
if [ -z "$JOB" ]; then
    N=$(squeue -u "$USER" -h -n roma_mifua -o "%i" | wc -l)
    if [ "$N" -ne 1 ]; then
        echo "ERROR: found $N roma_mifua jobs, cannot guess which to wait for."
        echo "       Pass the job id as the first argument. Candidates:"
        squeue -u "$USER" -n roma_mifua -o "%.10i %.12j %.2t %.10M %Z"
        exit 1
    fi
    JOB=$(squeue -u "$USER" -h -n roma_mifua -o "%i")
fi

[ -d "$SAVE_DIR" ] || { echo "ERROR: no save dir $SAVE_DIR"; exit 1; }
for f in regime_rollout.sbatch role_scene_icc_cpu.sbatch; do
    [ -f "$HERE/$f" ] || { echo "ERROR: missing $HERE/$f"; exit 1; }
done

echo "[queue] waiting on training job : $JOB"
squeue -j "$JOB" -o "%.10i %.12j %.2t %.10M %.12L %Z" 2>/dev/null || \
    echo "  (not in the queue -- already finished? the dependents start at once)"
echo "[queue] checkpoint              : $CK"
echo "[queue] reps                    : $REPS sweeps + $REPS ICC"
echo

SW=""
for i in $(seq 1 "$REPS"); do
    J=$(sbatch --parsable --dependency=afterany:"$JOB" \
        --export=ALL,SWEEP=1,CKPT="$CK",OUT="$SCRATCH_ROOT/regimes_rollout/${TAG}_rep$i" \
        "$HERE/regime_rollout.sbatch")
    echo "  sweep rep$i : $J"
    SW="$SW $J"
done
for i in $(seq 1 "$REPS"); do
    J=$(sbatch --parsable --dependency=afterany:"$JOB" --time=00:40:00 \
        --export=ALL,CKPT="$CK",OUT="$SCRATCH_ROOT/role_icc/${TAG}_rep$i" \
        "$HERE/role_scene_icc_cpu.sbatch")
    echo "  ICC   rep$i : $J"
done

cat <<MSG

Everything is queued and will not start until $JOB finishes. Go to sleep.

WHEN YOU WAKE UP, three commands.

  1. did it all run
     sacct -u \$USER --starttime=now-1day --format=JobID,JobName%14,State,Elapsed -X | tail -12

  2. the table, per sweep
     cd \$HOME/roma_film
     for d in $SCRATCH_ROOT/regimes_rollout/${TAG}_rep*/; do echo; echo "### \$(basename \$d)"; \\
       python alpha_table.py --rollout_dir \$d \\
         --gt_regimes $SCRATCH_ROOT/regimes/regimes_gt.csv \\
         --gt_conflicts $SCRATCH_ROOT/regimes/conflicts_gt.csv; done

  3. ICC, mean over reps
     python3 -c "
import csv, glob, statistics as st
r=[]
for f in glob.glob('$SCRATCH_ROOT/role_icc/${TAG}_rep*/role_scene_icc.csv'):
    rows=list(csv.DictReader(open(f)))
    r.append([float(x['icc']) for x in rows if x['kind']=='role_summary'][0])
print('role ICC %+.3f +- %.3f over %d reps' % (st.mean(r), st.pstdev(r) if len(r)>1 else 0, len(r)))
"

REFERENCE POINTS for the morning:
  ICC             0.18 +- 0.03 across this run's five checkpoints
  headway at a=-2 2.011 s at 4.75B, against a human 2.059
  accel_ff        3.87 flat across alpha, human 2.512 -- the role does not
                  touch smoothness, and that gap is what BC+KL is for
MSG
