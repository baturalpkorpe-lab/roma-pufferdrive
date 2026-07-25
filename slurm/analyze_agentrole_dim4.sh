#!/bin/bash
# AGENT-CENTRIC dim-4, div-on (TRAINING FINISHED -- runnable right now).
# ICC skipped: the training chain already ran it ->
#   $SCRATCH_ROOT/role_icc/agentrole_dim4_r16p64_ep
RUN_DIR=roma_agentrole_dim4_r16p64_ep
TAG=agentrole_dim4
SKIP_ICC=1
. "$(dirname "$0")/_analyze_common.sh"
