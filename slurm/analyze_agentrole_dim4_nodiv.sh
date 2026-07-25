#!/bin/bash
# AGENT-CENTRIC dim-4, div_weight=0 arm (job 10518580). Run when it finishes.
# ICC skipped: chained to the training job already (10518582).
RUN_DIR=roma_agentrole_dim4_r16p64_ep_nodiv
TAG=agentrole_dim4_nodiv
SKIP_ICC=1
. "$(dirname "$0")/_analyze_common.sh"
