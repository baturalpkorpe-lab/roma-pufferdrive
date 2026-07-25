#!/bin/bash
# BASELINE dim-4 -- the CONTROL every other run is read against.
# Its scene ICC is the reference number for the agent-centric comparison.
RUN_DIR=roma_baseline_dim4
TAG=baseline_dim4
ROLE_DIM=${ROLE_DIM:-4}
SKIP_ICC=0
. "$(dirname "$0")/_analyze_common.sh"
