#!/bin/bash
# DIVERSITY ABLATION dim-4 (div_weight=0). run_nodiv_analysis.sh covers the
# same ground; this launcher is the one-liner form.
RUN_DIR=roma_nodiv_dim4
TAG=nodiv_dim4
ROLE_DIM=${ROLE_DIM:-4}
SKIP_ICC=0
. "$(dirname "$0")/_analyze_common.sh"
