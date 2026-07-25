#!/bin/bash
# DIVERSITY ABLATION dim-1. ROLE_DIM=1: the checkpoint is roma_dim1_final.pt,
# and the 1-D role gets bimodality analysis rather than PCA (eval_roma).
RUN_DIR=roma_nodiv_dim1
TAG=nodiv_dim1
ROLE_DIM=${ROLE_DIM:-1}
SKIP_ICC=0
. "$(dirname "$0")/_analyze_common.sh"
