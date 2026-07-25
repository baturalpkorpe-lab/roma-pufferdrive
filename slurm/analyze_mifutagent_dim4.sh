#!/bin/bash
# COMBINED future-MI x agent-centric, div-on arm.
# ICC skipped: run_mi_future.sh already chains a scene-ICC job to the training.
RUN_DIR=roma_mifutagent_dim4_H8_r16p64_ep
TAG=mifutagent_dim4
SKIP_ICC=1
. "$(dirname "$0")/_analyze_common.sh"
