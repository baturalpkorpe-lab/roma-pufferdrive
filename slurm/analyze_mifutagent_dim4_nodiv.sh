#!/bin/bash
# COMBINED future-MI x agent-centric, div_weight=0 arm.
# ICC skipped: chained to the training job already.
RUN_DIR=roma_mifutagent_dim4_H8_r16p64_ep_nodiv
TAG=mifutagent_dim4_nodiv
SKIP_ICC=1
. "$(dirname "$0")/_analyze_common.sh"
