#!/bin/bash
# COMBINED future-MI x agent-centric, div-on arm (job 10518754). Run when it
# finishes. ICC skipped: chained to the training job already (10518756).
RUN_DIR=roma_mifutagent_dim4_H8_r16p64_ep
TAG=mifutagent_dim4
SKIP_ICC=1
. "$(dirname "$0")/_analyze_common.sh"
