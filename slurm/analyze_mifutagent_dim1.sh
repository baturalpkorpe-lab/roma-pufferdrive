#!/bin/bash
# FUTURE-MI x AGENT-CENTRIC at ROLE_DIM=1, div loss ON. Launched by:
#
#   bash slurm/run_mi_future.sh 1 8
#
# Pairs with analyze_mifutagent_dim1_nodiv.sh. BOTH arms are required: the
# question is whether the ICC's dependence on the diversity loss (large at
# dim-4: 0.300 div-on vs 0.383 nodiv, t=-8.2) survives at dim-1. One arm
# measures a value, not a dependence.
#
# READ THE BIMODALITY COEFFICIENT, NOT ONLY THE ICC. A 1-D role has been
# measured collapsing to a binary switch (BC~0.99) where dim-2's PC1 is a
# continuum (BC~0.54). A binary latent cannot be swept: intermediate alphas
# land in the empty valley between the two clumps, so "easier to control"
# would be exactly backwards. eval_roma logs role/bimodality_bc for dim-1.
# The open hypothesis these two arms test: the div loss may be what CREATES
# the split, since in one dimension pushing roles apart has nowhere to go but
# two clumps -- in which case the NODIV arm is the one that gets a usable dial.
#
# ICC skipped: run_mi_future.sh already chains a scene-ICC job to the training.
RUN_DIR=roma_mifutagent_dim1_H8_r16p64_ep
TAG=mifutagent_dim1
ROLE_DIM=1
SKIP_ICC=1
. "$(dirname "$0")/_analyze_common.sh"
