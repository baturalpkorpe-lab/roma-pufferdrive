#!/bin/bash
# PLAIN future-MI, div loss ON, dim-4 -- the arm Saeed trained and WOSAC'd.
# Role pairing + a 3-repeat scene ICC. NO video render.
#
#   bash slurm/analyze_mifuture_dim4_saeed.sh
#
# THIS IS THE MISSING CELL. Everything else in the 2x2 is measured (role ICC,
# mean +- sd over 3-6 repeats, current 33-metric code):
#
#     arm                        role ICC        div  agent-centric
#     baseline                   0.410 +- 0.019   -        -
#     mifuture_nodiv             0.379 +- 0.014   no      no
#     mifutagent_nodiv           0.383 +- 0.012   no      yes
#     agentrole_div              0.382 +- 0.005   yes     yes(no MI)
#     mifutagent_div             0.300 +- 0.018   yes     yes
#     mifuture_div  <- THIS      ?                yes     no
#
# The whole headline rests on this number. mifutagent_div is 0.300 while every
# other arm sits in a flat 0.379-0.410 band, and the div loss is worth -0.083
# in the combined architecture but only -0.010 in agentrole. So either:
#   ~0.30  -> future-MI + div does it alone, agent-centric is redundant, and
#             the result simplifies to two ingredients.
#   ~0.38  -> the three-way interaction is real and required, which is the
#             stronger claim but needs all three components to state.
# Nothing else currently distinguishes those two stories.
#
# The checkpoint is on Saeed's scratch (readable). Everything written lands on
# whichever account runs this, so either of us can run it.
CKPT=${CKPT:-/scratch/srahmani/checkpoints/roma_mifuture_dim4_H8/roma_dim4_final.pt}
TAG=${TAG:-mifuture_dim4_div}
ROLE_DIM=${ROLE_DIM:-4}
REPEATS=${REPEATS:-3}
export CKPT TAG ROLE_DIM REPEATS
exec bash "$(dirname "$0")/run_pairing_icc3.sh"
