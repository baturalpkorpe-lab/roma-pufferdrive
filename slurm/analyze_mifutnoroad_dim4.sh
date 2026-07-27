#!/bin/bash
# FUTURE-MI x AGENT-CENTRIC with the road removed from the role entirely
# (role_road_dim=0), div loss ON. Launched by:
#
#   ROLE_ROAD_DIM=0 bash slurm/run_mi_future.sh 4 8
#
# which puts the arch in the path, so this stub only has to name that dir.
#
# WHY THIS ARM: role ICC vs road bandwidth into the role encoder is NOT
# monotonic. Measured at n=3 each, div-on:
#     road 64 (baseline)   0.410 +- 0.019
#     road 16 (agentrole)  0.382 +- 0.005
#     road  0 (noroad)     0.414 +- 0.021
# Squeezing the road channel helps; removing it entirely undoes the gain,
# presumably because the role then reconstructs map context from the partner
# and ego streams, which are just as scene-correlated. This arm asks whether
# future-MI changes that -- if it does, road 0 + future-MI beats the 0.300 of
# road 16 + future-MI; if it does not, 16 is the operating point and the
# non-monotonicity is a property of the role input, not of the MI target.
#
# ICC skipped: run_mi_future.sh already chains a scene-ICC job to the training.
RUN_DIR=roma_mifutagent_dim4_H8_r0p64_ep
TAG=mifutnoroad_dim4
SKIP_ICC=1
. "$(dirname "$0")/_analyze_common.sh"
