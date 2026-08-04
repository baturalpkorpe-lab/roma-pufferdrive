#!/bin/bash
# FiLM ON, compliance OFF, role_dim=1, diversity loss OFF.
#
# The control arm for the dim-1 FiLM pair. dim-1 is the current best arm on the
# pre-registered metric (role ICC 0.170 +- 0.018 vs 0.300 for the dim-4 div-on
# arm) and its single axis needs no PC rotation to interpret, so a forced sweep
# is directly readable. FiLM is the one architectural change that has never been
# tested at dim-1: with only ONE role input against a 128-wide env embedding,
# concatenation makes the role an additive bias outnumbered 128:1, which is the
# regime where feature-wise modulation should matter most.
#
# ICC skipped: run_mi_future.sh already chains a scene-ICC job to the training.
RUN_DIR=roma_mifutagent_dim1_H8_r16p64_ep_film_nodiv
TAG=mifutfilm_dim1_nodiv
ROLE_DIM=1
SKIP_ICC=1
. "$(dirname "$0")/_analyze_common.sh"
