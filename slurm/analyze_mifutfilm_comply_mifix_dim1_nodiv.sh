#!/bin/bash
# Compliance + perturbation with the MI-DECODER FIX, FiLM ON,
# role_dim=1, div loss OFF.
#
# The fix: --mi_exclude_perturbed=1 fits the MI decoder on NATURAL roles only.
# Without it the decoder was trained on the SHIFTED role against the perturbed
# agent's own realised future, so it learned "this shift means whatever just
# happened" and compliance was satisfied with no behaviour change. The decoder
# wins that race by construction -- direct supervised gradient against a scalar
# PPO reward -- which is why the dim-4 compliance arm shrank every causal effect
# 4-11x without changing a single sign.
#
# ICC skipped: run_mi_future.sh chains one. Use 3 CPU repeats afterwards.
RUN_DIR=roma_mifutagent_dim1_H8_r16p64_ep_film_comply_mifix_nodiv
TAG=mifutfilm_comply_mifix_dim1_nodiv
ROLE_DIM=1
SKIP_ICC=1
. "$(dirname "$0")/_analyze_common.sh"
