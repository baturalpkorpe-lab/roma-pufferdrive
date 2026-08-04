#!/bin/bash
# FiLM ON + compliance reward + role perturbation, role_dim=1, div loss OFF.
#
# The treatment arm of the dim-1 FiLM pair. Compare against
# analyze_mifutfilm_dim1_nodiv.sh, which differs ONLY in the compliance and
# perturbation terms -- FiLM is held constant, so the contrast isolates
# compliance rather than confounding it with the conditioning mechanism.
#
#   FiLM         lets the role matter at all; at dim-1 it is one input against
#                a 128-wide embedding.
#   compliance   rewards ||MIDecoder(z) - BE(realised future)||^2 through PPO,
#                so the POLICY learns to obey the role instead of only the
#                encoder learning to describe it.
#   perturbation puts SHIFTED roles in the training distribution, so the
#                off-manifold response a sweep probes is trained rather than
#                extrapolated. Without compliance it teaches the policy to
#                IGNORE the role, and train_mi_future.sbatch refuses that pair.
#
# SUCCESS METRIC, pre-registered: role_causal_axis.py coherence -> +1. Watch
# WOSAC realism alongside it -- compliance competes with the driving reward.
#
# ICC skipped: run_mi_future.sh already chains a scene-ICC job to the training.
RUN_DIR=roma_mifutagent_dim1_H8_r16p64_ep_film_comply_nodiv
TAG=mifutfilm_comply_dim1_nodiv
ROLE_DIM=1
SKIP_ICC=1
. "$(dirname "$0")/_analyze_common.sh"
