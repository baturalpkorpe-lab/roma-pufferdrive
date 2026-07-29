#!/bin/bash
# FiLM + COMPLIANCE REWARD, div loss ON. The full fix for the descriptive vs
# causal gap, launched by:
#
#   ROLE_FILM=1 COMPLIANCE_WEIGHT=0.05 PERTURB_FRAC=0.15 \
#       bash slurm/run_mi_future.sh 4 8
#
# The two halves only work together:
#   FiLM        makes the role ABLE to matter -- concatenation lets it act only
#               as an additive bias, 4 inputs against 128.
#   compliance  makes the policy USE it in the direction the encoder means, by
#               rewarding || MIDecoder(z) - BehaviourExtractor(realised future)
#               ||^2 through PPO rather than as a detached loss.
#   perturbation puts SHIFTED roles in the training distribution, so the
#               off-manifold response a sweep probes is trained rather than
#               extrapolated. Alone it teaches the policy to IGNORE the role
#               (ignoring noise maximises reward) -- the sbatch refuses that
#               combination outright.
#
# SUCCESS METRIC, pre-registered: role_causal_axis.py coherence -> +1. Watch
# WOSAC realism alongside it; the compliance term competes with the driving
# reward and likelihood_linear_speed is already the weakest number at ~0.11.
#
# ICC skipped: run_mi_future.sh already chains a scene-ICC job to the training.
RUN_DIR=roma_mifutagent_dim4_H8_r16p64_ep_film_comply
TAG=mifutfilm_comply_dim4
SKIP_ICC=1
. "$(dirname "$0")/_analyze_common.sh"
