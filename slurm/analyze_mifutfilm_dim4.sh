#!/bin/bash
# FUTURE-MI x AGENT-CENTRIC with FiLM role conditioning, div loss ON.
#
#   ROLE_FILM=1 bash slurm/run_mi_future.sh 4 8
#
# WHY: concatenating the role to the env embedding lets it act only as an
# ADDITIVE BIAS in the GRU's first layer -- h = W_env@e + W_z@z. It shifts the
# pre-activation and nothing else, and it is 4 inputs against 128, outnumbered
# 32:1 even at equal per-dimension weight. FiLM has the role emit a per-feature
# scale and shift so it controls HOW each env feature is used:
#     e' = e * (1 + gamma(z)) + beta(z)
#
# The measured problem it targets: the encoder assigns high roles to fast
# agents while forcing that same role SLOWS them (agentrole PC1, within-scene
# r +0.634 against a causal -4.265). A bias term's sign is whatever
# optimisation happened to land on, because nothing during training ever pairs
# a SHIFTED role with an observation -- forced_role is used only by the video
# logger, never in the PPO rollout. FiLM does not fix that by itself; it makes
# the role able to matter enough for a compliance signal to have something to
# act on.
#
# Zero-init: gamma = beta = 0 at step 0, so the run starts from exactly the
# function the non-FiLM run starts from.
#
# ICC skipped: run_mi_future.sh already chains a scene-ICC job to the training.
RUN_DIR=roma_mifutagent_dim4_H8_r16p64_ep_film
TAG=mifutfilm_dim4
SKIP_ICC=1
. "$(dirname "$0")/_analyze_common.sh"
