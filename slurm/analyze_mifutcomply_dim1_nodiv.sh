#!/bin/bash
# Compliance + perturbation WITHOUT FiLM, role_dim=1, diversity loss OFF.
#
# Built on the dim-4 ICC result: FiLM ALONE made map-dependence markedly worse
# (mifutfilm_dim4 0.507 +- 0.029 against 0.300 +- 0.018 for the same arch
# without FiLM), and compliance + perturbation recovered part of that
# (mifutfilm_comply_dim4 0.389 +- 0.013) but did not get back to the non-FiLM
# number. So the two mechanisms pull in opposite directions on ICC, and they
# have never been separated: compliance has only ever been tested WITH FiLM.
#
# This arm is that separation, on the best base we have (dim-1 nodiv, ICC
# 0.170 +- 0.018). No code change is involved -- --compliance_weight,
# --perturb_frac and --role_film are independent arguments.
#
# ICC skipped: run_mi_future.sh already chains a scene-ICC job to the training.
# Prefer 3 CPU repeats via slurm/role_scene_icc_cpu.sbatch afterwards -- a
# single ICC run is not a result at sd 0.005-0.021.
RUN_DIR=roma_mifutagent_dim1_H8_r16p64_ep_comply_nodiv
TAG=mifutcomply_dim1_nodiv
ROLE_DIM=1
SKIP_ICC=1
. "$(dirname "$0")/_analyze_common.sh"
