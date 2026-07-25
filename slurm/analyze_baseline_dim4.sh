#!/bin/bash
# BASELINE dim-4 -- the CONTROL every other analysis is read against. Its
# scene ICC has never been run; this launcher finally produces it. Run this
# one FIRST: agentrole/mifutagent ICC numbers mean nothing without it.
# Checkpoint has existed for weeks -- runnable right now, no waiting.
RUN_DIR=roma_baseline_dim4
TAG=baseline_dim4
SKIP_ICC=0
. "$(dirname "$0")/_analyze_common.sh"
