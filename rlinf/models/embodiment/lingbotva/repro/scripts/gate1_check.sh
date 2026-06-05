#!/bin/bash
# ============================================================================
# Gate #1: RL-core recompute-consistency sanity check (single GPU, no FSDP).
# ============================================================================
# Verifies that the rollout log-prob == the gradient-recompute log-prob at the
# SAME weights (proximal ratio == 1.0, max|delta| == 0). This validates the
# whole RL core: SDE action sampling, the diagonal-Gaussian log-prob, the
# sigma/velocity conventions, and KV-cache replay from the stored final video
# latents. Run this FIRST on any new checkpoint / after touching the RL math.
#
# Expected: "ratio=1.0000  max|delta|=0.0" (bit-exact under SDPA / attn_mode=torch).
# ============================================================================
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

export LINGBOT_VA_MODEL_PATH="$BASE_MODEL"
export LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH="$SFT_TRANSFORMER"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" examples/embodiment/check_lingbotva_rl_gate1.py \
  actor.model.lingbotva.attn_mode=torch
