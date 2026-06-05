#!/bin/bash
# ============================================================================
# Standard-scale GRPO training (64 envs / group 8) across ALL 10 Libero-Object
# tasks. This is the native config scale (the focused run CLI-shrinks it).
# ============================================================================
# NOTE: this run uses the BIASED synced gradient (recompute_kv_replay=false) for
# distributed safety across the ragged all-10-task buffer. In our experiments
# the biased gradient was FLAT (the exact gradient in train_exact.sh is what
# moved SR). Use this script to scale up AFTER confirming the exact-gradient
# gain — ideally combine with ignore_terminations=true + recompute_kv_replay=true
# once you've validated the rank-symmetry holds at 64 envs / 10 tasks. See
# NEXT_SESSION.md "open items".
#
# ~22 min/step (rollout-dominated). Each checkpoint full_weights.pt ~9.5 GB;
# the dcp_checkpoint resume-state adds ~29 GB -> budget ~38 GB/checkpoint.
#
# Usage:  train_proper.sh [MAX_STEPS]
# ============================================================================
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

MAX_STEPS="${1:-50}"
export LINGBOT_VA_MODEL_PATH="$BASE_SFT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" examples/embodiment/train_embodied_agent.py \
  --config-path "$REPO_PATH/examples/embodiment/config" \
  --config-name libero_object_grpo_lingbotva \
  runner.max_steps="$MAX_STEPS" runner.val_check_interval=-1 runner.save_interval=10 \
  actor.model.lingbotva.enable_offload=true rollout.model.lingbotva.enable_offload=true \
  env.train.total_num_envs=64 algorithm.group_size=8 env.train.group_size=8 \
  algorithm.rollout_epoch=1 actor.global_batch_size=64 actor.micro_batch_size=1 \
  actor.model.lingbotva.recompute_kv_replay=false actor.optim.lr=3.0e-6 \
  env.train.max_steps_per_rollout_epoch=240 env.train.max_episode_steps=240 \
  +weight_syncer.use_ring_sync=true \
  runner.logger.log_path="$LOG_ROOT/grpo_proper_logs"
