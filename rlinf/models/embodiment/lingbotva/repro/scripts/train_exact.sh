#!/bin/bash
# ============================================================================
# THE WINNING RUN: GRPO RL with the (near-)exact gradient on LingBot-VA.
# ============================================================================
# Focused 2-task (Libero-Object tasks 2 & 6) GRPO from the SFT checkpoint.
# Produces the reference trajectory in REFERENCE_RESULTS.md
#   (SFT 40% -> step 20 66.7% -> step 30 70.0%, two z>2.1 peaks).
#
# Why this config works where earlier ones were flat:
#   recompute_kv_replay=true  -> exact KV-cache-replay recompute (proximal
#                                ratio ~0.91, vs the biased no-replay ~0.6).
#   ignore_terminations=true  -> every env runs the full 240 steps -> the
#                                rollout buffer is a non-ragged [chunks x envs]
#                                grid -> per-rank FSDP collectives are
#                                symmetric by construction -> no distributed
#                                deadlock (the ragged-buffer failure, see
#                                NEXT_SESSION.md "gotchas").
#
# Disaggregated placement (actor 0-3 / rollout 4-7) + ring weight sync is
# REQUIRED on this container: collocated CUDA-IPC weight sync hits pidfd_getfd
# EPERM (ptrace_scope locked). See ENV_PREP.md.
#
# Usage:  train_exact.sh [MAX_STEPS] [RESUME_DIR]
#   MAX_STEPS   default 30
#   RESUME_DIR  optional global_step_<N> dir to resume from (e.g. to extend a run)
#
# ~25-30 min/step (ignore_terminations keeps envs stepping after success).
# Checkpoints (every 5 steps) under $LOG_ROOT/grpo_exact_train_logs/.../checkpoints/.
# Each checkpoint is ~38 GB (dcp_checkpoint 29 GB resume-state + full_weights.pt
# 9.5 GB eval-weights); trim the dcp_checkpoint dir after each save if disk is
# tight — eval only needs full_weights.pt.
# ============================================================================
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

MAX_STEPS="${1:-30}"
RESUME_DIR="${2:-}"
export LINGBOT_VA_MODEL_PATH="$BASE_SFT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # OK here: disaggregated => ring sync, no IPC
export TORCH_NCCL_BLOCKING_WAIT=1                          # surface NCCL desync instead of silent hang

RESUME_ARG=()
[ -n "$RESUME_DIR" ] && RESUME_ARG=("+runner.resume_dir=$RESUME_DIR")

"$PY" examples/embodiment/train_embodied_agent.py \
  --config-path "$REPO_PATH/examples/embodiment/config" \
  --config-name libero_object_grpo_lingbotva \
  runner.max_steps="$MAX_STEPS" runner.val_check_interval=-1 runner.save_interval=5 \
  "${RESUME_ARG[@]}" \
  actor.model.lingbotva.enable_offload=true rollout.model.lingbotva.enable_offload=true \
  env.train.total_num_envs=32 algorithm.group_size=8 env.train.group_size=8 \
  "+env.train.task_id_filter=[2,6]" env.train.ignore_terminations=true \
  algorithm.rollout_epoch=1 actor.global_batch_size=32 actor.micro_batch_size=8 \
  actor.model.lingbotva.recompute_kv_replay=true actor.optim.lr=3.0e-6 \
  actor.model.lingbotva.rl_noise_level=1.0 rollout.model.lingbotva.rl_noise_level=1.0 \
  actor.model.lingbotva.kv_replay_max_history=24 actor.model.lingbotva.kv_replay_max_frames=96 \
  env.train.max_steps_per_rollout_epoch=240 env.train.max_episode_steps=240 \
  +weight_syncer.use_ring_sync=true \
  runner.logger.log_path="$LOG_ROOT/grpo_exact_train_logs"
