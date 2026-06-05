#!/bin/bash
# ============================================================================
# Deterministic (noise=0) checkpoint eval on a chosen set of Libero-Object
# tasks. THIS is the real success-rate signal. The training-time `success_once`
# metric is measured at rl_noise_level=1.0 (full SDE exploration) and is
# noisy/misleading — always judge a policy by deterministic eval of its
# checkpoint, not by the training curve.
#
# Usage:  eval_tasks.sh <label> <transformer_path> [tasks] [offsets] [nenvs]
#   label            output subdir name (e.g. exact30)
#   transformer_path .../actor/model_state_dict/full_weights.pt  (an RL ckpt)
#                    OR the SFT dir $SFT_TRANSFORMER for the baseline
#   tasks            comma list, default "2,6" (the trained tasks)
#   offsets          comma list of init-state start indices, default "0,3,6,9,12"
#   nenvs            envs per (task,offset) process, default 3  (>3 risks OOM:
#                    the 240-step KV-replay history grows O(n^2) per chunk)
#
# Layout: tasks x offsets processes spread over 8 GPUs, one episode per env
# (eval_rollout_epoch=1 — multi-episode reuse is BROKEN, see NEXT_SESSION.md).
# Default 2 tasks x 5 offsets x 3 envs = 15 ep/task = 30 ep/checkpoint.
# Aggregate with: python agg_trajectory.py   (reads $LOG_ROOT/valt26/*)
# ============================================================================
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

LABEL="$1"; TRANSFORMER="$2"
IFS=',' read -ra TASKS   <<< "${3:-2,6}"
IFS=',' read -ra OFFSETS <<< "${4:-0,3,6,9,12}"
NENVS="${5:-3}"

export LINGBOT_VA_MODEL_PATH="$BASE_MODEL"
export LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH="$TRANSFORMER"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="$LOG_ROOT/valt26/$LABEL"; mkdir -p "$OUT"
JOBS=(); for t in "${TASKS[@]}"; do for off in "${OFFSETS[@]}"; do JOBS+=("$t:$off"); done; done

run_job() { local t=$1 off=$2 gpu=$3
  CUDA_VISIBLE_DEVICES=$gpu "$PY" examples/embodiment/eval_lingbotva.py \
    --config-path "$REPO_PATH/examples/embodiment/config" \
    --config-name libero_object_eval_lingbotva \
    env.eval.total_num_envs="$NENVS" algorithm.eval_rollout_epoch=1 \
    "env.eval.task_id_filter=[$t]" "+env.eval.eval_reset_start_idx=$off" \
    env.eval.video_cfg.save_video=False \
    "actor.model.lingbotva.save_root=$LOG_ROOT/valt26_rt/${LABEL}_t${t}_o${off}" \
    runner.logger.log_path="$OUT/task_${t}_off_${off}" > "$OUT/task_${t}_off_${off}.log" 2>&1; }

echo "[eval_tasks] $LABEL tasks=${TASKS[*]} offsets=${OFFSETS[*]} nenvs=$NENVS jobs=${#JOBS[@]}"
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for gpu in 0 1 2 3 4 5 6 7; do
    [ $i -lt ${#JOBS[@]} ] || break
    IFS=':' read -r t off <<< "${JOBS[$i]}"; run_job "$t" "$off" "$gpu" & i=$((i+1))
  done
  wait; echo "[eval_tasks] wave done ($i/${#JOBS[@]})"
done
echo "[eval_tasks] $LABEL DONE -> $OUT"
