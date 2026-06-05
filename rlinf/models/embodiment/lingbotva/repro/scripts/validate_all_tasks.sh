#!/bin/bash
# ============================================================================
# Large-N deterministic validation across ALL 10 Libero-Object tasks for one
# checkpoint. Use this to validate a full (all-task) training run; for the
# focused 2-task run use eval_tasks.sh instead.
#
# Usage:  validate_all_tasks.sh <label> <transformer_path> [nenvs] [offsets]
#   nenvs    default 3   (>3 risks OOM on the hard tasks ~step 140)
#   offsets  default "0,3,6"  -> 3 envs x 3 offsets x 10 tasks = 90 ep/checkpoint
#
# Aggregate with: python agg_val.py   (reads $LOG_ROOT/val2/*, Wilson CI + z-test)
# ============================================================================
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

LABEL="$1"; TRANSFORMER="$2"; NENVS="${3:-3}"
IFS=',' read -ra OFFSETS <<< "${4:-0,3,6}"

export LINGBOT_VA_MODEL_PATH="$BASE_MODEL"
export LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH="$TRANSFORMER"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="$LOG_ROOT/val2/$LABEL"; mkdir -p "$OUT"
JOBS=(); for t in 0 1 2 3 4 5 6 7 8 9; do for off in "${OFFSETS[@]}"; do JOBS+=("$t:$off"); done; done

run_job() { local t=$1 off=$2 gpu=$3
  CUDA_VISIBLE_DEVICES=$gpu "$PY" examples/embodiment/eval_lingbotva.py \
    --config-path "$REPO_PATH/examples/embodiment/config" \
    --config-name libero_object_eval_lingbotva \
    env.eval.total_num_envs="$NENVS" algorithm.eval_rollout_epoch=1 \
    "env.eval.task_id_filter=[$t]" "+env.eval.eval_reset_start_idx=$off" \
    env.eval.video_cfg.save_video=False \
    "actor.model.lingbotva.save_root=$LOG_ROOT/val2_rt/${LABEL}_t${t}_o${off}" \
    runner.logger.log_path="$OUT/task_${t}_off_${off}" > "$OUT/task_${t}_off_${off}.log" 2>&1; }

echo "[validate_all] $LABEL nenvs=$NENVS offsets=${OFFSETS[*]} jobs=${#JOBS[@]}"
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for gpu in 0 1 2 3 4 5 6 7; do
    [ $i -lt ${#JOBS[@]} ] || break
    IFS=':' read -r t off <<< "${JOBS[$i]}"; run_job "$t" "$off" "$gpu" & i=$((i+1))
  done
  wait; echo "[validate_all] wave done ($i/${#JOBS[@]})"
done
echo "[validate_all] $LABEL DONE -> $OUT"
