# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stateful rollout worker for LingBot-VA GRPO — SCAFFOLDING.

The generic ``MultiStepRolloutWorker`` is stateless across chunk steps: it
keeps only the last step's wrapped obs and drops the per-step raw obs that
LingBot-VA needs to maintain its cross-chunk KV-cache. That is exactly why
PR #1220 shipped a dedicated ``eval_lingbotva.py`` driver for evaluation.

For RL the same gap means a vanilla rollout collapses LingBot-VA's success
rate to ~0% (no KV-cache replay), so the policy would never see reward and
never learn. This worker is the RL-path equivalent of ``eval_lingbotva.py``:
it maintains per-environment ``LingbotVAEpisodeState``, records chunk
observations for KV-cache replay, and resets episode state on done — while
still emitting ``RolloutResult``s onto the channels like the base worker.

STATUS: not implemented. See
``rlinf/models/embodiment/lingbotva/RL_DESIGN.md`` §4B. The substantive logic
(per-step obs capture + KV replay on the channel-based path) requires the
lingbot-va repo and a checkpoint to validate against gate #2 (RL rollout SR ≈
eval SR before any update).
"""

from __future__ import annotations

from omegaconf import DictConfig

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class LingbotVARolloutWorker(MultiStepRolloutWorker):
    """Rollout worker for LingBot-VA GRPO.

    LingBot-VA's RL rollout (``_rl_predict_action_batch`` ->
    ``backend.infer_batch_with_logprob``) runs the first-chunk
    (``frame_st_id == 0``) path: each ``predict_action_batch`` call is
    self-contained and returns real ``prev_logprobs`` + ``forward_inputs``. So
    the generic per-chunk loop of :class:`MultiStepRolloutWorker` already drives
    it correctly and we inherit it unchanged.

    Phase 2 (cross-chunk KV-cache replay for higher rollout SR) will override
    the per-chunk generation to capture per-step raw obs and feed
    ``model.record_chunk_observations`` / ``model.reset_episode``, mirroring
    ``eval_lingbotva.py``. Until then the per-env state below is unused but kept
    as the hook point. See RL_DESIGN.md §4B.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # Reserved for Phase 2 stateful KV-replay (per global env index).
        self._episode_states: dict = {}
