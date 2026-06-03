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

"""Stateful KV-replay rollout worker for LingBot-VA GRPO.

LingBot-VA conditions each chunk on the prior chunks' observed key frames via a
transformer KV cache, which is what gives it non-zero success rate (without
replay SR collapses to ~0%). The generic ``MultiStepRolloutWorker`` is
stateless across chunks and only forwards the chunk-boundary obs, so this
subclass:

  * reads the per-step key-frame images the env worker now forwards (sibling
    keys ``chunk_keyframe_main`` / ``chunk_keyframe_wrist`` inside the obs, plus
    ``chunk_dones``), and
  * before each prediction, feeds them to the policy's
    ``record_chunk_wrapped_observations`` (or resets the episode on done),

so the model's per-env KV-replay state is maintained across the distributed
rollout — the channel-based equivalent of what ``eval_lingbotva.py`` does
in-process. See RL_DESIGN.md §4B.
"""

from __future__ import annotations

from typing import Any, Literal

from omegaconf import DictConfig

from rlinf.scheduler import Channel
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class LingbotVARolloutWorker(MultiStepRolloutWorker):
    """KV-replay rollout worker for LingBot-VA (GRPO)."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    async def recv_env_output(
        self, input_channel: Channel, mode: Literal["train", "eval"] = "train"
    ) -> dict[str, Any]:
        """Receive the env output, then update the policy's KV-replay state.

        The env worker rides the chunk's per-step key-frame images + dones in
        the obs dict. We record them into the (per-env) episode history of the
        policy before the next prediction, and reset episodes that just ended.
        The extra keys are stripped so the downstream predict path is unchanged.
        """
        env_output = await super().recv_env_output(input_channel, mode)
        obs = env_output.get("obs") if isinstance(env_output, dict) else None
        if not isinstance(obs, dict) or "chunk_keyframe_main" not in obs:
            return env_output

        kf_main = obs.pop("chunk_keyframe_main")  # [B, T, H, W, 3]
        kf_wrist = obs.pop("chunk_keyframe_wrist")
        dones = obs.pop("chunk_dones", None)  # [B] bool
        model = self.hf_model
        if not hasattr(model, "record_chunk_wrapped_observations"):
            return env_output

        batch_size = kf_main.shape[0]
        for env_idx in range(batch_size):
            # chunk_dones may be [B] or [B, chunk_steps]; an env "finished" this
            # chunk if it terminated/truncated at any step.
            done = bool(dones[env_idx].any()) if dones is not None else False
            if done:
                # Episode just ended (env auto-reset); drop its history so the
                # next prediction starts a fresh first chunk.
                model.reset_episode(env_idx)
                continue
            state = model._get_state(env_idx)
            prev_action = getattr(state, "prev_model_action", None)
            if prev_action is None:
                continue  # no chunk produced yet (first step)
            model.record_chunk_wrapped_observations(
                env_idx,
                kf_main[env_idx],
                kf_wrist[env_idx],
                prev_action,
            )
        # Debug-level trace of the per-env replay state (history should grow
        # 1..N within an episode and reset to 0 at the boundary).
        try:
            st0 = model._get_state(0)
            self.log_debug(
                f"[KV-replay] env0 history_len={len(st0.kv_cache_history)} "
                f"kf_shape={tuple(kf_main.shape)} prompt={st0.prompt is not None} "
                f"replay_on={model.enable_kv_cache_replay}"
            )
        except Exception:
            pass
        return env_output
