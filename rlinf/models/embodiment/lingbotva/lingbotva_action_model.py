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

"""LingBot-VA action model adapter for RLinf Libero evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.lingbotva.history_buffer import LingbotVAEpisodeState
from rlinf.models.embodiment.lingbotva.native_backend import LingbotVALiberoBackend
from rlinf.models.embodiment.lingbotva.observation_adapter import (
    LingbotVALiberoObservationAdapter,
)


class LingbotVAActionModel(nn.Module, BasePolicy):
    """LingBot-VA inference-only adapter for the Libero suite.

    Each :meth:`predict_action_batch` call runs one diffusion-based inference.
    For the very first chunk of an episode we follow LingBot-VA's first-chunk
    semantics (``start_idx=1``: skip the conditioning frame, producing
    ``(frame_chunk_size - 1) * action_per_frame`` env actions). On subsequent
    chunks we either:
      * Replay the previously cached key-frame history via the model's KV
        cache (the official LingBot-VA flow), or
      * Reuse the cached ``first_obs`` and start fresh from ``frame_st_id=0``
        when no chunk observations have been recorded yet.
    """

    def __init__(self, cfg: Any, torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = cfg
        self.torch_dtype = torch_dtype
        self.action_dim = int(getattr(cfg, "action_dim", 7))
        self.action_per_frame = int(getattr(cfg.lingbotva, "action_per_frame", 4))
        # Number of action steps to actually execute per inference. Defaults to
        # ``(frame_chunk_size - 1) * action_per_frame`` (12 for Libero), but
        # callers can lower it to force shorter chunks with more frequent
        # replanning.
        self.exec_steps_per_chunk = int(
            getattr(cfg, "num_action_chunks", 3 * self.action_per_frame)
        )
        # When ``False`` we always restart from ``frame_st_id=0`` and treat each
        # chunk as the model's first chunk. When ``True`` we maintain
        # per-environment KV-cache history (chunked obs + prior action) and
        # replay it on each call so the model keeps long-horizon context.
        self.enable_kv_cache_replay = bool(
            getattr(cfg.lingbotva, "enable_kv_cache_replay", False)
        )

        self._backend: LingbotVALiberoBackend | None = None
        self._episode_states: dict[int, LingbotVAEpisodeState] = {}

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(
            f"LingBot-VA does not support forward_type={forward_type}."
        )

    def default_forward(self, **kwargs):
        del kwargs
        raise NotImplementedError(
            "LingBot-VA default_forward is not supported in the eval integration. "
            "Use predict_action_batch."
        )

    def _ensure_backend(self) -> LingbotVALiberoBackend:
        if self._backend is None:
            self._backend = LingbotVALiberoBackend(self.config, self.torch_dtype)
        return self._backend

    def _get_state(self, env_idx: int) -> LingbotVAEpisodeState:
        if env_idx not in self._episode_states:
            self._episode_states[env_idx] = LingbotVAEpisodeState()
        return self._episode_states[env_idx]

    @staticmethod
    def _get_prompt(env_obs: dict[str, Any], env_idx: int) -> str:
        prompts = env_obs.get("task_descriptions")
        if prompts is None:
            raise ValueError(
                "LingBot-VA requires task_descriptions in env observations."
            )
        return str(prompts[env_idx])

    @staticmethod
    def _select_executable_actions(
        raw_action: np.ndarray, first_chunk: bool
    ) -> np.ndarray:
        """Convert raw model output to a flat sequence of env actions.

        Args:
            raw_action: array of shape
                ``(action_dim, frame_chunk_size, action_per_frame)``.
            first_chunk: if True the leading frame is skipped to match the
                LingBot-VA Libero client behaviour.

        Returns:
            Array of shape ``(num_actions, action_dim)``.
        """
        start_idx = 1 if first_chunk else 0
        selected = raw_action[:, start_idx:, :]
        return np.transpose(selected, (1, 2, 0)).reshape(-1, raw_action.shape[0])

    def reset_episode(self, env_idx: int, prompt: str | None = None) -> None:
        """Drop cached state for the given env so the next call is a first chunk."""
        self._get_state(env_idx).reset(prompt or "")

    def record_chunk_observations(
        self,
        env_idx: int,
        chunk_obs_list: list[dict[str, Any]],
        prev_model_action: np.ndarray,
    ) -> None:
        """Push the previous chunk's observations into the KV-cache history.

        ``chunk_obs_list`` is the per-step raw libero observation dicts (with
        ``agentview_image`` and ``robot0_eye_in_hand_image`` keys). We pick the
        configured key frames (every ``action_per_frame`` steps) and pair them
        with the raw model action that produced them so the backend can replay
        them into the transformer's KV cache on the next inference.
        """
        if not self.enable_kv_cache_replay:
            return
        state = self._get_state(env_idx)
        if state.prompt is None:
            return
        # The LingBot-VA clients sample key frames every ``action_per_frame // 4``
        # env steps (so they record 4 key frames per video frame predicted by
        # the model). For Libero that period is 1 (every step), for RoboTwin
        # 4. Fall back to 1 if the integer division would round to 0.
        period = max(1, self.action_per_frame // 4)
        key_frames: list[dict[str, Any]] = []
        for step_idx, raw_obs in enumerate(chunk_obs_list):
            if (step_idx + 1) % period != 0:
                continue
            key_frames.append(
                LingbotVALiberoObservationAdapter.format_raw_step_observation(
                    raw_obs=raw_obs,
                    prompt=state.prompt,
                )
            )
        if not key_frames:
            return
        state.kv_cache_history.append(
            (key_frames, np.asarray(prev_model_action, dtype=np.float32).copy())
        )

    def predict_action_batch(
        self, env_obs: dict[str, Any], mode: str = "eval", **_: Any
    ):
        if mode != "eval":
            raise NotImplementedError(
                "LingBot-VA Libero adapter only supports eval mode."
            )
        states_tensor = env_obs.get("states")
        if states_tensor is None:
            raise ValueError("LingBot-VA requires batched states in env observations.")
        batch_size = states_tensor.shape[0]

        backend = self._ensure_backend()

        prompts: list[str] = []
        obs_batch: list[dict[str, Any]] = []
        kv_cache_histories: list[list[tuple[list[dict[str, Any]], np.ndarray]]] = []
        first_chunk_flags: list[bool] = []

        for env_idx in range(batch_size):
            prompt = self._get_prompt(env_obs, env_idx)
            state = self._get_state(env_idx)
            if state.prompt != prompt:
                state.reset(prompt)
            obs = LingbotVALiberoObservationAdapter.format_observation(
                env_obs, env_idx, prompt
            )
            if state.first_obs is None:
                state.first_obs = obs
            prompts.append(prompt)
            # When KV-cache replay is enabled we reuse the cached first_obs to
            # keep the obs grid consistent with the replayed history.
            obs_batch.append(state.first_obs if self.enable_kv_cache_replay else obs)
            kv_cache_histories.append(list(state.kv_cache_history))
            first_chunk_flags.append(state.first_chunk)

        replay_groups_match = all(
            len(history) == len(kv_cache_histories[0]) for history in kv_cache_histories
        )
        if (
            self.enable_kv_cache_replay
            and replay_groups_match
            and any(len(h) > 0 for h in kv_cache_histories)
        ):
            raw_actions = backend.infer_batch(
                obs_batch, prompts, kv_cache_histories=kv_cache_histories
            )
            current_first_chunk = False
        else:
            raw_actions = backend.infer_batch(obs_batch, prompts)
            current_first_chunk = any(first_chunk_flags)

        chunks: list[torch.Tensor] = []
        for env_idx, raw_action in enumerate(raw_actions):
            state = self._get_state(env_idx)
            # Without KV-cache replay every call rebuilds context from the
            # current observation, so the model is effectively starting a
            # fresh "first chunk" each time and we must drop the leading
            # placeholder frame. Once we are using KV replay we follow the
            # LingBot-VA client semantics: skip the frame only on the very
            # first inference per episode.
            is_first_chunk_for_selection = (
                state.first_chunk if self.enable_kv_cache_replay else True
            )
            env_actions = self._select_executable_actions(
                raw_action, first_chunk=is_first_chunk_for_selection
            )
            limit = min(self.exec_steps_per_chunk, env_actions.shape[0])
            chunks.append(torch.from_numpy(env_actions[:limit]))
            # Track this chunk's raw model action so callers can hand it back
            # via ``record_chunk_observations`` for KV replay on the next call.
            state.prev_model_action = raw_action.astype(np.float32)
            state.last_action_per_frame = raw_action.shape[2]
            state.first_chunk = False

        # In multi-env mode envs can be in different first-chunk states; we
        # truncate to the shortest chunk so we can stack into a single tensor.
        common_len = min(c.shape[0] for c in chunks)
        chunks = [c[:common_len] for c in chunks]

        action_tensor = torch.stack(chunks, dim=0).to(dtype=torch.float32)
        zeros = torch.zeros(action_tensor.shape[:2], dtype=torch.float32)
        result = {
            "prev_logprobs": zeros,
            "prev_values": zeros,
            "forward_inputs": {"action": action_tensor},
        }
        return action_tensor, result
