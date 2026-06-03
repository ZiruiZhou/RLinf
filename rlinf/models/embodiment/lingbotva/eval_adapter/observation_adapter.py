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

"""Observation conversion utilities for LingBot-VA on Libero."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class LingbotVALiberoObservationAdapter:
    """Convert RLinf LiberoEnv observations into LingBot-VA Libero format.

    LiberoEnv applies ``img[::-1, ::-1]`` (a 180-degree rotation) to the
    rendered MuJoCo frames. LingBot-VA's Libero evaluation client only flips
    the vertical axis (``[::-1]``). We therefore restore the horizontal axis
    so the model sees observations in its training-time orientation.
    """

    @staticmethod
    def _select_image(image_tensor: Any, env_idx: int) -> np.ndarray:
        if image_tensor is None:
            raise ValueError("LingBot-VA requires image observations, but got None.")
        if isinstance(image_tensor, torch.Tensor):
            arr = image_tensor[env_idx].detach().cpu().numpy()
        else:
            arr = np.asarray(image_tensor[env_idx])
        return np.ascontiguousarray(arr[:, ::-1, :])

    @staticmethod
    def _select_state(state_tensor: Any, env_idx: int) -> np.ndarray:
        if state_tensor is None:
            return np.zeros((7,), dtype=np.float32)
        if isinstance(state_tensor, torch.Tensor):
            if state_tensor.dtype == torch.bfloat16:
                state_tensor = state_tensor.to(torch.float32)
            arr = state_tensor[env_idx].detach().cpu().numpy()
        else:
            arr = np.asarray(state_tensor[env_idx])
        return arr.astype(np.float32)

    @classmethod
    def format_observation(
        cls,
        env_obs: dict[str, Any],
        env_idx: int,
        prompt: str,
    ) -> dict[str, Any]:
        """Convert one batch element to the official LingBot-VA Libero format."""
        agentview = cls._select_image(env_obs.get("main_images"), env_idx)
        eye_in_hand = cls._select_image(env_obs.get("wrist_images"), env_idx)
        state = cls._select_state(env_obs.get("states"), env_idx)
        return {
            "observation.images.agentview_rgb": agentview,
            "observation.images.eye_in_hand_rgb": eye_in_hand,
            "observation.state": state,
            "task": prompt,
        }

    @classmethod
    def format_raw_step_observation(
        cls,
        raw_obs: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Convert a raw libero per-step obs dict (from chunk_step) to LingBot-VA format.

        ``raw_obs`` is the dict returned by Libero's ``OffScreenRenderEnv.step``
        before RLinf wraps it. It includes ``agentview_image`` and
        ``robot0_eye_in_hand_image`` arrays in their raw MuJoCo orientation
        (upside-down). LingBot-VA expects them vertically flipped.
        """
        agentview = raw_obs.get("agentview_image")
        eye_in_hand = raw_obs.get("robot0_eye_in_hand_image")
        if agentview is None or eye_in_hand is None:
            raise ValueError(
                "LingBot-VA raw libero step observation expects "
                "'agentview_image' and 'robot0_eye_in_hand_image' keys."
            )
        return {
            "observation.images.agentview_rgb": np.ascontiguousarray(
                np.asarray(agentview)[::-1]
            ),
            "observation.images.eye_in_hand_rgb": np.ascontiguousarray(
                np.asarray(eye_in_hand)[::-1]
            ),
            "task": prompt,
        }

    @staticmethod
    def _flip_wrapped_image(img: Any) -> np.ndarray:
        """Apply the same horizontal flip as ``format_observation`` to one
        already-wrapped (180-rotated) image of shape ``[H, W, 3]``."""
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            arr = np.asarray(img)
        return np.ascontiguousarray(arr[:, ::-1, :])

    @classmethod
    def format_single_wrapped(
        cls, main_img: Any, wrist_img: Any, prompt: str
    ) -> dict[str, Any]:
        """Format one single-env WRAPPED key-frame (main/wrist ``[H, W, 3]``).

        Used by the distributed KV-replay path: the env worker forwards the
        per-step wrapped images (``main_images``/``wrist_images``), which carry
        the same orientation ``format_observation`` consumes, so we apply the
        same horizontal flip. Equivalent to ``format_raw_step_observation`` but
        for the wrapped (rather than raw libero) image layout.
        """
        return {
            "observation.images.agentview_rgb": cls._flip_wrapped_image(main_img),
            "observation.images.eye_in_hand_rgb": cls._flip_wrapped_image(wrist_img),
            "task": prompt,
        }
