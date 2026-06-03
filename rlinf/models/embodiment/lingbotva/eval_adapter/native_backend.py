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

"""In-process LingBot-VA runtime backend for Libero evaluation."""

from __future__ import annotations

import atexit
import copy
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.models.embodiment.lingbotva._utils import (
    extend_import_path,
    load_transformer_state_dict,
)
from rlinf.models.embodiment.lingbotva.rl_engine import (
    RLForwardInputs,
    normalized_action_sigmas,
    reduce_chain_logprob,
    sde_mean_std,
    select_denoise_index,
)
from rlinf.models.embodiment.lingbotva.rl_utils import (
    gaussian_entropy,
    gaussian_logprob,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


def _resolve_cuda_local_rank() -> int:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            return int(local_rank)
        except ValueError as exc:
            raise ValueError(f"Invalid LOCAL_RANK value: {local_rank!r}") from exc
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return 0


def _resolve_runtime_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    repo_root = os.environ.get("REPO_PATH")
    if repo_root:
        return (Path(repo_root) / path).resolve()
    return path.resolve()


class LingbotVALiberoBackend:
    """Single-process backend using the official LingBot-VA modules for Libero."""

    _BATCH_CACHE_NAME = "rlinf_libero_batch"

    def __init__(self, cfg: Any, torch_dtype: torch.dtype) -> None:
        self.cfg = cfg
        self.torch_dtype = torch_dtype
        self.config_name = getattr(cfg.lingbotva, "config_name", "libero")
        self.repo_path = Path(getattr(cfg.lingbotva, "repo_path"))
        self.model_path = Path(cfg.model_path)
        transformer_state_dict_path = getattr(
            cfg.lingbotva, "transformer_state_dict_path", None
        )
        self.transformer_state_dict_path = (
            Path(transformer_state_dict_path)
            if transformer_state_dict_path is not None
            else None
        )
        self.save_root = _resolve_runtime_path(
            getattr(cfg.lingbotva, "save_root", "./runtime/lingbotva")
        )
        self._get_mesh_id = None
        self._data_seq_to_patch = None
        self._prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        self._validate_runtime_paths()
        self._server = self._build_server()
        atexit.register(self.close)

    def _validate_runtime_paths(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"LingBot-VA repo path does not exist: {self.repo_path}"
            )
        if not self.repo_path.is_dir():
            raise NotADirectoryError(
                f"LingBot-VA repo path is not a directory: {self.repo_path}"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"LingBot-VA model path does not exist: {self.model_path}"
            )
        if not self.model_path.is_dir():
            raise NotADirectoryError(
                f"LingBot-VA model path is not a directory: {self.model_path}"
            )

    def _build_server(self):
        extend_import_path(self.repo_path)

        from wan_va.configs import VA_CONFIGS
        from wan_va.utils import data_seq_to_patch, get_mesh_id
        from wan_va.wan_va_server import VA_Server

        job_config = copy.deepcopy(VA_CONFIGS[self.config_name])
        job_config.wan22_pretrained_model_name_or_path = str(self.model_path)
        job_config.param_dtype = self.torch_dtype
        job_config.enable_offload = bool(
            getattr(self.cfg.lingbotva, "enable_offload", False)
        )
        job_config.save_root = str(self.save_root)

        # Allow callers to override the diffusion step counts so that the
        # 25-step video / 50-step action defaults can be reduced for faster
        # evaluation when needed.
        override_video_steps = getattr(self.cfg.lingbotva, "num_inference_steps", None)
        if override_video_steps is not None:
            job_config.num_inference_steps = int(override_video_steps)
        override_action_steps = getattr(
            self.cfg.lingbotva, "action_num_inference_steps", None
        )
        if override_action_steps is not None:
            job_config.action_num_inference_steps = int(override_action_steps)

        current_device = _resolve_cuda_local_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(current_device)
        job_config.rank = 0
        job_config.local_rank = current_device
        job_config.world_size = 1
        server = VA_Server(job_config)
        self._data_seq_to_patch = data_seq_to_patch
        self._get_mesh_id = get_mesh_id
        if self.transformer_state_dict_path is not None:
            load_transformer_state_dict(
                server.transformer, self.transformer_state_dict_path
            )
        logger.info(
            "Initialized in-process LingBot-VA Libero runtime on device %s "
            "(CUDA_VISIBLE_DEVICES=%s).",
            current_device,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        return server

    def _cfg_batch_size(self, batch_size: int) -> int:
        return batch_size * (2 if self._server.use_cfg else 1)

    def _ensure_observation_runtime_shape(self) -> None:
        server = self._server
        if all(
            hasattr(server, attr)
            for attr in ("height", "width", "latent_height", "latent_width")
        ):
            return

        server.action_per_frame = server.job_config.action_per_frame
        server.height, server.width = server.job_config.height, server.job_config.width
        server.latent_height, server.latent_width = (
            server.height // 16,
            server.width // 16 * len(server.job_config.obs_cam_keys),
        )

    def _setup_batch_runtime(self, batch_size: int) -> None:
        """Prompt-free per-batch runtime setup (cache, masks, norm stats).

        Extracted from :meth:`_reset_batch_runtime` so the RL recompute path
        (which already holds cached prompt embeddings and must not touch the
        text encoder) can reuse the exact same cache/scheduler/mask setup.
        """
        server = self._server
        server.cache_name = self._BATCH_CACHE_NAME
        server.use_cfg = (server.job_config.guidance_scale > 1) or (
            server.job_config.action_guidance_scale > 1
        )
        server.frame_st_id = 0
        server.init_latent = None
        server.transformer.clear_cache(server.cache_name)
        server.streaming_vae.clear_cache()

        self._ensure_observation_runtime_shape()

        patch_size = server.job_config.patch_size
        latent_token_per_chunk = (
            server.job_config.frame_chunk_size
            * server.latent_height
            * server.latent_width
        ) // (patch_size[0] * patch_size[1] * patch_size[2])
        action_token_per_chunk = (
            server.job_config.frame_chunk_size * server.action_per_frame
        )
        server.transformer.create_empty_cache(
            server.cache_name,
            server.job_config.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            dtype=server.dtype,
            device=server.device,
            batch_size=self._cfg_batch_size(batch_size),
        )

        server.action_mask = torch.zeros([server.job_config.action_dim]).bool()
        server.action_mask[server.job_config.used_action_channel_ids] = True
        server.actions_q01 = torch.tensor(
            server.job_config.norm_stat["q01"], dtype=torch.float32
        ).reshape(-1, 1, 1)
        server.actions_q99 = torch.tensor(
            server.job_config.norm_stat["q99"], dtype=torch.float32
        ).reshape(-1, 1, 1)
        server.action_norm_method = server.job_config.action_norm_method

    def _reset_batch_runtime(self, prompts: list[str]) -> None:
        if not prompts:
            raise ValueError("LingBot-VA batch runtime requires at least one prompt.")

        self._setup_batch_runtime(len(prompts))
        server = self._server
        # Cache prompt embeddings by prompt text so we only run the 5.7B-param
        # UMT5 text encoder when we encounter a new prompt. After encoding we
        # offload the text encoder back to CPU (or even free it entirely) to
        # leave room for the transformer's diffusion-inference activations.
        do_cfg = server.job_config.guidance_scale > 1
        pos_list: list[torch.Tensor] = []
        neg_list: list[torch.Tensor | None] = []
        missing_prompts = [
            prompt for prompt in prompts if prompt not in self._prompt_cache
        ]
        if missing_prompts:
            text_encoder = server.text_encoder
            text_encoder_orig_device = next(text_encoder.parameters()).device
            if text_encoder_orig_device.type != server.device.type:
                text_encoder.to(server.device)
            try:
                with torch.no_grad():
                    new_pos, new_neg = server.encode_prompt(
                        prompt=missing_prompts,
                        negative_prompt=None,
                        do_classifier_free_guidance=do_cfg,
                        num_videos_per_prompt=1,
                        prompt_embeds=None,
                        negative_prompt_embeds=None,
                        max_sequence_length=512,
                        device=server.device,
                        dtype=server.dtype,
                    )
                for idx, prompt in enumerate(missing_prompts):
                    pos_emb = new_pos[idx : idx + 1].clone()
                    neg_emb = (
                        new_neg[idx : idx + 1].clone()
                        if (do_cfg and new_neg is not None)
                        else None
                    )
                    self._prompt_cache[prompt] = (pos_emb, neg_emb)
            finally:
                if text_encoder_orig_device.type != server.device.type:
                    text_encoder.to(text_encoder_orig_device)
                torch.cuda.empty_cache()
        for prompt in prompts:
            pos_emb, neg_emb = self._prompt_cache[prompt]
            pos_list.append(pos_emb)
            neg_list.append(neg_emb)
        server.prompt_embeds = torch.cat(pos_list, dim=0).to(server.device)
        if do_cfg and all(neg is not None for neg in neg_list):
            server.negative_prompt_embeds = torch.cat(neg_list, dim=0).to(server.device)
        else:
            server.negative_prompt_embeds = None
        server.exp_name = "rlinf_libero_batch"
        server.exp_save_root = str(self.save_root / "real")
        os.makedirs(server.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    @staticmethod
    def _normalize_obs_sequences(
        obs_batch: list[dict[str, Any]] | list[list[dict[str, Any]]],
    ) -> list[list[dict[str, Any]]]:
        if not obs_batch:
            raise ValueError("LingBot-VA batch infer requires non-empty observations.")
        if isinstance(obs_batch[0], dict):
            return [[obs] for obs in obs_batch]
        return obs_batch  # type: ignore[return-value]

    def _encode_obs_batch(
        self,
        obs_batch: list[dict[str, Any]] | list[list[dict[str, Any]]],
    ) -> torch.Tensor | None:
        server = self._server
        self._ensure_observation_runtime_shape()
        obs_sequences = self._normalize_obs_sequences(obs_batch)
        if not obs_sequences:
            return None

        # Validate uniform sequence lengths.
        lengths = [len(sequence) for sequence in obs_sequences]
        if len(set(lengths)) != 1:
            raise ValueError(
                "LingBot-VA Libero encoding requires uniform sequence lengths "
                f"across the batch, got {lengths}."
            )

        videos = []
        for camera_key in server.job_config.obs_cam_keys:
            height_i, width_i = server.height, server.width
            history_video = torch.stack(
                [
                    torch.from_numpy(np.stack([frame[camera_key] for frame in seq]))
                    .float()
                    .permute(3, 0, 1, 2)
                    for seq in obs_sequences
                ],
                dim=0,
            )
            history_video = F.interpolate(
                history_video.flatten(0, 1),
                size=(height_i, width_i),
                mode="bilinear",
                align_corners=False,
            ).unflatten(0, (history_video.shape[0], history_video.shape[1]))
            videos.append(history_video)

        videos_tensor = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
        vae = server.streaming_vae.vae
        vae_orig_device = next(vae.parameters()).device
        if vae_orig_device.type != server.device.type:
            vae.to(server.device)
        try:
            enc_out = server.streaming_vae.encode_chunk(
                videos_tensor.to(server.device).to(server.dtype)
            )
        finally:
            if vae_orig_device.type != server.device.type:
                vae.to(vae_orig_device)
                torch.cuda.empty_cache()

        mu, _logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(server.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(server.vae.config.latents_std).to(mu.device)
        mu_norm = server.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        # Concatenate cameras along the width dimension.
        video_latent = torch.cat(mu_norm.split(len(obs_sequences), dim=0), dim=-1).to(
            server.device
        )
        return video_latent

    def _prepare_batch_input(
        self,
        *,
        latent_model_input: torch.Tensor | None,
        action_model_input: torch.Tensor | None,
        latent_t: float = 0,
        action_t: float = 0,
        latent_cond: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        frame_st_id: int = 0,
    ) -> dict[str, dict[str, torch.Tensor]]:
        server = self._server
        batch_size = (
            latent_model_input.shape[0]
            if latent_model_input is not None
            else action_model_input.shape[0]
        )
        input_dict: dict[str, dict[str, torch.Tensor]] = {}

        if latent_model_input is not None:
            timesteps = (
                torch.ones(
                    [latent_model_input.shape[2]],
                    dtype=torch.float32,
                    device=server.device,
                )
                * latent_t
            )
            grid_id = self._get_mesh_id(
                latent_model_input.shape[-3] // server.job_config.patch_size[0],
                latent_model_input.shape[-2] // server.job_config.patch_size[1],
                latent_model_input.shape[-1] // server.job_config.patch_size[2],
                0,
                1,
                frame_st_id,
            ).to(server.device)
            noisy_latents = latent_model_input.clone()
            if latent_cond is not None:
                noisy_latents[:, :, 0:1] = latent_cond[:, :, 0:1]
                timesteps[0:1] *= 0
            input_dict["latent_res_lst"] = {
                "noisy_latents": noisy_latents,
                "timesteps": timesteps,
                "grid_id": grid_id,
                "text_emb": server.prompt_embeds.to(server.dtype).clone(),
            }

        if action_model_input is not None:
            timesteps = (
                torch.ones(
                    [action_model_input.shape[2]],
                    dtype=torch.float32,
                    device=server.device,
                )
                * action_t
            )
            grid_id = self._get_mesh_id(
                action_model_input.shape[-3],
                action_model_input.shape[-2],
                action_model_input.shape[-1],
                1,
                1,
                frame_st_id,
                action=True,
            ).to(server.device)
            noisy_actions = action_model_input.clone()
            if action_cond is not None:
                noisy_actions[:, :, 0:1] = action_cond[:, :, 0:1]
                timesteps[0:1] *= 0
            noisy_actions[:, ~server.action_mask] *= 0
            input_dict["action_res_lst"] = {
                "noisy_latents": noisy_actions,
                "timesteps": timesteps,
                "grid_id": grid_id,
                "text_emb": server.prompt_embeds.to(server.dtype).clone(),
            }

        for input_value in input_dict.values():
            if server.use_cfg:
                input_value["noisy_latents"] = input_value["noisy_latents"].repeat(
                    2, 1, 1, 1, 1
                )
                input_value["text_emb"] = torch.cat(
                    [
                        server.prompt_embeds.to(server.dtype).clone(),
                        server.negative_prompt_embeds.to(server.dtype).clone(),
                    ],
                    dim=0,
                )
                input_value["grid_id"] = input_value["grid_id"][None].repeat(
                    self._cfg_batch_size(batch_size), 1, 1
                )
                input_value["timesteps"] = input_value["timesteps"][None].repeat(
                    self._cfg_batch_size(batch_size), 1
                )
            else:
                input_value["grid_id"] = input_value["grid_id"][None].repeat(
                    batch_size, 1, 1
                )
                input_value["timesteps"] = input_value["timesteps"][None].repeat(
                    batch_size, 1
                )
        return input_dict

    def _postprocess_action_batch(self, action: torch.Tensor) -> list[np.ndarray]:
        server = self._server
        action = action.detach().cpu()[..., 0]
        if server.action_norm_method == "quantiles":
            action = (action + 1) / 2 * (
                server.actions_q99 - server.actions_q01 + 1e-6
            ) + server.actions_q01
        else:
            raise NotImplementedError
        action_np = action.numpy()
        used = action_np[:, server.job_config.used_action_channel_ids]
        return [used[idx].astype(np.float32) for idx in range(used.shape[0])]

    def _preprocess_action_batch(self, state_batch: np.ndarray) -> torch.Tensor:
        server = self._server
        tensors = [
            server.preprocess_action(np.asarray(state, dtype=np.float32))
            for state in state_batch
        ]
        return torch.cat(tensors, dim=0)

    def _infer_batch_impl(
        self,
        obs_batch: list[dict[str, Any]] | list[list[dict[str, Any]]],
        *,
        frame_st_id: int = 0,
    ) -> list[np.ndarray]:
        server = self._server
        obs_sequences = self._normalize_obs_sequences(obs_batch)
        batch_size = len(obs_sequences)

        if frame_st_id == 0 and server.init_latent is None:
            server.init_latent = self._encode_obs_batch(obs_sequences)

        latents = torch.randn(
            batch_size,
            48,
            server.job_config.frame_chunk_size,
            server.latent_height,
            server.latent_width,
            device=server.device,
            dtype=server.dtype,
        )
        actions = torch.randn(
            batch_size,
            server.job_config.action_dim,
            server.job_config.frame_chunk_size,
            server.action_per_frame,
            1,
            device=server.device,
            dtype=server.dtype,
        )

        server.scheduler.set_timesteps(server.job_config.num_inference_steps)
        server.action_scheduler.set_timesteps(
            server.job_config.action_num_inference_steps
        )
        timesteps = torch.nn.functional.pad(
            server.scheduler.timesteps, (0, 1), mode="constant", value=0
        )
        if server.job_config.video_exec_step != -1:
            timesteps = timesteps[: server.job_config.video_exec_step]
        action_timesteps = torch.nn.functional.pad(
            server.action_scheduler.timesteps, (0, 1), mode="constant", value=0
        )

        with torch.no_grad():
            for step_idx, timestep in enumerate(timesteps):
                last_step = step_idx == len(timesteps) - 1
                latent_cond = (
                    server.init_latent[:, :, 0:1] if frame_st_id == 0 else None
                )
                input_dict = self._prepare_batch_input(
                    latent_model_input=latents,
                    action_model_input=None,
                    latent_t=float(timestep),
                    action_t=float(timestep),
                    latent_cond=latent_cond,
                    action_cond=None,
                    frame_st_id=frame_st_id,
                )
                video_noise_pred = server.transformer(
                    input_dict["latent_res_lst"],
                    update_cache=1 if last_step else 0,
                    cache_name=server.cache_name,
                    action_mode=False,
                )
                if not last_step or server.job_config.video_exec_step != -1:
                    video_noise_pred = self._data_seq_to_patch(
                        server.job_config.patch_size,
                        video_noise_pred,
                        server.job_config.frame_chunk_size,
                        server.latent_height,
                        server.latent_width,
                        batch_size=self._cfg_batch_size(batch_size),
                    )
                    if server.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[
                            batch_size:
                        ] + server.job_config.guidance_scale * (
                            video_noise_pred[:batch_size]
                            - video_noise_pred[batch_size:]
                        )
                    else:
                        video_noise_pred = video_noise_pred[:batch_size]
                    latents = server.scheduler.step(
                        video_noise_pred, timestep, latents, return_dict=False
                    )
                if latent_cond is not None:
                    latents[:, :, 0:1] = latent_cond

            for step_idx, timestep in enumerate(action_timesteps):
                last_step = step_idx == len(action_timesteps) - 1
                action_cond = (
                    torch.zeros(
                        [
                            batch_size,
                            server.job_config.action_dim,
                            1,
                            server.action_per_frame,
                            1,
                        ],
                        device=server.device,
                        dtype=server.dtype,
                    )
                    if frame_st_id == 0
                    else None
                )
                input_dict = self._prepare_batch_input(
                    latent_model_input=None,
                    action_model_input=actions,
                    latent_t=float(timestep),
                    action_t=float(timestep),
                    latent_cond=None,
                    action_cond=action_cond,
                    frame_st_id=frame_st_id,
                )
                action_noise_pred = server.transformer(
                    input_dict["action_res_lst"],
                    update_cache=1 if last_step else 0,
                    cache_name=server.cache_name,
                    action_mode=True,
                )
                if not last_step:
                    action_noise_pred = (
                        action_noise_pred.unflatten(
                            1,
                            (
                                server.job_config.frame_chunk_size,
                                server.action_per_frame,
                            ),
                        )
                        .permute(0, 3, 1, 2)
                        .unsqueeze(-1)
                    )
                    if server.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[
                            batch_size:
                        ] + server.job_config.action_guidance_scale * (
                            action_noise_pred[:batch_size]
                            - action_noise_pred[batch_size:]
                        )
                    else:
                        action_noise_pred = action_noise_pred[:batch_size]
                    actions = server.action_scheduler.step(
                        action_noise_pred, timestep, actions, return_dict=False
                    )
                if action_cond is not None:
                    actions[:, :, 0:1] = action_cond

        actions[:, ~server.action_mask] *= 0
        torch.cuda.empty_cache()
        return self._postprocess_action_batch(actions)

    def _replay_kv_cache_entry(
        self,
        latent_model_input: torch.Tensor,
        action_model_input: torch.Tensor,
        frame_st_id: int,
    ) -> int:
        """Write one history entry into the KV cache from PRE-COMPUTED tensors.

        The cache-only forwards (``update_cache=2``) the action step later reads
        depend solely on ``latent_model_input`` (post VAE-encode and, on the
        first entry, the prepended ``init_latent``) and ``action_model_input``
        (post action preprocessing) — NOT on the raw obs. So the RL recompute
        can reproduce the exact same cache from the stored tensors without
        re-running the VAE. Returns the advanced ``frame_st_id``.
        """
        server = self._server
        server.transformer.clear_pred_cache(server.cache_name)
        input_dict = self._prepare_batch_input(
            latent_model_input=latent_model_input,
            action_model_input=action_model_input,
            frame_st_id=frame_st_id,
        )
        with torch.no_grad():
            server.transformer(
                input_dict["latent_res_lst"],
                update_cache=2,
                cache_name=server.cache_name,
                action_mode=False,
            )
            server.transformer(
                input_dict["action_res_lst"],
                update_cache=2,
                cache_name=server.cache_name,
                action_mode=True,
            )
        return frame_st_id + int(latent_model_input.shape[2])

    def _compute_kv_cache_batch_impl(
        self,
        *,
        obs_batch: list[list[dict[str, Any]]],
        state_batch: np.ndarray,
        frame_st_id: int,
        capture: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> int:
        server = self._server
        latent_model_input = self._encode_obs_batch(obs_batch)
        if frame_st_id == 0:
            latent_model_input = (
                torch.cat([server.init_latent, latent_model_input], dim=2)
                if latent_model_input is not None
                else server.init_latent
            )

        action_model_input = self._preprocess_action_batch(state_batch).to(
            latent_model_input
        )
        if capture is not None:
            # Stash the exact cache-writing tensors so the RL recompute can
            # replay this entry deterministically (see _replay_kv_cache_entry).
            capture.append(
                (
                    latent_model_input.detach().to("cpu"),
                    action_model_input.detach().to("cpu"),
                )
            )
        new_frame_st_id = self._replay_kv_cache_entry(
            latent_model_input, action_model_input, frame_st_id
        )
        torch.cuda.empty_cache()
        return new_frame_st_id

    def infer_batch(
        self,
        obs_batch: list[dict[str, Any]],
        prompts: list[str],
        *,
        kv_cache_histories: (
            list[list[tuple[list[dict[str, Any]], np.ndarray]]] | None
        ) = None,
    ) -> list[np.ndarray]:
        """Run inference, optionally replaying past chunk history.

        If ``kv_cache_histories`` is provided, we encode the cached
        ``first_obs`` to populate ``init_latent`` then sequentially feed the
        history's key-frame batches through the transformer's KV cache before
        running the actual inference at the resulting ``frame_st_id``. This
        matches the official LingBot-VA loop where every chunk benefits from
        the prior chunks' video/action context.
        """
        if len(obs_batch) != len(prompts):
            raise ValueError(
                "LingBot-VA batch infer expects equal numbers of obs and prompts, got "
                f"{len(obs_batch)} and {len(prompts)}."
            )
        self._reset_batch_runtime(prompts)
        frame_st_id = 0
        if kv_cache_histories and any(len(h) > 0 for h in kv_cache_histories):
            history_lengths = {len(history) for history in kv_cache_histories}
            if len(history_lengths) != 1:
                raise ValueError(
                    "LingBot-VA batched follow-up infer requires uniform kv history "
                    f"lengths across the batch, got {history_lengths}."
                )
            history_len = history_lengths.pop()
            self._server.init_latent = self._encode_obs_batch(obs_batch)
            for history_idx in range(history_len):
                obs_sequences = [
                    history[history_idx][0] for history in kv_cache_histories
                ]
                state_batch = np.stack(
                    [history[history_idx][1] for history in kv_cache_histories], axis=0
                )
                frame_st_id = self._compute_kv_cache_batch_impl(
                    obs_batch=obs_sequences,
                    state_batch=state_batch,
                    frame_st_id=frame_st_id,
                )
        return self._infer_batch_impl(obs_batch, frame_st_id=frame_st_id)

    # ------------------------------------------------------------------
    # RL (GRPO) — Phase-1 DRAFT (not yet validated against the repo)
    # ------------------------------------------------------------------
    # These reproduce the loops in `_infer_batch_impl` with two changes:
    #   * sampling makes ONE action-denoise step a stochastic SDE and records
    #     its Gaussian log-prob + the minimal state needed to replay it;
    #   * recompute re-commits only the single video-cache entry (deterministic)
    #     then re-runs that one action step with gradients.
    # The SDE/log-prob math comes from rl_engine / rl_utils (unit-tested). The
    # `wan_va`-coupled calls are kept byte-for-byte consistent with the eval
    # loops above; `# VALIDATE:` marks assumptions to confirm with the repo.
    # Helpers are duplicated (not refactored into the eval path) so the
    # validated eval/SFT behaviour is untouched until these are tested.

    def _run_video_loop(
        self, latents: torch.Tensor, batch_size: int, frame_st_id: int
    ) -> torch.Tensor:
        """Deterministic video diffusion; populates the KV cache. See the video
        loop in `_infer_batch_impl` — keep in sync. Returns the final latents.
        """
        server = self._server
        server.scheduler.set_timesteps(server.job_config.num_inference_steps)
        timesteps = torch.nn.functional.pad(
            server.scheduler.timesteps, (0, 1), mode="constant", value=0
        )
        if server.job_config.video_exec_step != -1:
            # VALIDATE: the draft assumes video_exec_step == -1 so the single
            # cache-commit step is the final t=0 forward (see _commit_video_cache).
            timesteps = timesteps[: server.job_config.video_exec_step]
        for step_idx, timestep in enumerate(timesteps):
            last_step = step_idx == len(timesteps) - 1
            latent_cond = (
                server.init_latent[:, :, 0:1] if frame_st_id == 0 else None
            )
            input_dict = self._prepare_batch_input(
                latent_model_input=latents,
                action_model_input=None,
                latent_t=float(timestep),
                action_t=float(timestep),
                latent_cond=latent_cond,
                action_cond=None,
                frame_st_id=frame_st_id,
            )
            video_noise_pred = server.transformer(
                input_dict["latent_res_lst"],
                update_cache=1 if last_step else 0,
                cache_name=server.cache_name,
                action_mode=False,
            )
            if not last_step or server.job_config.video_exec_step != -1:
                video_noise_pred = self._data_seq_to_patch(
                    server.job_config.patch_size,
                    video_noise_pred,
                    server.job_config.frame_chunk_size,
                    server.latent_height,
                    server.latent_width,
                    batch_size=self._cfg_batch_size(batch_size),
                )
                if server.job_config.guidance_scale > 1:
                    video_noise_pred = video_noise_pred[
                        batch_size:
                    ] + server.job_config.guidance_scale * (
                        video_noise_pred[:batch_size]
                        - video_noise_pred[batch_size:]
                    )
                else:
                    video_noise_pred = video_noise_pred[:batch_size]
                latents = server.scheduler.step(
                    video_noise_pred, timestep, latents, return_dict=False
                )
            if latent_cond is not None:
                latents[:, :, 0:1] = latent_cond
        return latents

    def _commit_video_cache(
        self, video_latents: torch.Tensor, batch_size: int, frame_st_id: int
    ) -> None:
        """Re-run ONLY the final (t=0, update_cache=1) video forward.

        Only that forward writes the KV cache the action steps read, so this
        reproduces the action-conditioning deterministically from the stored
        final video latents — no need to replay the whole video diffusion.
        """
        server = self._server
        latent_cond = server.init_latent[:, :, 0:1] if frame_st_id == 0 else None
        input_dict = self._prepare_batch_input(
            latent_model_input=video_latents,
            action_model_input=None,
            latent_t=0.0,
            action_t=0.0,
            latent_cond=latent_cond,
            action_cond=None,
            frame_st_id=frame_st_id,
        )
        server.transformer(
            input_dict["latent_res_lst"],
            update_cache=1,
            cache_name=server.cache_name,
            action_mode=False,
        )

    def _action_velocity(
        self,
        actions: torch.Tensor,
        timestep: float,
        frame_st_id: int,
        batch_size: int,
        update_cache: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One action-transformer forward -> velocity reshaped to `actions`.

        Mirrors the per-step body of `_infer_batch_impl`'s action loop up to
        (but excluding) `action_scheduler.step`. Returns `(velocity, action_cond)`.
        """
        server = self._server
        action_cond = (
            torch.zeros(
                [batch_size, server.job_config.action_dim, 1, server.action_per_frame, 1],
                device=server.device,
                dtype=server.dtype,
            )
            if frame_st_id == 0
            else None
        )
        input_dict = self._prepare_batch_input(
            latent_model_input=None,
            action_model_input=actions,
            latent_t=float(timestep),
            action_t=float(timestep),
            latent_cond=None,
            action_cond=action_cond,
            frame_st_id=frame_st_id,
        )
        action_noise_pred = server.transformer(
            input_dict["action_res_lst"],
            update_cache=update_cache,
            cache_name=server.cache_name,
            action_mode=True,
        )
        action_noise_pred = (
            action_noise_pred.unflatten(
                1, (server.job_config.frame_chunk_size, server.action_per_frame)
            )
            .permute(0, 3, 1, 2)
            .unsqueeze(-1)
        )
        if server.job_config.action_guidance_scale > 1:
            action_noise_pred = action_noise_pred[
                batch_size:
            ] + server.job_config.action_guidance_scale * (
                action_noise_pred[:batch_size] - action_noise_pred[batch_size:]
            )
        else:
            action_noise_pred = action_noise_pred[:batch_size]
        return action_noise_pred, action_cond

    def _mask_action_std(
        self, std: torch.Tensor, action_cond: torch.Tensor | None
    ) -> torch.Tensor:
        """Zero the SDE std on coordinates with no randomness.

        Unused action channels (``~action_mask``) and, on the first chunk, the
        conditioned frame 0 are deterministic, so they must not contribute to
        the log-prob (``gaussian_logprob`` returns 0 where std == 0).
        """
        std = std.clone()
        std[:, ~self._server.action_mask] = 0.0
        if action_cond is not None:
            std[:, :, 0:1] = 0.0
        return std

    def _pack_history(
        self,
        history_capture: list[tuple[torch.Tensor, torch.Tensor]],
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Pack the captured per-entry cache tensors into fixed-shape blocks.

        The actor's trajectory pipeline concatenates ``forward_inputs`` across
        chunks, so the history block must have a constant shape regardless of
        how many entries this chunk actually replayed. We therefore pad to a
        config-fixed ``H_max`` entries / ``T_max`` total latent frames:

          * ``history_latents`` ``[B, C, T_max, h, w]`` — every entry's cache
            latent concatenated along the frame dim, zero-padded.
          * ``history_actions`` ``[B, H_max, A, FC, APF, 1]`` — per-entry action
            input, zero-padded.
          * ``history_lat_frames`` ``[B, H_max]`` long — frames per entry (0 for
            padding); used to slice ``history_latents`` and advance frame_st_id.
          * ``history_len`` ``[B]`` long — number of valid entries.
        """
        server = self._server
        H_max = int(getattr(self.cfg.lingbotva, "kv_replay_max_history", 24))
        T_max = int(getattr(self.cfg.lingbotva, "kv_replay_max_frames", 80))
        il = server.init_latent
        ch, lat_h, lat_w = int(il.shape[1]), int(il.shape[-2]), int(il.shape[-1])
        dtype = il.dtype
        H = len(history_capture)

        history_latents = torch.zeros(
            batch_size, ch, T_max, lat_h, lat_w, dtype=dtype
        )
        history_lat_frames = torch.zeros(batch_size, H_max, dtype=torch.long)
        if H > 0:
            act_shape = tuple(history_capture[0][1].shape[1:])  # [A, FC, APF, 1]
            history_actions = torch.zeros(
                batch_size, H_max, *act_shape, dtype=dtype
            )
            offset = 0
            for i, (lat_i, act_i) in enumerate(history_capture):
                t_i = int(lat_i.shape[2])
                if offset + t_i > T_max:
                    raise ValueError(
                        f"KV-replay history frames {offset + t_i} exceed "
                        f"kv_replay_max_frames={T_max}; raise the config knob."
                    )
                history_latents[:, :, offset : offset + t_i] = lat_i.to(dtype)
                history_lat_frames[:, i] = t_i
                history_actions[:, i] = act_i.to(dtype)
                offset += t_i
        else:
            a = int(server.job_config.action_dim)
            fc = int(server.job_config.frame_chunk_size)
            apf = int(server.action_per_frame)
            history_actions = torch.zeros(
                batch_size, H_max, a, fc, apf, 1, dtype=dtype
            )
        return {
            "history_latents": history_latents.cpu(),
            "history_actions": history_actions.cpu(),
            "history_lat_frames": history_lat_frames.cpu(),
            "history_len": torch.full((batch_size,), H, dtype=torch.long),
        }

    def infer_batch_with_logprob(
        self,
        obs_batch: list[dict[str, Any]],
        prompts: list[str],
        *,
        noise_level: float = 1.0,
        kv_cache_histories: (
            list[list[tuple[list[dict[str, Any]], np.ndarray]]] | None
        ) = None,
    ) -> dict[str, Any]:
        """RL rollout: sample the action chain as an SDE and return log-probs.

        Mirrors :meth:`infer_batch` (incl. cross-chunk KV-cache replay) but makes
        one action-denoise step stochastic and returns its Gaussian log-prob.
        When ``kv_cache_histories`` is given, the prior chunks' key-frame obs +
        actions are replayed into the transformer's KV cache (advancing
        ``frame_st_id``) before the current chunk is generated — this is what
        restores non-zero rollout success rate.

        Returns the raw pieces the caller folds into an ``RLForwardInputs``.
        NOTE: under replay (``frame_st_id > 0``) the action conditioning depends
        on the replayed cache; ``recompute_logprob`` currently reproduces only
        the single video commit, so recompute consistency under replay is a
        follow-up (RL_DESIGN.md; task: distributed recompute consistency).
        """
        if len(obs_batch) != len(prompts):
            raise ValueError(
                "infer_batch_with_logprob expects equal numbers of obs and "
                f"prompts, got {len(obs_batch)} and {len(prompts)}."
            )
        self._reset_batch_runtime(prompts)
        server = self._server
        frame_st_id = 0
        obs_sequences = self._normalize_obs_sequences(obs_batch)
        batch_size = len(obs_sequences)
        # Capture each replayed entry's exact cache-writing tensors so the actor
        # recompute can reproduce the same KV cache (see recompute_logprob).
        history_capture: list[tuple[torch.Tensor, torch.Tensor]] = []
        # Must equal _pack_history's H_max so the rollout replays exactly the
        # entries that get stored for recompute (recompute consistency).
        max_history = int(getattr(self.cfg.lingbotva, "kv_replay_max_history", 24))

        with torch.no_grad():
            # Replay prior chunks into the KV cache (mirrors infer_batch).
            if kv_cache_histories and any(len(h) > 0 for h in kv_cache_histories):
                history_lengths = {len(h) for h in kv_cache_histories}
                if len(history_lengths) != 1:
                    raise ValueError(
                        "infer_batch_with_logprob requires uniform kv history "
                        f"lengths across the batch, got {history_lengths}."
                    )
                history_len = history_lengths.pop()
                # Cap to the most recent ``max_history`` entries so both the
                # rollout AND the recompute (which stores a fixed-size history
                # block) replay the SAME context — recompute consistency.
                start = max(0, history_len - max_history)
                server.init_latent = self._encode_obs_batch(obs_batch)
                for history_idx in range(start, history_len):
                    hist_obs = [h[history_idx][0] for h in kv_cache_histories]
                    state_batch = np.stack(
                        [h[history_idx][1] for h in kv_cache_histories], axis=0
                    )
                    frame_st_id = self._compute_kv_cache_batch_impl(
                        obs_batch=hist_obs,
                        state_batch=state_batch,
                        frame_st_id=frame_st_id,
                        capture=history_capture,
                    )
            else:
                server.init_latent = self._encode_obs_batch(obs_sequences)
            latents = torch.randn(
                batch_size,
                48,
                server.job_config.frame_chunk_size,
                server.latent_height,
                server.latent_width,
                device=server.device,
                dtype=server.dtype,
            )
            actions = torch.randn(
                batch_size,
                server.job_config.action_dim,
                server.job_config.frame_chunk_size,
                server.action_per_frame,
                1,
                device=server.device,
                dtype=server.dtype,
            )
            video_latents = self._run_video_loop(latents, batch_size, frame_st_id)

            server.action_scheduler.set_timesteps(
                server.job_config.action_num_inference_steps
            )
            action_timesteps = torch.nn.functional.pad(
                server.action_scheduler.timesteps, (0, 1), mode="constant", value=0
            )
            num_action_steps = len(action_timesteps) - 1
            sigmas_padded = normalized_action_sigmas(
                server.action_scheduler, num_action_steps
            )
            chosen_step = select_denoise_index(num_action_steps)

            chains_pre = chains_next = None
            logprob = None
            for step_idx, timestep in enumerate(action_timesteps):
                last_step = step_idx == len(action_timesteps) - 1
                if last_step:
                    # Commit the action cache (matches eval's update_cache=1).
                    self._action_velocity(
                        actions, timestep, frame_st_id, batch_size, update_cache=1
                    )
                    break
                v_t, action_cond = self._action_velocity(
                    actions, timestep, frame_st_id, batch_size, update_cache=0
                )
                if step_idx == chosen_step:
                    mean, std = sde_mean_std(
                        actions, v_t, sigmas_padded, step_idx, noise_level
                    )
                    std = self._mask_action_std(std, action_cond)
                    next_actions = mean + torch.randn_like(mean) * std
                    logprob = reduce_chain_logprob(
                        gaussian_logprob(next_actions, mean, std)
                    )
                    chains_pre = actions.clone()
                    chains_next = next_actions.clone()
                    actions = next_actions
                else:
                    actions = server.action_scheduler.step(
                        v_t, timestep, actions, return_dict=False
                    )
                if action_cond is not None:
                    actions[:, :, 0:1] = action_cond

            actions[:, ~server.action_mask] *= 0

        if logprob is None:
            raise RuntimeError("RL rollout did not score any denoise step.")
        raw_actions = self._postprocess_action_batch(actions)
        history_pack = self._pack_history(history_capture, batch_size)
        torch.cuda.empty_cache()
        return {
            "actions": raw_actions,
            "logprob": logprob.detach().cpu(),
            # Pieces for RLForwardInputs.build (caller adds exec_steps).
            "action_chains": torch.stack([chains_pre, chains_next], dim=1).cpu(),
            "denoise_inds": torch.full((batch_size,), chosen_step, dtype=torch.long),
            "init_latent": server.init_latent.detach().cpu(),
            "video_latents": video_latents.detach().cpu(),
            "prompt_embeds": server.prompt_embeds.detach().cpu(),
            "negative_prompt_embeds": (
                server.negative_prompt_embeds.detach().cpu()
                if server.negative_prompt_embeds is not None
                else None
            ),
            "frame_st_id": frame_st_id,
            "noise_level": float(noise_level),
            "num_action_steps": num_action_steps,
            "action_dim": server.job_config.action_dim,
            # Cache-replay history for recompute consistency (task #4 / #2).
            **history_pack,
        }

    def _replay_history_cache(
        self, fi: RLForwardInputs, batch_size: int
    ) -> int:
        """Replay the stored history entries into the KV cache (no grad).

        Mirrors the rollout's per-entry ``_compute_kv_cache_batch_impl`` loop
        using the stored (already encoded + init-prepended) tensors. Returns the
        cumulative ``frame_st_id`` after replay (0 when there is no history).

        Assumes a homogeneous micro-batch (the RL config pins
        ``micro_batch_size == 1``), so the per-entry frame counts are read from
        sample 0.
        """
        if fi.history_len is None or fi.history_latents is None:
            return 0
        hist_len = int(fi.history_len[0].item())
        if hist_len <= 0:
            return 0
        frames = fi.history_lat_frames[0]  # [H_max] per-entry latent frames
        frame_st_id = 0
        offset = 0
        for i in range(hist_len):
            t_i = int(frames[i].item())
            lat_i = fi.history_latents[:, :, offset : offset + t_i]
            act_i = fi.history_actions[:, i]
            frame_st_id = self._replay_kv_cache_entry(lat_i, act_i, frame_st_id)
            offset += t_i
        return frame_st_id

    def recompute_logprob(
        self,
        forward_inputs: RLForwardInputs,
        *,
        compute_entropy: bool = False,
        compute_values: bool = False,
    ) -> dict[str, torch.Tensor]:
        """RL update: recompute the scored step's log-prob WITH gradients.

        Re-commits the single video-cache entry deterministically from the
        stored final video latents, then re-runs the one stored action-denoise
        step through the (trainable) transformer. Gradients flow through the
        action step only; the video conditioning is treated as fixed (run under
        ``no_grad``), mirroring how OpenPI/lingbotvla detach the prefix.
        """
        server = self._server
        fi = forward_inputs.to(server.device)
        batch_size = fi.action_chains.shape[0]

        # Group rows that share (frame_st_id, denoise_ind): they have identical
        # history structure and scored timestep, so each group recomputes as ONE
        # batched forward. This amortizes the FSDP all-gather over the whole
        # group (~chunk-size speedup vs the per-sample micro_batch_size=1 path)
        # while staying exact, and transparently supports heterogeneous
        # micro-batches (mixed chunks) without per-sample action timesteps.
        keys = (fi.frame_st_id.to(torch.long) << 20) + fi.denoise_inds.to(torch.long)
        order: list[torch.Tensor] = []
        logp_parts: list[torch.Tensor] = []
        ent_parts: list[torch.Tensor] = []
        for key in torch.unique(keys):
            idx = torch.nonzero(keys == key, as_tuple=False).flatten()
            out = self._recompute_group(
                fi.index_select(idx),
                compute_entropy=compute_entropy,
            )
            order.append(idx)
            logp_parts.append(out["logprobs"])
            ent_parts.append(out["entropy"])

        order_cat = torch.cat(order)
        inv = torch.empty_like(order_cat)
        inv[order_cat] = torch.arange(batch_size, device=order_cat.device)
        logprobs = torch.cat(logp_parts)[inv]
        entropy = torch.cat(ent_parts)[inv]
        values = torch.zeros(batch_size, device=server.device)  # GRPO: critic-free
        return {"logprobs": logprobs, "entropy": entropy, "values": values}

    def _recompute_group(
        self,
        fi: RLForwardInputs,
        *,
        compute_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Recompute one homogeneous group (single frame_st_id + denoise_ind).

        Re-commits the single video-cache entry deterministically (after
        replaying the stored history KV cache), then re-runs the one stored
        action-denoise step through the (trainable) transformer. Gradients flow
        through the action step only; the conditioning is run under ``no_grad``,
        mirroring how OpenPI/lingbotvla detach the prefix.
        """
        server = self._server
        batch_size = fi.action_chains.shape[0]
        denoise_inds = fi.denoise_inds.to(torch.long)
        num_action_steps = fi.scalar_num_action_steps
        frame_st_id = fi.scalar_frame_st_id
        noise_level = fi.scalar_noise_level

        self._setup_batch_runtime(batch_size)
        server.init_latent = fi.init_latent
        server.prompt_embeds = fi.prompt_embeds
        server.negative_prompt_embeds = fi.negative_prompt_embeds

        # Replaying the per-chunk history makes the recompute log-prob exact
        # (ratio==1), but it runs a DATA-DEPENDENT number of transformer forwards
        # (history grows with chunk index). Under FSDP each forward all-gathers
        # params, so different data-parallel ranks issue different collectives
        # and DEADLOCK (NCCL watchdog timeout). Until a rank-symmetric replay
        # (pad every rank to a global-max forward count) lands, this knob lets
        # distributed training skip the replay: the recompute then does a fixed
        # 2 forwards/sample (commit + action) -> identical across ranks -> synced,
        # at the cost of a mild history-conditioning bias in the ratio (~0.7).
        replay = bool(getattr(self.cfg.lingbotva, "recompute_kv_replay", True))
        with torch.no_grad():
            if replay:
                replay_frame_st_id = self._replay_history_cache(fi, batch_size)
                if replay_frame_st_id != frame_st_id:
                    raise RuntimeError(
                        "recompute history replay reached frame_st_id "
                        f"{replay_frame_st_id} but forward_inputs stored "
                        f"{frame_st_id}; KV-replay history pack is inconsistent."
                    )
            self._commit_video_cache(fi.video_latents, batch_size, frame_st_id)

        server.action_scheduler.set_timesteps(num_action_steps)
        action_timesteps = torch.nn.functional.pad(
            server.action_scheduler.timesteps, (0, 1), mode="constant", value=0
        )
        sigmas_padded = normalized_action_sigmas(
            server.action_scheduler, num_action_steps
        )
        chains_pre = fi.action_chains[:, 0]
        chains_next = fi.action_chains[:, 1]

        # Group is homogeneous in denoise_ind, so one scalar timestep applies.
        timestep = float(action_timesteps[int(denoise_inds[0].item())])

        # Gradients on: this is the only forward the GRPO update backprops.
        v_t, action_cond = self._action_velocity(
            chains_pre, timestep, frame_st_id, batch_size, update_cache=0
        )
        mean, std = sde_mean_std(
            chains_pre, v_t, sigmas_padded, denoise_inds, noise_level
        )
        std = self._mask_action_std(std, action_cond)
        logprobs = reduce_chain_logprob(gaussian_logprob(chains_next, mean, std))
        if compute_entropy:
            entropy = reduce_chain_logprob(gaussian_entropy(std))
        else:
            entropy = torch.zeros(batch_size, device=server.device)
        return {"logprobs": logprobs, "entropy": entropy}

    def close(self) -> None:
        if getattr(self, "_server", None) is None:
            return
        transformer = getattr(self._server, "transformer", None)
        if transformer is not None:
            try:
                transformer.clear_cache(self._BATCH_CACHE_NAME)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
