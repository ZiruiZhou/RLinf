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

"""Orchestration for LingBot-VA GRPO: turning the action-denoising loop into a
stochastic SDE policy and scoring it.

This module holds the parts of the RL path that do NOT depend on the external
``wan_va`` package, so they can be unit tested without the LingBot-VA repo,
weights, or a GPU:

  * choosing which denoising step is made stochastic,
  * extracting the rectified-flow sigma schedule from a scheduler,
  * the per-step SDE mean/std (delegating to :mod:`rl_utils`),
  * reducing the per-element Gaussian log-prob to the per-sample value GRPO
    consumes, and broadcasting it over the executed action steps,
  * the ``RLForwardInputs`` contract that ``predict_action_batch`` stores and
    ``get_log_prob_value`` replays.

The ``wan_va``-coupled steps (running the transformer, the cache, the VAE/text
encoder) live in ``eval_adapter/native_backend.py`` and the action model, and
call into the helpers here. Every place that depends on a ``wan_va`` behaviour
we could not verify offline is flagged ``# VALIDATE:``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rlinf.models.embodiment.lingbotva.rl_utils import (
    flow_sde_step,
    gaussian_entropy,
    gaussian_logprob,
    sde_sigmas_from_timesteps,
)

__all__ = [
    "RLForwardInputs",
    "select_denoise_index",
    "normalized_action_sigmas",
    "sde_mean_std",
    "reduce_chain_logprob",
    "broadcast_logprob_to_actions",
]


_RL_FI_FIELDS = (
    "action_chains",
    "denoise_inds",
    "init_latent",
    "video_latents",
    "prompt_embeds",
    "negative_prompt_embeds",
    "frame_st_id",
    "noise_level",
    "num_action_steps",
    "exec_steps",
    "action_dim",
    # Cross-chunk KV-cache replay (recompute consistency under frame_st_id>0).
    "history_latents",
    "history_actions",
    "history_lat_frames",
    "history_len",
)


@dataclass
class RLForwardInputs:
    """Everything ``get_log_prob_value`` needs to replay the scored step.

    Stored by the rollout (``_rl_predict_action_batch``) into
    ``result["forward_inputs"]`` and rebuilt on the actor for the GRPO update.

    **Every field is a batch-dim-0 tensor** (no python scalars, no nested
    dicts). This is a hard requirement: the actor's trajectory pipeline batches
    ``forward_inputs`` with ``split_dict_to_chunk`` / ``concat_batch``, which
    raise on / silently drop non-tensor values. Per-call scalars (frame_st_id,
    noise_level, num_action_steps, exec_steps, action_dim) are therefore stored
    as ``[B]`` tensors and read back via the ``scalar_*`` accessors.

    ``denoise_inds`` is genuinely per-sample (``[B]``): trajectory concat can
    interleave rollout calls that scored different steps, so the recompute must
    handle a heterogeneous batch.

    Attributes:
        action_chains: ``[B, 2, *action_latent_shape]`` — the scored transition
            ``chains[:, 0] (pre) -> chains[:, 1] (next)``.
        denoise_inds: ``[B]`` long — the scored step index per sample.
        init_latent: VAE-encoded obs latent (``latent_cond`` pinning frame 0).
        video_latents: FINAL video-denoising latents. Only the last video
            forward (``update_cache=1``) writes the KV cache the action steps
            read; intermediate action steps are read-only (``update_cache=0``).
            So the actor re-commits that single cache entry DETERMINISTICALLY
            from these — no need to replay video diffusion. Keeps
            rollout/recompute log-probs consistent (gate #1).
        prompt_embeds / negative_prompt_embeds: cached UMT5 text embeddings.
        frame_st_id / noise_level / num_action_steps / exec_steps / action_dim:
            per-call scalars broadcast to ``[B]``.
    """

    action_chains: torch.Tensor
    denoise_inds: torch.Tensor
    init_latent: torch.Tensor
    video_latents: torch.Tensor
    prompt_embeds: torch.Tensor
    frame_st_id: torch.Tensor
    noise_level: torch.Tensor
    num_action_steps: torch.Tensor
    exec_steps: torch.Tensor
    action_dim: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None = None
    # Cross-chunk KV-cache replay (optional; absent -> recompute skips replay).
    history_latents: torch.Tensor | None = None
    history_actions: torch.Tensor | None = None
    history_lat_frames: torch.Tensor | None = None
    history_len: torch.Tensor | None = None

    @classmethod
    def build(
        cls,
        *,
        action_chains: torch.Tensor,
        denoise_inds: torch.Tensor,
        init_latent: torch.Tensor,
        video_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        frame_st_id: int,
        noise_level: float,
        num_action_steps: int,
        exec_steps: int,
        action_dim: int,
        history_latents: torch.Tensor | None = None,
        history_actions: torch.Tensor | None = None,
        history_lat_frames: torch.Tensor | None = None,
        history_len: torch.Tensor | None = None,
    ) -> "RLForwardInputs":
        """Construct from tensors + per-call scalars (scalars -> ``[B]`` tensors)."""
        batch = action_chains.shape[0]

        def _full(value, dtype):
            return torch.full((batch,), value, dtype=dtype)

        return cls(
            action_chains=action_chains,
            denoise_inds=denoise_inds.reshape(batch).to(torch.long),
            init_latent=init_latent,
            video_latents=video_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            frame_st_id=_full(int(frame_st_id), torch.long),
            noise_level=_full(float(noise_level), torch.float32),
            num_action_steps=_full(int(num_action_steps), torch.long),
            exec_steps=_full(int(exec_steps), torch.long),
            action_dim=_full(int(action_dim), torch.long),
            history_latents=history_latents,
            history_actions=history_actions,
            history_lat_frames=history_lat_frames,
            history_len=history_len,
        )

    # Per-call scalars are batch-constant; read the first element.
    @property
    def scalar_frame_st_id(self) -> int:
        return int(self.frame_st_id[0].item())

    @property
    def scalar_noise_level(self) -> float:
        return float(self.noise_level[0].item())

    @property
    def scalar_num_action_steps(self) -> int:
        return int(self.num_action_steps[0].item())

    @property
    def scalar_exec_steps(self) -> int:
        return int(self.exec_steps[0].item())

    @property
    def scalar_action_dim(self) -> int:
        return int(self.action_dim[0].item())

    def to(self, device) -> "RLForwardInputs":
        def _m(t):
            return t.to(device) if torch.is_tensor(t) else t

        return RLForwardInputs(**{k: _m(getattr(self, k)) for k in _RL_FI_FIELDS})

    def index_select(self, idx: torch.Tensor) -> "RLForwardInputs":
        """Return a sub-batch view selecting rows ``idx`` along dim 0.

        Used by the recompute to process a homogeneous group (same scored step +
        history structure) as one batched forward.
        """

        def _m(t):
            return t[idx] if torch.is_tensor(t) else t

        return RLForwardInputs(**{k: _m(getattr(self, k)) for k in _RL_FI_FIELDS})

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in _RL_FI_FIELDS}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RLForwardInputs":
        return cls(
            action_chains=d["action_chains"],
            denoise_inds=d["denoise_inds"],
            init_latent=d["init_latent"],
            video_latents=d["video_latents"],
            prompt_embeds=d["prompt_embeds"],
            negative_prompt_embeds=d.get("negative_prompt_embeds"),
            frame_st_id=d["frame_st_id"],
            noise_level=d["noise_level"],
            num_action_steps=d["num_action_steps"],
            exec_steps=d["exec_steps"],
            action_dim=d["action_dim"],
            history_latents=d.get("history_latents"),
            history_actions=d.get("history_actions"),
            history_lat_frames=d.get("history_lat_frames"),
            history_len=d.get("history_len"),
        )


def select_denoise_index(
    num_steps: int, generator: torch.Generator | None = None
) -> int:
    """Pick the single action-denoising step to make stochastic.

    Mirrors OpenPI/lingbotvla, which keep one sampled step per rollout (rather
    than summing log-probs over all steps) to bound RL cost. The final clean
    step (index ``num_steps``) has no stochastic transition, so we draw from
    ``[0, num_steps - 1]``.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}.")
    return int(torch.randint(0, num_steps, (1,), generator=generator).item())


def normalized_action_sigmas(scheduler: Any, num_steps: int) -> torch.Tensor:
    """Return the rectified-flow sigma schedule in ``[0, 1]``, padded with 0.

    LingBot-VA's action ``FlowMatchScheduler`` Euler step is
    ``x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v``; the SDE math needs those
    per-step sigmas (the flow "t") of length ``num_steps + 1`` with a trailing
    clean step at 0.

    ``# VALIDATE:`` we read ``scheduler.sigmas`` when present (FlowMatch stores
    sigmas in ``[0, 1]``); otherwise we fall back to
    ``scheduler.timesteps / num_train_timesteps``. Confirm the convention
    against the real scheduler.
    """
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None:
        timesteps = scheduler.timesteps
        denom = float(getattr(scheduler, "num_train_timesteps", 1000))
        sigmas = timesteps.to(torch.float32) / denom
    sigmas = torch.as_tensor(sigmas, dtype=torch.float32).flatten()
    sigmas = sigmas[: num_steps + 1] if sigmas.numel() >= num_steps + 1 else sigmas
    if sigmas.numel() == num_steps:  # not yet padded with the trailing clean step
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
    return sigmas


def sde_mean_std(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    sigmas_padded: torch.Tensor,
    idx: int | torch.Tensor,
    noise_level: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-step Gaussian transition ``(mean, std)`` for action step ``idx``.

    Builds the inputs that ``rl_utils.flow_sde_step`` expects from the sigma
    schedule and the transformer's velocity prediction ``v_t`` (the same tensor
    the deterministic scheduler would integrate). Supports a scalar ``idx``
    (rollout, all batch elements share the step) or a per-sample index tensor
    (recompute).
    """
    batch = x_t.shape[0]
    sigmas_padded = sigmas_padded.to(device=x_t.device, dtype=x_t.dtype)
    all_sigmas = sde_sigmas_from_timesteps(sigmas_padded, noise_level)

    def _per_sample(value: torch.Tensor) -> torch.Tensor:
        # Normalise a scalar (shared step) or [B] (per-sample step) to [B].
        value = torch.as_tensor(value, device=x_t.device, dtype=x_t.dtype)
        return value.expand(batch) if value.dim() == 0 else value

    t_input = _per_sample(sigmas_padded[idx])
    t_next = _per_sample(sigmas_padded[idx + 1])
    sigma_i = _per_sample(all_sigmas[idx])
    delta = t_input - t_next

    # Broadcast the per-sample scalars to x_t's shape.
    view = (batch,) + (1,) * (x_t.dim() - 1)
    t_input = t_input.reshape(view).expand_as(x_t)
    delta = delta.reshape(view).expand_as(x_t)
    sigma_i = sigma_i.reshape(view).expand_as(x_t)
    return flow_sde_step(x_t, v_t, t_input, delta, sigma_i)


def reduce_chain_logprob(per_elem_logprob: torch.Tensor) -> torch.Tensor:
    """Reduce a per-element Gaussian log-density to one log-prob per sample.

    The action latent has shape ``[B, ...]``; the diffusion-step transition is a
    joint Gaussian over all latent coordinates. The joint log-prob is the SUM
    over coordinates, but in the distributed GRPO setting the rollout samples
    with the unsharded bf16 transformer while the actor recomputes with the
    FSDP-sharded copy. Those two bf16 forwards agree only to ~1e-3 per element;
    summed over ~112 unmasked coordinates that becomes a multi-nat difference,
    and ``exp(logprob_new - logprob_old)`` blows the PPO ratio up to 1e4-1e15
    even at identical weights (the rollout/recompute consistency gate).

    We therefore return the per-coordinate MEAN over the *active* (non-masked)
    coordinates instead of the sum. ``gaussian_logprob`` returns exactly 0 where
    ``std == 0`` (masked / deterministic coordinates), so we average over the
    nonzero entries. Rollout and recompute apply this identically and mask the
    same coordinates, so the ratio is unchanged in expectation but ~112x less
    sensitive to bf16 forward noise. Returns ``[B]``.
    """
    flat = per_elem_logprob.flatten(1)
    count = (flat != 0).sum(dim=1).clamp(min=1)
    return flat.sum(dim=1) / count


def broadcast_logprob_to_actions(
    logprob_b: torch.Tensor, exec_steps: int, action_dim: int
) -> torch.Tensor:
    """Expand a per-sample log-prob ``[B]`` to ``[B, exec_steps, action_dim]``.

    A single denoising-step log-prob represents the whole action chunk, so for
    ``logprob_type: action_level`` we broadcast it across the executed steps
    and action dims. The ``token-mean`` loss aggregator then averages back to a
    per-sample contribution, so the broadcast does not change the GRPO ratio's
    scale relative to ``old_logprobs`` (which is broadcast identically).
    """
    return logprob_b.reshape(-1, 1, 1).expand(-1, exec_steps, action_dim).contiguous()


def chain_logprob_and_entropy(
    sample: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample log-prob and entropy of one Gaussian denoising transition."""
    logp = reduce_chain_logprob(gaussian_logprob(sample, mean, std))
    ent = reduce_chain_logprob(gaussian_entropy(std))
    return logp, ent
