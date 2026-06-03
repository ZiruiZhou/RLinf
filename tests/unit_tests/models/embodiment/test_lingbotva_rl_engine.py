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

"""Unit tests for the wan_va-independent LingBot-VA RL orchestration.

Covers the Phase-1 draft helpers that do not touch the external LingBot-VA
package: denoise-index selection, sigma extraction, the SDE mean/std wiring,
log-prob reduction/broadcast, and the RLForwardInputs round-trip.
"""

from __future__ import annotations

import pytest
import torch

from rlinf.models.embodiment.lingbotva.rl_engine import (
    RLForwardInputs,
    broadcast_logprob_to_actions,
    chain_logprob_and_entropy,
    normalized_action_sigmas,
    reduce_chain_logprob,
    sde_mean_std,
    select_denoise_index,
)
from rlinf.models.embodiment.lingbotva.rl_utils import flow_ode_step

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# select_denoise_index
# ---------------------------------------------------------------------------
def test_select_denoise_index_in_range():
    for _ in range(200):
        idx = select_denoise_index(10)
        assert 0 <= idx <= 9  # never selects the trailing clean step (== 10)


def test_select_denoise_index_rejects_zero_steps():
    with pytest.raises(ValueError):
        select_denoise_index(0)


def test_select_denoise_index_is_reproducible_with_generator():
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = [select_denoise_index(50, generator=g1) for _ in range(5)]
    b = [select_denoise_index(50, generator=g2) for _ in range(5)]
    assert a == b


# ---------------------------------------------------------------------------
# normalized_action_sigmas
# ---------------------------------------------------------------------------
class _FakeScheduler:
    def __init__(self, sigmas=None, timesteps=None, num_train_timesteps=1000):
        if sigmas is not None:
            self.sigmas = torch.tensor(sigmas)
        if timesteps is not None:
            self.timesteps = torch.tensor(timesteps)
        self.num_train_timesteps = num_train_timesteps


def test_sigmas_from_scheduler_sigmas_attr():
    sched = _FakeScheduler(sigmas=[0.9, 0.6, 0.3])  # num_steps = 3, unpadded
    out = normalized_action_sigmas(sched, num_steps=3)
    assert out.shape == (4,)
    assert out[-1] == 0.0
    assert torch.allclose(out[:3], torch.tensor([0.9, 0.6, 0.3]))


def test_sigmas_fallback_to_timesteps():
    # No .sigmas -> normalize timesteps by num_train_timesteps.
    sched = _FakeScheduler(timesteps=[900, 600, 300], num_train_timesteps=1000)
    out = normalized_action_sigmas(sched, num_steps=3)
    assert torch.allclose(out[:3], torch.tensor([0.9, 0.6, 0.3]), atol=1e-6)
    assert out[-1] == 0.0


# ---------------------------------------------------------------------------
# sde_mean_std  (wiring of rl_utils into the schedule)
# ---------------------------------------------------------------------------
def test_sde_mean_matches_euler_when_noise_zero():
    # With noise_level=0 the SDE mean must equal the deterministic rectified-
    # flow Euler step x - delta*v (== scheduler.step), the eval-time update.
    B = 3
    x_t = torch.randn(B, 7, 4, 4, 1)
    v_t = torch.randn_like(x_t)
    sigmas = torch.tensor([0.9, 0.6, 0.3, 0.0])
    idx = 1
    mean, std = sde_mean_std(x_t, v_t, sigmas, idx, noise_level=0.0)
    delta = sigmas[idx] - sigmas[idx + 1]
    euler = x_t - delta * v_t
    assert torch.allclose(mean, euler, atol=1e-5)
    assert torch.allclose(std, torch.zeros_like(std), atol=1e-7)


def test_sde_mean_std_matches_flow_ode_helper():
    B = 2
    x_t = torch.randn(B, 7, 4, 4, 1)
    v_t = torch.randn_like(x_t)
    sigmas = torch.tensor([0.8, 0.5, 0.2, 0.0])
    idx = 0
    mean, _ = sde_mean_std(x_t, v_t, sigmas, idx, noise_level=0.0)
    t = sigmas[idx].expand_as(x_t)
    delta = (sigmas[idx] - sigmas[idx + 1]).expand_as(x_t)
    ode_mean, _ = flow_ode_step(x_t, v_t, t, delta)
    assert torch.allclose(mean, ode_mean, atol=1e-6)


def test_sde_std_positive_with_noise():
    x_t = torch.randn(2, 7, 4, 4, 1)
    v_t = torch.randn_like(x_t)
    sigmas = torch.tensor([0.9, 0.6, 0.3, 0.0])
    _, std = sde_mean_std(x_t, v_t, sigmas, 1, noise_level=1.0)
    assert torch.all(std > 0)


def test_sde_mean_std_per_sample_index():
    # Recompute path passes a per-sample index tensor.
    B = 4
    x_t = torch.randn(B, 7, 4, 4, 1)
    v_t = torch.randn_like(x_t)
    sigmas = torch.tensor([0.9, 0.6, 0.3, 0.0])
    idx = torch.tensor([0, 1, 2, 0])
    mean, std = sde_mean_std(x_t, v_t, sigmas, idx, noise_level=0.5)
    assert mean.shape == x_t.shape and std.shape == x_t.shape
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()


# ---------------------------------------------------------------------------
# logprob reduction / broadcast
# ---------------------------------------------------------------------------
def test_reduce_chain_logprob_sums_non_batch_dims():
    x = torch.ones(3, 7, 4, 4, 1)
    out = reduce_chain_logprob(x)
    assert out.shape == (3,)
    assert torch.allclose(out, torch.full((3,), float(7 * 4 * 4 * 1)))


def test_broadcast_logprob_shape_and_values():
    lp = torch.tensor([1.0, -2.0])
    out = broadcast_logprob_to_actions(lp, exec_steps=12, action_dim=7)
    assert out.shape == (2, 12, 7)
    assert torch.all(out[0] == 1.0) and torch.all(out[1] == -2.0)


def test_chain_logprob_and_entropy_consistency():
    mean = torch.randn(2, 7, 4)
    std = torch.rand(2, 7, 4) + 0.1
    sample = mean + torch.randn_like(mean) * std
    logp, ent = chain_logprob_and_entropy(sample, mean, std)
    assert logp.shape == (2,) and ent.shape == (2,)
    # entropy is sample-independent and positive for std in this range
    ref_ent = reduce_chain_logprob(
        0.5 * torch.log(2 * torch.pi * torch.e * std**2)
    )
    assert torch.allclose(ent, ref_ent, atol=1e-5)


# ---------------------------------------------------------------------------
# RLForwardInputs round-trip
# ---------------------------------------------------------------------------
def test_rl_forward_inputs_build_and_roundtrip():
    fi = RLForwardInputs.build(
        action_chains=torch.randn(2, 2, 7, 4, 4, 1),
        denoise_inds=torch.tensor([3, 3]),
        init_latent=torch.randn(2, 48, 1, 8, 16),
        video_latents=torch.randn(2, 48, 4, 8, 16),
        prompt_embeds=torch.randn(2, 512, 768),
        negative_prompt_embeds=None,
        frame_st_id=4,
        noise_level=0.8,
        num_action_steps=5,
        exec_steps=12,
        action_dim=7,
    )
    # Per-call scalars are stored as [B] tensors and exposed via accessors.
    assert fi.frame_st_id.shape == (2,) and fi.exec_steps.shape == (2,)
    assert fi.scalar_frame_st_id == 4 and fi.scalar_exec_steps == 12
    assert abs(fi.scalar_noise_level - 0.8) < 1e-6
    assert fi.scalar_num_action_steps == 5 and fi.scalar_action_dim == 7

    d = fi.as_dict()
    # Every value must be a tensor (or None) for the actor's batching pipeline.
    for k, v in d.items():
        assert v is None or torch.is_tensor(v), f"{k} is non-tensor: {type(v)}"

    fi2 = RLForwardInputs.from_dict(d)
    assert torch.equal(fi2.action_chains, fi.action_chains)
    assert torch.equal(fi2.denoise_inds, fi.denoise_inds)
    assert torch.equal(fi2.video_latents, fi.video_latents)
    assert fi2.scalar_exec_steps == 12 and fi2.scalar_num_action_steps == 5
    assert torch.equal(fi.to("cpu").prompt_embeds, fi.prompt_embeds)


def test_rl_forward_inputs_survives_chunk_and_concat():
    # The forward_inputs dict must round-trip through the actor's batching
    # helpers (split_dict_to_chunk / concat_batch) without raising or dropping
    # scalar fields — the reason every field is a tensor.
    from rlinf.utils.nested_dict_process import concat_batch, split_dict_to_chunk

    def _make(b, step):
        return RLForwardInputs.build(
            action_chains=torch.randn(b, 2, 7, 4, 4, 1),
            denoise_inds=torch.full((b,), step),
            init_latent=torch.randn(b, 48, 1, 8, 16),
            video_latents=torch.randn(b, 48, 4, 8, 16),
            prompt_embeds=torch.randn(b, 8, 16),
            negative_prompt_embeds=None,
            frame_st_id=0,
            noise_level=1.0,
            num_action_steps=5,
            exec_steps=12,
            action_dim=7,
        ).as_dict()

    a, b = _make(2, 1), _make(2, 4)
    merged = concat_batch(a, b)  # interleaves different denoise steps
    assert merged["denoise_inds"].tolist() == [1, 1, 4, 4]
    assert merged["exec_steps"].tolist() == [12, 12, 12, 12]
    chunks = split_dict_to_chunk(merged, 4)
    assert len(chunks) == 4
    assert chunks[3]["denoise_inds"].item() == 4
    assert chunks[0]["num_action_steps"].item() == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
