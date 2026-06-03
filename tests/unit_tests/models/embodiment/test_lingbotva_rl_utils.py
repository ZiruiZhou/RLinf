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

"""Unit tests for the LingBot-VA RL (flow-SDE) math.

These cover the checkpoint-independent core of the planned GRPO support:
the diagonal-Gaussian log-prob/entropy and the rectified-flow ODE->SDE
transition kernel. They require only ``torch`` (no ``wan_va``, no weights),
so they run in CI without the external LingBot-VA repo or a GPU.

Run with::

    /venv/main/bin/python3.10 -m pytest \
        tests/unit_tests/models/embodiment/test_lingbotva_rl_utils.py -q
"""

from __future__ import annotations

import math

import pytest
import torch

from rlinf.models.embodiment.lingbotva.rl_utils import (
    flow_ode_step,
    flow_sde_step,
    gaussian_entropy,
    gaussian_logprob,
    sde_sigmas_from_timesteps,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# gaussian_logprob
# ---------------------------------------------------------------------------
def test_gaussian_logprob_matches_torch_distribution():
    mean = torch.randn(4, 7)
    std = torch.rand(4, 7) + 0.1  # strictly positive
    sample = mean + torch.randn_like(mean) * std

    ours = gaussian_logprob(sample, mean, std)
    ref = torch.distributions.Normal(mean, std).log_prob(sample)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_gaussian_logprob_maximised_at_mean():
    mean = torch.randn(3, 5)
    std = torch.rand(3, 5) + 0.2
    at_mean = gaussian_logprob(mean, mean, std)
    off_mean = gaussian_logprob(mean + 0.3, mean, std)
    assert torch.all(at_mean >= off_mean)
    # peak density equals -log(std) - 0.5*log(2*pi)
    expected_peak = -torch.log(std) - 0.5 * math.log(2 * math.pi)
    assert torch.allclose(at_mean, expected_peak, atol=1e-6)


def test_gaussian_logprob_zero_std_is_deterministic_zero():
    mean = torch.randn(2, 4)
    std = torch.zeros(2, 4)
    # zero-std coordinates contribute 0 (not -inf / nan), regardless of sample
    lp = gaussian_logprob(mean + 5.0, mean, std)
    assert torch.all(lp == 0.0)
    assert torch.isfinite(lp).all()


def test_gaussian_logprob_safe_mode():
    mean = torch.randn(2, 3)
    std = torch.full_like(mean, 1e-9)
    sample = mean + 0.5
    lp = gaussian_logprob(sample, mean, std, safe=True)
    assert torch.allclose(lp, -(sample - mean) ** 2, atol=1e-7)
    assert torch.isfinite(lp).all()


# ---------------------------------------------------------------------------
# gaussian_entropy
# ---------------------------------------------------------------------------
def test_gaussian_entropy_matches_closed_form():
    std = torch.rand(5) + 0.05
    ours = gaussian_entropy(std)
    ref = torch.distributions.Normal(torch.zeros_like(std), std).entropy()
    assert torch.allclose(ours, ref, atol=1e-6)


def test_gaussian_entropy_monotonic_in_std():
    std = torch.tensor([0.1, 0.5, 1.0, 2.0])
    ent = gaussian_entropy(std)
    assert torch.all(ent[1:] > ent[:-1])


def test_gaussian_entropy_zero_std_is_zero():
    std = torch.zeros(3)
    assert torch.all(gaussian_entropy(std) == 0.0)


# ---------------------------------------------------------------------------
# flow ODE / SDE transition kernel
# ---------------------------------------------------------------------------
def _rand_step(shape=(2, 7, 4)):
    x_t = torch.randn(*shape)
    v_t = torch.randn(*shape)
    t_input = torch.full(shape, 0.6)
    delta = torch.full(shape, 0.1)
    return x_t, v_t, t_input, delta


def test_sde_reduces_to_ode_when_noise_vanishes():
    # With sigma_i -> 0 the SDE mean must equal the deterministic ODE mean
    # and the std must be ~0. This anchors the SDE to the eval-time sampler.
    x_t, v_t, t_input, delta = _rand_step()
    ode_mean, ode_std = flow_ode_step(x_t, v_t, t_input, delta)
    sde_mean, sde_std = flow_sde_step(
        x_t, v_t, t_input, delta, sigma_i=torch.zeros_like(x_t)
    )
    assert torch.allclose(sde_mean, ode_mean, atol=1e-6)
    assert torch.all(ode_std == 0)
    assert torch.allclose(sde_std, torch.zeros_like(sde_std), atol=1e-7)


def test_sde_std_is_sqrt_delta_times_sigma():
    x_t, v_t, t_input, delta = _rand_step()
    sigma_i = torch.rand_like(x_t) * 0.5 + 0.1
    _, std = flow_sde_step(x_t, v_t, t_input, delta, sigma_i)
    assert torch.allclose(std, torch.sqrt(delta) * sigma_i, atol=1e-6)
    assert torch.all(std > 0)


def test_sde_mean_shifts_below_ode_by_extra_x1_term():
    # The SDE x1_weight subtracts sigma^2 * delta / (2 * t) relative to ODE,
    # so (sde_mean - ode_mean) == -(that term) * x1_pred. Verify exactly.
    x_t, v_t, t_input, delta = _rand_step()
    sigma_i = torch.rand_like(x_t) * 0.3 + 0.1
    ode_mean, _ = flow_ode_step(x_t, v_t, t_input, delta)
    sde_mean, _ = flow_sde_step(x_t, v_t, t_input, delta, sigma_i)
    x1_pred = x_t + v_t * (1 - t_input)
    extra = sigma_i**2 * delta / (2 * t_input)
    assert torch.allclose(sde_mean - ode_mean, -extra * x1_pred, atol=1e-5)


def test_sample_then_logprob_is_finite_and_differentiable():
    # End-to-end: sample a step, score it, backprop into the velocity.
    x_t, _, t_input, delta = _rand_step()
    v_t = torch.randn_like(x_t).requires_grad_(True)
    sigma_i = torch.full_like(x_t, 0.3)
    mean, std = flow_sde_step(x_t, v_t, t_input, delta, sigma_i)
    eps = torch.randn_like(mean)
    sample = (mean + eps * std).detach()  # behaviour sample (no grad)
    logp = gaussian_logprob(sample, mean, std)
    assert torch.isfinite(logp).all()
    logp.sum().backward()
    assert v_t.grad is not None and torch.isfinite(v_t.grad).all()


# ---------------------------------------------------------------------------
# sde_sigmas_from_timesteps
# ---------------------------------------------------------------------------
def test_sde_sigmas_shape_and_positivity():
    # num_steps + 1 timesteps (descending, trailing 0) -> num_steps sigmas
    timesteps = torch.tensor([0.9, 0.6, 0.3, 0.0])
    sigmas = sde_sigmas_from_timesteps(timesteps, noise_level=1.0)
    assert sigmas.shape == (3,)
    assert torch.all(sigmas >= 0)


def test_sde_sigmas_scale_linearly_with_noise_level():
    timesteps = torch.tensor([0.9, 0.6, 0.3, 0.0])
    s1 = sde_sigmas_from_timesteps(timesteps, noise_level=1.0)
    s2 = sde_sigmas_from_timesteps(timesteps, noise_level=2.0)
    assert torch.allclose(s2, 2.0 * s1, atol=1e-6)


def test_sde_sigmas_handle_timestep_equal_one():
    # timesteps == 1 would divide by zero in 1 - t; the guard replaces the
    # denominator with timesteps[1]. Result must stay finite.
    timesteps = torch.tensor([1.0, 0.5, 0.0])
    sigmas = sde_sigmas_from_timesteps(timesteps, noise_level=1.0)
    assert torch.isfinite(sigmas).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
