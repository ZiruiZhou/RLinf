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

"""Stochastic-sampler (SDE) math for LingBot-VA reinforcement learning.

LingBot-VA produces an action chunk through a deterministic flow-matching
(rectified-flow) ODE: each denoising step is ``x_{t+1} = mean(x_t, v_t)`` with
zero variance. That gives no tractable per-action log-probability, so RL is
impossible on the ODE as-is.

Following the approach already used in RLinf for OpenPI (pi0/pi05) and the
``lingbotvla`` model, we turn the **action** denoising ODE into a stochastic
SDE: every step becomes a diagonal-Gaussian transition
``x_{t+1} ~ N(mean_t, std_t**2)`` whose closed-form log-density is the action
log-probability used by GRPO/PPO. The video-latent denoising stays
deterministic and is treated as world-context (not part of the policy).

The functions here are deliberately **pure tensor math with no dependency on
the external ``wan_va`` package or on any model weights**, so they can be unit
tested in isolation. They are faithful re-implementations of the corresponding
methods in ``rlinf/models/embodiment/openpi/openpi_action_model.py``
(``sample_mean_var_val`` / ``get_logprob_norm`` / ``gaussian_entropy``), the
canonical flow-matching RL implementation in this repo.

Symbols
-------
``v_t``      velocity prediction from the transformer at the current latent.
``t_input``  the (broadcast) flow timestep of the current step, in ``[0, 1]``.
``delta``    ``t_input - t_next``, the step size toward clean data.
``sigma_i``  the per-step SDE noise scale derived from the timestep schedule.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "gaussian_logprob",
    "gaussian_entropy",
    "flow_sde_step",
    "flow_ode_step",
    "sde_sigmas_from_timesteps",
]


def gaussian_logprob(
    sample: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    safe: bool = False,
) -> torch.Tensor:
    """Element-wise diagonal-Gaussian log-density ``log N(sample | mean, std**2)``.

    Mirrors ``OpenPIActionModel.get_logprob_norm``. A ``std`` entry equal to
    exactly ``0`` denotes a deterministic coordinate (e.g. masked / ODE) and
    contributes ``0`` to the log-probability rather than ``-inf``.

    Args:
        sample: the realised next latent ``x_{t+1}``.
        mean: the predicted transition mean ``mean_t``.
        std: the per-coordinate transition std ``std_t`` (``>= 0``).
        safe: if True use the numerically forgiving surrogate
            ``-(sample - mean)**2`` (matches OpenPI's ``safe_get_logprob``),
            which keeps the PPO ratio well-defined when ``std`` is tiny.

    Returns:
        Tensor of the same shape as ``sample`` holding the per-coordinate
        log-density. Callers reduce it (sum/mean over action dims) according
        to ``algorithm.logprob_type``.
    """
    if safe:
        return -torch.pow(sample - mean, 2)
    zero_std = std == 0
    std_safe = torch.where(zero_std, torch.ones_like(std), std)
    constant_term = -torch.log(std_safe) - 0.5 * math.log(2 * math.pi)
    exponent_term = -0.5 * torch.pow((sample - mean) / std_safe, 2)
    log_prob = constant_term + exponent_term
    return torch.where(zero_std, torch.zeros_like(log_prob), log_prob)


def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    """Element-wise differential entropy of ``N(., std**2)``.

    Mirrors ``OpenPIActionModel.gaussian_entropy``:
    ``0.5 * log(2 * pi * e * std**2)``. Coordinates with ``std == 0`` are
    treated as deterministic and contribute ``0``.
    """
    zero_std = std == 0
    std_safe = torch.where(zero_std, torch.ones_like(std), std)
    entropy = 0.5 * torch.log(2 * math.pi * math.e * std_safe.pow(2))
    return torch.where(zero_std, torch.zeros_like(entropy), entropy)


def sde_sigmas_from_timesteps(
    timesteps: torch.Tensor,
    noise_level: torch.Tensor | float,
) -> torch.Tensor:
    """Per-step SDE noise scales ``sigma_i`` from a rectified-flow schedule.

    Faithful to the ``flow_sde`` branch of
    ``OpenPIActionModel.sample_mean_var_val``::

        denom = where(timesteps == 1, timesteps[1], timesteps)
        sigma_ratio = timesteps / (1 - denom)
        sigmas = noise_level * sqrt(sigma_ratio)[:-1]

    Args:
        timesteps: 1-D tensor of flow timesteps in ``[0, 1]`` of length
            ``num_steps + 1`` (descending toward 0), as produced by the
            action scheduler padded with a trailing 0.
        noise_level: scalar (or broadcastable) SDE temperature.

    Returns:
        1-D tensor of length ``num_steps`` (one ``sigma`` per usable step;
        the final clean step is dropped, matching ``[:-1]``).
    """
    if not torch.is_tensor(noise_level):
        noise_level = torch.as_tensor(
            noise_level, dtype=timesteps.dtype, device=timesteps.device
        )
    denom_timesteps = torch.where(timesteps == 1, timesteps[1], timesteps)
    sigma_ratio = timesteps / (1 - denom_timesteps)
    return noise_level * torch.sqrt(sigma_ratio)[:-1]


def _x0_x1_predictions(
    x_t: torch.Tensor, v_t: torch.Tensor, t_input: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean-data (``x0``) and noise (``x1``) endpoints from a velocity.

    Rectified flow defines ``x_t = (1 - t) * x0 + t * x1`` with constant
    velocity ``v = x1 - x0``, so ``x0 = x_t - v * t`` and
    ``x1 = x_t + v * (1 - t)``. Matches OpenPI's ``x0_pred`` / ``x1_pred``.
    """
    x0_pred = x_t - v_t * t_input
    x1_pred = x_t + v_t * (1 - t_input)
    return x0_pred, x1_pred


def flow_ode_step(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    t_input: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic flow-matching step (``flow_ode``); std is all zeros.

    Provided for parity / eval and as the ``noise_level -> 0`` reference that
    :func:`flow_sde_step` must reduce to. Returns ``(mean, std)``.
    """
    x0_pred, x1_pred = _x0_x1_predictions(x_t, v_t, t_input)
    x0_weight = 1 - (t_input - delta)
    x1_weight = t_input - delta
    mean = x0_pred * x0_weight + x1_pred * x1_weight
    std = torch.zeros_like(mean)
    return mean, std


def flow_sde_step(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    t_input: torch.Tensor,
    delta: torch.Tensor,
    sigma_i: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stochastic flow-matching step (``flow_sde``): the RL transition kernel.

    Faithful to the ``flow_sde`` branch of
    ``OpenPIActionModel.sample_mean_var_val``::

        x0_weight = 1 - (t_input - delta)
        x1_weight = (t_input - delta) - sigma_i**2 * delta / (2 * t_input)
        x_t_mean  = x0_pred * x0_weight + x1_pred * x1_weight
        x_t_std   = sqrt(delta) * sigma_i

    The sampled next latent is ``mean + eps * std`` with ``eps ~ N(0, I)``;
    its log-prob (for the PPO/GRPO ratio) is :func:`gaussian_logprob`.

    All tensor args must broadcast against ``x_t``. Returns ``(mean, std)``.
    """
    x0_pred, x1_pred = _x0_x1_predictions(x_t, v_t, t_input)
    x0_weight = torch.ones_like(t_input) - (t_input - delta)
    x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
    mean = x0_pred * x0_weight + x1_pred * x1_weight
    std = torch.sqrt(delta) * sigma_i
    return mean, std
