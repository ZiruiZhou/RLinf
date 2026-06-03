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

"""Gate #1 for LingBot-VA GRPO: rollout vs. recompute log-prob consistency.

This is the single make-or-break test for the RL draft. It runs ONE model
instance (so the weights are identical), samples an action chunk with the
stochastic SDE rollout (`_rl_predict_action_batch` -> `prev_logprobs` +
`forward_inputs`), then recomputes the log-prob of that exact sampled step
(`get_log_prob_value`). If the draft's conditioning replay is faithful, the two
log-probs must match — i.e. the initial PPO/GRPO ratio is ~1.0. A mismatch
means the recompute does not reproduce the rollout's denoising step (usually a
KV-cache / sigma-convention / velocity-sign bug); see RL_DESIGN.md.

It uses SYNTHETIC observations (random images/state) — gate #1 tests
rollout/recompute self-consistency, not task success, so the simulator is not
needed. Only the LingBot-VA model (transformer + VAE + text encoder) is built.

Run on a box with a GPU, the lingbot-va repo, the base model, the SFT
checkpoint, and the rlinf env installed::

    export LINGBOT_VA_REPO_PATH=/path/to/lingbot-va
    export LINGBOT_VA_MODEL_PATH=/path/to/lingbot-va-base
    export LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH=/path/to/checkpoint_step_3000_extra
    export EMBODIED_PATH=$PWD/examples/embodiment
    python examples/embodiment/check_lingbotva_rl_gate1.py \
        actor.model.lingbotva.attn_mode=torch

The checkpoint path may be either a directory with
``transformer/diffusion_pytorch_model.safetensors`` or a single
``full_weights.pt`` (``_utils.load_transformer_state_dict`` handles both).
"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from rlinf.models import get_model


def _synthetic_env_obs(batch_size: int, height: int, width: int, device: str) -> dict:
    """Fabricate a LiberoEnv-style observation batch (random pixels/state)."""
    return {
        "states": torch.randn(batch_size, 8, device=device),
        "main_images": (torch.rand(batch_size, height, width, 3) * 255).to(
            torch.uint8
        ),
        "wrist_images": (torch.rand(batch_size, height, width, 3) * 255).to(
            torch.uint8
        ),
        "task_descriptions": ["pick up the object and place it"] * batch_size,
    }


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_object_grpo_lingbotva",
)
def main(cfg: DictConfig) -> None:
    if str(cfg.actor.model.model_type) != "lingbotva":
        raise ValueError("Gate #1 expects actor.model.model_type=lingbotva.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # One instance plays both roles: rollout sampling and the actor recompute.
    # rl_mode builds the backend and exposes the transformer.
    model = get_model(cfg.actor.model).to(device)
    model.eval()

    batch_size = 1  # matches the RL config's micro_batch_size (homogeneous step)
    env_obs = _synthetic_env_obs(batch_size, height=256, width=256, device=device)

    # --- rollout: stochastic SDE sample + behaviour log-prob ----------------
    with torch.no_grad():
        _, result = model.predict_action_batch(env_obs, mode="train")
    prev_logprobs = result["prev_logprobs"].to(torch.float64).cpu()  # [B, exec, adim]
    forward_inputs = result["forward_inputs"]

    # --- recompute: log-prob of the SAME stored step ------------------------
    with torch.no_grad():
        out = model.get_log_prob_value(
            forward_inputs=forward_inputs,
            compute_logprobs=True,
            compute_entropy=True,
        )
    new_logprobs = out["logprobs"].to(torch.float64).cpu()

    # --- compare ------------------------------------------------------------
    abs_diff = (prev_logprobs - new_logprobs).abs()
    ratio = (new_logprobs - prev_logprobs).exp()  # the initial GRPO ratio
    max_abs = float(abs_diff.max())
    print("=" * 64)
    print("LingBot-VA RL gate #1 — rollout vs. recompute log-prob")
    print(f"  prev_logprobs  (sample): {prev_logprobs.flatten()[:4].tolist()}")
    print(f"  new_logprobs   (recomp): {new_logprobs.flatten()[:4].tolist()}")
    print(f"  max |Δ logprob|        : {max_abs:.3e}")
    print(
        f"  ratio exp(Δ)  min/mean/max: "
        f"{float(ratio.min()):.4f} / {float(ratio.mean()):.4f} / "
        f"{float(ratio.max()):.4f}  (want ~1.0)"
    )

    # bf16 forwards make exact equality unrealistic; require the ratio close to
    # 1 (a few percent). Tighten once attn_mode/precision are settled.
    tol = float(cfg.actor.model.lingbotva.get("gate1_ratio_tol", 0.05))
    ok = bool((ratio - 1.0).abs().max() < tol)
    print(f"  PASS (|ratio-1| < {tol}): {ok}")
    print("=" * 64)
    if not ok:
        raise SystemExit(
            "GATE #1 FAILED: recompute does not reproduce the rollout log-prob. "
            "Check the # VALIDATE: markers in native_backend.py (cache replay, "
            "sigma convention, velocity sign) and RL_DESIGN.md."
        )


if __name__ == "__main__":
    main()
