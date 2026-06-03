# LingBot-VA Reinforcement Learning (GRPO) — Design & Status

This document tracks the design for adding RL (GRPO) to the LingBot-VA model
(`model_type: lingbotva`) on top of PR #1220's SFT + eval. It is the handoff
spec for the phases that need the external `lingbot-va` repo and an SFT-trained
checkpoint (~30–50% SR on Libero-Object).

## 1. Core idea

LingBot-VA produces an action chunk via two diffusion loops (in
`eval_adapter/native_backend.py::_infer_batch_impl`): a **video-latent** loop
then an **action-latent** loop. For RL:

- **Video-latent diffusion stays deterministic** — treated as world-context /
  feature extraction, not part of the policy distribution.
- **The action-denoising chain is made stochastic (SDE, `flow_sde`)**: each
  denoising step becomes a diagonal Gaussian `x_{t+1} ~ N(mean_t, std_t²)`
  whose closed-form log-density is the action log-probability. This is the same
  trick RLinf already uses for OpenPI (pi0/pi05) and the `lingbotvla` model.

**Algorithm: GRPO** — `adv_type: grpo`, `loss_type: actor`, critic-free. The
advantage is the group-normalized episode return (binary Libero-Object
success). No value head on the 5B transformer. (PPO + value head is a possible
later extension.)

Template files in-repo:
- `rlinf/models/embodiment/openpi/openpi_action_model.py`
  (`sample_mean_var_val`, `get_logprob_norm`, `gaussian_entropy`,
  `get_log_prob_value`) — canonical flow-matching RL.
- `rlinf/models/embodiment/lingbotvla/lingbotvla_action_model.py`
  (`sample_actions` ↔ `get_log_prob_value`, `forward_inputs` contract).

## 2. The RL model interface (what the workers require)

A policy must implement (see `rlinf/models/embodiment/base_policy.py`):

- `predict_action_batch(env_obs, mode, **sampling) -> (actions, result)` —
  rollout. `result` must contain:
  - `prev_logprobs`: behaviour log-prob of the sampled action chain step(s).
  - `prev_values`: `zeros` under GRPO.
  - `forward_inputs`: everything `default_forward` needs to reproduce the step.
- `default_forward(forward_inputs, compute_logprobs, compute_entropy,
  compute_values) -> {logprobs, entropy, values}` — gradient-bearing recompute
  for the GRPO ratio.

## 3. Status

| Piece | File | Status |
|---|---|---|
| SDE/Gaussian math (mean/std, log-prob, entropy, sigmas) | `rl_utils.py` | **Done + unit-tested** (`test_lingbotva_rl_utils.py`) |
| RL orchestration (denoise-index, sigma schedule, SDE wiring, logprob reduce/broadcast, `RLForwardInputs`) | `rl_engine.py` | **Done + unit-tested** (`test_lingbotva_rl_engine.py`) |
| `rl_mode` plumbing + forward dispatch | `lingbotva_action_model.py` | **Done** (gated, no behavior change to SFT/eval) |
| `get_log_prob_value` (recompute) | `lingbotva_action_model.py` → backend | **Draft, unvalidated** |
| `_rl_predict_action_batch` (rollout) | `lingbotva_action_model.py` → backend | **Draft, unvalidated** |
| `infer_batch_with_logprob` / `recompute_logprob` + helpers | `eval_adapter/native_backend.py` | **Draft, unvalidated** (`# VALIDATE:` markers) |
| GRPO config | `examples/embodiment/config/libero_object_grpo_lingbotva.yaml` | **Draft** |
| Stateful rollout worker | `rlinf/workers/rollout/hf/lingbotva_rollout_worker.py` | **Stub** (Phase 2) |
| `model_type` worker branches | rollout/actor workers | **TODO** (Phase 1 wiring) |

### How recompute stays consistent (the key draft decision)
Only the final video forward (`update_cache=1`) writes the KV cache the action
steps read; intermediate action steps are read-only (`update_cache=0`). So the
rollout stores the **final video latents** and `recompute_logprob` re-commits
just that one cache entry deterministically, then re-runs the single scored
action step with gradients. No need to replay the whole video diffusion, and
the conditioning is bit-identical → rollout and recompute log-probs match
(gate #1). Video conditioning is run under `no_grad` (treated as fixed, like
OpenPI/lingbotvla detaching the prefix); gradients flow only through the scored
action step.

## 4. Two hard problems (need the repo + checkpoint)

### A. Recompute conditioning (correctness gate #1)
The action transformer is conditioned on the video-latent KV cache built by the
expensive video loop — unlike `lingbotvla`'s cheap VLM prefix. `default_forward`
must reconstruct that conditioning to recompute the one stored denoise step.
- *Preferred:* store post-video conditioning (init/KV state, prompt embeds,
  grid ids, frame_st_id, action_mask) in `forward_inputs` and replay only the
  single action-denoise forward with gradients.
- *Fallback:* re-run VAE-encode + deterministic video diffusion from stored obs
  (correct but Nx compute).
Resolve against the real `wan_va` transformer. **Gate:** at the behaviour
weights, `default_forward` log-prob must equal rollout `prev_logprobs`
(PPO ratio ≈ 1).

### B. Stateful rollout (the central engineering problem)
RL goes through `MultiStepRolloutWorker`, which is **stateless across chunks**
and drops per-step raw obs — the same gap that forced PR #1220's dedicated
`eval_lingbotva.py`. LingBot-VA needs KV-cache replay for non-zero SR. Plan: a
`LingbotVARolloutWorker(MultiStepRolloutWorker)` that keeps per-env
`LingbotVAEpisodeState`, records chunk observations, and resets on done —
generalizing `eval_lingbotva.py` onto the channel-based RL path. Selected by
`model_type` in `examples/embodiment/train_embodied_agent.py`.
**Gate:** RL rollout SR ≈ eval SR (~30–50%) before any update.

## 5. Wiring checklist (Phase 1)

1. **Done** — Rollout worker `huggingface_worker.py::predict`: `LINGBOTVA`
   added to the `{"mode": mode}` branch so `predict_action_batch` gets
   `mode="train"` (→ `_rl_predict_action_batch`). No `return_obs` needed.
2. **Done (no change needed)** — Actor worker `fsdp_actor_worker.py`: the
   generic path already calls
   `self.model(forward_inputs=..., compute_logprobs=True, ...)` and reads
   `output_dict["logprobs"/"values"/"entropy"]`. LingBot-VA needs no
   temperature/top_k (OpenVLA) or prev_logprobs-readback (GR00T) branch, and
   GRPO sets `compute_values=False`.
3. **Done** — `train_embodied_agent.py`: selects `LingbotVARolloutWorker`
   when `cfg.rollout.model.model_type == "lingbotva"`.
4. **Done** — `forward_inputs` is a flat **tensor-only** dict
   (`RLForwardInputs`) so it survives `split_dict_to_chunk`/`concat_batch`;
   `denoise_inds` is per-sample; log-prob is `action_level`
   (`[B, exec_steps, action_dim]`).
5. **TODO (validate)** — Weight sync: confirm `state_dict`/`load_state_dict`
   over the transformer is uniform across the actor (FSDP child) and rollout
   (backend) copies, and that fsdp2 `fully_shard` keeps
   `self.transformer is backend._server.transformer`.
6. **TODO** — `LingbotVARolloutWorker.generate_one_epoch` (stateful KV-replay);
   currently a sentinel that raises (Phase 2).

## 6. Phased roadmap

- **Phase 0 (done):** math + tests, config, skeletons, this doc. ✅
- **Phase 1 (drafted; needs `lingbot-va` repo to validate):**
  `_rl_predict_action_batch` + `get_log_prob_value` + backend
  `infer_batch_with_logprob`/`recompute_logprob` are written. Remaining: run
  against the real `wan_va`, clear the `# VALIDATE:` markers (FSDP transformer
  sharing, scheduler sigma convention, velocity sign, logprob shape vs. the
  actor worker), add the `model_type` worker branches, and pass gate #1.
- **Phase 2 (needs checkpoint):** `LingbotVARolloutWorker`; tiny 1-GPU GRPO on
  a few tasks; pass gates #2–3.
- **Phase 3:** scale to 8×A100, full Libero-Object, tune
  (`temperature_train`, `group_size`, `reward_coef`, entropy/KL); measure SR
  lift; add e2e CI config + EN/ZH docs.
