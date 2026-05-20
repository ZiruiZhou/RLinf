# LingBot-VA × RLinf integration — design notes

## Goal & non-goals

**Goal (M1, achieved):** Drive `robbyant/lingbot-va-posttrain-libero-long` through RLinf's standalone embodied eval pipeline on LIBERO-Long. Reproducibility was validated at 10/10 = 100% success at the LingBot-VA repo's default LIBERO inference settings (`num_inference_steps=20`, `action_num_inference_steps=50`, 800-step episodes), consistent with the repo README's reported 98.5% on the full 500-episode protocol.

**Non-goals (next milestones, separate work):**

- RL fine-tuning of LingBot-VA via RLinf — requires a native `BasePolicy` reimplementation with autograd, replacing the VA_Server wrapper.
- Vectorized eval beyond `total_num_envs=1` — VA_Server holds per-episode KV-cache, so multiple envs would need either per-env server instances or a refactor that exposes per-env state tensors.
- Upstream PR submission to `RLinf/RLinf` — commits are structured to support this; opening the PR is a separate decision.

## Architecture

```
RLinf EmbodiedEvalRunner
        |
        v
MultiStepRolloutWorker (Ray actor)
    .predict_action_batch(env_obs)
        |
        v
LingbotvaWanActionModel  ← new wrapper, this PR
    .predict_action_batch(env_obs, mode="eval")
        |
        +-- (a) prompt change OR _elapsed_steps decrease -> server.infer({reset, prompt})
        +-- (b) otherwise & not first chunk     -> server.infer({obs=keyframes, compute_kv_cache, state=last_action})
        +-- always                              -> server.infer({obs=current, prompt})
        |
        v
wan_va.wan_va_server.VA_Server  (lingbot-va, imported via sys.path)
    .infer(...) -> dict("action": np.ndarray [7, F, H])
        |
        v
returns [num_envs=1, num_action_chunks=16, action_dim=7]
        |
        v
EnvWorker.env_evaluate_step
    .chunk_step(16 actions)
        - executes per-step in LiberoEnv
        - returns obs_list (16 obs dicts)
        - attaches _chunk_main_keyframes, _chunk_wrist_keyframes,
          _elapsed_steps to obs_list[-1] before sending back
```

The wrapper is **eval-only**. `default_forward` raises NotImplementedError (no autograd through VA_Server).

## Per-episode KV-cache lifecycle

VA_Server keeps two pieces of stateful context per episode: the T5 prompt embeddings (UMT5-XXL) and the transformer's auto-regressive KV cache. Both are populated on `infer({reset: True})` and grow chunk by chunk via `compute_kv_cache`. Carrying them across episode boundaries would corrupt subsequent rollouts.

RLinf's env worker auto-resets the LIBERO env on truncation but doesn't surface a `dones`/`is_reset` flag in `env_obs`. We solve this by passing per-env `_elapsed_steps` through the obs dict. The wrapper compares the current value to the previous and treats a decrease as an episode boundary (triggers `server.infer({reset, prompt})`).

The order of `_reset_metrics()` vs `_wrap_obs()` inside `LiberoEnv.reset()` matters: `_reset_metrics()` must run **before** `_wrap_obs()` so the obs reflects the zeroed `_elapsed_steps`. The pre-existing order had this backwards, which is why our first three eval attempts saw `Reset server` fire only once across multiple episodes despite the wrapper logic being correct.

## Per-frame keyframe threading

`VA_Server._compute_kv_cache(obs)` takes a LIST of per-frame obs dicts and runs them through the Wan2.2 VAE (3D conv with kernel size 3 in T axis), producing latents that update the KV cache. LingBot-VA's eval client passes one obs per env-step (16 per chunk for libero @ `num_action_chunks=16`). RLinf's `env_evaluate_step` originally surfaced only the final post-chunk obs to the rollout worker.

Fix: in `env_evaluate_step`, after `chunk_step`, sample the per-step obs tensors and stack into `[num_envs, T, H, W, C]` tensors (`_chunk_main_keyframes` and `_chunk_wrist_keyframes`). These are sent through the same channel as the regular obs. The wrapper unstacks them into a list of dicts and passes to `server.infer({obs=keyframes, compute_kv_cache, state=last_action})`.

Storing keyframes as **tensors** (not as a Python list of dicts) is important: `split_dict` asserts `len(value) == total_size` for list values, which would crash for a 16-element list when `total_size=num_envs=1`. Tensors get split on dim-0 like other obs.

## The `prepare_observations` allowlist bug

`rlinf/data/embodied_io_struct.py::EnvOutput.prepare_observations` was a **hardcoded allowlist** that silently dropped any obs key not in `{main_images, wrist_images, extra_view_images, states, task_descriptions}`. The above two improvements (elapsed_steps and keyframes) were therefore invisible to the rollout worker — the wrapper kept seeing only the five known keys.

This was the root-cause of `success_once == 0` for the first three v1-v3 eval attempts after the wrapper was written. Diagnosing it required a `print(sorted(env_obs.keys()))` in the wrapper. Fix: extend the allowlist with the three new underscore-prefixed keys. Other models ignore unknown obs keys, so this is a strict superset.

## The `flash_attn` stub

`lingbot-va/wan_va/modules/model.py` does an unconditional `from flash_attn import flash_attn_func` at module import, and `diffusers 0.36` independently calls `importlib.util.find_spec("flash_attn")` during its own import. Building flash-attn from source is slow and brittle (the LingBot-VA repo's `requirements.txt` lists it but the user couldn't get it to compile).

We use `attn_mode="torch"` everywhere, so the function is never actually called. The wrapper's `_stub_flash_attn` installs a minimal `flash_attn` module in `sys.modules` with:

- a proper `importlib.machinery.ModuleSpec` (so `find_spec` returns non-None);
- a `__version__` attribute;
- a `flash_attn_func` that raises `RuntimeError` if invoked (so misconfiguration fails loud).

This must run before `from wan_va_server import VA_Server`, since loading that module triggers both the diffusers probe and the lingbot-va import. The wrapper's `_import_wan_va()` orders them correctly.

## Container PID cap

Our host container caps PIDs at `/sys/fs/cgroup/pids.max=3840`. Ray's default raylet spawns 15 prestart Python workers with many gRPC threads each, plus dashboard subprocesses, and the cumulative pthread count exhausts the budget. Workers then crash with `pthread_create failed: Resource temporarily unavailable`.

Pre-starting Ray with `ray start --head --num-cpus=4 --include-dashboard=false` keeps the prestart count down (4) and skips dashboard. The eval driver attaches to this lean cluster instead of letting Ray auto-init. This is operational guidance, not a code change — documented in `REPRODUCE.md`.

## Bugs found and fixed (chronological)

1. **`flash_attn` import on module load** — diffusers' `find_spec` raised `ValueError: __spec__ is None` with a bare `types.ModuleType` stub. Fixed by attaching a proper `ModuleSpec`.
2. **`Could not override 'rollout.model.enable_offload'`** — Hydra struct mode rejects new keys via override. The rollout model config is built from `cfg.actor.model` inside `huggingface_worker.init_worker`, so `actor.model.enable_offload=False` alone is sufficient.
3. **T5 prompt encoding on CPU hangs the eval** — `enable_offload=True` (lingbot-va default) puts UMT5 on CPU; encoding a 512-token prompt takes 5+ minutes. Fixed by setting `enable_offload=False` for the wrapper (we have 80 GB VRAM available).
4. **`VAE conv3d: Calculated padded input size per channel (2 x 16 x 16). Kernel size (3 x 1 x 1) can't be greater than actual input size`** — original wrapper passed `[obs]` (length 1) to `compute_kv_cache`. VAE temporal kernel needs ≥3 frames. Fixed first by replicating the obs 4x; later replaced with the real per-step keyframes after the env_worker change.
5. **`state=action` type confusion** — VA_Server's `preprocess_action` does `torch.from_numpy(action)` internally and expects shape `[C=7, F, H]`. An earlier draft wrapped in `torch.from_numpy(...).unsqueeze(0)` and broke the shape unpack. Fixed by passing the raw numpy.
6. **`bddl` missing** — LIBERO's `setup.py` has `install_requires=[]` so `bddl` isn't pulled. Documented in REPRODUCE.md as an explicit install step.
7. **`_reset_metrics()` after `_wrap_obs()`** — `LiberoEnv.reset()` zeroed `_elapsed_steps` after wrapping the obs, so the fresh-reset obs carried the pre-reset value. The wrapper's "elapsed_steps decreased" trigger never fired. Fixed by swapping the two lines.
8. **`prepare_observations` allowlist silently dropped new keys** — root-cause of `success_once == 0` in v1-v3. See above section.
9. **Subsampled 4 keyframes vs lingbot-va's 16** — initial keyframe implementation subsampled every 4th step, producing 4 keyframes per chunk. LingBot-VA's eval client passes all 16. Different number → different VAE latent count → different KV cache content → degraded model behavior. Fixed by sending all per-step obs.

## Module identifier — `lingbotva_wan` vs `lingbotvla`

RLinf already ships a model named `lingbotvla` at `rlinf/models/embodiment/lingbotvla/`. It is a Pi0/Qwen2.5-VL-3B architecture used for RoboTwin (configs: `robotwin_*_lingbotvla*.yaml`), imports from a different Python package called `lingbotvla`, and is **not the same model** as `robbyant/lingbot-va`. We chose the identifier `lingbotva_wan` (the "wan" hints at the Wan2.2 video backbone) to disambiguate. The existing `lingbotvla` integration is left untouched.

## Followups (not in this PR)

- **RL training path**: replace the wrapper with a native `BasePolicy` that owns the Wan2.2 transformer and exposes `default_forward` returning `(logprobs, values, entropy)`. The current wrapper's lack of autograd makes any RL run impossible.
- **Vectorized eval (`total_num_envs > 1`)**: maintain a `dict[env_id, VA_Server]` or refactor VA_Server state to per-env tensors. Today we serialize.
- **Subsequent-task scaling**: gripper sign verification on tasks beyond the 10 we tested; we pass the raw output through (no `prepare_actions_for_libero` branch for `lingbotva_wan`) which appears correct, but spot-checking on more tasks would harden confidence.
- **First-chunk frame-0 skip**: LingBot-VA's eval client starts execution at frame 1 of the first chunk (the model's frame-0 actions are designed as model-warming artifacts). We execute all 16. Didn't hurt the 10/10 result, but matching the client exactly might squeeze out the last 1.5%.
- **Caching T5 prompt embeds by task language**: VA_Server re-encodes the prompt on every `_reset`. With only 10 unique prompts on LIBERO-Long, a per-prompt cache saves ~50 T5 encodes in a 500-episode eval.
