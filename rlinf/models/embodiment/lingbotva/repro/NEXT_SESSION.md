# NEXT_SESSION — handoff notes (LingBot-VA GRPO)

Notes-to-self for resuming this effort. Read `REPRODUCE.md` for the runbook,
`REFERENCE_RESULTS.md` for the numbers, `../RL_RESULTS.md` for the full story,
`../RL_DESIGN.md` for the RL core design.

## Where it stands (as of this archive)
- **Goal met:** GRPO RL with the (near-)exact gradient raises Libero-Object SR
  over SFT on the focused tasks {2,6}: SFT 40% → **step 30 70.0% (+30.0%,
  z=2.45)**, step 20 also significant (66.7%, z=2.15). Rising noisy envelope.
- **Branch:** `feat/lingbotva-rl-grpo` (off `pr-1220`), remote `RLinf/RLinf`.
- **Everything code-side is committed.** The result is reproducible from the
  scripts in `./scripts/`. The trained checkpoints were **not** backed up (per
  decision) — they lived in `/tmp/grpo_exact_train_logs/...` and are gone on
  teardown. Retrain from `base_sft` via `scripts/train_exact.sh` to regenerate.

## Commit map (the implementation, in order)
| commit | what |
| --- | --- |
| `a6083e6` | SFT + eval for LingBot-VA (PR #1220 baseline this builds on) |
| `779c17d` | GRPO RL core (SDE sampling, Gaussian log-prob, recompute) |
| `7ab19c1` | distributed KV-cache replay for the rollout |
| `0ce5e03` | **mean-reduce** log-prob (over active coords) — ratio stability |
| `0c5ce2c` | recompute log-prob **under KV-replay** + batched (grouped) recompute |
| `d9763a5` | `recompute_kv_replay` flag + `reshard_after_forward=False` |
| `5bf2ee1` | eval one-episode-per-env counting fix |
| `8b5a774` | `eval_reset_start_idx` knob (disjoint init shards) |
| `41ed6b1` | build `loss_mask` when `ignore_terminations` (the exact-gradient enabler) |
| `48375e3…7e74a1a` | the experimental writeup (`../RL_RESULTS.md`) |

## Gotchas that cost real time — don't re-derive
1. **Checkpoint size = ~38 GB, not 9.5 GB.** `actor/dcp_checkpoint` (29 GB,
   resume-state) + `actor/model_state_dict/full_weights.pt` (9.5 GB, eval). Two
   runs filled the 160 GB disk and crashed mid-save. Trim `dcp_checkpoint` after
   each save if you don't need mid-run resume.
2. **Config divisibility:** `total_num_envs // env_world_size(4) // pipeline(1)`
   must be `% group_size(8) == 0` → `total_num_envs` must be a **multiple of 32**
   (48 is invalid; the exact run uses 32).
3. **FSDP ragged-buffer deadlock** is the central distributed hazard. The exact
   recompute replays a *data-dependent* number of FSDP forwards (history depth
   grows with chunk index). If episodes terminate early the buffer is ragged →
   per-rank forward counts diverge → mismatched all-gathers → NCCL spin-wait
   deadlock (actor GPUs 100% util, frozen memory, no progress). **Workaround in
   use:** `ignore_terminations=true` → every env runs full 240 steps →
   non-ragged `[chunks × envs]` grid → rank-symmetric by construction. This is
   why `train_exact.sh` sets it.
4. **Must run disaggregated** (collocated weight sync → `pidfd_getfd` EPERM) and
   **must keep `expandable_segments`** only in disaggregated mode. See
   `ENV_PREP.md` "container constraints".
5. **Eval is non-deterministic and OOM-prone.** ≥30 episodes, `nenvs ≤ 3`/proc,
   `eval_rollout_epoch=1` (multi-episode reuse is broken — only the first episode
   per env slot counts). The `success_once` training metric (noise=1.0) is
   unreliable; trust only deterministic checkpoint eval.

## Open items (in priority order)
1. **Firm up the peak.** Re-eval steps 20 and 30 at n=60–90 (more offsets /
   tasks split across GPUs) to shrink the ±17 pt CIs and nail the significance.
   Cheap, high-value.
2. **Scale the exact gradient.** Carry `recompute_kv_replay=true` +
   `ignore_terminations=true` to 64 envs / all 10 tasks (`train_proper.sh`
   currently uses the *biased* gradient for distributed safety). Need to confirm
   rank-symmetry holds at that scale and that ignore_terminations' rollout-speed
   cost is acceptable.
3. **True ratio = 1.0 (rank-symmetric padded replay).** The current 0.91 (bf16
   drift over deep history) works via the `ignore_terminations` shortcut. A true
   1.0 — and removal of the rollout-speed cost — needs: `all_gather` the
   chunk-key union across actor DP ranks; every rank iterates the **same** global
   sorted keys; for keys a rank lacks, run a padding `_recompute_group` on a
   synthetic 1-row batch with matching `frame_st_id` + fabricated zero-history;
   route the padding action log-prob through a **0-weight loss sink** so the
   *backward* collectives also match (the hard trap — replay is no_grad so only
   forward syncs, but the scored action forward's backward must match too).
   Code anchors: `native_backend.py::recompute_logprob` (groups rows by
   `key=(frame_st_id<<20)+denoise_ind`, `torch.unique` → sorted, one batched
   `_recompute_group` per key; replay gated at the `recompute_kv_replay` check);
   actor DP group = `torch.distributed.group.WORLD`
   (`fsdp_actor_worker.py`, FSDP full_shard, no TP). `denoise_ind` is
   **chunk-shared** (`select_denoise_index` called once/chunk, broadcast via
   `torch.full`), so with an even env split every rank already holds all ~20
   keys in the same order — the desync is a micro-batch *ordering* problem, not
   fundamental.
4. **Re-sweep tuning on the now-working exact gradient.** All prior lr/steps/
   group-size/noise sweeps were on the biased gradient (all flat) and are not
   informative. Re-do lr ∈ {1e-6,3e-6,5e-6} and steps on the exact gradient.
5. **Fix eval-during-training for lingbotva** (optional): `env_evaluate_step`
   doesn't build the KV-replay keyframes → in-loop val SR ≈ 0. We validate
   offline instead. Wiring `_build_chunk_keyframes` into the eval step would give
   a clean in-loop SR curve.

## Key file anchors
- RL core: `../eval_adapter/native_backend.py` — `infer_batch_with_logprob`,
  `recompute_logprob`, `_recompute_group`, `_replay_history_cache`, `_pack_history`.
- RL math/engine: `../rl_utils.py`, `../rl_engine.py` (`RLForwardInputs`).
- Action model: `../lingbotva_action_model.py` (`_init_rl`, `get_log_prob_value`,
  `_rl_predict_action_batch`).
- Rollout worker: `rlinf/workers/rollout/hf/lingbotva_rollout_worker.py`.
- Actor loss-mask fix: `rlinf/workers/actor/fsdp_actor_worker.py` (~the
  `loss_mask` build gated on `not auto_reset`).
- Config: `examples/embodiment/config/libero_object_grpo_lingbotva.yaml`,
  `examples/embodiment/config/libero_object_eval_lingbotva.yaml`.
- Gate-1: `examples/embodiment/check_lingbotva_rl_gate1.py`.
- Unit tests: `tests/unit_tests/models/embodiment/test_lingbotva_rl_*.py`
  (29 tests; run standalone — the container's `/venv/main` lacks omegaconf/ray
  so importing the full `rlinf` package fails; the RL math has no rlinf deps).

## Secrets
The SFT checkpoint is on a **private** HF repo (`kzrzhou/...`). Fetch with your
own token at setup. **Never** commit a token or write one to memory/files.
