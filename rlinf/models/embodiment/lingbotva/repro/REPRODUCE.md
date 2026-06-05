# REPRODUCE — LingBot-VA GRPO RL on Libero-Object

End-to-end runbook to reproduce the headline result: **GRPO RL with the
(near-)exact gradient raises Libero-Object success rate over the SFT checkpoint**
(focused tasks {2,6}: SFT 40.0% → step 30 **70.0%**, +30.0%, z=2.45). Full
numbers and caveats in `REFERENCE_RESULTS.md`; the narrative/diagnosis in
`../RL_RESULTS.md`; the RL design in `../RL_DESIGN.md`.

All scripts live in `./scripts/` and source `./scripts/env.sh` (paths are
overridable env vars — see `ENV_PREP.md`). `chmod +x scripts/*.sh` once.

---

## 0. Prerequisites
Build the environment per **`ENV_PREP.md`** (repos, `/venv/main` deps, the
`base` / SFT / `base_sft` checkpoints, env vars, container constraints). Confirm
8× A100 are visible and idle (`nvidia-smi`).

---

## 1. Sanity gate — RL-core recompute consistency (single GPU, ~2 min)
Verifies the rollout log-prob equals the gradient-recompute log-prob at the same
weights (proximal ratio = 1.0). Run this first; if it fails, nothing downstream
is meaningful.
```bash
bash scripts/gate1_check.sh
# expect: ratio=1.0000  max|delta|=0.0
```

---

## 2. SFT baseline — deterministic eval on the trained tasks (~15 min)
Establish the pre-RL success rate the RL run is measured against.
```bash
bash scripts/eval_tasks.sh sft "$SFT_TRANSFORMER" 2,6
python scripts/agg_trajectory.py sft
# expect ~40% on tasks {2,6} at 30 episodes (noisy; see REFERENCE_RESULTS.md)
```
> Eval is **non-deterministic** (CUDA/bf16 over the 240-step closed loop) — the
> same checkpoint and inits can give 0/3 vs 1/3. Use ≥30 episodes; don't read a
> single offset. Keep `nenvs ≤ 3` per process (the 240-step KV-replay history
> grows O(n²) and OOMs at 5 envs on the hard tasks).

---

## 3. The RL run — exact-gradient GRPO, tasks {2,6}, to step 30 (~13–15 h)
```bash
# Fully detach so it survives the shell; clean any stale Ray first (see ENV_PREP).
bash scripts/train_exact.sh 30
```
What this run is (and why it works) — see header of `scripts/train_exact.sh`:
- `recompute_kv_replay=true` → exact KV-replay recompute (ratio ≈ 0.91 vs the
  biased no-replay ≈ 0.6). **The biased gradient was the wall**: every earlier
  run was flat at every tuning; this is the change that moved SR.
- `ignore_terminations=true` → every env runs the full 240 steps → non-ragged
  `[chunks × envs]` buffer → FSDP rank-symmetric → no distributed deadlock (the
  ragged-buffer failure documented in `NEXT_SESSION.md`). It also requires a
  one-line actor fix (build `loss_mask` when `ignore_terminations`) that is
  already committed.
- 32 envs, group 8, tasks {2,6}, lr 3e-6, noise 1.0, disaggregated + ring sync,
  offload on. Checkpoints every 5 steps under
  `$LOG_ROOT/grpo_exact_train_logs/.../checkpoints/global_step_<N>/`.

Notes while it runs:
- The training-time `success_once` metric is at `rl_noise_level=1.0` (full SDE
  exploration). **It is noisy and misleading** — it stayed ~0.40 the whole run
  while the deterministic policy improved. Do **not** judge progress by it.
- Disk: each checkpoint is ~38 GB. If tight, delete the `actor/dcp_checkpoint`
  subdir after each save (eval needs only `actor/model_state_dict/full_weights.pt`).
  You then lose mid-run resume for that step but keep the eval weights.
- To extend an existing run instead of starting fresh:
  `bash scripts/train_exact.sh 40 $LOG_ROOT/grpo_exact_train_logs/.../global_step_30`
  (the published trajectory was produced by 10→20→30 resume increments; a single
  0→30 run with this config reproduces it up to eval noise).

---

## 4. Deterministic eval of the RL checkpoints (~15 min each)
After training stops (free the GPUs — `ray stop --force` etc.), eval each saved
checkpoint deterministically on the trained tasks and aggregate the trajectory.
```bash
CK=$LOG_ROOT/grpo_exact_train_logs/libero_object_grpo_lingbotva/checkpoints
for s in 10 15 20 25 30; do
  bash scripts/eval_tasks.sh "exact$s" "$CK/global_step_$s/actor/model_state_dict/full_weights.pt" 2,6
done
python scripts/agg_trajectory.py        # prints the full SFT→s30 table + z-tests
```
Compare against `REFERENCE_RESULTS.md`. The signal is the **rising envelope and
the two z>2.1 peaks (steps 20 and 30)** — the curve is noisy, not monotone (step
25 dips), which is expected at n=30.

---

## 5. (Optional) Scale up — standard 64-env / all-10-task run
```bash
bash scripts/train_proper.sh 50
# validate any checkpoint across all 10 tasks (90 ep):
bash scripts/validate_all_tasks.sh p20 "$CK/.../global_step_20/actor/model_state_dict/full_weights.pt"
python scripts/agg_val.py
```
Caveat: `train_proper.sh` uses the **biased** synced gradient
(`recompute_kv_replay=false`) for distributed safety across the ragged all-task
buffer, and the biased gradient was flat in our runs. To carry the exact-gradient
gain to scale you need `ignore_terminations=true` + `recompute_kv_replay=true`
validated at 64 envs / 10 tasks — see `NEXT_SESSION.md` "open items".

---

## Time/résource budget (reference box)
| step | wall-clock | GPUs |
| --- | --- | --- |
| gate-1 | ~2 min | 1 |
| SFT eval (30 ep) | ~15 min | 8 |
| exact RL → step 30 | ~13–15 h | 8 (4 actor + 4 rollout) |
| per-checkpoint eval (30 ep) | ~15 min | 8 |
| proper 64-env → step 50 | ~18 h | 8 |
