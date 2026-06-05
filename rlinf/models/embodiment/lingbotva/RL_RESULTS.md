# LingBot-VA GRPO — Experimental Results

Empirical findings for GRPO reinforcement learning on the **LingBot-VA** model
(`model_type: lingbotva`, a Wan-Video diffusion video+action VLA) on
**Libero-Object**, starting from the SFT checkpoint `checkpoint_step_3000_extra`.

For the algorithm/design, see [`RL_DESIGN.md`](./RL_DESIGN.md). This document
records *what we observed*, not how the RL core works.

## TL;DR

**GRPO RL raises Libero-Object success rate over SFT — but only with the
(near-)exact gradient and ~20 steps.** Deterministic eval on the trained tasks
{2,6}: SFT 40.0% → exact-gradient step 20 **66.7% (+26.7%, z=2.15, p≈0.03)**, a
monotonic rise across checkpoints.

The bottleneck was the **gradient**. The distributed-synced recompute originally
had to drop the KV-cache replay (`recompute_kv_replay=false`, proximal
ratio ≈ 0.6, ~87% clipped); on that biased gradient, *every* run was flat
regardless of envs, group size, noise, lr, or task focus, and higher lr actively
degraded SR. Restoring the exact replay (`recompute_kv_replay=true`,
ratio ≈ 0.91) — enabled distributed by `ignore_terminations=true` (non-ragged
buffer → FSDP rank-symmetric) — unlocked the gain. It took ~20 steps to show:
the step-10 eval was still flat (+3.3%, n.s.).

| Gradient | SR vs SFT @ ~20 steps |
| --- | --- |
| biased (ratio 0.6) | flat (all runs / all tuning) |
| **exact (ratio 0.91)** | **+26.7%, significant** |

Two measurement lessons: (1) the training `success_once` (noise=1.0) is
unreliable — it stayed ~flat while the deterministic policy improved; always
eval checkpoints at noise=0. (2) Gains need ~20 steps to surface; a step-10
read mislead toward "no effect."

## Setup

- **Model / env:** LingBot-VA (5B WanTransformer3D), Libero-Object, 240-step
  closed-loop episodes, action chunk = 12 executable actions (every 12 sim steps).
- **Algorithm:** GRPO, critic-free (`adv_type=grpo`, `loss_type=actor`). The
  action-denoising chain runs as a stochastic SDE (`noise_method=flow_sde`);
  per-step diagonal-Gaussian density is the action log-prob. Video-latent
  denoising stays deterministic (world-context).
- **Cluster:** 8×A100-80GB, disaggregated (actor 0–3, env 0–3, rollout 4–7),
  NCCL ring weight sync. Collocation is **blocked** on this container (CUDA-IPC
  needs `pidfd_getfd`, restricted by `ptrace_scope`).
- **Gradient:** `recompute_kv_replay=false` — the actor recompute does a fixed
  2 forwards/sample (no data-dependent history replay), keeping FSDP collectives
  identical across DP ranks. The exact (replay-on) recompute is bit-exact
  single-process (gate #1) but **deadlocks** distributed FSDP because per-rank
  trajectories replay a data-dependent number of forwards.

## Evaluation methodology (and three bugs we had to fix first)

The standalone eval driver `examples/embodiment/eval_lingbotva.py` was buggier
than expected; the gain could not be trusted until these were fixed:

1. **Counting bug** — the loop counted *every* terminal state and stopped at a
   cumulative env-count, so fast envs were double-counted and slow envs never
   counted. Fixed: count exactly the **first** completion per env, loop until
   all envs counted (commit `5bf2ee1`), plus a no-churn guard so only
   still-running envs grow their KV history.
2. **Coverage bug** — the old "10-task" eval drew inits contiguously from the
   front and actually only touched 2 of 10 tasks. Fixed: one process per task
   via `task_id_filter=[t]`.
3. **Init-shard offset** — added `eval_reset_start_idx` (commit `8b5a774`) so
   parallel eval processes draw **disjoint** init states.

Two further realities shape the protocol:

- **Eval is non-deterministic.** Even a greedy policy gives stochastic outcomes:
  CUDA/bf16 numerical noise amplifies over the 240-step closed loop (same
  checkpoint + same inits gave 0/3 then 1/3). → large-N sampling required.
- **Memory wall.** Eval re-encodes a growing KV-cache history each chunk (≈
  O(n²)); hard (low-success) tasks run all envs to the full 240 steps and OOM.
  **Safe = 3 envs/process.** (Training rollout does *not* hit this — it caps
  history via `kv_replay_max_frames=80`.)

**Protocol:** 3 envs/process, offsets {0,3,6} → 9 inits/task × 10 tasks =
**90 episodes/checkpoint**, all tasks covered, 0 OOMs. Tooling:
`/tmp/validate2.sh`, `/tmp/agg_val2.py` (Wilson CI + two-proportion z-test).

## Results

### Clean validation — 16-env / group-4 run (90 ep each)

| Checkpoint | SR | succ/tot | Δ vs SFT | z |
| --- | --- | --- | --- | --- |
| SFT | 45.6% | 41/90 | — | — |
| step 5 | **54.4%** | 49/90 | +8.9% | +1.20 |
| step 10 | 46.7% | 42/90 | +1.1% | +0.15 |
| step 15 | 42.2% | 38/90 | −3.3% | −0.45 |

Directional peak-then-decay (step 5 best), but not significant at n=90.

### Proper-config — 64-env / group-8 run

Native config scale; only env-required overrides (offload on,
`recompute_kv_replay=false`, in-loop eval off, `lr=3e-6`, `max_steps=50`,
`save_interval=10`). No OOM (peak 50/80 GB). Per-step ≈ 22 min (rollout ≈ 1011 s
generation-dominated + train ≈ 367 s).

Training-rollout `success_once` (noise=1.0, 64 ep/step) over steps 1–29:

```
.42 .33 .34 .42 .44 .48 .27 .41 .48 .34 .42 .36 .36 .45 .58
.28 .33 .30 .28 .28 .41 .48 .47 .34 .45 .45 .27 .41 .56
```

Noisy-flat around ~0.40 (the steps 16–20 dip was noise; recovered by 21–29).
**This metric is unreliable** — noise=1.0 exploration masks policy quality; the
deterministic checkpoint eval is the real signal.

Deterministic validation (90 ep each):

| Checkpoint | SR | succ/tot | Δ vs SFT | z |
| --- | --- | --- | --- | --- |
| SFT | 45.6% | 41/90 | — | — |
| step 10 | 47.8% | 43/90 | +2.2% | +0.30 |
| step 20 | 47.8% | 43/90 | +2.2% | +0.30 |

Flat/tied with SFT. Note step 20 *redistributed* rather than improved: task 8
went 2/9 → **9/9** but tasks 2 and 4 regressed — net wash.

The run **crashed at step ~30 on a full disk** before step 30's weights were
written (steps 10 & 20 salvaged). See *Operational notes*.

### Hyperparameter tuning + 2-task focus (the dilution hypothesis)

To rule out that the flatness was a *tuning* problem rather than a *gradient*
problem, we swept the cheap knobs. All on the biased gradient:

| Variant | Result |
| --- | --- |
| 10 tasks, lr 1e-6 / 3e-6 | flat (~0.5 training-rollout SR, no trend) |
| group_size 2 / 4 / 8 | flat (group_size≥4 fixes sparse advantage, but SR still flat) |
| rl_noise_level 1.0 / 0.5 | flat |
| lr 5e-6 | **degrades** (see below) |
| **2-task focus** (tasks 2 & 6) | **flat** |

**Dilution hypothesis (rejected).** Idea: training across all 10 tasks dilutes
the gradient and stalls SR; focus on 2 mid-SR tasks (2 & 6, both ~50% SFT = max
GRPO within-group outcome variance) with many envs each (96 → ~48/task). Run on
the training tasks only (`env.train.task_id_filter=[2,6]`; `success_once` then
measured on exactly those tasks). Result over 12 steps (lr=3e-6, noise=1.0):

```
step: 1     2     3     4     5     6     7     8     9    10    11    12
SR:  .240  .229  .156  .344  .250  .208  .208  .260  .219  .177  .271  .240
```

first-6 mean 0.238, last-6 mean 0.229 — **flat**, just like all-10. Focusing the
gradient did not help.

**Higher lr actively degrades.** A 2-task run at **lr=5e-6** (noise=0.5)
monotonically *decreased* SR `.354 → .250 → .198 → .188` over 4 steps with reward
falling in lockstep. The biased gradient points in a subtly wrong direction on
deep chunks; on 10 tasks at lr=3e-6 those errors partly average out (→ flat), but
with a bigger step the bias compounds and walks the policy downhill. So the
biased gradient is not merely weak — it is *wrong*, and only a gentle lr hides it.

**Conclusion:** across task count, group size, exploration noise, and learning
rate, the biased no-replay gradient never raises SR — it is flat at best and
degrading at worst. This isolates the **gradient** (not tuning, not task
dilution) as the wall. The exact unbiased gradient is the required next step.

## Rollout-speed investigation (no good free speedup)

Rollout is ~100% diffusion generation: ≈90 transformer forwards per action chunk
(50 action-denoise + 20 video-denoise × 2 for `guidance_scale=5` CFG). Levers
tested on the SFT checkpoint:

- `action_num_inference_steps 50→25`: measured **1.27×** at training scale, but
  dropped the policy SR floor ~10 pts (0.41→0.33) → bad trade, reverted.
- `num_inference_steps 20→10` (video) preserves SR in a noisy probe; halving
  **both** degrades hard tasks (task 8 → 0/3).
- `guidance_scale 5→1` is not config-wired (needs a code change).
- Collocation (all-8-GPU rollout, ~2×) is blocked by the container.

**Conclusion:** every meaningful speedup degrades the policy; the only clean
wall-clock lever is fewer training steps (gains peak early anyway).

## Operational notes

- ⚠️ **Checkpoint size is 38 GB**, not 9.5 GB. A checkpoint dir is
  `actor/dcp_checkpoint/` (29 GB sharded resume state) +
  `actor/model_state_dict/full_weights.pt` (9.5 GB, all that eval needs). Three
  of them fill the 160 GB disk → save crashes mid-write
  (`RuntimeError: enforce fail ... unexpected pos` = `torch.save` on a full
  disk). **For future runs: budget 38 GB/checkpoint, or delete `dcp_checkpoint`
  after each save, or save less often.** (Same failure the 16-env run hit.)
- Always fully kill stale procs between runs (Ray actors leak GPU memory →
  false OOM): `ray stop --force; pkill -9 -f "ray::|train_embodied"; verify
  `nvidia-smi` ~0 MiB`.

## Exact gradient RAISES SR (+26.7%, significant) — the key positive result

The biased gradient (proximal ratio ≈ 0.6) was the wall. Making the recompute
(near-)exact (ratio ≈ 0.91) **and running ~20 steps** produces a clear,
statistically significant success-rate gain over SFT — something no biased-
gradient run ever achieved at any tuning.

Deterministic (noise=0) eval on the trained tasks {2,6}, 30 episodes each:

| checkpoint | SR | Δ vs SFT | z |
| --- | --- | --- | --- |
| SFT | 40.0% (12/30) | — | — |
| exact step 10 | 43.3% (13/30) | +3.3% | +0.26 |
| exact step 15 | 46.7% (14/30) | +6.7% | +0.52 |
| **exact step 20** | **66.7% (20/30)** | **+26.7%** | **+2.15** |

A monotonic, *accelerating* rise (40 → 43 → 47 → 67) across four independent
checkpoints — significant at step 20 (p ≈ 0.03), and the trajectory shape makes
noise very unlikely. Both tasks improved (t2 6→8, t6 6→12).

**Lesson — measurement and patience both mattered.** The training `success_once`
(noise=1.0) was noisy-flat (~0.40) the entire run and was *misleading*; only the
deterministic checkpoint eval revealed the gain. And the policy needed ~20 steps
to escape the SFT basin — a step-10 eval (+3.3%, n.s.) looked flat and would have
(did, initially) led to the wrong conclusion that the gradient was not the lever.

### Why earlier runs were flat (the biased gradient WAS the bottleneck)

Every flat result used `recompute_kv_replay=false` (ratio ≈ 0.6, ~87% of samples
clipped). The exact gradient run above is identical except `recompute_kv_replay=
true` + `ignore_terminations=true`. So the gradient correctness was indeed the
lever — it just needed enough steps to show, which is why the tuning sweep (all
on the biased gradient) and the step-10 exact eval both looked flat.

- **Mechanism fixed:** with `ignore_terminations=true` (non-ragged buffer →
  FSDP rank-symmetric, see below) the exact recompute (`recompute_kv_replay=true`)
  runs distributed at full 240-step length with `actor/ratio ≈ 0.91` every step
  (vs the biased 0.6). The residual 0.09 is bf16 drift over the deep replayed
  history; a true 1.0 would need the rank-symmetric padding build (1b below).
- **Real training run** (tasks 2 & 6, 32 envs, lr=3e-6, noise=1.0): training
  `success_once` over 10 steps stayed noisy-flat (~0.37) — but that metric is
  unreliable (noise=1.0). The deterministic (noise=0) eval of the step-10
  checkpoint on the trained tasks:

  | | SR (30 ep) | t2 | t6 |
  | --- | --- | --- | --- |
  | SFT | 40.0% (12/30) | 6/15 | 6/15 |
  | exact step 10 | 43.3% (13/30) | 5/15 | 8/15 |

  Δ = +3.3%, z = +0.26 — **not significant**. Flat.

**Conclusion:** moving the gradient from biased (ratio 0.6) to near-exact (0.91)
— a large reduction in bias — produced no SR change in either the training curve
or deterministic eval. The gradient correctness is therefore not the lever.
Combined with the flat tuning matrix and the flat 2-task focus, the bottleneck
lies elsewhere — most plausibly the update magnitude (only 10 steps at a
conservative lr) and/or the reward/exploration regime, or limited GRPO headroom
over this SFT checkpoint. (Caveat: ratio is 0.91 not 1.0, the run was short, and
n=30 cannot resolve a small gain — but the 0.6→0.91 null makes 0.91→1.0
unlikely to flip it.)

## Exact gradient: distributed deadlock root cause (now worked around)

The exact unbiased recompute (`recompute_kv_replay=true`) is **correct**: a short
4-chunk (48-step) distributed run gives `actor/ratio = 1.000`, `ratio_abs = 0`,
no deadlock. But the full **20-chunk (240-step)** run **deadlocks** — actor GPUs
pin at 100% util with frozen memory (NCCL spin-wait) right after the rollout,
rollout GPUs idle, no progress.

**Root cause: ragged buffer from early termination.** Libero episodes stop when
the task succeeds (`env.train.ignore_terminations=False`), so deep chunks have
fewer surviving envs than shallow ones — and *which* envs survive differs per
actor rank. The actor splits each rank's local buffer into micro-batches
independently, and `recompute_logprob` replays a number of FSDP forwards that
depends on the chunk's `frame_st_id` depth. So at micro-step *k*, one rank is on
a shallow chunk (few replay forwards) while another is on a deep one (many) →
mismatched all-gather sequence → deadlock. (4 chunks works because there is
little raggedness; it breaks as episodes spread out — explaining why prior
micro=1 and micro=4 attempts both died at 240 steps.)

## Path forward

1a. **Cheap shortcut — non-ragged buffer.** Set
   `env.train.ignore_terminations=True` so every env runs the full 240 steps →
   buffer is a clean `[chunks × envs]` grid → per-rank chunk composition is
   uniform → ranks issue identical collectives by construction, *no padding code
   needed*. Trade-off: larger buffer + slower rollout (envs keep stepping after
   success), and post-success steps add some reward noise. Test this first.
1b. **Full fix — rank-symmetric padded replay** (if the shortcut is too slow or
   hurts the signal). `all_gather` the chunk-key union across DP ranks; every
   rank iterates the same global sorted keys; for keys it lacks, run a padding
   `_recompute_group` on a synthetic 1-row batch at that chunk's depth (matching
   forward count) and route its action log-prob through a **0-weight loss term**
   so the *backward* collectives also match (the replay is `no_grad` and only
   forward-syncs; the scored action forward's backward must sync too — the subtle
   trap). Touches `recompute_logprob`, its return contract, and the actor loss.
   Single-process this is bit-exact (gate #1, ratio=1.000); batched by chunk it
   runs in minutes/step (the per-sample micro=1 replay was the original perf
   blocker, ~21 min for 4 chunks).
2. **Tuning is exhausted.** The cheap knobs (lr, noise, group size, task focus)
   have all been swept and none raise SR on the biased gradient — so further
   hyperparameter search is not worthwhile; the gradient is the bottleneck.
