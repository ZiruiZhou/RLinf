# LingBot-VA GRPO — Experimental Results

Empirical findings for GRPO reinforcement learning on the **LingBot-VA** model
(`model_type: lingbotva`, a Wan-Video diffusion video+action VLA) on
**Libero-Object**, starting from the SFT checkpoint `checkpoint_step_3000_extra`.

For the algorithm/design, see [`RL_DESIGN.md`](./RL_DESIGN.md). This document
records *what we observed*, not how the RL core works.

## TL;DR

The RL pipeline is **correct and runs at scale** (gate #1 bit-exact, 64-env /
group-8 training stable end-to-end). But with the **biased no-replay recompute
gradient** (the only variant that stays FSDP-synced in distributed training),
RL does **not** produce a statistically significant success-rate gain over SFT.
Two independent runs land flat-to-marginal:

| Run | Best RL checkpoint | SR vs SFT (deterministic, 90 ep) | Significance |
| --- | --- | --- | --- |
| 16-env / group-4 | step 5 | 54.4% vs 45.6% (**+8.9%**) | z = +1.20 (n.s.) |
| 64-env / group-8 | step 10 & 20 | 47.8% vs 45.6% (**+2.2%**) | z = +0.30 (n.s.) |

The prime suspect is the biased gradient (`recompute_kv_replay=false`,
proximal ratio ≈ 0.6). The indicated next step is the **exact unbiased
gradient** (rank-symmetric padded replay) — see *Path forward*.

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

## Path forward

1. **Exact unbiased gradient (rank-symmetric padded replay)** — the principled
   fix and most likely lever to actually raise SR. Set
   `recompute_kv_replay=true`, then in the actor recompute `all_reduce(MAX)` the
   per-rank replay-forward count and have every rank do `global_max` replay
   iterations (real entries write the real cache; padding iterations run
   `update_cache=0` no-op forwards into a scratch cache) → identical FSDP
   collectives across ranks *and* an exact ratio = 1.0.
2. **Cheaper experiments meanwhile:** resume from step 20 (DCP kept) to
   step 40–50, and tune `lr` / `rl_noise_level` / `group_size`, still on the
   biased gradient. Lower expected value given two flat runs, but fast.
