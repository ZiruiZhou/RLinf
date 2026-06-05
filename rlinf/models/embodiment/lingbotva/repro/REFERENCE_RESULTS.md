# REFERENCE_RESULTS — numbers to validate against

Canonical results for GRPO RL on **LingBot-VA** / **Libero-Object**, starting
from the SFT checkpoint `checkpoint_step_3000_extra`. Full narrative and the
diagnosis path are in `../RL_RESULTS.md`; how to regenerate these in
`REPRODUCE.md`.

## Headline: the exact gradient raises SR over SFT

Deterministic (noise=0) eval on the **trained tasks {2,6}**, 30 episodes each
(`scripts/eval_tasks.sh` + `scripts/agg_trajectory.py`):

```
sft     40.0% (12/30)  CI[25,58]  t2:6/15  t6:6/15
exact10 43.3% (13/30)  CI[27,61]  t2:5/15  t6:8/15   Δ=+3.3%  z=0.26
exact15 46.7% (14/30)  CI[30,64]  t2:4/15  t6:10/15  Δ=+6.7%  z=0.52
exact20 66.7% (20/30)  CI[49,81]  t2:8/15  t6:12/15  Δ=+26.7% z=2.15   *
exact25 46.7% (14/30)  CI[30,64]  t2:4/15  t6:10/15  Δ=+6.7%  z=0.52
exact30 70.0% (21/30)  CI[52,83]  t2:11/15 t6:10/15  Δ=+30.0% z=2.45   *
```
`*` = significant (|z| > 1.96). **Two independent checkpoints (steps 20 and 30)
both clear z > 2.1; peak p ≈ 0.014.**

### How to read it
- The curve is a **rising noisy envelope, not a monotone climb** — step 25 falls
  back to 46.7%. This mirrors the step-to-step oscillation of the training
  metric and is expected at n=30 (per-checkpoint 95% CI is ±~17 pt). The robust
  evidence is the two significant peaks plus the rising floor, **not** any single
  step.
- Both trained tasks improve at the peak (step 30: t2 6→11, t6 6→10), so it is
  not a single-task fluke.
- Run-to-run reproduction will not match these counts exactly: eval is
  non-deterministic (CUDA/bf16 over the 240-step closed loop). Expect the same
  *shape* (SFT ~40%, peaks ~65–70% around steps 20–30), not identical integers.

## The key contrast: biased vs exact gradient

| gradient | recompute | proximal ratio | SR vs SFT @ ~20 steps |
| --- | --- | --- | --- |
| biased | `recompute_kv_replay=false` (no KV replay) | ≈ 0.6 (~87% clipped) | **flat** at every tuning (lr, group size, noise, env count, task focus) |
| **exact** | `recompute_kv_replay=true` + `ignore_terminations=true` | ≈ 0.91 | **+26.7%…+30.0%, significant** |

The biased gradient was the bottleneck the whole time. The residual 0.09 in the
exact ratio is bf16 drift over the deep replayed history (a true 1.0 would need
the rank-symmetric padded-replay build — see `NEXT_SESSION.md`).

## Measurement lessons (so the next run isn't misread)
1. **Training `success_once` is unreliable.** Measured at `rl_noise_level=1.0`
   (full SDE exploration) over 16–64 episodes, it stayed noisy-flat (~0.40) the
   entire run while the deterministic policy improved by +30%. Judge a policy by
   **deterministic checkpoint eval**, never by the training curve.
2. **Gains need ~20+ steps to surface.** A step-10 eval (+3.3%, n.s.) looked
   flat and initially led to a wrong "gradient isn't the lever" conclusion.
   Push past step 20 before judging.
3. **Use enough episodes.** n=30 gives ±~17 pt CIs. To firm up a peak (e.g.
   confirm step 30 > SFT beyond doubt) run n=60–90.

## Earlier supporting evidence (under-powered biased runs, for context)
- 16-env / group-4 biased run, all 10 tasks, deterministic 10-ep/task eval:
  SFT 60% → step5 70% → step10 60% → step15 80% — suggestive +20% but n=10
  (±0.15), not significant.
- Clean 90-ep all-task validation of a biased run: SFT 45.6% → step5 54.4%
  (+8.9%, z=1.20) → decaying after — directional, not significant at n=90.

These motivated the move to the exact gradient + focused tasks, which produced
the significant result above.
