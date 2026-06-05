# LingBot-VA GRPO — reproduction & handoff

Self-contained guide to reproduce (or hand off for validation) the GRPO RL
result for **LingBot-VA** (`model_type: lingbotva`) on **Libero-Object**:
RL with the (near-)exact gradient raises success rate over the SFT checkpoint
(tasks {2,6}: SFT 40.0% → step 30 **70.0%**, +30.0%, z=2.45).

## Read in this order
1. **[ENV_PREP.md](ENV_PREP.md)** — build the dev environment (hardware, deps,
   repos, model checkpoints, env vars, container constraints).
2. **[REPRODUCE.md](REPRODUCE.md)** — end-to-end runbook: gate → SFT eval →
   RL training → checkpoint eval → aggregation, with commands and budgets.
3. **[REFERENCE_RESULTS.md](REFERENCE_RESULTS.md)** — the numbers to validate
   against and how to read them.
4. **[NEXT_SESSION.md](NEXT_SESSION.md)** — handoff notes: state, commit map,
   gotchas, open items, file anchors.

Background (not needed to reproduce): **[../RL_RESULTS.md](../RL_RESULTS.md)**
(full experimental narrative) and **[../RL_DESIGN.md](../RL_DESIGN.md)** (RL core
design and contracts).

## scripts/
All source `scripts/env.sh` (paths via overridable env vars). `chmod +x` once.

| script | purpose |
| --- | --- |
| `env.sh` | shared env block (repos, checkpoints, interpreter) — sourced by the rest |
| `gate1_check.sh` | RL-core recompute-consistency sanity (ratio=1.0), single GPU |
| `train_exact.sh` | **the winning run** — exact-gradient GRPO, tasks {2,6}, to step 30 |
| `train_proper.sh` | standard 64-env / all-10-task scale (biased synced gradient) |
| `eval_tasks.sh` | deterministic (noise=0) eval on chosen tasks (default {2,6}) |
| `validate_all_tasks.sh` | large-N deterministic validation across all 10 tasks |
| `agg_trajectory.py` | aggregate `eval_tasks.sh` output → SR trajectory + z-tests |
| `agg_val.py` | aggregate `validate_all_tasks.sh` output → SR + Wilson CI + z-test |

## TL;DR commands
```bash
chmod +x scripts/*.sh
bash scripts/gate1_check.sh                                  # sanity
bash scripts/eval_tasks.sh sft "$SFT_TRANSFORMER" 2,6        # SFT baseline (~40%)
bash scripts/train_exact.sh 30                               # RL run (~13-15 h)
CK=$LOG_ROOT/grpo_exact_train_logs/libero_object_grpo_lingbotva/checkpoints
for s in 10 15 20 25 30; do
  bash scripts/eval_tasks.sh exact$s "$CK/global_step_$s/actor/model_state_dict/full_weights.pt" 2,6
done
python scripts/agg_trajectory.py                            # SFT→s30 table
```
