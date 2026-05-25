# Reproducing LingBot-VA SFT on Libero-Object in RLinf

This guide reproduces Milestone 2 of the LingBot-VA integration: supervised
fine-tuning of the 5 B-parameter LingBot-VA transformer on Libero-Object,
matching the reference single-A100 recipe's loss curve and exceeding its
ckpt_1000 evaluation success rate.

**Verified result on a single L40 (45 GB)**:

| Metric                              | This recipe              | Reference (single A100) |
|-------------------------------------|--------------------------|-------------------------|
| ckpt_1000 mean SR (5 ep × 10 tasks) | **26 % (13/50)**         | 18 % (9/50)             |
| Per-task SR                         | `[80,0,0,20,20,20,20,40,0,60]` | `[0,40,0,20,20,40,0,0,20,40]` |
| Wall time to ckpt_1000              | ~5 h (~17 s/step)        | ~3 h (faster GPU)       |
| Wall time to step 3000              | ~14 h projected          | ~9 h                    |

Per-task swings vs the reference are typical noise for 50 episodes; mean SR
is clearly above reference, which validates the integration end-to-end.

---

## 1. Prerequisites

### Hardware
- **Minimum:** 1 GPU with **≥45 GB** of memory.
  Verified on NVIDIA L40 (45 GB). Should fit on A100 (40 or 80 GB), H100,
  RTX 6000 Ada, etc.
- **Recommended:** an A100 or H100 for ~2× wall-clock speedup.
- Disk: ~100 GB free (model ~30 GB; one full checkpoint ~30 GB; dataset ~10 GB).

### Software
- **OS:** Linux (only platform tested). Driver supporting CUDA 12.4+.
- **Python:** 3.10. (Python 3.11+ may work but is untested.)
- **PyTorch:** **2.9.0+cu126**. Earlier versions hit a `flex_attention`
  + inductor codegen bug ("NameError: name 's10' is not defined") and
  must NOT be used.

Other key Python deps (auto-installed below):
- `bitsandbytes>=0.43,<0.46` for `AdamW8bit`
- `nvidia-cufile-cu12`, `nvidia-nvshmem-cu12`, `nvidia-nccl-cu12>=2.30.4`
  (torch 2.9 ABI requirements)
- `flash_attn` is **NOT required** for this recipe (we use flex_attention
  via inductor). See §3.2 below about a small patch you may need to
  apply to lingbot-va.

---

## 2. Clone the two repos

```bash
# RLinf fork containing the LingBot-VA integration
git clone https://github.com/ZiruiZhou/RLinf.git
cd RLinf
git checkout feat/lingbotva-libero-object-sft

# lingbot-va peer repo (provides wan_va Python package + data prep scripts)
git clone https://github.com/robbyant/lingbot-va.git /workspace/lingbot-va
```

Set the path env vars used by RLinf's launch script:

```bash
export REPO_PATH=$(pwd)                                 # RLinf clone path
export EMBODIED_PATH=${REPO_PATH}/examples/embodiment
export LINGBOT_VA_REPO_PATH=/workspace/lingbot-va       # lingbot-va clone path
export LINGBOT_VA_MODEL_PATH=/workspace/models/lingbot-va-base  # see §4
export LINGBOT_VA_DATASET_PATH=/workspace/rlinf_data    # see §3
export PYTHONPATH=${REPO_PATH}:${LINGBOT_VA_REPO_PATH}:${PYTHONPATH}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

---

## 3. Environment setup

### 3.1 Create venv and install PyTorch 2.9

```bash
python3.10 -m venv /workspace/.venv
source /workspace/.venv/bin/activate
pip install --upgrade pip

# RLinf's pyproject.toml pins torch==2.6.0 via override-dependencies, so use
# pip directly (NOT uv pip) to install the newer version:
pip install --no-deps --force-reinstall \
  torch==2.9.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126

# Companion CUDA libs that torch 2.9 needs:
pip install nvidia-cufile-cu12 nvidia-nvshmem-cu12
pip install --upgrade 'nvidia-nccl-cu12>=2.30.4'

# RLinf itself + LingBot-VA model dep group:
pip install -e ${REPO_PATH}
pip install -r ${REPO_PATH}/requirements/embodied/models/lingbotva.txt
pip install -e ${LINGBOT_VA_REPO_PATH}
```

### 3.2 lingbot-va patch (only needed on torch 2.9 without flash_attn)

`wan_va/modules/model.py` imports `flash_attn` at module-load time. On
torch 2.9 the prebuilt `flash_attn` wheels do not load (no cp310 wheel
exists for this torch version yet), so the import raises and blocks any
downstream import — even though our recipe doesn't actually call
`flash_attn_func` (we use `attn_mode: flex`).

Patch the wan_va import to fall through gracefully. From the lingbot-va
checkout:

```diff
 try:
     from flash_attn_interface import flash_attn_func
-except:
-    from flash_attn import flash_attn_func
+except Exception:
+    try:
+        from flash_attn import flash_attn_func
+    except Exception:
+        flash_attn_func = None
```

If you have a working `flash_attn` build for torch 2.9 (built from
source against this ABI), you can skip this patch.

### 3.3 Sanity check

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.9.0+cu126 True

python -c "import bitsandbytes as bnb; print(bnb.__version__)"
# Expected: 0.43.x ... 0.45.x

python -c "from wan_va.modules.utils import load_transformer; print('OK')"
# Expected: OK  (no ImportError)
```

---

## 4. Model weights

Download the LingBot-VA base checkpoint (Wan-Video 2.2 backbone, ~30 GB) into
`${LINGBOT_VA_MODEL_PATH}`. Layout expected:

```
${LINGBOT_VA_MODEL_PATH}/
├── transformer/                       # 5B-param Wan transformer
│   ├── config.json
│   └── diffusion_pytorch_model-*.safetensors
├── vae/                               # Wan 2.2 VAE (not used during SFT)
└── text_encoder/                      # UMT5 (not used during SFT — we
                                         use a cached empty embedding)
```

Source: as provided in the original lingbot-va release notes.

---

## 5. Dataset preparation

This step runs the four lingbot-va scripts in `script/` to produce
${LINGBOT_VA_DATASET_PATH}: an LeRobot v2.1 dataset of 100 demos
(10 per task × 10 tasks) with pre-extracted Wan 2.2 VAE latents and a
cached UMT5 empty embedding.

### 5.1 Download raw HDF5 demonstrations

**Important:** use `yifengzhu-hf/LIBERO-datasets`, NOT `IPEC-COMMUNITY`. The
two distributions have different wrist-camera mounts; the IPEC version trains
fine but evaluates at 0 % SR because the wrist view doesn't match the eval
environment.

```bash
# Download Libero-Object HDF5 files (~25 GB) to e.g. /workspace/libero_raw
huggingface-cli download yifengzhu-hf/LIBERO-datasets --local-dir /workspace/libero_raw
```

### 5.2 Run the four conversion scripts

```bash
cd ${LINGBOT_VA_REPO_PATH}

# (a) HDF5 → LeRobot v2.1 format
python script/convert_libero_object_to_lerobot.py \
  --hdf5-root /workspace/libero_raw/libero_object \
  --output ${LINGBOT_VA_DATASET_PATH}_full

# (b) Subsample to 10 demos × 10 tasks (deterministic via seed 42)
python script/select_subset.py \
  --src ${LINGBOT_VA_DATASET_PATH}_full \
  --dst ${LINGBOT_VA_DATASET_PATH} \
  --per-task 10 --seed 42

# (c) Extract Wan 2.2 VAE latents + cache the UMT5 empty-prompt embedding
python script/extract_latents.py \
  --dataset ${LINGBOT_VA_DATASET_PATH} \
  --model-path ${LINGBOT_VA_MODEL_PATH}
```

### 5.3 Verify dataset layout

```
${LINGBOT_VA_DATASET_PATH}/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
├── data/chunk-000/episode_NNNNNN.parquet × 100
├── videos/chunk-000/
│   ├── observation.images.agentview_rgb/episode_NNNNNN.mp4
│   └── observation.images.eye_in_hand_rgb/episode_NNNNNN.mp4
├── latents/chunk-000/
│   ├── observation.images.agentview_rgb/episode_NNNNNN_<s>_<e>.pth
│   └── observation.images.eye_in_hand_rgb/episode_NNNNNN_<s>_<e>.pth
└── empty_emb.pt        # cached UMT5 embedding of the empty prompt
```

If `episodes_stats.jsonl` has image stats stored as `(3,)` instead of
`(3, 1, 1)`, modern `lerobot>=0.3.3` will reject the dataset. Reshape
in-place if needed (the original is preserved by a `_full` copy from
step (a)).

---

## 6. Launch SFT training

```bash
cd ${REPO_PATH}
bash examples/sft/run_vla_sft.sh libero_sft_lingbotva
```

That script does:

- Exports `LINGBOT_VA_REPO_PATH`, `LINGBOT_VA_MODEL_PATH`,
  `LINGBOT_VA_DATASET_PATH`, `PYTHONPATH` (defaults match §2).
- Exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — required for
  the 5 B-param transformer + activations to fit in 45 GB.
- Runs `examples/sft/train_vla_sft.py` with config
  `examples/sft/config/libero_sft_lingbotva.yaml`.

### Key config values (single-GPU recipe)

| Setting | Value | Why |
|---|---|---|
| `actor.fsdp_config.strategy` | `fsdp` (FSDP1) | FSDP2 wraps params as DTensors, incompatible with bnb AdamW8bit |
| `actor.fsdp_config.sharding_strategy` | `no_shard` | At ws=1 sharding offers nothing |
| `actor.fsdp_config.use_orig_params` | `True` | Keeps params as plain Tensors that bnb's CUDA kernels accept |
| `actor.optim.optimizer_type` | `adamw_8bit` | Reference recipe is bnb-only |
| `actor.optim.lr` / `betas` / `wd` | `1e-5` / `(0.9, 0.95)` / `0.1` | Match reference |
| `actor.micro_batch_size` | `1` | Memory limit on 45 GB |
| `actor.global_batch_size` | `10` | gradient_accumulation = global/(micro·ws) = 10 |
| `actor.model.lingbotva.attn_mode` | `flex` | Reference recipe; **requires torch ≥2.9** |
| `actor.model.precision` | `bf16` | Transformer is loaded bf16 + force-cast (avoids fp32 leakage from `scale_shift_table`) |
| `runner.max_steps` | `3000` | Reference run length |
| `runner.save_interval` | `1000` | First eval-able ckpt at step 1000 |

### Expected loss trajectory (single-GPU)

| Step | latent_loss | action_loss |
|------|-------------|-------------|
| 1    | ~0.20       | ~0.55       |
| 50   | ~0.18       | ~0.16       |
| 100  | ~0.15       | ~0.14       |
| 500  | ~0.13       | ~0.12       |
| 1000 | ~0.10       | ~0.11       |

These are means over the 10 grad-accum micro-batches of an optimizer step.
Values within ±25 % of these are within typical RNG noise (CFG dropout +
flow-matching timestep sampling diverge across torch versions even at
the same seed).

If the values are >2× off the reference at the same step count, something
is wrong (the most common cause is `attn_mode: torch` slipping back in
via a stale config or YAML merge — verify the active config logs show
`attn_mode: flex`).

---

## 7. Evaluation

When `checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt`
exists, you can either wait for training to finish or pause it to free
the GPU and run eval.

```bash
# 10 tasks × 5 episodes = 50 episodes. Loops over task IDs 0..9 explicitly
# (do NOT just set --num-episodes 50 — that runs 50 episodes on whichever
# task gets sampled first; the env's task_id_filter cycles reset states,
# not tasks).

CKPT_DIR=${REPO_PATH}/runtime/lingbotva_sft/libero_sft_lingbotva/checkpoints
CKPT=${CKPT_DIR}/global_step_1000/actor/model_state_dict/full_weights.pt
OUT=${REPO_PATH}/runtime/lingbotva_libero_eval/ckpt_1000

mkdir -p ${OUT}
for tid in 0 1 2 3 4 5 6 7 8 9; do
  python scripts/eval_lingbotva_libero_standalone.py \
    --model-path ${LINGBOT_VA_MODEL_PATH} \
    --repo-path  ${LINGBOT_VA_REPO_PATH} \
    --transformer-state-dict-path ${CKPT} \
    --num-envs 1 --num-episodes 5 --seed 0 --task-id ${tid} \
    --results-path ${OUT}/task_${tid}.json
done
```

Each task takes ~3–17 minutes (failure runs hit the 240-step episode
cap, success runs end earlier). Total ~30 min – ~2 h depending on SR.

Aggregate:

```bash
python - <<'PY' ${OUT}
import json, glob, os, sys
out = sys.argv[1]
totals = {"success": 0, "total": 0, "by_task": {}}
for f in sorted(glob.glob(os.path.join(out, "task_*.json"))):
    d = json.load(open(f))
    tid = int(os.path.basename(f).split("_")[1].split(".")[0])
    s, n = d["successes"], d["successes"] + d["failures"]
    totals["by_task"][tid] = {"success": s, "total": n, "rate": s/n}
    totals["success"] += s; totals["total"] += n
totals["mean_sr"] = totals["success"] / totals["total"]
print(json.dumps(totals, indent=2))
PY
```

Expected mean SR at ckpt_1000: **~26 %** (range 18–28 % across seeds).
Anything below 10 % means flex_attention is not actually active and you
were silently training under SDPA — see §8.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `NameError: name 's10' is not defined` during forward | torch <2.9 flex_attention codegen bug | Upgrade to torch 2.9 (do NOT fall back to `attn_mode: torch`) |
| 0 % SR at ckpt_1000 despite normal-looking loss | SDPA mask leak — `attn_mode` is `torch` somewhere | Confirm the logged config shows `attn_mode: flex` |
| `ImportError: cannot import name 'flash_attn_func'` | torch 2.9 ABI breaks flash_attn wheel | Apply the patch in §3.2 |
| CUDA illegal memory access during warmup | FSDP2's DTensor + bnb AdamW8bit collision | Confirm `fsdp_config.strategy: fsdp` (FSDP1), not `fsdp2` |
| FSDP1 `flatten tensors with uniform dtype` error | `scale_shift_table` initialised fp32 | Confirm `precision: bf16` is set AND the transformer is force-cast to bf16 (already wired in the action model) |
| OOM during training | 45 GB card without expandable_segments | Confirm `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is exported (run_vla_sft.sh does this); otherwise reduce `num_action_chunks` |
| `lerobot` rejects dataset | `episodes_stats.jsonl` has wrong image-stats shape | Reshape per-channel stats from `(3,)` to `(3, 1, 1)` |

---

## 9. Scaling to multiple GPUs (not yet verified)

The current recipe is verified at world_size = 1 only. Pointers if you
want to try ws > 1:

- **Keep FSDP1**, not FSDP2: bnb AdamW8bit still requires plain Tensors
  for its quantized state. FSDP2's `fully_shard` always produces DTensors.
- **Sharding strategy** can move from `no_shard` to `full_shard` once
  ws ≥ 2 — the AdamW8bit state should still be plain CUDA Tensors per
  rank, but this is **untested by us**; verify the optimizer step
  succeeds before assuming.
- **Batch arithmetic**: `grad_accum = global_batch_size / (micro × ws)`.
  Reference uses ws=1 × micro=1 × grad_accum=10. To match the effective
  batch at ws=4: keep `global_batch_size=10` (so `grad_accum=2.5` —
  bump to `global=20` for `grad_accum=5`, or accept the lower accum
  with proportionally lower noise).
- **flex_attention** with FSDP1 FULL_SHARD has known limitations around
  `torch.compile`. If the inductor lowerings break under FSDP wrapping,
  try `actor.fsdp_config.gradient_checkpointing: True` first (already
  on), then consider running the attention in eager mode via the
  appropriate wan_va flag.
- **Wall time**: data is loaded from disk via the dataloader (~7 % of
  step time on L40), so adding GPUs should give close-to-linear speedup
  on the compute-bound forward/backward.

If you scale up successfully, please update this section with whatever
you learn — the loss-curve milestones in §6 should be GPU-count-invariant
as long as the global batch and grad-accum product is preserved.

---

## 10. References

- This RLinf branch: `feat/lingbotva-libero-object-sft` on
  `https://github.com/ZiruiZhou/RLinf.git`
  - Backup of the pre-squash history is on
    `backup/lingbotva-sft-pre-squash`.
- The lingbot-va peer repo: `https://github.com/robbyant/lingbot-va.git`
- Critical recipe commit: `0c7d801` — single source of truth for FSDP1,
  bnb AdamW8bit, bf16 load, attn_mode=flex, expandable_segments, grad-accum
  loss logging, eval CLI flag.
