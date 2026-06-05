# ENV_PREP — LingBot-VA GRPO development environment

How to build the container used to develop and run GRPO RL for **LingBot-VA**
(`model_type: lingbotva`, a Wan-Video diffusion video+action VLA) on
**Libero-Object**. Everything below was validated on the dev box; commands are
illustrative — pin to the same versions if you want bit-comparable numbers.

## Hardware
- **8× NVIDIA A100-80GB** (the exact run uses 4 actor + 4 rollout GPUs,
  disaggregated). A single GPU is enough for the gate-1 sanity check and small
  deterministic evals.
- ~160 GB local scratch disk. **Checkpoints are ~38 GB each** (29 GB DCP
  resume-state + 9.5 GB `full_weights.pt` eval-weights) — budget accordingly or
  trim the DCP dir after each save (eval needs only `full_weights.pt`).

## Repositories (3, wired via PYTHONPATH — no pip-install of the two externals)
```bash
# RLinf (this repo) — branch feat/lingbotva-rl-grpo
git clone https://github.com/RLinf/RLinf.git /workspace/RLinf

# LingBot-VA model code (public). Imported as the `wan_va` package via PYTHONPATH.
# (Its pyproject lists packages=["lingbot_va"] which is wrong — the real package
#  is `wan_va`; do NOT `pip install -e`, just put it on PYTHONPATH.)
git clone https://github.com/robbyant/lingbot-va.git /workspace/lingbot-va

# LIBERO simulator (RLinf fork)
git clone https://github.com/RLinf/LIBERO.git /workspace/LIBERO
pip install -e /workspace/LIBERO --no-deps
```

## Python env (`/venv/main`, Python 3.10)
Key pins (the reference box; newer combos may work but were not validated):
- `torch==2.6.0+cu126` — **attn_mode=torch (SDPA)** is used throughout; this
  avoids the flex-attention bug that needs torch 2.9. (The lingbot-va REPRODUCE
  references torch 2.9; 2.6 + SDPA is what we validated and what gate-1 passes on.)
- `numpy==1.26.4` (pinned **<2**)
- `diffusers==0.36`, `transformers==4.55.2`, `accelerate`
- `einops easydict safetensors imageio opencv-python ftfy thop`
- RL/infra: `ray[default] hydra-core omegaconf websockets msgpack peft datasets
  qwen-vl-utils gymnasium pandas tensorboard`
- sim: `robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 mujoco==2.3.0 termcolor numba
  pynput h5py scipy future cloudpickle matplotlib`

### flash_attn stub (important)
`wan_va` imports `flash_attn` at load time, but with `attn_mode=torch` it never
calls it. Install a stub so the import succeeds:
```bash
python - <<'PY'
import os, site
p = os.path.join(site.getsitepackages()[0], "flash_attn")
os.makedirs(p, exist_ok=True)
open(os.path.join(p, "__init__.py"), "w").write(
    "def flash_attn_func(*a, **k):\n    raise RuntimeError('flash_attn stub — use attn_mode=torch')\n"
)
PY
```

## Models / checkpoints (`$CKPT_ROOT`, default `/root/ckpts`)
The SFT checkpoint lives on a **private** HF repo. Fetch it once with **your own**
HF token (never commit a token; never write one to memory/files).

```bash
# Base model (public): VAE + text encoder + transformer (~23 GB)
huggingface-cli download robbyant/lingbot-va-base --local-dir /root/ckpts/base

# SFT checkpoint (PRIVATE — needs your token). Use subfolder checkpoint_step_3000_extra
# (config.json + diffusion_pytorch_model.safetensors, ~9.5 GB).
HF_TOKEN=<your-token> huggingface-cli download \
  kzrzhou/lingbot-va-libero-object-sft-0522 \
  --include "checkpoint_step_3000_extra/*" \
  --local-dir /root/ckpts/sft
```
`checkpoint_step_3000_extra` is the winning SFT (all 10 Libero-Object tasks
non-zero; ~70% mean SR at 1 ep/task in our smoke eval; `_extra` = a 1000-step
warm-start on the 4 weak tasks).

### The `base_sft` merged dir (required by distributed training)
Under multi-rank FSDP, `wan_va` shards the transformer **during build**, so
loading the SFT weights *after* the build via
`LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH` fails ("mixed Tensor and DTensor").
Fix: build a merged model dir whose `transformer/` already points at the SFT
weights, and set `LINGBOT_VA_MODEL_PATH=base_sft` (no separate transformer path):
```bash
mkdir -p /root/ckpts/base_sft
ln -s /root/ckpts/base/vae          /root/ckpts/base_sft/vae
ln -s /root/ckpts/base/text_encoder /root/ckpts/base_sft/text_encoder
ln -s /root/ckpts/base/tokenizer    /root/ckpts/base_sft/tokenizer
ln -s /root/ckpts/sft/checkpoint_step_3000_extra /root/ckpts/base_sft/transformer
```
(Eval, which runs single-GPU with no dist-init, loads fine via
`LINGBOT_VA_MODEL_PATH=base` + `LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH=<sft>`.)

## Environment variables
All repro scripts source `scripts/env.sh`, which sets these with overridable
defaults. The essential block:
```bash
export REPO_PATH=/workspace/RLinf
export LINGBOT_VA_REPO_PATH=/workspace/lingbot-va
export PYTHONPATH=$REPO_PATH:$LINGBOT_VA_REPO_PATH:/workspace/LIBERO
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export LINGBOT_VA_MODEL_PATH=/root/ckpts/base_sft   # training; =base for eval
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
```

## Container constraints learned the hard way (don't re-derive these)
1. **Collocated weight sync is unsupported here.** Collocated actor→rollout
   CUDA-IPC weight sync hits `pidfd_getfd: Operation not permitted` (the
   container's `ptrace_scope` is locked and cannot be relaxed). **Must run
   disaggregated** (actor and rollout on disjoint GPUs) so weight sync uses the
   NCCL ring path. The configs/scripts here already do this
   (`+weight_syncer.use_ring_sync=true`, placement actor 0-3 / rollout 4-7).
2. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` breaks legacy CUDA-IPC.**
   Harmless in disaggregated mode (ring sync, no IPC), so the training scripts
   set it for memory headroom. If you ever go collocated, you must remove it.
3. **Ray actors leak GPU memory between runs** → false OOM. Always fully clean
   up between runs:
   ```bash
   ray stop --force
   pkill -9 -f "ray::|train_embodied|raylet|gcs_server|CollectiveManager"
   nvidia-smi   # confirm ~0 MiB before relaunching
   ```
4. **Rendering:** `MUJOCO_GL=egl` works headless on A100.

Next: `REPRODUCE.md` for the end-to-end runbook.
