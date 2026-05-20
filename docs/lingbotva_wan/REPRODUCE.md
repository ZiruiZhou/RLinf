# Reproducing LingBot-VA × RLinf on LIBERO-Long

This guide takes you from a clean machine to a 10/10 success rate on LIBERO-Long with `robbyant/lingbot-va-posttrain-libero-long` running through RLinf's eval pipeline. Followed step by step on an A100-class GPU it takes ~2 hours (mostly checkpoint download + the 10-episode eval).

## 0. Prerequisites

- Linux x86_64 host with an NVIDIA GPU (≥40 GB VRAM recommended; tested on A100-SXM4-80GB).
- CUDA 12.x driver available on the host (driver only — wheels come from PyPI).
- Python 3.11 toolchain available somewhere `uv` can find (we install via `uv venv --python 3.11`).
- `/workspace` (or any directory) with ≥50 GB free for the two checkpoints and the venv.
- `gh` CLI optional, `git`/`curl`/`hf` (huggingface-cli) required.

## 1. Clone the repos

```bash
cd /workspace
# This fork's feature branch
git clone -b feat/lingbotva-wan-libero-eval https://github.com/ZiruiZhou/RLinf.git
# LingBot-VA source — needed for the wan_va Python package
git clone https://github.com/Robbyant/lingbot-va.git
```

Do NOT modify `lingbot-va`. The RLinf wrapper imports `VA_Server` via `sys.path` injection driven by the `LINGBOT_VA_REPO` environment variable.

## 2. Build the venv

A dedicated venv is required because `lingbot-va` pins `torch 2.9 / diffusers 0.36 / transformers ≥4.55` which conflict with RLinf's `openvla-oft` venv.

```bash
mkdir -p /workspace/venvs
uv venv --python 3.11 /workspace/venvs/lingbotva_wan
source /workspace/venvs/lingbotva_wan/bin/activate

# Torch: cu126 wheels top out at 2.6.0 but lingbot-va works fine there.
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu126

# Lingbot-VA Python deps (skip flash_attn — wrapper stubs it).
uv pip install diffusers==0.36.0 transformers==4.55.2 accelerate einops easydict \
  "numpy<2" opencv-python Pillow "imageio[ffmpeg]" safetensors ftfy matplotlib \
  tqdm tokenizers msgpack websockets

# RLinf infra (no torch override).
uv pip install ray hydra-core omegaconf wandb tensorboard PyYAML rich psutil \
  pynvml requests py-spy

# LIBERO sim deps + bddl + robomimic (NOT pulled by libero's setup.py).
uv pip install bddl==1.0.1 future cloudpickle thop robomimic \
  robosuite==1.4.1 mujoco "gym==0.26.2" dm_control timm peft

# Editable installs — no --deps so the lingbot-va pins are preserved.
uv pip install -e /workspace/RLinf --no-deps

# Clone LIBERO inside the venv directory (matches RLinf's openvla-oft layout).
git clone https://github.com/RLinf/LIBERO.git /workspace/venvs/lingbotva_wan/libero
uv pip install -e /workspace/venvs/lingbotva_wan/libero --no-deps
echo "export PYTHONPATH=/workspace/venvs/lingbotva_wan/libero:\$PYTHONPATH" \
  >> /workspace/venvs/lingbotva_wan/bin/activate
```

Re-source the venv to pick up the new `PYTHONPATH` export:

```bash
deactivate && source /workspace/venvs/lingbotva_wan/bin/activate
python -c "import torch, transformers, diffusers, libero, bddl; print('ok')"
```

## 3. Download the checkpoint and patch `attn_mode`

```bash
mkdir -p /workspace/ckpts
hf download robbyant/lingbot-va-posttrain-libero-long \
  --local-dir /workspace/ckpts/lingbotva_libero_long

# The shipped config sets attn_mode=flex (used for training) which crashes
# at inference. Switch to torch SDPA.
python -c "
import json
p = '/workspace/ckpts/lingbotva_libero_long/transformer/config.json'
c = json.load(open(p))
c['attn_mode'] = 'torch'
json.dump(c, open(p, 'w'), indent=2)
print('attn_mode patched ->', c['attn_mode'])
"
```

## 4. Start Ray in a low-resource mode (avoids container PID cap)

```bash
ray stop --force 2>/dev/null
ray start --head --num-cpus=4 --port=6379 --include-dashboard=false
```

Background: the host container in our setup has `/sys/fs/cgroup/pids.max=3840`. Ray's default raylet spawns 15 prestart Python workers, each with many gRPC threads, which exhausts the PID budget and the workers crash with `pthread_create failed: Resource temporarily unavailable`. Limiting Ray to 4 CPUs and disabling the dashboard keeps the PID count under ~1500 throughout the eval.

## 5. Run the eval

```bash
source /workspace/venvs/lingbotva_wan/bin/activate
cd /workspace/RLinf

export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export REPO_PATH=/workspace/RLinf
export PYTHONPATH=/workspace/RLinf:$PYTHONPATH
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HYDRA_FULL_ERROR=1
export RLINF_NODE_RANK=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LINGBOT_VA_REPO=/workspace/lingbot-va

LOG_DIR=/workspace/RLinf/logs/m1_eval_10ep
mkdir -p "$LOG_DIR"

python examples/embodiment/eval_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config/ \
  --config-name libero_10_lingbotva_wan_eval \
  runner.logger.log_path=$LOG_DIR \
  runner.logger.experiment_name=m1_eval_10ep \
  actor.model.enable_offload=False \
  actor.model.num_inference_steps=20 \
  actor.model.action_num_inference_steps=50 \
  env.eval.max_episode_steps=800 \
  env.eval.max_steps_per_rollout_epoch=800 \
  env.eval.use_ordered_reset_state_ids=False \
  algorithm.eval_rollout_epoch=10
```

Why the overrides:

- `actor.model.enable_offload=False`: keeps the UMT5 text encoder on the GPU. With offload, T5 encoding runs on CPU and takes ~5 minutes per `_reset`, hanging the eval. We have plenty of VRAM (~24 GB peak with everything on GPU).
- `num_inference_steps=20`, `action_num_inference_steps=50`: the lingbot-va repo's libero defaults.
- `max_episode_steps=800`, `max_steps_per_rollout_epoch=800`: matches the lingbot-va eval client.
- `use_ordered_reset_state_ids=False`: random sampling across the 10 LIBERO-Long tasks so a small eval covers diverse tasks instead of cycling all 50 init states of task 0 first.
- `eval_rollout_epoch=10`: 10 trajectories. Bump to 100+ for tighter statistics.

## 6. Expected result

After ~60 minutes (≈6 min/episode at default inference steps) you should see:

```
[INFO ... RLinf] {'eval/success_once': 1.0,
                  'eval/success_at_end': 1.0,
                  'eval/return': 5.0,
                  'eval/episode_len': 800.0,
                  'eval/reward': 0.0177,
                  'eval/num_trajectories': 10}
```

`success_once: 1.0` is a mean over the 10 trajectories → 10/10 = 100%. This matches the LingBot-VA repo's README result of 98.5% over 500 episodes (10 episodes is statistically noisy but consistent within that envelope).

Videos for each trajectory are saved to `${LOG_DIR}/video/eval/seed_0/<idx>.mp4`. TensorBoard events under `${LOG_DIR}/tensorboard/`.

## 7. Sanity checks if something's off

| Symptom | Likely cause | Fix |
|---|---|---|
| Hangs in T5 encode (`_get_t5_prompt_embeds`) for minutes | `enable_offload=True` puts UMT5 on CPU | Set `actor.model.enable_offload=False` |
| `pthread_create failed: Resource temporarily unavailable` | Ray's default workers hit container PID cap | Pre-start Ray with `--num-cpus=4 --include-dashboard=false` |
| `ModuleNotFoundError: bddl` | LIBERO setup.py has empty install_requires | `uv pip install bddl==1.0.1` |
| `ModuleNotFoundError: flash_attn` from `diffusers/utils/import_utils.py` | venv was created with old wrapper | Rebuild venv off this branch; wrapper stubs the module with proper `__spec__` |
| `attn_mode==flex` error at first infer | Forgot to patch the checkpoint config | Re-run the `attn_mode` patch in §3 |
| `eval/success_once: 0.0` | `_chunk_main_keyframes` / `_elapsed_steps` not reaching the wrapper | Verify `rlinf/data/embodied_io_struct.py` allowlist includes the three underscore-prefixed keys |
| `AssertionError: List field '_chunk_obs_list' expected length 1, got 16` | An older version of `env_evaluate_step` is on path | Stash + checkout `feat/lingbotva-wan-libero-eval` fresh; the keyframes are sent as tensors now, not as a raw list of dicts |

## 8. Standalone wrapper smoke (no Ray, no LIBERO)

For diagnosing wrapper construction in isolation:

```bash
LINGBOT_VA_REPO=/workspace/lingbot-va \
  python scripts/m1_test_wrapper_import.py
```

Expected last line: `[smoke] PASS`.
