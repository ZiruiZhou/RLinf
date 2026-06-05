#!/bin/bash
# Shared environment block for all LingBot-VA GRPO repro scripts.
# Source this from the other scripts: `source "$(dirname "$0")/env.sh"`.
#
# Every path is overridable from the caller's environment; the defaults match
# the development container documented in ENV_PREP.md (8x A100-80GB).
#
# NOTE: nothing secret lives here. The SFT checkpoint is on a *private* HF repo;
# fetch it once with your own token (see ENV_PREP.md) — do NOT bake a token in.

# --- repos -------------------------------------------------------------------
export REPO_PATH="${REPO_PATH:-/workspace/RLinf}"
export LINGBOT_VA_REPO_PATH="${LINGBOT_VA_REPO_PATH:-/workspace/lingbot-va}"
export LIBERO_PATH="${LIBERO_PATH:-/workspace/LIBERO}"
export EMBODIED_PATH="${EMBODIED_PATH:-$REPO_PATH/examples/embodiment}"
export PYTHONPATH="$REPO_PATH:$LINGBOT_VA_REPO_PATH:$LIBERO_PATH"

# --- checkpoints / models ----------------------------------------------------
# CKPT_ROOT holds: base/ (robbyant/lingbot-va-base), sft/checkpoint_step_3000_extra
# (the private SFT transformer), and base_sft/ (a merged dir: vae/text_encoder/
# tokenizer symlinked to base, transformer symlinked to the SFT checkpoint).
export CKPT_ROOT="${CKPT_ROOT:-/root/ckpts}"
export BASE_MODEL="${BASE_MODEL:-$CKPT_ROOT/base}"                       # full base (VAE+TE+transformer)
export SFT_TRANSFORMER="${SFT_TRANSFORMER:-$CKPT_ROOT/sft/checkpoint_step_3000_extra}"
export BASE_SFT="${BASE_SFT:-$CKPT_ROOT/base_sft}"                       # merged base+SFT (see ENV_PREP.md)

# --- python interpreter ------------------------------------------------------
export PY="${PY:-/venv/main/bin/python3.10}"

# --- runtime knobs the env needs --------------------------------------------
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Where training/eval logs and checkpoints go (override for a persistent disk).
export LOG_ROOT="${LOG_ROOT:-/tmp}"

cd "$REPO_PATH"
