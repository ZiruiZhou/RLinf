"""Tiny smoke test for the lingbotva_wan wrapper. Verifies that the model can be
constructed against the downloaded checkpoint and that one predict_action_batch
call succeeds, without going through Ray / EnvWorker / EvalRunner.

Run inside the lingbotva_wan venv:
    LINGBOT_VA_REPO=/workspace/lingbot-va python scripts/m1_test_wrapper_import.py
"""

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/workspace/RLinf")

from rlinf.models.embodiment.lingbotva_wan import get_model  # noqa: E402

cfg = OmegaConf.create(
    {
        "model_path": "/workspace/ckpts/lingbotva_libero_long",
        "model_type": "lingbotva_wan",
        "precision": "bf16",
        "action_dim": 7,
        "num_action_chunks": 16,
        "add_value_head": False,
        "is_lora": False,
        "lora_path": None,
        "enable_offload": True,
        "frame_chunk_size": 4,
        "action_per_frame": 4,
        # use shorter inference steps for the smoke test
        "num_inference_steps": 4,
        "action_num_inference_steps": 8,
    }
)

print("[smoke] constructing LingbotvaWanActionModel ...", flush=True)
model = get_model(cfg, torch_dtype=torch.bfloat16)
print("[smoke] constructed.", flush=True)

# Fabricate a 1-env LIBERO-style obs dict.
H, W = 256, 256
env_obs = {
    "main_images": torch.zeros((1, H, W, 3), dtype=torch.uint8),
    "wrist_images": torch.zeros((1, H, W, 3), dtype=torch.uint8),
    "states": torch.zeros((1, 7), dtype=torch.float32),
    "task_descriptions": ["pick up the apple"],
}

print("[smoke] calling predict_action_batch ...", flush=True)
actions, info = model.predict_action_batch(env_obs, mode="eval")
print(f"[smoke] actions shape: {actions.shape} dtype: {actions.dtype}")
print(f"[smoke] sample: {actions[0, 0]}")
assert actions.shape == (1, 16, 7), actions.shape
print("[smoke] PASS")
