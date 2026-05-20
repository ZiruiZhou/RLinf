"""Eval-only RLinf wrapper around wan_va.wan_va_server.VA_Server (LingBot-VA Wan2.2).

VA_Server keeps per-episode state (prompt embeddings + transformer KV cache).
For milestone 1 we constrain `env.eval.total_num_envs == 1` so a single server
instance is sufficient. Vectorized eval requires per-env servers and is deferred.

VA_Server.infer() lifecycle per episode (mirrors lingbot-va/evaluation/libero/client.py):
  1. infer({"reset": True, "prompt": task_lang})                    -- episode start
  2. infer({"obs": obs_dict, "prompt": task_lang})                  -- predict 4x4=16 actions
  3. infer({"obs": [obs_dict], "compute_kv_cache": True,            -- push executed
            "imagine": False, "state": action_tensor})                 chunk into cache
Repeat 2-3 until done or step budget exhausted.

Action tensor returned by step 2 has shape `[7, frame_chunk(4), action_per_frame(4)]`
after postprocess_action (7 used channels of the 30-D Wan action space). We flatten
the last two dims to expose `num_action_chunks=16` actions of dim 7 to RLinf's env
worker (matches the `chunk_step` contract `[num_envs, chunk, action_dim]`).
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy

LINGBOT_VA_REPO = os.environ.get("LINGBOT_VA_REPO", "/workspace/lingbot-va")


def _stub_flash_attn():
    """lingbot-va/wan_va/modules/model.py does an unconditional `from flash_attn
    import flash_attn_func` at module load, AND diffusers calls
    importlib.util.find_spec("flash_attn") (needs a proper ModuleSpec, not just a
    sys.modules entry). We use attn_mode="torch" everywhere so the function is
    never actually called -- install a stub that satisfies both."""
    import importlib.machinery
    import types

    if "flash_attn" in sys.modules:
        return

    m = types.ModuleType("flash_attn")
    m.__version__ = "0.0.0+stub"
    m.__spec__ = importlib.machinery.ModuleSpec(
        name="flash_attn", loader=None, is_package=False
    )

    def _unavailable(*args, **kwargs):
        raise RuntimeError(
            "flash_attn is stubbed in this venv. Make sure attn_mode='torch' "
            "is set so this code path is not exercised."
        )

    m.flash_attn_func = _unavailable
    sys.modules["flash_attn"] = m


def _import_wan_va():
    """Import VA_Server from the cloned lingbot-va repo (its modules use bare
    `from configs import ...` / `from modules.utils import ...` imports that
    only resolve when wan_va/ is on sys.path)."""
    _stub_flash_attn()
    wan_va_pkg = os.path.join(LINGBOT_VA_REPO, "wan_va")
    if wan_va_pkg not in sys.path:
        sys.path.insert(0, wan_va_pkg)
    from configs import VA_CONFIGS  # noqa: E402
    from wan_va_server import VA_Server  # noqa: E402

    return VA_Server, VA_CONFIGS


class LingbotvaWanActionModel(nn.Module, BasePolicy):
    def __init__(self, cfg, torch_dtype):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.torch_dtype = torch_dtype

        VA_Server, VA_CONFIGS = _import_wan_va()

        # Use libero config from lingbot-va as the base; override model path/dtype.
        job_cfg = copy.deepcopy(VA_CONFIGS["libero"])
        job_cfg.wan22_pretrained_model_name_or_path = cfg.model_path
        job_cfg.save_root = cfg.get("save_root", "/tmp/lingbotva_wan_save")
        job_cfg.param_dtype = torch_dtype
        job_cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        job_cfg.enable_offload = bool(cfg.get("enable_offload", True))
        # Optional overrides via Hydra config.
        for key in (
            "num_inference_steps",
            "action_num_inference_steps",
            "video_exec_step",
            "frame_chunk_size",
            "action_per_frame",
            "height",
            "width",
            "guidance_scale",
            "action_guidance_scale",
        ):
            if cfg.get(key, None) is not None:
                setattr(job_cfg, key, cfg.get(key))

        os.makedirs(job_cfg.save_root, exist_ok=True)

        self.server = VA_Server(job_cfg)
        self.job_cfg = job_cfg

        # RLinf bookkeeping
        self.action_dim = int(cfg.get("action_dim", 7))
        # frame_chunk * action_per_frame -> per-call action count
        self.num_action_chunks = int(
            cfg.get(
                "num_action_chunks",
                self.job_cfg.frame_chunk_size * self.job_cfg.action_per_frame,
            )
        )

        self._last_prompt = None
        self._last_action = None  # last raw [7, F, H] action tensor (numpy)
        self._last_elapsed_steps = None  # int, for intra-task reset detection

    # ------------------------------------------------------------------
    # Observation translation
    # ------------------------------------------------------------------
    def _to_wan_obs(self, env_obs):
        """RLinf libero env yields:
            main_images:  [num_envs, H, W, 3] tensor (already rotated 180 deg)
            wrist_images: [num_envs, H, W, 3] tensor (already rotated 180 deg)
            task_descriptions: list[str] (len=num_envs)
        lingbot-va expects per-cam keys 'observation.images.agentview_rgb' and
        '...eye_in_hand_rgb' as HxWx3 uint8 numpy arrays. RLinf does [::-1,::-1]
        (rotate 180) but lingbot-va training used only [::-1] (vertical flip),
        so we undo the extra horizontal flip via [:, ::-1, :].
        """
        return self._single_to_wan_obs(env_obs["main_images"][0], env_obs["wrist_images"][0])

    @staticmethod
    def _single_to_wan_obs(main, wrist):
        if torch.is_tensor(main):
            main = main.cpu().numpy()
        if torch.is_tensor(wrist):
            wrist = wrist.cpu().numpy()
        # Undo RLinf's horizontal flip to match lingbot-va training preprocessing.
        main = np.ascontiguousarray(main[:, ::-1, :])
        wrist = np.ascontiguousarray(wrist[:, ::-1, :])
        return {
            "observation.images.agentview_rgb": main,
            "observation.images.eye_in_hand_rgb": wrist,
        }

    def _keyframes_from_env_obs(self, env_obs):
        """Build the list of per-step obs for compute_kv_cache.

        The env_worker attaches `_chunk_main_keyframes` and
        `_chunk_wrist_keyframes` as tensors of shape `[num_envs, T, H, W, C]`
        where T == chunk_size (= 16 for libero @ num_action_chunks=16). This
        matches `lingbot-va/evaluation/libero/client.py:124`, which passes one
        obs per env-step in `key_frame_list` (since `action_per_frame=1` in the
        client convention). If the tensors aren't surfaced (different env or
        older worker), fall back to replicating the final obs `frame_chunk_size`
        times.
        """
        main_kf = env_obs.get("_chunk_main_keyframes", None)
        wrist_kf = env_obs.get("_chunk_wrist_keyframes", None)
        if main_kf is None or wrist_kf is None:
            return [self._to_wan_obs(env_obs)] * int(self.job_cfg.frame_chunk_size)
        T = main_kf.shape[1]
        keyframes = []
        for t in range(T):
            keyframes.append(self._single_to_wan_obs(main_kf[0, t], wrist_kf[0, t]))
        return keyframes

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_action_batch(self, env_obs, mode="eval", **kwargs):
        if mode != "eval":
            raise NotImplementedError(
                "lingbotva_wan is eval-only for milestone 1. mode={!r}".format(mode)
            )

        num_envs = (
            env_obs["main_images"].shape[0]
            if torch.is_tensor(env_obs["main_images"])
            else len(env_obs["main_images"])
        )
        if num_envs != 1:
            raise RuntimeError(
                f"lingbotva_wan wraps a stateful VA_Server; expects num_envs=1 (got {num_envs}). "
                "Reduce env.eval.total_num_envs to 1."
            )

        prompt = env_obs["task_descriptions"][0]
        wan_obs = self._to_wan_obs(env_obs)

        # Detect episode boundary. Two reset triggers:
        #   (a) prompt changed -> new task entirely
        #   (b) elapsed_steps decreased -> same task, fresh episode after auto_reset
        elapsed = env_obs.get("_elapsed_steps", None)
        elapsed_now = int(elapsed[0].item()) if elapsed is not None and torch.is_tensor(elapsed) else (int(elapsed[0]) if elapsed is not None else None)

        is_reset = False
        if prompt != self._last_prompt:
            is_reset = True
        elif elapsed_now is not None and self._last_elapsed_steps is not None:
            if elapsed_now < self._last_elapsed_steps:
                is_reset = True

        if is_reset:
            self.server.infer({"reset": True, "prompt": prompt})
            self._last_prompt = prompt
            self._last_action = None
        elif self._last_action is not None:
            # Push the previously-executed action chunk + per-frame keyframes
            # into the KV cache (mirrors evaluation/libero/client.py:124).
            keyframes = self._keyframes_from_env_obs(env_obs)
            self.server.infer(
                {
                    "obs": keyframes,
                    "compute_kv_cache": True,
                    "imagine": False,
                    "state": self._last_action,
                }
            )

        ret = self.server.infer({"obs": wan_obs, "prompt": prompt})
        action = ret["action"]  # numpy: [7, frame_chunk(4), action_per_frame(4)]
        self._last_action = action
        self._last_elapsed_steps = elapsed_now

        # [C=7, F, H] -> [num_envs=1, F*H, C=7] for RLinf chunk_step contract.
        action_chunk = np.transpose(action, (1, 2, 0)).reshape(
            1, self.num_action_chunks, self.action_dim
        ).astype(np.float32, copy=False)

        return action_chunk, {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {},
        }

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "lingbotva_wan wraps VA_Server (no autograd). Training is not implemented; "
            "use only with runner.only_eval: True."
        )

    # RLinf calls model.to(device) after construction. VA_Server already manages
    # device placement internally, so make this a no-op.
    def to(self, *args, **kwargs):
        return self

    # Disable enable_offload by RLinf rollout worker (VA_Server has its own offload).
    def cpu(self):
        return self
