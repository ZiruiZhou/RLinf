"""Standalone evaluation driver for LingBot-VA on Libero-Object.

This bypasses RLinf's multi-worker pipeline so we can validate the
integration end-to-end. It uses the same LingBot-VA adapter that the
huggingface_worker would instantiate.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.models import get_model


def build_libero_env(
    num_envs: int, seed: int = 0, task_id_filter: list[int] | None = None
) -> LiberoEnv:
    env_cfg = OmegaConf.create(
        {
            "env_type": "libero",
            "task_suite_name": "libero_object",
            "total_num_envs": num_envs,
            "auto_reset": True,
            # We need the env to stop the episode the moment the task succeeds
            # so we can detect it. With ignore_terminations=True the termination
            # signal is zeroed out and a successful pick can later be "undone"
            # if the robot keeps moving, producing false negatives.
            "ignore_terminations": False,
            "max_steps_per_rollout_epoch": 240,
            "max_episode_steps": 240,
            "use_rel_reward": False,
            "reward_coef": 1.0,
            "reset_gripper_open": True,
            "is_eval": True,
            "seed": seed,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": True,
            "specific_reset_id": None,
            "task_id_filter": task_id_filter,
            "video_cfg": OmegaConf.create(
                {
                    "save_video": False,
                    "info_on_video": False,
                    "video_base_dir": "/tmp/libero_videos",
                }
            ),
            "init_params": {"camera_heights": 128, "camera_widths": 128},
        }
    )
    return LiberoEnv(
        cfg=env_cfg,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def build_model(
    model_path: str, repo_path: str, transformer_state_dict_path: str | None = None
) -> torch.nn.Module:
    cfg = OmegaConf.create(
        {
            "model_type": "lingbotva",
            "model_path": model_path,
            "precision": "bf16",
            "num_action_chunks": int(os.environ.get("LINGBOT_VA_NUM_ACTION_CHUNKS", 16)),
            "action_dim": 7,
            "is_lora": False,
            "lora_rank": 4,
            "lora_path": None,
            "lingbotva": {
                "repo_path": repo_path,
                "config_name": "libero",
                "enable_offload": True,
                "action_per_frame": 4,
                "transformer_state_dict_path": transformer_state_dict_path,
                "save_root": "./runtime/lingbotva_libero_eval",
                "num_inference_steps": int(
                    os.environ.get("LINGBOT_VA_VIDEO_STEPS", 20)
                ),
                "action_num_inference_steps": int(
                    os.environ.get("LINGBOT_VA_ACTION_STEPS", 50)
                ),
                "enable_kv_cache_replay": os.environ.get(
                    "LINGBOT_VA_KV_REPLAY", "0"
                )
                == "1",
            },
        }
    )
    return get_model(cfg)


def _chunk_step_with_obs(env: LiberoEnv, chunk_actions: np.ndarray):
    """Run libero's chunk-step manually so we can capture per-step raw obs."""
    num_envs, chunk_size, _ = chunk_actions.shape
    raw_obs_history: list[list[dict]] = [[] for _ in range(num_envs)]
    obs_list = []
    rewards = []
    terms = []
    truncs = []
    for i in range(chunk_size):
        action = chunk_actions[:, i]
        wrapped, r, t, tr, _ = env.step(action, auto_reset=False)
        for env_idx in range(num_envs):
            raw_obs_history[env_idx].append(env.current_raw_obs[env_idx])
        obs_list.append(wrapped)
        rewards.append(r)
        terms.append(t)
        truncs.append(tr)
    rewards = torch.stack(rewards, dim=1)
    terms = torch.stack(terms, dim=1)
    truncs = torch.stack(truncs, dim=1)
    past_dones = (terms | truncs).any(dim=1)
    if past_dones.any() and env.auto_reset:
        obs_list[-1], _ = env._handle_auto_reset(
            past_dones.cpu().numpy(), obs_list[-1], {}
        )
    return obs_list[-1], rewards, terms, truncs, raw_obs_history, past_dones


def evaluate(
    model: torch.nn.Module,
    env: LiberoEnv,
    num_episodes: int,
) -> dict:
    """Run num_episodes per task and report Libero-Object success rate."""
    print("[lingbotva-libero-eval] resetting env...", flush=True)
    obs, _ = env.reset()
    print("[lingbotva-libero-eval] env reset complete", flush=True)
    total_steps = 0
    total_episodes_done = 0
    successes = 0
    failures = 0
    task_stats: dict[int, dict[str, int]] = {}
    cur_episode_task_ids = list(env.task_ids.copy())

    target_total_episodes = num_episodes
    max_steps_safety = num_episodes * 260

    chunk_idx = 0
    enable_kv_replay = getattr(model, "enable_kv_cache_replay", False)
    while total_episodes_done < target_total_episodes and total_steps < max_steps_safety:
        t0 = time.time()
        action_tensor, _ = model.predict_action_batch(obs, mode="eval")
        infer_sec = time.time() - t0

        t1 = time.time()
        chunk_actions = action_tensor.detach().cpu().numpy()
        final_obs, rewards, chunk_terminations, chunk_truncations, raw_obs_history, past_dones = (
            _chunk_step_with_obs(env, chunk_actions)
        )
        step_sec = time.time() - t1

        successes_this_chunk = chunk_terminations.any(dim=1)
        for env_idx in range(past_dones.shape[0]):
            if past_dones[env_idx].item():
                task_id = int(cur_episode_task_ids[env_idx])
                stats = task_stats.setdefault(task_id, {"success": 0, "total": 0})
                stats["total"] += 1
                if successes_this_chunk[env_idx].item():
                    stats["success"] += 1
                    successes += 1
                else:
                    failures += 1
                total_episodes_done += 1
                # next episode uses the freshly reset task id
                cur_episode_task_ids[env_idx] = int(env.task_ids[env_idx])
                # Drop history so the next episode starts as a first chunk.
                if hasattr(model, "reset_episode"):
                    model.reset_episode(env_idx)
            elif enable_kv_replay:
                state = model._episode_states.get(env_idx)
                if state is not None and state.prev_model_action is not None:
                    model.record_chunk_observations(
                        env_idx=env_idx,
                        chunk_obs_list=raw_obs_history[env_idx],
                        prev_model_action=state.prev_model_action,
                    )
        obs = final_obs
        total_steps += chunk_actions.shape[1]
        chunk_idx += 1
        print(
            f"  chunk={chunk_idx:3d} step={total_steps:4d} "
            f"infer={infer_sec:6.2f}s env={step_sec:5.2f}s "
            f"episodes={total_episodes_done} succ={successes} fail={failures}",
            flush=True,
        )

    total = successes + failures
    success_rate = successes / total if total > 0 else 0.0
    return {
        "success_rate": success_rate,
        "successes": successes,
        "failures": failures,
        "task_stats": {
            tid: {**stats, "rate": stats["success"] / stats["total"] if stats["total"] else 0.0}
            for tid, stats in task_stats.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "LINGBOT_VA_MODEL_PATH",
            "/workspace/zirui/models/lingbot-va-libero-hybrid",
        ),
    )
    parser.add_argument(
        "--repo-path",
        default=os.environ.get(
            "LINGBOT_VA_REPO_PATH", "/workspace/zirui/lingbot-va"
        ),
    )
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="If set, restrict the env to this libero_object task id (0-9).",
    )
    parser.add_argument(
        "--results-path",
        default="./runtime/lingbotva_libero_eval/results.json",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    task_id_filter = [args.task_id] if args.task_id is not None else None
    print(
        f"[lingbotva-libero-eval] num_envs={args.num_envs} num_episodes={args.num_episodes} "
        f"task_id_filter={task_id_filter}"
    )
    model = build_model(args.model_path, args.repo_path)
    model = model.cuda()
    env = build_libero_env(
        num_envs=args.num_envs, seed=args.seed, task_id_filter=task_id_filter
    )

    t0 = time.time()
    metrics = evaluate(model, env, num_episodes=args.num_episodes)
    elapsed = time.time() - t0
    metrics["elapsed_sec"] = elapsed
    metrics["num_envs"] = args.num_envs
    metrics["num_episodes"] = args.num_episodes

    Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_path).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
