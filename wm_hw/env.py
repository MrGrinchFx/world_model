"""Locked Gymnasium InvertedPendulum wrapper."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


ENV_ID = "InvertedPendulum-v5"
OBS_DIM = 4
ACT_DIM = 1
ACTION_LOW = -3.0
ACTION_HIGH = 3.0


def make_env(seed: int | None = None, reset_noise_scale: float = 0.01, max_episode_steps: int | None = None):
    kwargs = {"reset_noise_scale": float(reset_noise_scale)}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    env = gym.make(ENV_ID, **kwargs)
    if seed is not None:
        env.reset(seed=int(seed))
        env.action_space.seed(int(seed))
    return env


def reset_env(env, seed: int | None = None) -> np.ndarray:
    obs, _info = env.reset(seed=seed)
    return np.asarray(obs, dtype=np.float32)


def step_env(env, action: np.ndarray) -> tuple[np.ndarray, bool]:
    action = np.asarray(action, dtype=np.float32).reshape(ACT_DIM)
    obs, _reward, terminated, truncated, _info = env.step(action)
    return np.asarray(obs, dtype=np.float32), bool(terminated or truncated)


def clip_action(action: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(action, dtype=np.float32).reshape(ACT_DIM), ACTION_LOW, ACTION_HIGH)
