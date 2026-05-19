"""Train-split normalization for observations, actions, and deltas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch


EPS = 1e-6


@dataclass
class Normalizer:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray
    delta_mean: np.ndarray
    delta_std: np.ndarray

    @classmethod
    def from_train(cls, states: np.ndarray, actions: np.ndarray) -> "Normalizer":
        obs = states[:, :-1].reshape(-1, states.shape[-1]).astype(np.float32)
        act = actions.reshape(-1, actions.shape[-1]).astype(np.float32)
        delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1]).astype(np.float32)
        return cls(
            obs_mean=obs.mean(axis=0),
            obs_std=obs.std(axis=0) + EPS,
            act_mean=act.mean(axis=0),
            act_std=act.std(axis=0) + EPS,
            delta_mean=delta.mean(axis=0),
            delta_std=delta.std(axis=0) + EPS,
        )

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self._t(self.obs_mean, obs)) / self._t(self.obs_std, obs)

    def normalize_act(self, act: torch.Tensor) -> torch.Tensor:
        return (act - self._t(self.act_mean, act)) / self._t(self.act_std, act)

    def normalize_delta(self, delta: torch.Tensor) -> torch.Tensor:
        return (delta - self._t(self.delta_mean, delta)) / self._t(self.delta_std, delta)

    def denormalize_delta(self, delta_norm: torch.Tensor) -> torch.Tensor:
        return delta_norm * self._t(self.delta_std, delta_norm) + self._t(self.delta_mean, delta_norm)

    def _t(self, array: np.ndarray, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(array, dtype=like.dtype, device=like.device)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "obs_mean": self.obs_mean.tolist(),
            "obs_std": self.obs_std.tolist(),
            "act_mean": self.act_mean.tolist(),
            "act_std": self.act_std.tolist(),
            "delta_mean": self.delta_mean.tolist(),
            "delta_std": self.delta_std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> "Normalizer":
        return cls(**{key: np.asarray(value, dtype=np.float32) for key, value in payload.items()})

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
