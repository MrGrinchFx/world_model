"""Locked prediction helpers shared by training and evaluation."""

from __future__ import annotations

import torch

from .normalizer import Normalizer


def predict_next(model, obs: torch.Tensor, action: torch.Tensor, hidden, normalizer: Normalizer):
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(action)
    delta_norm, hidden = model(obs_norm, act_norm, hidden)
    delta = normalizer.denormalize_delta(delta_norm)
    return obs + delta, hidden
