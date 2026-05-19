"""Locked open-loop rollout used by official evaluation."""

from __future__ import annotations

import torch

from .model_utils import predict_next


@torch.no_grad()
def official_open_loop_rollout(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    *,
    warmup_steps: int,
    horizon: int,
) -> torch.Tensor:
    """Predict `horizon` steps after ground-truth warmup without future leakage."""
    warmup_steps = int(warmup_steps)
    horizon = int(horizon)
    if states.ndim != 3 or actions.ndim != 3:
        raise ValueError("states/actions must be [B, T, D] tensors.")
    if states.shape[1] < warmup_steps + horizon + 1:
        raise ValueError("states tensor is too short for requested warmup/horizon.")
    if actions.shape[1] < warmup_steps + horizon:
        raise ValueError("actions tensor is too short for requested warmup/horizon.")

    hidden = model.initial_hidden(states.shape[0], states.device)
    for t in range(warmup_steps):
        _, hidden = predict_next(model, states[:, t], actions[:, t], hidden, normalizer)

    cur = states[:, warmup_steps]
    preds = []
    for h in range(horizon):
        cur, hidden = predict_next(model, cur, actions[:, warmup_steps + h], hidden, normalizer)
        preds.append(cur)
    return torch.stack(preds, dim=1)
