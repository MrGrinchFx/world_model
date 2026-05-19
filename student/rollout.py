"""Student open-loop rollout implementation."""

from __future__ import annotations

import torch

from wm_hw.model_utils import predict_next


def open_loop_rollout(model, states: torch.Tensor, actions: torch.Tensor, normalizer, warmup_steps: int, horizon: int):
    """Roll out `horizon` steps after a ground-truth warmup.

    Future ground-truth states after `warmup_steps` must not be read.
    """
    batch_size = states.shape[0]
    hidden = model.initial_hidden(batch_size, states.device)
    for t in range(int(warmup_steps)):
        _, hidden = predict_next(model, states[:, t], actions[:, t], hidden, normalizer)
    cur = states[:, int(warmup_steps)]
    preds = []
    for h in range(int(horizon)):
        cur, hidden = predict_next(model, cur, actions[:, int(warmup_steps) + h], hidden, normalizer)
        preds.append(cur)
    return torch.stack(preds, dim=1)
