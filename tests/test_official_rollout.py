from __future__ import annotations

import numpy as np
import torch

from wm_hw.normalizer import Normalizer
from wm_hw.official_rollout import official_open_loop_rollout


class ActionOnlyModel(torch.nn.Module):
    def initial_hidden(self, batch_size, device):
        return None

    def forward(self, obs_norm, act_norm, hidden=None):
        delta = torch.cat([act_norm, torch.zeros(obs_norm.shape[0], 3, device=obs_norm.device)], dim=-1)
        return delta, hidden


def test_official_rollout_no_future_state_leak_and_off_by_one():
    warmup = 10
    horizon = 4
    states = torch.zeros(2, warmup + horizon + 1, 4)
    actions = torch.ones(2, warmup + horizon, 1) * 0.1
    states[:, warmup + 1 :] = 999.0
    norm = Normalizer(
        obs_mean=np.zeros(4, dtype=np.float32),
        obs_std=np.ones(4, dtype=np.float32),
        act_mean=np.zeros(1, dtype=np.float32),
        act_std=np.ones(1, dtype=np.float32),
        delta_mean=np.zeros(4, dtype=np.float32),
        delta_std=np.ones(4, dtype=np.float32),
    )
    preds = official_open_loop_rollout(ActionOnlyModel(), states, actions, norm, warmup_steps=warmup, horizon=horizon)
    assert preds.shape == (2, horizon, 4)
    assert torch.allclose(preds[:, :, 0], torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]))
    assert torch.all(preds[..., 0] < 999.0)
