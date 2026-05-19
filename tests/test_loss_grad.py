from __future__ import annotations

import numpy as np
import torch

from student.losses import compute_loss
from student.model import StudentWorldModel
from wm_hw.normalizer import Normalizer


def test_loss_backward_has_gradients():
    states = torch.randn(4, 12, 4)
    actions = torch.randn(4, 11, 1)
    norm = Normalizer.from_train(states.numpy(), actions.numpy())
    model = StudentWorldModel(hidden_dim=32)
    cfg = {"loss": {"one_step_weight": 1.0, "rollout_weight": 0.3, "rollout_train_horizon": 5}, "eval": {"warmup_steps": 5}}
    loss, metrics = compute_loss(model, {"states": states, "actions": actions}, norm, cfg)
    loss.backward()
    assert "loss/rollout" in metrics
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
