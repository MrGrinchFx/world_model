from __future__ import annotations

import numpy as np
import torch

from wm_hw.train import _batch


def test_training_batch_uses_random_subwindow():
    data = {
        "states": np.zeros((3, 1011, 4), dtype=np.float32),
        "actions": np.zeros((3, 1010, 1), dtype=np.float32),
    }
    batch = _batch(data, np.array([0, 1]), torch.device("cpu"), sequence_length=64, rng=np.random.default_rng(0))
    assert batch["states"].shape == (2, 65, 4)
    assert batch["actions"].shape == (2, 64, 1)
