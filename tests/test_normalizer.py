from __future__ import annotations

import numpy as np

from wm_hw.normalizer import Normalizer


def test_normalizer_shapes_and_roundtrip():
    warmup = 5
    horizon = 37
    states = np.random.randn(5, warmup + horizon + 1, 4).astype(np.float32)
    actions = np.random.randn(5, warmup + horizon, 1).astype(np.float32)
    norm = Normalizer.from_train(states, actions)
    assert norm.obs_mean.shape == (4,)
    assert norm.act_mean.shape == (1,)
    assert norm.delta_mean.shape == (4,)
    restored = Normalizer.from_dict(norm.to_dict())
    np.testing.assert_allclose(restored.obs_mean, norm.obs_mean)
