from __future__ import annotations

from wm_hw.env import make_env


def test_inverted_pendulum_env_smoke():
    env = make_env(seed=0, reset_noise_scale=0.01)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,)
    assert env.action_space.shape == (1,)
    assert float(env.action_space.low[0]) == -3.0
    assert float(env.action_space.high[0]) == 3.0
    env.close()
