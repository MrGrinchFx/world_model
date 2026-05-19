from __future__ import annotations

from wm_hw.config import load_config


def test_locked_official_eval_config():
    cfg = load_config("configs/official_eval.yaml")["eval"]
    assert cfg["warmup_steps"] == 10
    assert cfg["max_horizon"] == 1000
    assert cfg["report_horizons"] == [1, 5, 10, 90, 100, 200, 500, 1000]
    assert cfg["primary_metric"] == "VPT80@0.25"
