from __future__ import annotations

import numpy as np
import pytest
import torch

from wm_hw.normalizer import Normalizer
from wm_hw.official_metrics import compute_official_metrics


def _normalizer():
    return Normalizer(
        obs_mean=np.zeros(4, dtype=np.float32),
        obs_std=np.ones(4, dtype=np.float32),
        act_mean=np.zeros(1, dtype=np.float32),
        act_std=np.ones(1, dtype=np.float32),
        delta_mean=np.zeros(4, dtype=np.float32),
        delta_std=np.ones(4, dtype=np.float32),
    )


def test_official_nmse_and_vpt_metrics():
    preds = torch.zeros(4, 12, 4)
    targets = torch.zeros_like(preds)
    preds[0, 4:, :] = 1.0
    metrics = compute_official_metrics(preds, targets, _normalizer(), report_horizons=[1, 5, 10], vpt_thresholds=[0.25])
    assert metrics["nMSE@1"] == 0.0
    assert metrics["nMSE@5"] == pytest.approx(0.05)
    assert metrics["step_nMSE@5"] == pytest.approx(0.25)
    assert metrics["VPT80@0.25"] == 4
    assert metrics["VPT50@0.25"] == 12
    assert metrics["per_window_vpt@0.25"].shape == (4,)
    assert "nMSE@1000" not in metrics


def test_official_vpt_drops_when_many_windows_cross_threshold():
    preds = torch.zeros(5, 10, 4)
    targets = torch.zeros_like(preds)
    preds[:2, 3:, :] = 1.0
    metrics = compute_official_metrics(preds, targets, _normalizer(), report_horizons=[10], vpt_thresholds=[0.25])
    assert metrics["VPT80@0.25"] == 3
    assert metrics["VPT50@0.25"] == 10


def test_official_nmse_rollout_average_differs_from_step_nmse():
    preds = torch.zeros(1, 4, 4)
    targets = torch.zeros_like(preds)
    preds[:, 3, :] = 2.0
    metrics = compute_official_metrics(preds, targets, _normalizer(), report_horizons=[4], vpt_thresholds=[0.25])
    assert metrics["step_nMSE@4"] == pytest.approx(4.0)
    assert metrics["nMSE@4"] == pytest.approx(1.0)
