"""Locked SmallWorlds-style nMSE/VPT metrics for the public scoreboard."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch


DEFAULT_REPORT_HORIZONS = (1, 5, 10, 90, 100, 200, 500, 1000)
DEFAULT_VPT_THRESHOLDS = (0.10, 0.25, 0.50)


def _as_list(values: Iterable[int | float] | str | None, *, cast):
    if values is None:
        return None
    if isinstance(values, str):
        return [cast(part) for part in values.split(",") if part.strip()]
    return [cast(value) for value in values]


def normalized_mse_curve(preds: torch.Tensor, targets: torch.Tensor, obs_std: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-window/per-step nMSE and rollout-average nMSE curve."""
    pred_t = torch.as_tensor(preds, dtype=torch.float32)
    target_t = torch.as_tensor(targets, dtype=torch.float32, device=pred_t.device)
    std_t = torch.as_tensor(obs_std, dtype=pred_t.dtype, device=pred_t.device).clamp_min(1e-6)
    per_window_step = torch.mean(((pred_t - target_t) / std_t) ** 2, dim=-1)
    steps = torch.arange(1, per_window_step.shape[1] + 1, dtype=per_window_step.dtype, device=per_window_step.device)
    per_window_rollout_mean = torch.cumsum(per_window_step, dim=1) / steps
    curve = torch.mean(per_window_rollout_mean, dim=0)
    return per_window_step, curve


def vpt_at_threshold(per_window_step_nmse: torch.Tensor, threshold: float, percentile: float) -> tuple[int, np.ndarray]:
    """Valid prediction time: largest h where `percentile` of windows survive."""
    valid = torch.as_tensor(per_window_step_nmse <= float(threshold), dtype=torch.bool)
    survival = []
    for row in valid.detach().cpu().numpy():
        bad = np.flatnonzero(~row)
        survival.append(int(bad[0]) if len(bad) else int(row.shape[0]))
    survival_np = np.asarray(survival, dtype=np.int32)
    horizon = int(valid.shape[1])
    rates = [(survival_np >= h).mean() for h in range(1, horizon + 1)]
    ok = [i + 1 for i, rate in enumerate(rates) if rate >= float(percentile)]
    return int(max(ok) if ok else 0), survival_np


def compute_official_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    normalizer,
    *,
    report_horizons: Iterable[int] | str | None = None,
    vpt_thresholds: Iterable[float] | str | None = None,
) -> dict[str, Any]:
    """Compute nMSE curve, AUC, and VPT scoreboard metrics."""
    per_window_step, curve = normalized_mse_curve(preds, targets, normalizer.obs_std)
    step_curve = torch.mean(per_window_step, dim=0)
    horizon = int(curve.shape[0])
    requested_horizons = _as_list(report_horizons, cast=int) or list(DEFAULT_REPORT_HORIZONS)
    requested_horizons = sorted({h for h in requested_horizons if 1 <= h <= horizon})
    requested_horizons.append(horizon)
    report = sorted(set(requested_horizons))
    thresholds = _as_list(vpt_thresholds, cast=float) or list(DEFAULT_VPT_THRESHOLDS)
    if not any(abs(float(value) - 0.25) < 1e-9 for value in thresholds):
        thresholds.append(0.25)
    thresholds = sorted(set(float(value) for value in thresholds))

    metrics: dict[str, Any] = {
        "max_horizon": horizon,
        "nMSE_AUC": float(torch.mean(curve).detach().cpu()),
        "nMSE_curve": curve.detach().cpu().numpy().astype(np.float32).tolist(),
        "step_nMSE_curve": step_curve.detach().cpu().numpy().astype(np.float32).tolist(),
    }
    for h in report:
        metrics[f"nMSE@{h}"] = float(curve[h - 1].detach().cpu())
        metrics[f"step_nMSE@{h}"] = float(step_curve[h - 1].detach().cpu())

    primary_survival = None
    for threshold in thresholds:
        vpt80, survival = vpt_at_threshold(per_window_step, threshold, 0.80)
        vpt50, _ = vpt_at_threshold(per_window_step, threshold, 0.50)
        suffix = f"{threshold:.2f}"
        metrics[f"VPT80@{suffix}"] = int(vpt80)
        metrics[f"VPT50@{suffix}"] = int(vpt50)
        if abs(float(threshold) - 0.25) < 1e-9:
            primary_survival = survival

    if primary_survival is None:
        _, primary_survival = vpt_at_threshold(per_window_step, thresholds[0], 0.80)
    metrics["primary_metric"] = "VPT80@0.25"
    metrics["per_window_vpt@0.25"] = primary_survival.astype(np.int32)
    return metrics
