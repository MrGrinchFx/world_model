"""Locked official nMSE/VPT scoreboard evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .config import load_config, save_json
from .dataset import load_metadata, load_split, validate_split_against_metadata
from .horizon import resolve_eval_horizon
from .model_utils import predict_next
from .normalizer import Normalizer
from .official_metrics import compute_official_metrics
from .official_rollout import official_open_loop_rollout


@torch.no_grad()
def _one_step_rmse(model, states: torch.Tensor, actions: torch.Tensor, normalizer: Normalizer, *, max_action_steps: int | None = None) -> float:
    hidden = model.initial_hidden(states.shape[0], states.device)
    preds = []
    steps = actions.shape[1] if max_action_steps is None else min(int(max_action_steps), actions.shape[1])
    for t in range(steps):
        pred, hidden = predict_next(model, states[:, t], actions[:, t], hidden, normalizer)
        preds.append(pred)
    pred_t = torch.stack(preds, dim=1)
    return float(torch.sqrt(torch.mean((pred_t - states[:, 1 : 1 + steps]) ** 2)).detach().cpu())


def _merged_eval_settings(cfg: dict[str, Any], eval_cfg: dict[str, Any] | None) -> dict[str, Any]:
    settings = dict(cfg.get("eval", {}))
    if eval_cfg is not None:
        settings.update(eval_cfg.get("eval", eval_cfg))
    return settings


@torch.no_grad()
def evaluate_model_on_split(
    model,
    data: dict[str, np.ndarray],
    normalizer: Normalizer,
    cfg: dict[str, Any],
    *,
    device: torch.device,
    max_windows: int | None = None,
    warmup_steps: int | None = None,
    horizon: str | int | None = None,
    eval_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    states_np = data["states"][:max_windows]
    actions_np = data["actions"][:max_windows]
    states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)

    eval_settings = _merged_eval_settings(cfg, eval_cfg)
    merged_cfg = {**cfg, "eval": eval_settings}
    warmup, horizon = resolve_eval_horizon(
        states_shape=tuple(states.shape),
        actions_shape=tuple(actions.shape),
        cfg=merged_cfg,
        warmup_override=warmup_steps,
        horizon_override=horizon,
    )
    preds = official_open_loop_rollout(model, states, actions, normalizer, warmup_steps=warmup, horizon=horizon)
    targets = states[:, warmup + 1 : warmup + 1 + horizon]
    metrics = compute_official_metrics(
        preds,
        targets,
        normalizer,
        report_horizons=eval_settings.get("report_horizons"),
        vpt_thresholds=eval_settings.get("vpt_thresholds"),
    )
    metrics.update(
        {
            "warmup_steps": warmup,
            "max_horizon": horizon,
            "one_step_rmse": _one_step_rmse(model, states, actions, normalizer, max_action_steps=warmup + horizon),
            "open_loop_rmse@horizon": float(torch.sqrt(torch.mean((preds - targets) ** 2)).detach().cpu()),
        }
    )
    model.train()
    return metrics


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    dataset_dir: str | Path,
    split: str,
    output_dir: str | Path,
    *,
    warmup_steps: int | None = None,
    horizon: str | int | None = "auto",
    eval_config: str | Path | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint_dir, device=device)
    data = load_split(dataset_dir, split)
    metadata = load_metadata(dataset_dir)
    validate_split_against_metadata(data, metadata, split)
    normalizer = Normalizer.from_dict(payload["normalizer"])
    eval_cfg = load_config(eval_config) if eval_config is not None else None
    metrics = evaluate_model_on_split(
        model,
        data,
        normalizer,
        payload["config"],
        device=device,
        warmup_steps=warmup_steps,
        horizon=horizon,
        eval_cfg=eval_cfg,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_vpt = np.asarray(metrics.pop("per_window_vpt@0.25"), dtype=np.int32)
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "scoreboard_summary.json", _scoreboard_summary(metrics, payload))
    np.save(output_dir / "per_window_vpt_0p25.npy", primary_vpt)
    return metrics


def _scoreboard_summary(metrics: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    max_horizon = int(metrics["max_horizon"])
    summary = {
        "model_name": payload["model_name"],
        "checkpoint_step": int(payload["step"]),
        "max_horizon": max_horizon,
        "primary_metric": "VPT80@0.25",
        "VPT80@0.25": int(metrics["VPT80@0.25"]),
        "VPT50@0.25": int(metrics["VPT50@0.25"]),
        "VPT80_fraction": float(metrics["VPT80@0.25"]) / max_horizon,
        "nMSE_AUC": float(metrics["nMSE_AUC"]),
    }
    for key in ("nMSE@10", "nMSE@90", "nMSE@100", "nMSE@1000", "step_nMSE@10", "step_nMSE@100", "step_nMSE@1000"):
        if key in metrics:
            summary[key] = float(metrics[key])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--horizon", default="auto", help="Use an integer horizon or 'auto' for the dataset maximum.")
    parser.add_argument("--eval-config", default=None, help="Locked official eval config, e.g. configs/official_eval.yaml.")
    args = parser.parse_args()
    metrics = evaluate_checkpoint(
        args.checkpoint_dir,
        args.dataset_dir,
        args.split,
        args.output_dir,
        warmup_steps=args.warmup,
        horizon=args.horizon,
        eval_config=args.eval_config,
    )
    summary_keys = (
        "max_horizon",
        "VPT80@0.25",
        "VPT50@0.25",
        "nMSE_AUC",
        "nMSE@10",
        "nMSE@90",
        "nMSE@100",
        "nMSE@1000",
        "step_nMSE@10",
        "step_nMSE@100",
        "step_nMSE@1000",
        "one_step_rmse",
        "open_loop_rmse@horizon",
    )
    print(
        json.dumps(
            {
                "metrics_summary": {key: metrics[key] for key in summary_keys if key in metrics},
                "metrics_json": str(Path(args.output_dir) / "metrics.json"),
                "scoreboard_summary_json": str(Path(args.output_dir) / "scoreboard_summary.json"),
                "per_window_vpt_0p25": str(Path(args.output_dir) / "per_window_vpt_0p25.npy"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
