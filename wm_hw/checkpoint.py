"""Checkpoint utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def build_model(model_name: str, cfg: dict[str, Any]) -> torch.nn.Module:
    mcfg = cfg.get("model", {})
    if model_name == "student":
        from student.model import StudentWorldModel

        return StudentWorldModel(hidden_dim=int(mcfg.get("hidden_dim", 128)), num_layers=int(mcfg.get("num_layers", 2)), use_gru=bool(mcfg.get("use_gru", False)))
    raise KeyError(f"Unknown model '{model_name}'.")


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model,
    model_name: str,
    config: dict[str, Any],
    normalizer: dict[str, Any],
    step: int,
    metrics: dict[str, float],
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": model_name,
            "config": config,
            "normalizer": normalizer,
            "step": int(step),
            "metrics": metrics,
        },
        path,
    )
    return path


def load_checkpoint(checkpoint_dir: str | Path, device: str | torch.device = "cpu"):
    path = Path(checkpoint_dir) / "checkpoint.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(payload["model_name"], payload["config"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload
