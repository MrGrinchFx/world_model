"""Evaluate a TorchScript-exported world model without importing student code."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

import numpy as np
import torch

from .config import load_config, save_json
from .dataset import load_metadata, load_split, validate_split_against_metadata
from .eval_horizon import _scoreboard_summary, evaluate_model_on_split
from .normalizer import Normalizer


class CompiledWorldModel:
    """Small adapter that gives a TorchScript module the locked model interface."""

    def __init__(self, module, *, hidden_dim: int):
        self.module = module
        self.hidden_dim = int(hidden_dim)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.hidden_dim <= 0:
            return torch.empty(batch_size, 0, dtype=torch.float32, device=device)
        return torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32, device=device)

    def eval(self):
        self.module.eval()
        return self

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def __call__(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden):
        if hidden is None:
            hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
        delta, next_hidden = self.module(obs_norm, act_norm, hidden)
        if next_hidden is None:
            next_hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
        return delta, next_hidden


def _load_metadata(compiled_dir: Path) -> dict[str, Any]:
    with (compiled_dir / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_compiled_model(compiled_dir: str | Path, *, device: str | torch.device):
    compiled_dir = Path(compiled_dir)
    device = torch.device(device)
    metadata = _load_metadata(compiled_dir)
    module = torch.jit.load(str(compiled_dir / "model_compiled.pt"), map_location=device)
    model = CompiledWorldModel(module, hidden_dim=int(metadata.get("hidden_dim", 0)))
    normalizer = Normalizer.load(compiled_dir / "normalizer.json")
    return model, normalizer, metadata


def evaluate_compiled(
    compiled_dir: str | Path,
    dataset_dir: str | Path,
    split: str,
    output_dir: str | Path,
    *,
    warmup_steps: int | None = None,
    horizon: str | int | None = "auto",
    eval_config: str | Path | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, metadata = load_compiled_model(compiled_dir, device=device)
    data = load_split(dataset_dir, split)
    dataset_metadata = load_metadata(dataset_dir)
    validate_split_against_metadata(data, dataset_metadata, split)
    eval_cfg = load_config(eval_config) if eval_config is not None else None
    cfg = metadata.get("config", {})
    metrics = evaluate_model_on_split(
        model,
        data,
        normalizer,
        cfg,
        device=device,
        warmup_steps=warmup_steps,
        horizon=horizon,
        eval_cfg=eval_cfg,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_vpt = np.asarray(metrics.pop("per_window_vpt@0.25"), dtype=np.int32)
    payload = {"model_name": metadata.get("model_name", "compiled"), "step": int(metadata.get("checkpoint_step", 0))}
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "scoreboard_summary.json", _scoreboard_summary(metrics, payload))
    np.save(output_dir / "per_window_vpt_0p25.npy", primary_vpt)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--horizon", default="auto")
    parser.add_argument("--eval-config", default=None)
    args = parser.parse_args()
    metrics = evaluate_compiled(
        args.compiled_dir,
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
        "nMSE@100",
        "nMSE@1000",
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
