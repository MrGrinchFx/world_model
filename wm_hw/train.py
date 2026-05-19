"""Locked training script for the starter student world model."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

import numpy as np
import torch

from .checkpoint import build_model, save_checkpoint
from .config import load_config, save_json, set_seed
from .dataset import load_split
from .eval_horizon import evaluate_model_on_split
from .horizon import available_horizon
from .normalizer import Normalizer


def _device(cfg: dict[str, Any]) -> torch.device:
    if cfg.get("device", "auto") == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    *,
    sequence_length: int | None,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    states_np = data["states"][indices]
    actions_np = data["actions"][indices]
    if sequence_length is not None and int(sequence_length) > 0 and int(sequence_length) < actions_np.shape[1]:
        length = int(sequence_length)
        start = int(rng.integers(0, actions_np.shape[1] - length + 1))
        states_np = states_np[:, start : start + length + 1]
        actions_np = actions_np[:, start : start + length]
    return {
        "states": torch.as_tensor(states_np, dtype=torch.float32, device=device),
        "actions": torch.as_tensor(actions_np, dtype=torch.float32, device=device),
    }


def _compute_loss(model_name: str, model, batch: dict[str, torch.Tensor], normalizer: Normalizer, cfg: dict[str, Any]):
    if model_name != "student":
        raise KeyError(f"Unknown model '{model_name}'. This release trains only the student starter model.")
    from student.losses import compute_loss

    return compute_loss(model, batch, normalizer, cfg)


def _checkpoint_score(metrics: dict[str, Any], metric_name: str) -> float:
    key = metric_name.split("/", 1)[-1]
    if key not in metrics:
        raise KeyError(f"checkpoint_metric '{metric_name}' not found in eval metrics. Available keys include: {sorted(metrics)[:12]}")
    return float(metrics[key])


def _is_better(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value <= best
    if mode == "max":
        return value >= best
    raise ValueError(f"checkpoint_mode must be 'max' or 'min', got {mode!r}.")


def train(config_path: str | Path, model_name: str, dataset_dir: str | Path, output_dir: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 0)))
    torch.set_num_threads(int(cfg.get("torch_num_threads", 1)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data = load_split(dataset_dir, "train")
    val_data = load_split(dataset_dir, "val")
    normalizer = Normalizer.from_train(train_data["states"], train_data["actions"])
    normalizer.save(output_dir / "normalizer.json")
    device = _device(cfg)
    model = build_model(model_name, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    updates = int(cfg["training"]["smoke_updates"] if smoke else cfg["training"]["updates"])
    batch_size = int(cfg["training"]["batch_size"])
    eval_every = int(cfg["training"]["smoke_eval_every"] if smoke else cfg["training"]["eval_every"])
    sequence_length = int(cfg["training"].get("train_sequence_length", 64))
    val_horizon = int(cfg["training"].get("val_horizon", cfg.get("eval", {}).get("horizon", 100)))
    val_horizon = min(val_horizon, available_horizon(val_data["states"].shape, val_data["actions"].shape, int(cfg.get("eval", {}).get("warmup_steps", 10))))
    max_val_windows = int(cfg["training"].get("max_val_windows", 128))
    checkpoint_metric = str(cfg["training"].get("checkpoint_metric", "val/VPT80@0.25"))
    checkpoint_mode = str(cfg["training"].get("checkpoint_mode", "max"))
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 17)
    best_score: float | None = None
    best_metrics: dict[str, float] = {}
    print(f"[train] model={model_name} device={device} updates={updates} smoke={smoke}")
    for update in range(1, updates + 1):
        indices = rng.integers(0, len(train_data["states"]), size=batch_size)
        batch = _batch(train_data, indices, device, sequence_length=sequence_length, rng=rng)
        loss, metrics = _compute_loss(model_name, model, batch, normalizer, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip_norm"]))
        opt.step()
        if update == 1 or update % eval_every == 0 or update == updates:
            eval_metrics = evaluate_model_on_split(
                model,
                val_data,
                normalizer,
                cfg,
                device=device,
                max_windows=min(max_val_windows, len(val_data["states"])),
                horizon=val_horizon,
            )
            eval_metrics.pop("per_window_vpt@0.25", None)
            line = " ".join([f"{k}={v:.4f}" for k, v in {**metrics, **eval_metrics}.items() if isinstance(v, (float, int))])
            print(f"[train] update={update} {line}")
            score = _checkpoint_score(eval_metrics, checkpoint_metric)
            if _is_better(score, best_score, checkpoint_mode):
                best_score = score
                best_metrics = {**metrics, **eval_metrics}
                save_checkpoint(
                    output_dir / "best_checkpoint",
                    model=model,
                    model_name=model_name,
                    config=cfg,
                    normalizer=normalizer.to_dict(),
                    step=update,
                    metrics=best_metrics,
                )
    summary = {
        "model": model_name,
        "updates": updates,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_mode": checkpoint_mode,
        "best_score": best_score,
        "metrics": best_metrics,
    }
    save_json(output_dir / "train_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["student"], default="student")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.model, args.dataset_dir, args.output_dir, smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
