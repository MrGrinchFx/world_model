"""Export a trained student checkpoint to a TorchScript artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

import torch

from .checkpoint import load_checkpoint
from .config import save_json
from .normalizer import Normalizer


def _example_hidden(model, batch_size: int, device: torch.device) -> torch.Tensor:
    hidden = model.initial_hidden(batch_size, device)
    if hidden is None:
        return torch.empty(batch_size, 0, dtype=torch.float32, device=device)
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(f"initial_hidden must return a Tensor or None, got {type(hidden)!r}.")
    return hidden


def _validate_compiled(compiled, obs: torch.Tensor, act: torch.Tensor, hidden: torch.Tensor) -> None:
    delta, next_hidden = compiled(obs, act, hidden)
    if not isinstance(delta, torch.Tensor) or list(delta.shape) != [obs.shape[0], obs.shape[1]]:
        raise RuntimeError(f"compiled model returned invalid delta shape: {getattr(delta, 'shape', None)}")
    if next_hidden is not None and not isinstance(next_hidden, torch.Tensor):
        raise RuntimeError(f"compiled model returned invalid hidden type: {type(next_hidden)!r}")
    compiled.save_to_buffer()


def _compile_model(model, example_obs: torch.Tensor, example_act: torch.Tensor, example_hidden: torch.Tensor):
    try:
        compiled = torch.jit.script(model)
        _validate_compiled(compiled, example_obs, example_act, example_hidden)
        return compiled, "script"
    except Exception as script_error:
        try:
            compiled = torch.jit.trace(model, (example_obs, example_act, example_hidden), strict=False)
            _validate_compiled(compiled, example_obs, example_act, example_hidden)
            return compiled, "trace"
        except Exception as trace_error:
            raise RuntimeError(
                "Could not export model with torch.jit.script or torch.jit.trace. "
                "Keep StudentWorldModel.forward(obs_norm, act_norm, hidden) TorchScript-compatible."
            ) from trace_error


def export_compiled_model(checkpoint_dir: str | Path, output_dir: str | Path, *, device: str | torch.device = "cpu") -> dict[str, Any]:
    device = torch.device(device)
    model, payload = load_checkpoint(checkpoint_dir, device=device)
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    example_obs = torch.zeros(2, 4, dtype=torch.float32, device=device)
    example_act = torch.zeros(2, 1, dtype=torch.float32, device=device)
    example_hidden = _example_hidden(model, 2, device)
    compiled, backend = _compile_model(model, example_obs, example_act, example_hidden)
    compiled_path = output_dir / "model_compiled.pt"
    compiled.save(str(compiled_path))

    normalizer = Normalizer.from_dict(payload["normalizer"])
    normalizer.save(output_dir / "normalizer.json")

    model_cfg = payload.get("config", {}).get("model", {})
    metadata: dict[str, Any] = {
        "compile_backend": backend,
        "model_name": payload.get("model_name", "student"),
        "checkpoint_step": int(payload.get("step", 0)),
        "obs_dim": 4,
        "act_dim": 1,
        "hidden_dim": int(example_hidden.shape[-1]) if example_hidden.ndim >= 2 else 0,
        "use_gru": bool(model_cfg.get("use_gru", int(example_hidden.shape[-1]) > 0)),
        "model_config": model_cfg,
        "config": payload.get("config", {}),
        "checkpoint_metrics": payload.get("metrics", {}),
    }
    save_json(output_dir / "metadata.json", metadata)
    return {
        "compiled_model": str(compiled_path),
        "normalizer": str(output_dir / "normalizer.json"),
        "metadata": str(output_dir / "metadata.json"),
        "compile_backend": backend,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(export_compiled_model(args.checkpoint_dir, args.output_dir, device=args.device), indent=2))


if __name__ == "__main__":
    main()
