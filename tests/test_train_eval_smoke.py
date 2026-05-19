from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from wm_hw.dataset import generate_dataset
from wm_hw.eval_horizon import evaluate_checkpoint
from wm_hw.export_model import export_compiled_model
from wm_hw.plotting import plot_horizon_rmse, plot_survival_curve
from wm_hw.train import train


@pytest.mark.slow
def test_train_and_eval_smoke(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    generate_dataset("configs/dev.yaml", data_dir, smoke=True)
    out_dir = tmp_path / "student"
    train("configs/student.yaml", "student", data_dir, out_dir, smoke=True)
    metrics = evaluate_checkpoint(out_dir / "best_checkpoint", data_dir, "test", tmp_path / "eval", eval_config="configs/official_eval.yaml")
    assert "VPT80@0.25" in metrics
    assert metrics["max_horizon"] == 100
    assert "open_loop_rmse@horizon" in metrics
    assert (tmp_path / "eval" / "metrics.json").exists()
    assert (tmp_path / "eval" / "per_window_vpt_0p25.npy").exists()
    assert (tmp_path / "eval" / "scoreboard_summary.json").exists()
    override = evaluate_checkpoint(out_dir / "best_checkpoint", data_dir, "test", tmp_path / "eval20", horizon=20, eval_config="configs/official_eval.yaml")
    assert override["max_horizon"] == 20
    assert "nMSE@20" in override
    plot_dir = tmp_path / "plots"
    vpt = np.load(tmp_path / "eval" / "per_window_vpt_0p25.npy")
    assert plot_survival_curve(metrics, plot_dir, vpt).exists()
    assert plot_horizon_rmse(metrics, plot_dir).exists()

    compiled_dir = tmp_path / "compiled"
    export_payload = export_compiled_model(out_dir / "best_checkpoint", compiled_dir)
    assert (compiled_dir / "model_compiled.pt").exists()
    assert (compiled_dir / "normalizer.json").exists()
    assert (compiled_dir / "metadata.json").exists()
    assert export_payload["compile_backend"] in {"script", "trace"}

    for name in list(sys.modules):
        if name == "student" or name.startswith("student."):
            sys.modules.pop(name)
    sys.modules.pop("wm_hw.eval_compiled", None)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "student" or name.startswith("student."):
            raise AssertionError(f"compiled eval imported forbidden module {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    eval_compiled = importlib.import_module("wm_hw.eval_compiled")
    compiled_metrics = eval_compiled.evaluate_compiled(compiled_dir, data_dir, "test", tmp_path / "compiled_eval", eval_config="configs/official_eval.yaml")
    assert (tmp_path / "compiled_eval" / "metrics.json").exists()
    assert (tmp_path / "compiled_eval" / "scoreboard_summary.json").exists()
    assert compiled_metrics["VPT80@0.25"] == metrics["VPT80@0.25"]
    assert compiled_metrics["nMSE@10"] == pytest.approx(metrics["nMSE@10"], rel=1e-6, abs=1e-7)
    assert compiled_metrics["nMSE@100"] == pytest.approx(metrics["nMSE@100"], rel=1e-6, abs=1e-7)
    assert compiled_metrics["nMSE_AUC"] == pytest.approx(metrics["nMSE_AUC"], rel=1e-6, abs=1e-7)
