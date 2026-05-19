from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from wm_hw.config import load_config
from wm_hw.dataset import generate_dataset, generate_split, load_metadata
from wm_hw.horizon import dataset_window_spec


def test_dataset_split_schema_smoke():
    cfg = load_config("configs/dev.yaml")
    data = generate_split("train", cfg, smoke=True)
    spec = dataset_window_spec(cfg["dataset"])
    assert data["states"].shape == (cfg["smoke"]["train_windows"], spec["window_states"], 4)
    assert data["actions"].shape == (cfg["smoke"]["train_windows"], spec["window_actions"], 1)
    assert abs(data["states"][:, :, 1]).max() < 0.20


def test_dataset_schema_uses_configured_scoreboard_horizon():
    cfg = deepcopy(load_config("configs/dev.yaml"))
    cfg["dataset"]["warmup_steps"] = 10
    cfg["dataset"]["max_horizon"] = 37
    cfg["smoke"]["train_windows"] = 4
    data = generate_split("train", cfg, smoke=True)
    spec = dataset_window_spec(cfg["dataset"])
    assert spec["window_states"] == 48
    assert spec["window_actions"] == 47
    assert data["states"].shape == (4, 48, 4)
    assert data["actions"].shape == (4, 47, 1)


def test_dataset_supports_requested_hidden_splits(tmp_path: Path):
    cfg = deepcopy(load_config("configs/dev.yaml"))
    cfg["dataset"]["max_horizon"] = 12
    cfg["splits"]["hidden_test"] = deepcopy(cfg["splits"]["test"])
    cfg["smoke"]["train_windows"] = 2
    cfg["smoke"]["hidden_test_windows"] = 2
    config_path = tmp_path / "hidden.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    generate_dataset(config_path, tmp_path / "data", smoke=True, splits="train,hidden_test")
    metadata = load_metadata(tmp_path / "data")

    assert sorted(metadata["splits"]) == ["hidden_test", "train"]
    assert (tmp_path / "data" / "train.npz").exists()
    assert (tmp_path / "data" / "hidden_test.npz").exists()
    assert not (tmp_path / "data" / "val.npz").exists()
    assert metadata["split_configs"]["train"]["seed"] != metadata["split_configs"]["hidden_test"]["seed"]
    assert metadata["split_signatures"]["train"] != metadata["split_signatures"]["hidden_test"]
    assert metadata["data_signature"]
