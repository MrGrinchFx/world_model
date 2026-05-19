"""Locked dataset generation for InvertedPendulum windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import hashlib
import json

import numpy as np
from tqdm import tqdm

from .config import load_config
from .env import ACTION_HIGH, ACTION_LOW, clip_action, make_env, reset_env, step_env
from .horizon import dataset_window_spec


def sample_ar1_noise(rng: np.random.Generator, steps: int, sigma: float, rho: float = 0.9) -> np.ndarray:
    noise = np.zeros((steps, 1), dtype=np.float32)
    cur = np.zeros(1, dtype=np.float32)
    for t in range(steps):
        cur = rho * cur + float(sigma) * rng.standard_normal(1).astype(np.float32)
        noise[t] = cur
    return noise


def fixed_action_generator(obs: np.ndarray, noise: np.ndarray | float, gain: np.ndarray) -> np.ndarray:
    action = -float(np.dot(gain, obs)) + float(np.asarray(noise).reshape(1)[0])
    return clip_action(action)


def _valid_window(states: np.ndarray, max_abs_angle: float) -> bool:
    return bool(np.all(np.isfinite(states)) and np.max(np.abs(states[:, 1])) < float(max_abs_angle))


def collect_valid_window(env, split_cfg: dict[str, Any], data_cfg: dict[str, Any], filter_cfg: dict[str, Any], rng: np.random.Generator):
    steps = int(dataset_window_spec(data_cfg)["window_actions"])
    gain = np.asarray(data_cfg["lqr_gain"], dtype=np.float32)
    rho = float(data_cfg.get("ar1_rho", 0.9))
    max_abs_angle = float(filter_cfg["max_abs_true_angle"])
    noise = sample_ar1_noise(rng, steps, float(split_cfg["action_noise_sigma"]), rho=rho)
    obs = reset_env(env, seed=int(rng.integers(0, 2**31 - 1)))
    states = [obs]
    actions = []
    for t in range(steps):
        action = fixed_action_generator(obs, noise[t], gain)
        obs, done = step_env(env, action)
        states.append(obs)
        actions.append(action)
        if done:
            break
    if len(states) != steps + 1:
        return None
    states_arr = np.asarray(states, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.float32)
    if not _valid_window(states_arr, max_abs_angle):
        return None
    return states_arr, actions_arr


def split_seed(base_seed: int, split: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}:{split}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _smoke_windows(split: str, cfg: dict[str, Any], default_windows: int) -> int:
    smoke_cfg = cfg.get("smoke", {})
    exact = f"{split}_windows"
    if exact in smoke_cfg:
        return int(smoke_cfg[exact])
    if "ood" in split and "ood_windows" in smoke_cfg:
        return int(smoke_cfg["ood_windows"])
    if "test" in split and "test_windows" in smoke_cfg:
        return int(smoke_cfg["test_windows"])
    if "val" in split and "val_windows" in smoke_cfg:
        return int(smoke_cfg["val_windows"])
    if "train_windows" in smoke_cfg:
        return int(smoke_cfg["train_windows"])
    return int(default_windows)


def generate_split(name: str, cfg: dict[str, Any], *, smoke: bool = False) -> dict[str, np.ndarray]:
    split_cfg = dict(cfg["splits"][name])
    spec = dataset_window_spec(cfg["dataset"])
    if smoke:
        split_cfg["windows"] = _smoke_windows(name, cfg, int(split_cfg["windows"]))
    rng = np.random.default_rng(split_seed(int(cfg["seed"]), name))
    env = make_env(reset_noise_scale=float(split_cfg["reset_noise_scale"]), max_episode_steps=spec["window_actions"])
    windows = int(split_cfg["windows"])
    max_attempts = windows * int(cfg["filter"].get("max_attempts_multiplier", 80))
    states = []
    actions = []
    attempts = 0
    with tqdm(total=windows, desc=f"generate {name}", leave=False) as pbar:
        while len(states) < windows and attempts < max_attempts:
            attempts += 1
            item = collect_valid_window(env, split_cfg, cfg["dataset"], cfg["filter"], rng)
            if item is None:
                continue
            s, a = item
            states.append(s)
            actions.append(a)
            pbar.update(1)
    env.close()
    if len(states) < windows:
        raise RuntimeError(f"Only collected {len(states)}/{windows} valid {name} windows after {attempts} attempts.")
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
    }


def save_split(output_dir: Path, split: str, data: dict[str, np.ndarray]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split}.npz"
    np.savez_compressed(path, **data)
    return path


def load_split(dataset_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    path = Path(dataset_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_metadata(dataset_dir: str | Path) -> dict[str, Any]:
    path = Path(dataset_dir) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_split_against_metadata(data: dict[str, np.ndarray], metadata: dict[str, Any], split: str) -> None:
    expected = metadata.get("splits", {}).get(split)
    if expected is None:
        raise ValueError(f"metadata does not contain split '{split}'.")
    if list(data["states"].shape) != list(expected["states"]):
        raise ValueError(f"{split} states shape {data['states'].shape} does not match metadata {expected['states']}.")
    if list(data["actions"].shape) != list(expected["actions"]):
        raise ValueError(f"{split} actions shape {data['actions'].shape} does not match metadata {expected['actions']}.")
    if int(data["states"].shape[1]) != int(metadata["window_states"]):
        raise ValueError("states time dimension does not match metadata window_states.")
    if int(data["actions"].shape[1]) != int(metadata["window_actions"]):
        raise ValueError("actions time dimension does not match metadata window_actions.")


def _parse_splits(requested: str | list[str] | tuple[str, ...] | None, cfg: dict[str, Any]) -> list[str]:
    if requested is None:
        names = list(cfg["splits"].keys())
    elif isinstance(requested, str):
        names = [part.strip() for part in requested.split(",") if part.strip()]
    else:
        names = [str(part) for part in requested]
    missing = [name for name in names if name not in cfg["splits"]]
    if missing:
        raise KeyError(f"Requested split(s) not found in config: {missing}. Available: {sorted(cfg['splits'])}")
    return names


def _data_signature(data: dict[str, np.ndarray]) -> str:
    sha = hashlib.sha256()
    for key in sorted(data):
        arr = np.ascontiguousarray(data[key])
        sha.update(key.encode("utf-8"))
        sha.update(str(arr.shape).encode("utf-8"))
        sha.update(arr.view(np.uint8))
    return sha.hexdigest()


def _dataset_signature(split_signatures: dict[str, str]) -> str:
    sha = hashlib.sha256()
    for split in sorted(split_signatures):
        sha.update(split.encode("utf-8"))
        sha.update(split_signatures[split].encode("utf-8"))
    return sha.hexdigest()


def generate_dataset(config_path: str | Path, output_dir: str | Path, *, smoke: bool = False, splits: str | list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    spec = dataset_window_spec(cfg["dataset"])
    split_names = _parse_splits(splits, cfg)
    output_dir = Path(output_dir)
    written = {}
    summaries = {}
    split_configs = {}
    split_signatures = {}
    for split in split_names:
        data = generate_split(split, cfg, smoke=smoke)
        written[split] = str(save_split(output_dir, split, data))
        summaries[split] = {key: list(value.shape) for key, value in data.items()}
        split_cfg = dict(cfg["splits"][split])
        if smoke:
            split_cfg["windows"] = _smoke_windows(split, cfg, int(split_cfg["windows"]))
        split_cfg["seed"] = split_seed(int(cfg["seed"]), split)
        split_configs[split] = split_cfg
        split_signatures[split] = _data_signature(data)
    metadata = {
        "env_id": cfg["env"]["id"],
        "seed": int(cfg["seed"]),
        "smoke": smoke,
        "warmup_steps": spec["warmup_steps"],
        "max_horizon": spec["max_horizon"],
        "window_states": spec["window_states"],
        "window_actions": spec["window_actions"],
        "action_range": [ACTION_LOW, ACTION_HIGH],
        "splits": summaries,
        "split_configs": split_configs,
        "split_signatures": split_signatures,
        "data_signature": _dataset_signature(split_signatures),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return {"output_dir": str(output_dir), "written": written, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--output-dir", default="data/public")
    parser.add_argument("--splits", default=None, help="Comma-separated split names. Defaults to all splits in the config.")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate_dataset(args.config, args.output_dir, smoke=args.smoke, splits=args.splits), indent=2))


if __name__ == "__main__":
    main()
