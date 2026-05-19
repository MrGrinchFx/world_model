"""Shared horizon/window helpers for survival-scoreboard evaluation."""

from __future__ import annotations

from typing import Any, Iterable


DEFAULT_MILESTONES = (5, 10, 25, 50, 100, 200, 500, 1000)


def dataset_window_spec(dataset_cfg: dict[str, Any]) -> dict[str, int]:
    """Resolve warmup, max horizon, and required window sizes.

    The canonical config fields are `warmup_steps` and `max_horizon`. Legacy
    `eval_horizon`, `window_states`, and `window_actions` are accepted only when
    they agree with the inferred sizes.
    """
    warmup = int(dataset_cfg.get("warmup_steps", 5))
    if "max_horizon" in dataset_cfg:
        max_horizon = int(dataset_cfg["max_horizon"])
    elif "eval_horizon" in dataset_cfg:
        max_horizon = int(dataset_cfg["eval_horizon"])
    elif "window_actions" in dataset_cfg:
        max_horizon = int(dataset_cfg["window_actions"]) - warmup
    else:
        raise ValueError("dataset config must define max_horizon or a compatible legacy window size.")
    if warmup < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup}.")
    if max_horizon <= 0:
        raise ValueError(f"max_horizon must be positive, got {max_horizon}.")
    window_actions = warmup + max_horizon
    window_states = window_actions + 1
    _validate_optional_size(dataset_cfg, "window_actions", window_actions)
    _validate_optional_size(dataset_cfg, "window_states", window_states)
    return {
        "warmup_steps": warmup,
        "max_horizon": max_horizon,
        "window_actions": window_actions,
        "window_states": window_states,
    }


def available_horizon(states_shape: tuple[int, ...], actions_shape: tuple[int, ...], warmup_steps: int) -> int:
    """Return the largest valid autoregressive horizon for a dataset split."""
    if len(states_shape) < 2 or len(actions_shape) < 2:
        raise ValueError("states/actions must have time dimensions.")
    by_actions = int(actions_shape[1]) - int(warmup_steps)
    by_states = int(states_shape[1]) - int(warmup_steps) - 1
    horizon = min(by_actions, by_states)
    if horizon <= 0:
        raise ValueError(
            f"Dataset is too short for warmup={warmup_steps}: "
            f"states={states_shape}, actions={actions_shape}."
        )
    return horizon


def resolve_eval_horizon(
    *,
    states_shape: tuple[int, ...],
    actions_shape: tuple[int, ...],
    cfg: dict[str, Any],
    warmup_override: int | None = None,
    horizon_override: str | int | None = None,
) -> tuple[int, int]:
    """Resolve `(warmup_steps, horizon)` with CLI/config/auto precedence."""
    warmup = int(warmup_override if warmup_override is not None else cfg.get("eval", {}).get("warmup_steps", 5))
    max_available = available_horizon(states_shape, actions_shape, warmup)
    if horizon_override == "auto":
        horizon = min(max_available, int(cfg.get("eval", {}).get("max_horizon", max_available)))
    elif horizon_override is not None:
        horizon = int(horizon_override)
    else:
        config_horizon = cfg.get("eval", {}).get("horizon", "auto")
        if config_horizon == "auto":
            horizon = min(max_available, int(cfg.get("eval", {}).get("max_horizon", max_available)))
        else:
            horizon = int(config_horizon)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    if horizon > max_available:
        raise ValueError(f"Requested horizon={horizon}, but dataset only supports {max_available}.")
    return warmup, horizon


def resolve_milestones(milestones: Iterable[int] | str | None, horizon: int) -> list[int]:
    """Return sorted success-rate milestones, always including `horizon`."""
    if milestones is None:
        values = list(DEFAULT_MILESTONES)
    elif isinstance(milestones, str):
        values = [int(part) for part in milestones.split(",") if part.strip()]
    else:
        values = [int(value) for value in milestones]
    values = [value for value in values if 0 < value <= int(horizon)]
    values.append(int(horizon))
    return sorted(set(values))


def _validate_optional_size(cfg: dict[str, Any], key: str, expected: int) -> None:
    if key in cfg and int(cfg[key]) != int(expected):
        raise ValueError(f"{key}={cfg[key]} does not match inferred value {expected}.")
