"""Student loss computation with Cosine Annealing and Covariate Injection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from .rollout import open_loop_rollout


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    
    # Inject subtle noise during training to teach the model to error-correct
    if model.training:
        state_perturbation = torch.randn_like(obs_norm) * 0.015
        obs_norm = obs_norm + state_perturbation
        
    pred_norm, _ = model(obs_norm, act_norm, None)
    
    logvar = model._current_logvar
    if logvar is None:
        return F.mse_loss(pred_norm, target_norm)
        
    return 0.5 * torch.mean(torch.exp(-logvar) * (pred_norm - target_norm) ** 2 + logvar)


def rollout_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer, warmup_steps: int, horizon: int) -> torch.Tensor:
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            f"training.train_sequence_length is too short for rollout loss: need at least {needed_states - 1} steps."
        )
    max_start = states.shape[1] - needed_states
    start = int(torch.randint(0, max_start + 1, (), device=states.device).item()) if max_start > 0 else 0
    
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    return F.mse_loss(pred_norm, target_norm)


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    import gc
    import math

    # 1. Track persistent step execution metadata
    if not hasattr(model, "_step_counter"):
        model._step_counter = 0
    if not hasattr(model, "_optimizer_ref"):
        model._optimizer_ref = None
        for obj in gc.get_objects():
            if isinstance(obj, torch.optim.Optimizer):
                model._optimizer_ref = obj
                break

    model._step_counter += 1
    
    # 2. Compute dynamic Cosine Annealing Learning Rate
    initial_lr = float(cfg["training"].get("learning_rate", 1.0e-4))
    total_updates = int(cfg["training"].get("updates", 5000))
    min_lr = 1.0e-6
    
    current_step = min(model._step_counter, total_updates)
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (
        1.0 + math.cos(math.pi * current_step / total_updates)
    )
    
    if model._optimizer_ref is not None:
        for param_group in model._optimizer_ref.param_groups:
            param_group['lr'] = cosine_lr

    # 3. Aggregate Objective Losses
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    
    one = one_step_delta_loss(model, states, actions, normalizer)
    
    horizon = int(loss_cfg.get("rollout_train_horizon", 15))
    warmup = int(cfg["eval"].get("warmup_steps", 10))
    roll = rollout_loss(model, states, actions, normalizer, warmup_steps=warmup, horizon=horizon)
    
    total = float(loss_cfg.get("one_step_weight", 1.0)) * one + float(loss_cfg.get("rollout_weight", 1.0)) * roll
    
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }
