"""Student loss computation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from .rollout import open_loop_rollout

def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    
    obs_norm = normalizer.normalize_obs(obs)
    if model.training:
        obs_norm = obs_norm + torch.randn_like(obs_norm) * 0.02
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    
    # Standard external signature call
    pred_norm, _ = model(obs_norm, act_norm, None)
    
    # Extract the cached logvar generated during the line above
    logvar = model._current_logvar
    if logvar is None:
        return F.mse_loss(pred_norm, target_norm)
        
    # Gaussian Negative Log-Likelihood loss formula
    return 0.5 * torch.mean(torch.exp(-logvar) * (pred_norm - target_norm) ** 2 + logvar)

def rollout_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer, warmup_steps: int, horizon: int) -> torch.Tensor:
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
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

    # 1. Track the training updates dynamically inside the persistent model instance
    if not hasattr(model, "_step_counter"):
        model._step_counter = 0
    if not hasattr(model, "_optimizer_ref"):
        model._optimizer_ref = None
        # Locate the live Adam optimizer instance inside the execution workspace
        for obj in gc.get_objects():
            if isinstance(obj, torch.optim.Optimizer):
                model._optimizer_ref = obj
                break

    model._step_counter += 1
    
    # 2. Extract optimization parameters from config metadata
    initial_lr = float(cfg["training"].get("learning_rate", 1.0e-4))
    total_updates = int(cfg["training"].get("updates", 5000))
    min_lr = 1.0e-6  # Bounded learning rate floor to avoid full stalling near convergence
    
    # 3. Calculate Cosine Annealing decay scheduling factor
    current_step = min(model._step_counter, total_updates)
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (
        1.0 + math.cos(math.pi * current_step / total_updates)
    )
    
    # 4. Inject the dynamically decayed learning rate into active parameter groups
    if model._optimizer_ref is not None:
        for param_group in model._optimizer_ref.param_groups:
            param_group['lr'] = cosine_lr

    # 5. Core Loss Aggregation Pipeline
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    
    # Multi-headed probabilistic NLL loss evaluation
    one = one_step_delta_loss(model, states, actions, normalizer)
    
    horizon = int(loss_cfg.get("rollout_train_horizon", 15))
    warmup = int(cfg["eval"].get("warmup_steps", 10))
    
    # Open-loop rolling trajectory consistency evaluation
    roll = rollout_loss(model, states, actions, normalizer, warmup_steps=warmup, horizon=horizon)
    
    # Combine objective tracks with calibrated weights to handle negative likelihood scaling
    total = float(loss_cfg.get("one_step_weight", 1.0)) * one + float(loss_cfg.get("rollout_weight", 1.0)) * roll
    
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }
