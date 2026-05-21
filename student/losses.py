"""Student loss computation with Parity Symmetry and Adaptive Coordinate Weighting."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from .rollout import open_loop_rollout


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> tuple[torch.Tensor, torch.Tensor]:
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    
    if model.training:
        # Reflectional Parity Invariance Augmentation
        obs_norm_mirror = -obs_norm
        act_norm_mirror = -act_norm
        target_norm_mirror = -target_norm
        
        obs_norm = torch.cat([obs_norm, obs_norm_mirror], dim=0)
        act_norm = torch.cat([act_norm, act_norm_mirror], dim=0)
        target_norm = torch.cat([target_norm, target_norm_mirror], dim=0)
        
        # Coarse-to-Fine Adaptive Covariate Noise Injection
        progress = getattr(model, "_step_counter", 0) / 10000.0
        dynamic_noise = 0.005 + (0.020 * min(progress, 1.0))
        obs_norm = obs_norm + torch.randn_like(obs_norm) * dynamic_noise
        
    pred_norm, _ = model(obs_norm, act_norm, None)
    
    logvar = model._current_logvar
    if logvar is None:
        return F.mse_loss(pred_norm, target_norm), torch.tensor(0.0, device=states.device)
        
    # Adaptive dimension weights: prioritize unstable angular tracking channels
    sq_error = (pred_norm - target_norm) ** 2
    with torch.no_grad():
        error_variance = torch.mean(sq_error, dim=0, keepdim=True)
        dim_weights = error_variance / (torch.sum(error_variance) + 1e-6)
        dim_weights = torch.clamp(dim_weights * states.shape[-1], min=0.5, max=2.5)
        
    nll_loss = 0.5 * torch.mean(dim_weights * (torch.exp(-logvar) * sq_error + logvar))
    var_penalty = 0.05 * torch.mean(logvar ** 2)
    
    return nll_loss, var_penalty


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

    if not hasattr(model, "_step_counter"):
        model._step_counter = 0
    if not hasattr(model, "_optimizer_ref"):
        model._optimizer_ref = None
        for obj in gc.get_objects():
            if isinstance(obj, torch.optim.Optimizer):
                model._optimizer_ref = obj
                break

    model._step_counter += 1
    
    training_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    eval_cfg = cfg.get("eval", {})
    
    initial_lr = float(training_cfg.get("learning_rate", 1.0e-4))
    total_updates = int(training_cfg.get("updates", 10000))
    min_lr = 1.0e-6
    
    current_step = min(model._step_counter, total_updates)
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (
        1.0 + math.cos(math.pi * current_step / total_updates)
    )
    
    if model._optimizer_ref is not None:
        for param_group in model._optimizer_ref.param_groups:
            param_group['lr'] = cosine_lr

    # Dynamic Curriculum fully bound to maximum sequence lengths
    seq_len = int(training_cfg.get("train_sequence_length", 96))
    warmup = int(eval_cfg.get("warmup_steps", 10))
    base_horizon = int(loss_cfg.get("rollout_train_horizon", 15))
    
    # Max out the unroll curriculum to fully utilize the context width
    max_horizon = seq_len - warmup - 2  # Progresses smoothly up to 84 steps
    progress = current_step / total_updates
    current_horizon = int(base_horizon + math.floor(progress * (max_horizon - base_horizon)))
    
    states = batch["states"]
    actions = batch["actions"]
    
    one_step_nll, var_penalty = one_step_delta_loss(model, states, actions, normalizer)
    roll = rollout_loss(model, states, actions, normalizer, warmup_steps=warmup, horizon=current_horizon)
    
    total = float(loss_cfg.get("one_step_weight", 1.0)) * (one_step_nll + var_penalty) + float(loss_cfg.get("rollout_weight", 1.0)) * roll
    
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one_step_nll.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }
