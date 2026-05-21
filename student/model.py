"""Student world model.

Students may replace this residual MLP with a GRU or another dynamics model,
but the public interface must stay the same.
"""

from __future__ import annotations

import torch
from torch import nn


class LayerNormGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_size = hidden_dim
        # Combine gates for computational efficiency
        self.input_to_gates = nn.Linear(input_dim, 3 * hidden_dim, bias=False)
        self.hidden_to_gates = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        
        # Layer norms for each gate component to keep features bounded
        self.ln_input = nn.LayerNorm(3 * hidden_dim)
        self.ln_hidden = nn.LayerNorm(3 * hidden_dim)
        self.ln_candidate = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Compute gates with normalization applied prior to activation
        gates_x = self.ln_input(self.input_to_gates(x))
        gates_h = self.ln_hidden(self.hidden_to_gates(h))
        
        r_x, z_x, n_x = gates_x.chunk(3, dim=-1)
        r_h, z_h, n_h = gates_h.chunk(3, dim=-1)
        
        reset_gate = torch.sigmoid(r_x + r_h)
        update_gate = torch.sigmoid(z_x + z_h)
        
        # Normalize the candidate state transition
        candidate = torch.tanh(n_x + reset_gate * n_h)
        candidate = self.ln_candidate(candidate)
        
        next_h = (1.0 - update_gate) * h + update_gate * candidate
        return next_h


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_gru: bool = True,
        delta_limit: float = 3.0,
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        
        # 1. Define the multi-layer MLP Encoder block
        in_dim = obs_dim + act_dim  # Starts at 5 (4 + 1)
        layers: list[nn.Module] = []
        for _ in range(int(num_layers)):
            layers += [
                nn.Linear(in_dim, hidden_dim), 
                nn.LayerNorm(hidden_dim), 
                nn.SiLU()
            ]
            in_dim = hidden_dim  # Transitions to 256 for subsequent layers
        self.encoder = nn.Sequential(*layers)
        
        # 2. Recurrent core processing
        self.gru = LayerNormGRUCell(hidden_dim, hidden_dim) if self.use_gru else None
        
        # 3. Dual prediction heads (Input dimension: hidden_dim + obs_dim + act_dim = 261)
        head_in_dim = hidden_dim + obs_dim + act_dim
        self.mu_head = nn.Linear(head_in_dim, obs_dim)
        self.logvar_head = nn.Linear(head_in_dim, obs_dim)
        
        # 4. Pure linear kinematics baseline shortcut mapping (5 -> 4)
        self.kinematic_shortcut = nn.Linear(obs_dim + act_dim, obs_dim)
        
        # Internal state to cache log-variance safely across calls during training
        self._current_logvar: torch.Tensor | None = None

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru or self.gru is None:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        # Create the raw 5D input state-action tensor [B, 5]
        raw_state_action = torch.cat([obs_norm, act_norm], dim=-1)
        
        # Compute the baseline kinematic projection [B, 4]
        baseline_delta = self.kinematic_shortcut(raw_state_action)
        
        # Pass the 5D input safely into the encoder -> outputs a 256D feature tensor [B, 256]
        feat = self.encoder(raw_state_action)
        
        # Route features through the recurrent GRU loop
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            assert hidden is not None  # Eliminates TorchScript type warnings
            hidden = self.gru(feat, hidden)
            feat = hidden  # Retains 256 dimensions
            
        # Direct Input Conditioning: Concatenate GRU feature with original observations and actions [B, 261]
        head_input = torch.cat([feat, obs_norm, act_norm], dim=-1)
        
        # Predict the delta mean distribution by adding non-linear adjustments to the linear baseline
        raw_mu = baseline_delta + 0.1 * self.mu_head(head_input)
        delta = self.delta_limit * torch.tanh(raw_mu / self.delta_limit)
        
        # Predict distribution log-variance and cache it for compute_loss lookup
        logvar = torch.clamp(self.logvar_head(head_input), min=-10.0, max=2.0)
        self._current_logvar = logvar
        
        return delta, hidden
