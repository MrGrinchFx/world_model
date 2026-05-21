"""Student world model - Probabilistic Interface-Safe Edition."""

from __future__ import annotations

import torch
from torch import nn

class LayerNormGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_size = hidden_dim
        self.input_to_gates = nn.Linear(input_dim, 3 * hidden_dim, bias=False)
        self.hidden_to_gates = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        
        self.ln_input = nn.LayerNorm(3 * hidden_dim)
        self.ln_hidden = nn.LayerNorm(3 * hidden_dim)
        self.ln_candidate = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gates_x = self.ln_input(self.input_to_gates(x))
        gates_h = self.ln_hidden(self.hidden_to_gates(h))
        
        r_x, z_x, n_x = gates_x.chunk(3, dim=-1)
        r_h, z_h, n_h = gates_h.chunk(3, dim=-1)
        
        reset_gate = torch.sigmoid(r_x + r_h)
        update_gate = torch.sigmoid(z_x + z_h)
        
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
        
        in_dim = obs_dim + act_dim
        layers: list[nn.Module] = []
# Inside your StudentWorldModel.__init__ loop:
        for i in range(int(num_layers)):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1)  # Stochastically drops out 10% of features during training
            ]
    in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        
        self.gru = LayerNormGRUCell(hidden_dim, hidden_dim) if self.use_gru else None
        
        # Explicit separation of heads
        self.mu_head = nn.Linear(hidden_dim + obs_dim + act_dim, obs_dim)
        self.logvar_head = nn.Linear(hidden_dim + obs_dim + act_dim, obs_dim)
        
        self.kinematic_shortcut = nn.Linear(obs_dim + act_dim, obs_dim)
        
        # State variable to safely pass variance to our custom loss function
        self._current_logvar: torch.Tensor | None = None

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        raw_input = torch.cat([obs_norm, act_norm], dim=-1)
        baseline_delta = self.kinematic_shortcut(raw_input)
        
        feat = self.encoder(raw_input)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            assert hidden is not None
            hidden = self.gru(feat, hidden)
            feat = hidden
            
        head_input = torch.cat([feat, obs_norm, act_norm], dim=-1)
        
        # Compute mean prediction
        raw_mu = baseline_delta + 0.1 * self.mu_head(head_input)
        delta = self.delta_limit * torch.tanh(raw_mu / self.delta_limit)
        
        # Compute log-variance and cache it internally for the loss function
        logvar = torch.clamp(self.logvar_head(head_input), min=-10.0, max=2.0)
        self._current_logvar = logvar
        
        # Return only delta and hidden to keep external scripts completely functional!
        return delta, hidden
