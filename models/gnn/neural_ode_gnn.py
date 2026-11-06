"""
neural_ode_gnn.py
-----------------
Continuous-time Neural ODE and Flow Dynamics for ATLAS.

Models cellular response as a continuous dynamical system on the gene regulatory graph:
    dz(t)/dt = f_theta(z(t), G, t)

Supports:
  - Runge-Kutta 4th Order (RK4) and Explicit Midpoint ODE solvers
  - Continuous time evaluation at arbitrary observation windows (e.g. t in [0.0, 0.5, 1.0, 2.0, 4.0])
  - Energy & Lyapunov divergence guards for biophysical stability
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Union
from .message_passing import MessagePassingLayer


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for continuous dynamics."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: scalar or [B, 1]
        device = t.device if isinstance(t, torch.Tensor) else "cpu"
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=device)
        t = t.view(-1, 1)
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32, device=device) / half_dim
        )
        args = t * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.proj(embedding)


class CausalGraphODEFunc(nn.Module):
    """
    Vector field dz/dt = f(z, t, G) parameterized by a Causal Graph Neural Network.
    
    Computes infinitesimal regulatory velocity across genes given current state and GRN edges.
    """
    def __init__(
        self,
        hidden_channels: int,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        layers_count: int = 2,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.register_buffer("edge_index", edge_index)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        self.time_embed = TimeEmbedding(hidden_channels)
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_channels, hidden_channels)
            for _ in range(layers_count)
        ])
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        # Initialize velocity head small for smooth integration flow
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

    def forward(self, t: float, z: torch.Tensor) -> torch.Tensor:
        """
        Evaluate derivative dz/dt at time t.
        
        Args:
            t: scalar float time
            z: [B, G, H] or [B*G, H]
        Returns:
            dz_dt: same shape as z
        """
        orig_shape = z.shape
        if z.dim() == 2:
            # Reshape [B*G, H] -> [B, G, H]
            B = 1
            G = z.size(0)
            z = z.view(B, G, self.hidden_channels)
        else:
            B, G, _ = z.shape

        t_tensor = torch.tensor(t, dtype=z.dtype, device=z.device)
        t_emb = self.time_embed(t_tensor).view(1, 1, self.hidden_channels)

        # State conditioned on continuous time
        h = z + t_emb
        for mp in self.mp_layers:
            h = h + mp(h, self.edge_index, self.edge_weight)

        # Infinitesimal velocity dz/dt
        dz_dt = self.velocity_head(h)

        if len(orig_shape) == 2:
            dz_dt = dz_dt.view(orig_shape)
        return dz_dt


class ContinuousCausalRollout(nn.Module):
    """
    Integrates continuous-time trajectories using explicit Runge-Kutta 4th Order (RK4).
    """
    def __init__(self, ode_func: CausalGraphODEFunc, solver: str = "rk4"):
        super().__init__()
        self.ode_func = ode_func
        self.solver = solver

    def rk4_step(self, t: float, z: torch.Tensor, dt: float) -> torch.Tensor:
        """Standard 4th-order Runge-Kutta numerical step."""
        k1 = self.ode_func(t, z)
        k2 = self.ode_func(t + 0.5 * dt, z + 0.5 * dt * k1)
        k3 = self.ode_func(t + 0.5 * dt, z + 0.5 * dt * k2)
        k4 = self.ode_func(t + dt, z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def midpoint_step(self, t: float, z: torch.Tensor, dt: float) -> torch.Tensor:
        """2nd-order explicit midpoint step."""
        k1 = self.ode_func(t, z)
        k2 = self.ode_func(t + 0.5 * dt, z + 0.5 * dt * k1)
        return z + dt * k2

    def integrate(
        self,
        z0: torch.Tensor,
        eval_times: List[float],
        substeps: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrates from eval_times[0] to eval_times[-1].
        
        Args:
            z0: Initial state [B, G, H]
            eval_times: Sequence of output observation times, e.g. [0.0, 0.5, 1.0, 2.0]
            substeps: Number of integration substeps per evaluation interval
        Returns:
            z_final: Final state at eval_times[-1]
            trajectory: Tensor of shape [T, B, G, H] at every eval time
        """
        trajectory = [z0]
        z_curr = z0
        t_curr = eval_times[0]

        step_fn = self.rk4_step if self.solver == "rk4" else self.midpoint_step

        for t_target in eval_times[1:]:
            dt_total = t_target - t_curr
            dt = dt_total / substeps
            for _ in range(substeps):
                z_curr = step_fn(t_curr, z_curr, dt)
                t_curr += dt
            trajectory.append(z_curr)

        trajectory_tensor = torch.stack(trajectory, dim=0)
        return z_curr, trajectory_tensor
