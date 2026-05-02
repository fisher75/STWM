from __future__ import annotations

import torch
from torch import nn


class DenseToSemanticTraceUnitEncoder(nn.Module):
    """Pool object-internal point trajectories into one semantic trace-unit token."""

    def __init__(self, obs_len: int = 8, hidden_dim: int = 128, unit_dim: int = 128) -> None:
        super().__init__()
        self.obs_len = int(obs_len)
        self.point_mlp = nn.Sequential(
            nn.Linear(self.obs_len * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Sequential(nn.Linear(hidden_dim * 2, unit_dim), nn.GELU())

    def forward(self, obs_points: torch.Tensor, obs_valid: torch.Tensor) -> torch.Tensor:
        # obs_points: [B, M, Tobs, 2], obs_valid: [B, M, Tobs]
        b, m, t, _ = obs_points.shape
        valid = obs_valid.float().clamp(0, 1)
        x = torch.cat([obs_points, valid[..., None]], dim=-1).reshape(b, m, t * 3)
        h = self.point_mlp(x)
        w = (valid.mean(dim=-1) > 0).float()
        denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (h * w[..., None]).sum(dim=1) / denom
        masked = h.masked_fill(~(w[..., None] > 0), -1e4)
        maxv = masked.max(dim=1).values
        return self.out(torch.cat([mean, maxv], dim=-1))


class OSTFMultiTracePilot(nn.Module):
    """Small Phase-1 pilot head that predicts future object boxes then expands to M points."""

    def __init__(self, obs_len: int = 8, horizon: int = 8, unit_dim: int = 128) -> None:
        super().__init__()
        self.obs_len = int(obs_len)
        self.horizon = int(horizon)
        self.encoder = DenseToSemanticTraceUnitEncoder(obs_len=obs_len, hidden_dim=unit_dim, unit_dim=unit_dim)
        self.box_head = nn.Sequential(
            nn.Linear(unit_dim, unit_dim),
            nn.GELU(),
            nn.Linear(unit_dim, horizon * 4),
        )

    def forward(self, obs_points: torch.Tensor, obs_valid: torch.Tensor, rel_xy: torch.Tensor) -> torch.Tensor:
        unit = self.encoder(obs_points, obs_valid)
        boxes = self.box_head(unit).reshape(obs_points.shape[0], self.horizon, 4).sigmoid()
        center = boxes[..., :2]
        size = boxes[..., 2:].clamp_min(0.01)
        rel = rel_xy[:, None, :, :] - 0.5
        return center[:, :, None, :] + rel * size[:, :, None, :]
