from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


@dataclass
class TraceUnitBroadcastConfig:
    hidden_dim: int = 1152
    unit_dim: int = 384
    residual_weight: float = 0.35
    stopgrad_semantic: bool = False


class TraceUnitBroadcast(nn.Module):
    def __init__(self, cfg: TraceUnitBroadcastConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(int(cfg.unit_dim) * 2, int(cfg.hidden_dim))
        self.norm = nn.LayerNorm(int(cfg.hidden_dim))

    def forward(
        self,
        *,
        backbone_hidden: torch.Tensor,
        assignment: torch.Tensor,
        z_dyn: torch.Tensor,
        z_sem: torch.Tensor,
    ) -> Dict[str, torch.Tensor | Dict[str, float]]:
        sem = z_sem.detach() if bool(self.cfg.stopgrad_semantic) else z_sem
        unit_cat = torch.cat([z_dyn, sem], dim=-1)
        unit_residual = self.proj(unit_cat)
        token_residual = torch.einsum("btkm,btmd->btkd", assignment, unit_residual)
        enhanced = self.norm(backbone_hidden + float(self.cfg.residual_weight) * token_residual)
        metrics = {
            "broadcast_residual_norm_mean": float(torch.linalg.norm(token_residual, dim=-1).detach().mean().cpu().item()),
        }
        return {
            "enhanced_hidden": enhanced,
            "token_residual": token_residual,
            "metrics": metrics,
        }
