from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass
class SemanticFusionConfig:
    hidden_dim: int = 1152
    semantic_dim: int = 256
    dropout: float = 0.1


class SemanticFusion(nn.Module):
    def __init__(self, cfg: SemanticFusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.semantic_proj = nn.Linear(int(cfg.semantic_dim), int(cfg.hidden_dim))
        self.gate = nn.Linear(int(cfg.hidden_dim) * 2, int(cfg.hidden_dim))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.norm = nn.LayerNorm(int(cfg.hidden_dim))

    def forward(
        self,
        backbone_hidden: torch.Tensor,
        semantic_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if backbone_hidden.ndim != 4:
            raise ValueError(f"backbone_hidden must be [B,T,K,H], got {tuple(backbone_hidden.shape)}")
        if semantic_tokens.ndim != 3:
            raise ValueError(f"semantic_tokens must be [B,K,S], got {tuple(semantic_tokens.shape)}")

        b, t, k, h = backbone_hidden.shape
        if semantic_tokens.shape[0] != b or semantic_tokens.shape[1] != k:
            raise ValueError(
                "semantic token shape mismatch: "
                f"hidden={tuple(backbone_hidden.shape)} semantic={tuple(semantic_tokens.shape)}"
            )

        sem_h = self.semantic_proj(semantic_tokens)
        sem_h = sem_h[:, None, :, :].expand(b, t, k, h)

        gate = torch.sigmoid(self.gate(torch.cat([backbone_hidden, sem_h], dim=-1)))
        fused = backbone_hidden + self.dropout(gate * sem_h)
        fused = self.norm(fused)

        if token_mask is not None:
            mask = token_mask[:, None, :, None].to(torch.bool)
            fused = torch.where(mask, fused, torch.zeros_like(fused))

        aux = {
            "gate_mean": float(gate.detach().mean().cpu().item()),
            "gate_std": float(gate.detach().std().cpu().item()),
        }
        return fused, aux
