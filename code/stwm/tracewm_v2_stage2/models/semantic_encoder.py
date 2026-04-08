from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class SemanticEncoderConfig:
    input_dim: int = 10
    hidden_dim: int = 128
    output_dim: int = 256
    dropout: float = 0.1


class SemanticEncoder(nn.Module):
    def __init__(self, cfg: SemanticEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(int(cfg.input_dim), int(cfg.hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.hidden_dim), int(cfg.output_dim)),
            nn.LayerNorm(int(cfg.output_dim)),
        )

    def forward(self, semantic_features: torch.Tensor) -> torch.Tensor:
        if semantic_features.ndim != 3:
            raise ValueError(f"semantic_features must be [B,K,F], got {tuple(semantic_features.shape)}")
        return self.net(semantic_features)
