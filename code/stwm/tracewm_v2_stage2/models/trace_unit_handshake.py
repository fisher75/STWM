from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import torch
from torch import nn


@dataclass
class TraceUnitHandshakeConfig:
    unit_dim: int = 384
    handshake_dim: int = 128
    layers: int = 1
    writeback: str = "dyn_only"


class TraceUnitHandshake(nn.Module):
    def __init__(self, cfg: TraceUnitHandshakeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(int(cfg.unit_dim), int(cfg.handshake_dim))
        self.k_proj = nn.Linear(int(cfg.unit_dim), int(cfg.handshake_dim))
        self.v_proj = nn.Linear(int(cfg.unit_dim), int(cfg.handshake_dim))
        self.out_proj = nn.Linear(int(cfg.handshake_dim), int(cfg.unit_dim))
        self.gate = nn.Linear(int(cfg.unit_dim) * 2, int(cfg.unit_dim))
        self.norm = nn.LayerNorm(int(cfg.unit_dim))

    def forward(self, *, z_dyn: torch.Tensor, z_sem: torch.Tensor, unit_presence: torch.Tensor) -> Dict[str, torch.Tensor | Dict[str, float]]:
        if int(self.cfg.layers) <= 0:
            return {
                "z_dyn": z_dyn,
                "metrics": {
                    "handshake_residual_norm_mean": 0.0,
                    "handshake_attention_entropy_mean": 0.0,
                },
            }
        q = self.q_proj(z_dyn)
        k = self.k_proj(z_sem)
        v = self.v_proj(z_sem)
        logits = torch.einsum("btmd,btnd->btmn", q, k) / math.sqrt(max(int(self.cfg.handshake_dim), 1))
        unit_mask = unit_presence[:, :, None, :].expand_as(logits)
        logits = logits.masked_fill(~unit_mask, -1e9)
        attn = torch.softmax(logits, dim=-1)
        attn = torch.where(unit_mask, attn, torch.zeros_like(attn))
        context = torch.einsum("btmn,btnd->btmd", attn, v)
        residual = self.out_proj(context)
        gate = torch.sigmoid(self.gate(torch.cat([z_dyn, residual], dim=-1)))
        updated = self.norm(z_dyn + gate * residual)
        attn_entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1)
        mask = unit_presence
        denom = mask.float().sum().clamp_min(1.0)
        metrics = {
            "handshake_residual_norm_mean": float(torch.linalg.norm((gate * residual), dim=-1).sum().detach().cpu().item() / float(denom.detach().cpu().item())),
            "handshake_attention_entropy_mean": float((attn_entropy * mask.float()).sum().detach().cpu().item() / float(denom.detach().cpu().item())),
        }
        return {"z_dyn": updated, "metrics": metrics}
