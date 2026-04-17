from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import torch
from torch import nn


@dataclass
class TraceUnitTokenizerConfig:
    hidden_dim: int = 1152
    semantic_dim: int = 256
    state_dim: int = 8
    unit_dim: int = 384
    unit_count: int = 16
    slot_iters: int = 3
    assignment_topk: int = 2
    assignment_temperature: float = 0.7
    use_instance_prior_bias: bool = True


class TraceUnitTokenizer(nn.Module):
    def __init__(self, cfg: TraceUnitTokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        prior_dim = 4
        self.input_proj = nn.Linear(
            int(cfg.hidden_dim) + int(cfg.semantic_dim) + int(cfg.state_dim) + prior_dim,
            int(cfg.unit_dim),
        )
        self.unit_queries = nn.Parameter(
            torch.randn(int(cfg.unit_count), int(cfg.unit_dim)) / math.sqrt(max(int(cfg.unit_dim), 1))
        )
        self.update = nn.Sequential(
            nn.LayerNorm(int(cfg.unit_dim)),
            nn.Linear(int(cfg.unit_dim), int(cfg.unit_dim) * 2),
            nn.GELU(),
            nn.Linear(int(cfg.unit_dim) * 2, int(cfg.unit_dim)),
        )
        self.norm = nn.LayerNorm(int(cfg.unit_dim))

    def _sparse_assignment(self, logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        topk = max(min(int(self.cfg.assignment_topk), int(logits.shape[-1])), 1)
        top_idx = torch.topk(logits, k=topk, dim=-1, largest=True, sorted=False).indices
        sparse_mask = torch.zeros_like(logits, dtype=torch.bool)
        sparse_mask.scatter_(-1, top_idx, True)
        logits = logits.masked_fill(~sparse_mask, -1e9)
        logits = logits / max(float(self.cfg.assignment_temperature), 1e-4)
        assign = torch.softmax(logits, dim=-1)
        return torch.where(valid[..., None], assign, torch.zeros_like(assign))

    def forward(
        self,
        *,
        backbone_hidden: torch.Tensor,
        state_seq: torch.Tensor,
        semantic_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        semantic_objectness_score: torch.Tensor | None = None,
        semantic_instance_valid: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | Dict[str, float]]:
        if backbone_hidden.ndim != 4:
            raise ValueError(f"backbone_hidden must be [B,T,K,H], got {tuple(backbone_hidden.shape)}")
        if state_seq.shape[:3] != backbone_hidden.shape[:3]:
            raise ValueError(
                f"state/backbone mismatch: state={tuple(state_seq.shape)} hidden={tuple(backbone_hidden.shape)}"
            )
        if semantic_tokens.ndim != 3:
            raise ValueError(f"semantic_tokens must be [B,K,S], got {tuple(semantic_tokens.shape)}")

        bsz, t_len, k_len, _ = backbone_hidden.shape
        semantic_expanded = semantic_tokens[:, None, :, :].expand(bsz, t_len, k_len, semantic_tokens.shape[-1])
        time_norm = torch.linspace(
            0.0,
            1.0,
            steps=t_len,
            device=backbone_hidden.device,
            dtype=backbone_hidden.dtype,
        )[None, :, None, None].expand(bsz, t_len, k_len, 1)
        objectness = (
            semantic_objectness_score[:, None, :, None].expand(bsz, t_len, k_len, 1)
            if isinstance(semantic_objectness_score, torch.Tensor)
            else torch.zeros((bsz, t_len, k_len, 1), device=backbone_hidden.device, dtype=backbone_hidden.dtype)
        )
        instance_valid = (
            semantic_instance_valid[:, :, : min(t_len, semantic_instance_valid.shape[-1])]
            if isinstance(semantic_instance_valid, torch.Tensor)
            else None
        )
        if isinstance(instance_valid, torch.Tensor):
            if instance_valid.shape[-1] < t_len:
                pad = torch.zeros(
                    (bsz, k_len, t_len - int(instance_valid.shape[-1])),
                    device=instance_valid.device,
                    dtype=instance_valid.dtype,
                )
                instance_valid = torch.cat([instance_valid, pad], dim=-1)
            instance_valid = instance_valid[:, :, :t_len].permute(0, 2, 1)[..., None].to(backbone_hidden.dtype)
        else:
            instance_valid = torch.zeros((bsz, t_len, k_len, 1), device=backbone_hidden.device, dtype=backbone_hidden.dtype)
        token_valid = token_mask[:, None, :].expand(bsz, t_len, k_len)

        priors = torch.cat([objectness, instance_valid, time_norm, 1.0 - time_norm], dim=-1)
        token_features = self.input_proj(
            torch.cat([backbone_hidden, semantic_expanded, state_seq, priors], dim=-1)
        )

        units = self.unit_queries[None, :, :].expand(bsz, -1, -1)
        bias = 0.0
        if bool(self.cfg.use_instance_prior_bias):
            bias = 0.10 * (objectness[..., 0] + instance_valid[..., 0])

        assignment = torch.zeros((bsz, t_len, k_len, int(self.cfg.unit_count)), device=token_features.device, dtype=token_features.dtype)
        for _ in range(max(int(self.cfg.slot_iters), 1)):
            logits = torch.einsum("btkd,bmd->btkm", token_features, units) / math.sqrt(max(int(self.cfg.unit_dim), 1))
            if isinstance(bias, torch.Tensor):
                logits = logits + bias[..., None]
            assignment = self._sparse_assignment(logits, token_valid)
            denom = assignment.sum(dim=(1, 2)).unsqueeze(-1).clamp_min(1e-6)
            updates = torch.einsum("btkm,btkd->bmd", assignment, token_features) / denom
            units = self.norm(units + self.update(updates))

        assign_entropy = -(assignment.clamp_min(1e-8) * assignment.clamp_min(1e-8).log()).sum(dim=-1)
        assign_entropy = torch.where(token_valid, assign_entropy, torch.zeros_like(assign_entropy))
        top2_mass = torch.topk(assignment, k=min(2, assignment.shape[-1]), dim=-1, largest=True, sorted=True).values
        secondary_mass = top2_mass[..., 1] if top2_mass.shape[-1] > 1 else torch.zeros_like(assign_entropy)
        unit_mass = assignment.sum(dim=(1, 2))
        active_unit_count = (unit_mass > 1e-3).float().sum(dim=-1)
        metrics = {
            "assignment_entropy_mean": float(assign_entropy.sum().detach().cpu().item() / max(float(token_valid.float().sum().item()), 1.0)),
            "actual_top2_assignment_ratio": float(((secondary_mass > 0.05) & token_valid).float().sum().detach().cpu().item() / max(float(token_valid.float().sum().item()), 1.0)),
            "active_unit_count_mean": float(active_unit_count.detach().mean().cpu().item()),
            "assignment_sparsity_mean": float((assignment > 1e-5).float().sum().detach().cpu().item() / max(float(token_valid.float().sum().item() * assignment.shape[-1]), 1.0)),
        }
        return {
            "token_features": token_features,
            "assignment": assignment,
            "unit_queries": units,
            "token_valid": token_valid,
            "metrics": metrics,
        }
