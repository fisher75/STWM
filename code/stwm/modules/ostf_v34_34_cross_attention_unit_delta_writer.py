from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class CrossAttentionUnitDeltaWriterV3434(nn.Module):
    """Observed-only cross-attention writer for oracle unit-delta distillation.

    Each unit/horizon query reads the raw point/top-k evidence set through the
    current point-to-unit assignment.  Future teacher embeddings are never used
    as input; they are only supervision in the training script.
    """

    def __init__(
        self,
        trace_hidden_dim: int,
        semantic_dim: int = 768,
        hidden_dim: int = 256,
        max_delta_magnitude: float = 2.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.semantic_dim = int(semantic_dim)
        self.max_delta_magnitude = float(max_delta_magnitude)
        self.trace_proj = nn.Sequential(nn.LayerNorm(trace_hidden_dim), nn.Linear(trace_hidden_dim, hidden_dim))
        self.semantic_key = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden_dim))
        self.semantic_value = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden_dim))
        self.semantic_summary = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden_dim))
        self.old_unit_proj = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden_dim))
        self.query = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + 3),
            nn.Linear(hidden_dim * 4 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5 + 4),
            nn.Linear(hidden_dim * 5 + 4, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
        )
        self.direction = nn.Linear(hidden_dim * 2, semantic_dim)
        self.magnitude = nn.Linear(hidden_dim * 2, 1)

    @staticmethod
    def _unit_pool(assign: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        den = assign.sum(dim=1).clamp_min(1.0e-6)
        if values.dim() == 4:
            return torch.einsum("bmu,bmhd->buhd", assign, values) / den[:, :, None, None]
        if values.dim() == 3:
            return torch.einsum("bmu,bmd->bud", assign, values) / den[:, :, None]
        raise ValueError(f"unsupported value rank for unit pooling: {values.dim()}")

    def forward(self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], *, assignment: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        assign = assignment.float() if assignment is not None else out["point_to_unit_assignment"].float()
        future_trace = out["future_trace_hidden"].float()
        topk_raw = torch.nan_to_num(out["topk_raw_evidence"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        topk_weights = out["topk_weights"].float().clamp_min(1.0e-8)
        anchor = torch.nan_to_num(batch["obs_semantic_measurements"].float(), nan=0.0).mean(dim=2)
        if "topk_raw_evidence_embedding" in out:
            point_evidence = out["topk_raw_evidence_embedding"].float()
        else:
            point_evidence = (topk_raw * topk_weights[..., None]).sum(dim=3)
        old_unit = out["unit_memory"].float()
        b, m, h, k, _ = topk_raw.shape
        u = assign.shape[-1]

        unit_trace = self._unit_pool(assign, future_trace)
        unit_anchor = self._unit_pool(assign, anchor[:, :, None, :].expand(-1, -1, h, -1))
        unit_evidence = self._unit_pool(assign, point_evidence)
        unit_conf = out.get("unit_confidence", torch.ones((b, u), device=assign.device, dtype=assign.dtype)).float()
        unit_usage = out.get("assignment_usage_score", torch.ones((b, u, h), device=assign.device, dtype=assign.dtype)).float()
        point_sem_usage = out.get("semantic_measurement_usage_score", torch.ones((b, m, h), device=assign.device, dtype=assign.dtype)).float()
        unit_sem_usage = self._unit_pool(assign, point_sem_usage).clamp(0.0, 1.0)

        trace_h = self.trace_proj(unit_trace)
        anchor_h = self.semantic_summary(unit_anchor)
        evidence_h = self.semantic_summary(unit_evidence)
        old_h = self.old_unit_proj(old_unit)
        q_in = torch.cat(
            [
                trace_h,
                anchor_h,
                evidence_h,
                old_h,
                unit_conf[:, :, None, None].expand(-1, -1, h, 1),
                unit_usage[:, :, :, None].clamp(0.0, 1.0),
                unit_sem_usage[..., None],
            ],
            dim=-1,
        )
        query = self.query(q_in)
        key = self.semantic_key(topk_raw)
        value = self.semantic_value(topk_raw)
        score = torch.einsum("buhc,bmhkc->buhmk", query, key) / math.sqrt(float(self.hidden_dim))
        log_prior = (
            assign.clamp_min(1.0e-8).permute(0, 2, 1)[:, :, None, :, None].log()
            + topk_weights.clamp_min(1.0e-8).permute(0, 2, 1, 3)[:, None].log()
        )
        attn = torch.softmax((score + log_prior).flatten(start_dim=3), dim=-1).reshape(b, u, h, m, k)
        context = torch.einsum("buhmk,bmhkc->buhc", attn, value)
        attn_entropy = -(attn.flatten(start_dim=3).clamp_min(1.0e-8) * attn.flatten(start_dim=3).clamp_min(1.0e-8).log()).sum(dim=-1)
        attn_max = attn.flatten(start_dim=3).max(dim=-1).values
        assignment_mass = torch.einsum("bmu,bmh->buh", assign, point_sem_usage).clamp_min(0.0)
        assignment_mass = assignment_mass / assign.sum(dim=1).clamp_min(1.0e-6)[:, :, None]
        fuse_in = torch.cat(
            [
                query,
                context,
                trace_h,
                anchor_h,
                old_h,
                attn_entropy[..., None],
                attn_max[..., None],
                unit_usage[:, :, :, None].clamp(0.0, 1.0),
                assignment_mass[..., None].clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        hidden = self.fuse(fuse_in)
        direction = F.normalize(self.direction(hidden), dim=-1)
        magnitude = self.max_delta_magnitude * torch.sigmoid(self.magnitude(hidden))
        unit_delta = direction * magnitude
        return {
            "unit_delta": unit_delta,
            "attention_entropy": attn_entropy,
            "attention_max": attn_max,
            "assignment_mass": assignment_mass,
            "unit_query": query,
            "unit_context": context,
        }
