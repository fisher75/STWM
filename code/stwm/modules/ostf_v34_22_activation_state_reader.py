from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ActivationStateReaderV3422(nn.Module):
    """Observed-only activation-state reader for top-k residual memory。

    该模块只读取过去观测、未来 trace hidden、top-k observed semantic evidence、
    pointwise belief 与 residual proposal；不读取 future teacher embedding。
    它用于 predictability probe，不是 learned residual gate。
    """

    def __init__(self, trace_hidden_dim: int, semantic_dim: int = 768, hidden_dim: int = 192) -> None:
        super().__init__()
        self.trace_query = nn.Sequential(
            nn.LayerNorm(trace_hidden_dim),
            nn.Linear(trace_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.semantic_query = nn.Sequential(
            nn.LayerNorm(semantic_dim * 2 + 6),
            nn.Linear(semantic_dim * 2 + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.evidence_key = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden_dim))
        self.evidence_value = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.unit_value = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.state = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + 8),
            nn.Linear(hidden_dim * 4 + 8, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict(
            {
                "aligned": nn.Linear(hidden_dim, 1),
                "utility": nn.Linear(hidden_dim, 1),
                "benefit": nn.Linear(hidden_dim, 1),
            }
        )

    def forward(
        self,
        *,
        future_trace_hidden: torch.Tensor,
        topk_raw_evidence: torch.Tensor,
        topk_weights: torch.Tensor,
        pointwise_semantic_belief: torch.Tensor,
        assignment_bound_residual: torch.Tensor,
        unit_memory: torch.Tensor,
        point_to_unit_assignment: torch.Tensor,
        semantic_measurement_usage_score: torch.Tensor,
        selector_confidence: torch.Tensor,
        selector_entropy: torch.Tensor,
        selector_measurement_weight: torch.Tensor,
        assignment_usage_score: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pointwise = F.normalize(torch.nan_to_num(pointwise_semantic_belief.float()), dim=-1)
        residual = torch.nan_to_num(assignment_bound_residual.float())
        residual_dir = F.normalize(residual, dim=-1)
        raw = F.normalize(torch.nan_to_num(topk_raw_evidence.float()), dim=-1)
        weights = topk_weights.float().clamp_min(1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        trace_q = self.trace_query(future_trace_hidden.float())

        evidence_mean = F.normalize((raw * weights[..., None]).sum(dim=3), dim=-1)
        residual_norm = residual.norm(dim=-1, keepdim=True).clamp(0.0, 1.0)
        point_residual_cos = (pointwise * residual_dir).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        evidence_point_cos = (evidence_mean * pointwise).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        evidence_residual_cos = (evidence_mean * residual_dir).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        selector_max = selector_measurement_weight.float().max(dim=-1).values[..., None].clamp(0.0, 1.0)
        semantic_q_in = torch.cat(
            [
                pointwise,
                residual_dir,
                residual_norm,
                point_residual_cos,
                evidence_point_cos,
                evidence_residual_cos,
                selector_confidence.float()[..., None].clamp(0.0, 1.0),
                semantic_measurement_usage_score.float()[..., None].clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        semantic_q = self.semantic_query(semantic_q_in)
        query = trace_q + semantic_q

        key = self.evidence_key(raw)
        value = self.evidence_value(raw)
        logits = torch.einsum("bmhd,bmhkd->bmhk", query, key) / math.sqrt(float(query.shape[-1]))
        logits = logits + torch.log(weights.clamp_min(1e-8))
        attn = torch.softmax(logits, dim=-1)
        evidence_context = torch.einsum("bmhk,bmhkd->bmhd", attn, value)

        unit_context_sem = torch.einsum("bmu,buhd->bmhd", point_to_unit_assignment.float(), unit_memory.float())
        unit_context = self.unit_value(unit_context_sem)
        assignment_usage = torch.einsum("bmu,buh->bmh", point_to_unit_assignment.float(), assignment_usage_score.float()).clamp(0.0, 1.0)
        entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1) / math.log(max(attn.shape[-1], 2))
        max_attn = attn.max(dim=-1).values
        scalar = torch.cat(
            [
                residual_norm,
                point_residual_cos,
                evidence_point_cos,
                evidence_residual_cos,
                selector_confidence.float()[..., None].clamp(0.0, 1.0),
                selector_entropy.float()[..., None].clamp(0.0, 1.0),
                selector_max,
                assignment_usage[..., None],
            ],
            dim=-1,
        )
        activation_state = self.state(torch.cat([trace_q, semantic_q, evidence_context, unit_context, scalar], dim=-1))
        logits_out = {name: head(activation_state).squeeze(-1) for name, head in self.heads.items()}
        return {
            "activation_state": activation_state,
            "activation_logits": logits_out,
            "evidence_attention": attn,
            "evidence_attention_entropy": entropy,
            "evidence_attention_max": max_attn,
            "activation_reader_diagnostics": {
                "observed_only": True,
                "uses_future_trace_hidden": True,
                "uses_topk_raw_evidence_set": True,
                "future_teacher_embedding_input_allowed": False,
                "learned_gate_training_ran": False,
            },
        }
