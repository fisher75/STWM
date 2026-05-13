from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SelectorConditionedLocalEvidenceEncoderV3413(nn.Module):
    """由 non-oracle selector 调制的局部 semantic evidence encoder。"""

    def __init__(self, hidden_dim: int, semantic_dim: int, horizon: int = 32) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.semantic_value = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.semantic_key = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.query = nn.Sequential(nn.LayerNorm(hidden_dim * 2 + 4), nn.Linear(hidden_dim * 2 + 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.usage = nn.Sequential(nn.LayerNorm(hidden_dim + 6), nn.Linear(hidden_dim + 6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.evidence_to_semantic = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, semantic_dim))

    def forward(
        self,
        *,
        point_token: torch.Tensor,
        step_hidden: torch.Tensor,
        obs_semantic_measurements: torch.Tensor,
        obs_semantic_measurement_mask: torch.Tensor,
        obs_measurement_confidence: torch.Tensor | None,
        teacher_agreement_score: torch.Tensor | None,
        obs_vis: torch.Tensor,
        point_to_unit_assignment: torch.Tensor,
        selector_measurement_weight: torch.Tensor,
        selector_confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sem = torch.nan_to_num(obs_semantic_measurements.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mask = obs_semantic_measurement_mask.bool()
        b, m, t, _ = sem.shape
        h = step_hidden.shape[1]
        conf = obs_measurement_confidence.float().clamp(0.0, 1.0) if obs_measurement_confidence is not None else mask.float()
        agree = teacher_agreement_score.float().clamp(0.0, 1.0) if teacher_agreement_score is not None else conf
        if conf.dim() == 2:
            conf = conf[:, :, None].expand(-1, -1, t)
        if agree.dim() == 2:
            agree = agree[:, :, None].expand(-1, -1, t)
        selector_w = selector_measurement_weight.float().clamp_min(0.0) * mask.float()
        selector_w = selector_w / selector_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        selector_conf = selector_confidence.float().clamp(0.0, 1.0)
        vis = obs_vis.float().clamp(0.0, 1.0)
        key = self.semantic_key(sem)
        value = self.semantic_value(sem)
        t_pos = torch.linspace(0.0, 1.0, t, device=sem.device, dtype=sem.dtype).view(1, 1, t)
        h_pos = torch.linspace(0.0, 1.0, h, device=sem.device, dtype=sem.dtype).view(1, 1, h, 1)
        conf_mean = conf.mean(dim=2, keepdim=True)[:, :, None, :].expand(-1, -1, h, -1)
        agree_mean = agree.mean(dim=2, keepdim=True)[:, :, None, :].expand(-1, -1, h, -1)
        vis_mean = vis.mean(dim=2, keepdim=True)[:, :, None, :].expand(-1, -1, h, -1)
        sel_conf_h = selector_conf[:, :, None, None].expand(-1, -1, h, -1)
        q_in = torch.cat(
            [
                point_token[:, :, None, :].expand(-1, -1, h, -1),
                step_hidden[:, None, :, :].expand(-1, m, -1, -1),
                h_pos.expand(b, m, -1, -1),
                conf_mean,
                agree_mean * vis_mean,
                sel_conf_h,
            ],
            dim=-1,
        )
        query = self.query(q_in)
        logits = torch.einsum("bmhd,bmtd->bmht", query, key) / math.sqrt(query.shape[-1])
        logits = logits + 2.0 * torch.log(selector_w[:, :, None, :].clamp_min(1e-6))
        logits = logits + 1.0 * conf[:, :, None, :] + 1.0 * agree[:, :, None, :] + 0.15 * t_pos[:, :, None, :]
        logits = logits.masked_fill(~mask[:, :, None, :], -1e4)
        attn = torch.softmax(logits, dim=-1) * mask[:, :, None, :].float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        local_evidence = torch.einsum("bmht,bmtd->bmhd", attn, value)
        local_sem = F.normalize(self.evidence_to_semantic(local_evidence), dim=-1)
        entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1) / math.log(max(t, 2))
        max_attn = attn.max(dim=-1).values
        selector_alignment = (attn * selector_w[:, :, None, :]).sum(dim=-1)
        usage_in = torch.cat([local_evidence, entropy[..., None], max_attn[..., None], selector_alignment[..., None], conf_mean, agree_mean, sel_conf_h], dim=-1)
        usage_score = torch.sigmoid(self.usage(usage_in)).squeeze(-1)
        assign = point_to_unit_assignment.float()
        denom = assign.sum(dim=1).clamp_min(1e-6)
        unit_evidence = torch.einsum("bmu,bmhd->buhd", assign, local_evidence) / denom[:, :, None, None]
        unit_usage = torch.einsum("bmu,bmh->buh", assign, usage_score) / denom[:, :, None]
        return {
            "local_semantic_evidence": local_evidence,
            "local_semantic_evidence_embedding": local_sem,
            "semantic_evidence_attention_weights": attn,
            "semantic_measurement_usage_score": usage_score,
            "unit_semantic_evidence": unit_evidence,
            "unit_semantic_usage_score": unit_usage.clamp(0.0, 1.0),
            "attention_temporal_entropy": entropy,
            "attention_max_weight": max_attn,
            "selector_attention_alignment": selector_alignment,
        }
