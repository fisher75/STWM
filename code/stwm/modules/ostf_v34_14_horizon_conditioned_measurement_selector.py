from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class HorizonConditionedMeasurementSelectorV3414(nn.Module):
    """Horizon-conditioned observed-only semantic measurement reader。

    输入 future_trace_hidden [B,M,H,C] 作为 query，读取 observed semantic
    measurements [B,M,Tobs,D]，输出 measurement_weight [B,M,H,Tobs] 与
    selected_evidence [B,M,H,D]。future teacher embedding 不允许进入 forward。
    """

    def __init__(self, trace_hidden_dim: int, semantic_dim: int = 768, hidden_dim: int = 256) -> None:
        super().__init__()
        self.semantic_key = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.semantic_value = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, semantic_dim),
        )
        self.query = nn.Sequential(
            nn.LayerNorm(trace_hidden_dim + 8),
            nn.Linear(trace_hidden_dim + 8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.confidence = nn.Sequential(
            nn.LayerNorm(trace_hidden_dim + semantic_dim + 5),
            nn.Linear(trace_hidden_dim + semantic_dim + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        future_trace_hidden: torch.Tensor,
        rel_pred: torch.Tensor,
        future_visibility_prob: torch.Tensor,
        obs_semantic_measurements: torch.Tensor,
        obs_semantic_measurement_mask: torch.Tensor,
        obs_measurement_confidence: torch.Tensor,
        teacher_agreement_score: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sem = torch.nan_to_num(obs_semantic_measurements.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mask = obs_semantic_measurement_mask.bool()
        b, m, t, _ = sem.shape
        h = future_trace_hidden.shape[2]
        conf = obs_measurement_confidence.float().clamp(0.0, 1.0)
        agree = teacher_agreement_score.float().clamp(0.0, 1.0)
        if conf.dim() == 2:
            conf = conf[:, :, None].expand(-1, -1, t)
        if agree.dim() == 2:
            agree = agree[:, :, None].expand(-1, -1, t)
        obs_vis_f = obs_vis.float().clamp(0.0, 1.0)
        obs_conf_f = obs_conf.float().clamp(0.0, 1.0)
        coverage = mask.float().mean(dim=-1)
        obs_conf_mean = (obs_conf_f * obs_vis_f).sum(dim=-1) / obs_vis_f.sum(dim=-1).clamp_min(1.0)
        meas_conf_mean = conf.mean(dim=-1)
        meas_agree_mean = agree.mean(dim=-1)
        horizon_pos = torch.linspace(0.0, 1.0, h, device=sem.device, dtype=sem.dtype).view(1, 1, h, 1).expand(b, m, -1, -1)
        q_in = torch.cat(
            [
                future_trace_hidden,
                rel_pred.float().clamp(-16.0, 16.0),
                future_visibility_prob.float()[..., None].clamp(0.0, 1.0),
                horizon_pos,
                coverage[:, :, None, None].expand(-1, -1, h, -1),
                obs_conf_mean[:, :, None, None].expand(-1, -1, h, -1),
                meas_conf_mean[:, :, None, None].expand(-1, -1, h, -1),
                meas_agree_mean[:, :, None, None].expand(-1, -1, h, -1),
            ],
            dim=-1,
        )
        query = self.query(q_in)
        key = self.semantic_key(sem)
        value = self.semantic_value(sem)
        logits = torch.einsum("bmhd,bmtd->bmht", query, key) / math.sqrt(query.shape[-1])
        time_prior = torch.linspace(0.0, 1.0, t, device=sem.device, dtype=sem.dtype).view(1, 1, 1, t)
        reliability = conf * agree * obs_vis_f
        logits = logits + 1.2 * torch.log(reliability[:, :, None, :].clamp_min(1e-6)) + 0.1 * time_prior
        logits = logits.masked_fill(~mask[:, :, None, :], -1e4)
        weight = torch.softmax(logits, dim=-1) * mask[:, :, None, :].float()
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        selected_raw = torch.einsum("bmht,bmtd->bmhd", weight, sem)
        selected_residual = self.semantic_value(selected_raw)
        selected = F.normalize(selected_raw + selected_residual, dim=-1)
        entropy = -(weight.clamp_min(1e-8) * weight.clamp_min(1e-8).log()).sum(dim=-1) / math.log(max(t, 2))
        max_weight = weight.max(dim=-1).values
        conf_in = torch.cat(
            [
                future_trace_hidden,
                selected,
                entropy[..., None],
                max_weight[..., None],
                future_visibility_prob.float()[..., None].clamp(0.0, 1.0),
                meas_conf_mean[:, :, None, None].expand(-1, -1, h, -1),
                meas_agree_mean[:, :, None, None].expand(-1, -1, h, -1),
            ],
            dim=-1,
        )
        selector_confidence = torch.sigmoid(self.confidence(conf_in)).squeeze(-1)
        return {
            "measurement_weight": weight,
            "selected_evidence": selected,
            "selected_measurement_embedding": selected,
            "selector_confidence": selector_confidence,
            "selector_entropy": entropy,
            "selector_max_weight": max_weight,
            "selector_diagnostics": {
                "horizon_conditioned": True,
                "measurement_weight_shape": "B,M,H,Tobs",
                "selected_evidence_shape": "B,M,H,D",
                "future_teacher_embedding_input_allowed": False,
            },
        }
