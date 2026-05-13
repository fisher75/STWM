from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class NonOracleMeasurementSelectorV3413(nn.Module):
    """Observed-only semantic measurement selector。

    输入只包含 observed semantic measurements、confidence/agreement、真实 trace
    visibility/confidence 与 observed trace 运动统计。future teacher embedding 只允许在
    训练 loss 中作为 supervision，不允许进入 forward。
    """

    def __init__(self, semantic_dim: int = 768, hidden_dim: int = 256) -> None:
        super().__init__()
        self.semantic_value = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, semantic_dim),
        )
        self.score = nn.Sequential(
            nn.LayerNorm(semantic_dim + 10),
            nn.Linear(semantic_dim + 10, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.confidence = nn.Sequential(
            nn.LayerNorm(semantic_dim + 6),
            nn.Linear(semantic_dim + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _trace_stats(obs_points: torch.Tensor, obs_vis: torch.Tensor, obs_conf: torch.Tensor) -> torch.Tensor:
        vis = obs_vis.float()
        conf = obs_conf.float().clamp(0.0, 1.0)
        valid = vis * (conf > 0).float()
        velocity = torch.zeros_like(obs_points)
        velocity[:, :, 1:] = obs_points[:, :, 1:] - obs_points[:, :, :-1]
        speed = velocity.norm(dim=-1)
        coverage = valid.mean(dim=-1)
        conf_mean = (conf * vis).sum(dim=-1) / vis.sum(dim=-1).clamp_min(1.0)
        speed_mean = (speed * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        speed_std = torch.sqrt(((speed - speed_mean[..., None]).pow(2) * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0))
        return torch.stack([coverage, conf_mean, speed_mean, speed_std], dim=-1)

    def forward(
        self,
        *,
        obs_semantic_measurements: torch.Tensor,
        obs_semantic_measurement_mask: torch.Tensor,
        obs_measurement_confidence: torch.Tensor,
        teacher_agreement_score: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        obs_points: torch.Tensor,
        unit_assignment: torch.Tensor | None = None,
        unit_purity_proxy: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        sem = torch.nan_to_num(obs_semantic_measurements.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mask = obs_semantic_measurement_mask.bool()
        b, m, t, d = sem.shape
        conf = obs_measurement_confidence.float().clamp(0.0, 1.0)
        agree = teacher_agreement_score.float().clamp(0.0, 1.0)
        if conf.dim() == 2:
            conf = conf[:, :, None].expand(-1, -1, t)
        if agree.dim() == 2:
            agree = agree[:, :, None].expand(-1, -1, t)
        trace = self._trace_stats(obs_points.float(), obs_vis.bool(), obs_conf.float())
        temporal = torch.linspace(0.0, 1.0, t, device=sem.device, dtype=sem.dtype).view(1, 1, t, 1).expand(b, m, -1, -1)
        if unit_assignment is not None and unit_purity_proxy is not None:
            point_purity = torch.einsum("bmu,bu->bm", unit_assignment.float(), unit_purity_proxy.float()).clamp(0.0, 1.0)
        else:
            point_purity = torch.ones((b, m), device=sem.device, dtype=sem.dtype)
        feature = torch.cat(
            [
                sem,
                conf[..., None],
                agree[..., None],
                obs_vis.float()[..., None],
                obs_conf.float().clamp(0.0, 1.0)[..., None],
                trace[:, :, None, :].expand(-1, -1, t, -1),
                point_purity[:, :, None, None].expand(-1, -1, t, -1),
                temporal,
            ],
            dim=-1,
        )
        logits = self.score(feature).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=-1) * mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        selected_raw = torch.einsum("bmt,bmtd->bmd", weights, sem)
        selected_proj = self.semantic_value(selected_raw)
        selected = F.normalize(selected_raw + selected_proj, dim=-1)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1) / torch.log(torch.tensor(float(max(t, 2)), device=sem.device))
        conf_feature = torch.cat(
            [
                selected,
                (conf * weights).sum(dim=-1, keepdim=True),
                (agree * weights).sum(dim=-1, keepdim=True),
                weights.max(dim=-1).values[..., None],
                entropy[..., None],
                trace[..., :2],
            ],
            dim=-1,
        )
        selector_confidence = torch.sigmoid(self.confidence(conf_feature)).squeeze(-1)
        return {
            "measurement_weight": weights,
            "selected_measurement_embedding": selected,
            "selector_confidence": selector_confidence,
            "selector_entropy": entropy,
            "selector_diagnostics": {
                "future_teacher_embedding_input_allowed": False,
                "uses_observed_confidence": True,
                "uses_trace_motion_stats": True,
                "uses_teacher_agreement_score": True,
            },
        }
