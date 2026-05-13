from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v34_12_measurement_causal_residual_memory import MeasurementCausalResidualMemoryV3412


class LocalEvidenceResidualGateWorldModelV3412(MeasurementCausalResidualMemoryV3412):
    """V34.12 learned gate wrapper，仅在 oracle residual probe 通过后使用。"""

    def __init__(self, v30_checkpoint_path: str | Path, *, teacher_embedding_dim: int = 768, identity_dim: int = 64, units: int = 16, horizon: int = 32) -> None:
        super().__init__(v30_checkpoint_path, teacher_embedding_dim=teacher_embedding_dim, identity_dim=identity_dim, units=units, horizon=horizon)
        self.local_evidence_gate = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        out = super().forward(**kwargs)
        base_unc = out["pointwise_semantic_uncertainty"].float()
        residual_norm = out["assignment_bound_residual"].norm(dim=-1)
        semantic_usage = out["semantic_measurement_usage_score"].float()
        assign_conf = out["point_to_unit_assignment"].max(dim=-1).values[:, :, None].expand_as(base_unc)
        visibility = torch.sigmoid(out["visibility_logits"]).float()
        gate_in = torch.stack([base_unc, residual_norm, semantic_usage, assign_conf, visibility], dim=-1)
        gate = torch.sigmoid(self.local_evidence_gate(gate_in)).squeeze(-1)
        out["semantic_residual_gate"] = gate
        out["final_semantic_belief"] = F.normalize(out["pointwise_semantic_belief"] + gate[..., None] * out["semantic_measurement_usage_score"][..., None] * out["assignment_bound_residual"], dim=-1)
        out["future_semantic_belief"] = out["final_semantic_belief"]
        return out
