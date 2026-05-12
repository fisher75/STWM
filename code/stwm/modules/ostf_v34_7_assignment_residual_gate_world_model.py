from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v34_7_assignment_bound_residual_memory import AssignmentBoundResidualMemoryV347


class AssignmentResidualGateWorldModelV347(AssignmentBoundResidualMemoryV347):
    """Learned sparse gate over an already assignment-bound residual memory."""

    def __init__(
        self,
        v30_checkpoint_path: str | Path,
        *,
        teacher_embedding_dim: int = 768,
        identity_dim: int = 64,
        units: int = 16,
        horizon: int = 32,
    ) -> None:
        super().__init__(
            v30_checkpoint_path,
            teacher_embedding_dim=teacher_embedding_dim,
            identity_dim=identity_dim,
            units=units,
            horizon=horizon,
        )
        self.assignment_residual_gate_head = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.constant_(self.assignment_residual_gate_head[-1].bias, -3.0)

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        out = super().forward(**kwargs, intervention="force_gate_zero")
        base_unc = out["pointwise_semantic_uncertainty"]
        residual_norm = out["assignment_bound_residual"].norm(dim=-1)
        base_margin = (out["pointwise_semantic_belief"].abs().topk(2, dim=-1).values)
        ambiguity = (base_margin[..., 0] - base_margin[..., 1]).neg().sigmoid()
        assignment_conf = out["assignment_confidence"][:, :, None].expand_as(base_unc)
        gate_input = torch.stack([base_unc, residual_norm, ambiguity, assignment_conf], dim=-1)
        sem_gate = torch.sigmoid(self.assignment_residual_gate_head(gate_input)).squeeze(-1)
        final = F.normalize(out["pointwise_semantic_belief"] + sem_gate[..., None] * out["assignment_bound_residual"], dim=-1)
        out["semantic_residual_gate"] = sem_gate
        out["final_semantic_belief"] = final
        out["future_semantic_belief"] = final
        return out
