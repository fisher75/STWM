from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v34_8_causal_assignment_bound_residual_memory import CausalAssignmentBoundResidualMemoryV348


class CausalAssignmentResidualGateWorldModelV348(CausalAssignmentBoundResidualMemoryV348):
    """Learned gate wrapper for V34.8 causal assignment residual memory."""

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
        hidden = int(self.v30.cfg.hidden_dim)
        self.causal_state_to_hidden = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.semantic_gate_head = nn.Sequential(
            nn.LayerNorm(hidden + 4),
            nn.Linear(hidden + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.identity_gate_head = nn.Sequential(
            nn.LayerNorm(hidden + 4),
            nn.Linear(hidden + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.semantic_gate_head[-1].bias, -4.0)
        nn.init.constant_(self.identity_gate_head[-1].bias, -4.0)

    def forward(self, **kwargs: object) -> dict[str, torch.Tensor]:
        intervention = kwargs.get("intervention")
        out = super().forward(**kwargs)
        if intervention == "force_gate_zero":
            sem_gate = torch.zeros_like(out["pointwise_identity_belief"])
            id_gate = torch.zeros_like(out["pointwise_identity_belief"])
        elif intervention == "force_gate_one":
            sem_gate = torch.ones_like(out["pointwise_identity_belief"])
            id_gate = torch.ones_like(out["pointwise_identity_belief"])
        else:
            state_h = self.causal_state_to_hidden(out["causal_semantic_state"])
            state_h = state_h[:, :, None, :].expand(-1, -1, out["point_pred"].shape[2], -1)
            assignment_conf = out["assignment_confidence"][:, :, None].expand_as(out["pointwise_identity_belief"])
            base_unc = out["pointwise_semantic_uncertainty"]
            residual_norm = out["assignment_bound_residual"].norm(dim=-1).clamp(0.0, 10.0)
            vis_prob = torch.sigmoid(out["visibility_logits"].detach())
            gate_input = torch.cat([state_h, assignment_conf[..., None], base_unc[..., None], residual_norm[..., None], vis_prob[..., None]], dim=-1)
            sem_gate = torch.sigmoid(self.semantic_gate_head(gate_input)).squeeze(-1)
            id_gate = torch.sigmoid(self.identity_gate_head(gate_input)).squeeze(-1)
        out["semantic_residual_gate"] = sem_gate
        out["identity_residual_gate"] = id_gate
        out["final_semantic_belief"] = F.normalize(out["pointwise_semantic_belief"] + sem_gate[..., None] * out["assignment_bound_residual"], dim=-1)
        out["future_semantic_belief"] = out["final_semantic_belief"]
        out["future_identity_belief"] = out["pointwise_identity_belief"] + id_gate * out["unit_identity_residual"]
        return out
