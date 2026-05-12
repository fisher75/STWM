from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v34_3_pointwise_unit_residual_world_model import PointwiseUnitResidualWorldModelV343


class AssignmentBoundResidualMemoryV347(PointwiseUnitResidualWorldModelV343):
    """Assignment-bound residual memory: units produce memory, points only read it through assignment."""

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
        self.unit_residual_memory_head = nn.Sequential(
            nn.LayerNorm(hidden * 2 + identity_dim + 2),
            nn.Linear(hidden * 2 + identity_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, teacher_embedding_dim),
        )
        self.unit_identity_memory_head = nn.Sequential(
            nn.LayerNorm(hidden * 2 + identity_dim + 2),
            nn.Linear(hidden * 2 + identity_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        for p in self.v30.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        *,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        obs_semantic_measurements: torch.Tensor,
        obs_semantic_measurement_mask: torch.Tensor,
        semantic_id: torch.Tensor | None = None,
        intervention: str | None = None,
    ) -> dict[str, torch.Tensor]:
        base, aux, _ = self._pointwise_base(obs_points, obs_vis, obs_conf, obs_semantic_measurements, obs_semantic_measurement_mask, semantic_id)
        sem_obs = obs_semantic_measurements
        sem_mask = obs_semantic_measurement_mask
        if intervention == "zero_observed_semantic_measurements":
            sem_obs = torch.zeros_like(sem_obs)
            sem_mask = torch.zeros_like(sem_mask)
        tok = self.tokenizer(aux["point_token"], obs_points, obs_vis, obs_conf, sem_obs, sem_mask)
        assign = tok["point_to_unit_assignment"]
        if intervention == "permute_unit_assignment":
            assign = assign[..., torch.randperm(assign.shape[-1], device=assign.device)]
        state = self.factorized_state(tok["trace_unit_features"], tok["semantic_unit_features"])
        z_dyn = state["z_dyn"]
        z_sem = state["z_sem"]
        identity_key = state["identity_key"]
        unit_conf = tok["unit_confidence"]
        if intervention == "drop_z_dyn":
            z_dyn = torch.zeros_like(z_dyn)
        if intervention == "drop_z_sem":
            z_sem = torch.zeros_like(z_sem)
        unit_dyn_h, unit_sem_h = self.unit_rollout(z_dyn, z_sem, base["point_pred"].shape[2])
        b, u, h, d = unit_dyn_h.shape
        id_h = identity_key[:, :, None, :].expand(-1, -1, h, -1)
        conf_h = unit_conf[:, :, None, None].expand(-1, -1, h, 1)
        usage = assign.mean(dim=1)
        usage_h = usage[:, :, None, None].expand(-1, -1, h, 1)
        unit_memory_input = torch.cat([unit_dyn_h, unit_sem_h, id_h, conf_h, usage_h], dim=-1)
        unit_residual_memory = F.normalize(self.unit_residual_memory_head(unit_memory_input), dim=-1)
        unit_identity_memory = self.unit_identity_memory_head(unit_memory_input).squeeze(-1)
        if intervention == "zero_unit_residual":
            unit_residual_memory = torch.zeros_like(unit_residual_memory)
            unit_identity_memory = torch.zeros_like(unit_identity_memory)
        if intervention == "shuffle_unit_residual":
            idx = torch.randperm(unit_residual_memory.shape[1], device=unit_residual_memory.device)
            unit_residual_memory = unit_residual_memory[:, idx]
            unit_identity_memory = unit_identity_memory[:, idx]
        assignment_bound_residual = torch.einsum("bmu,buhd->bmhd", assign, unit_residual_memory)
        assignment_bound_identity = torch.einsum("bmu,buh->bmh", assign, unit_identity_memory)
        assignment_confidence = assign.max(dim=-1).values
        unit_purity_proxy = usage
        sem_gate = torch.zeros_like(base["pointwise_identity_belief"])
        id_gate = torch.zeros_like(base["pointwise_identity_belief"])
        if intervention == "force_gate_one":
            sem_gate = torch.ones_like(sem_gate)
            id_gate = torch.ones_like(id_gate)
        elif intervention == "force_gate_zero" or intervention is None:
            sem_gate = torch.zeros_like(sem_gate)
            id_gate = torch.zeros_like(id_gate)
        final_sem = F.normalize(base["pointwise_semantic_belief"] + sem_gate[..., None] * assignment_bound_residual, dim=-1)
        final_identity = base["pointwise_identity_belief"] + id_gate * assignment_bound_identity
        final_embed = base["pointwise_identity_embedding"]
        sem_unc = base["pointwise_semantic_uncertainty"]
        out = dict(base)
        out.update(
            {
                "unit_residual_memory": unit_residual_memory,
                "unit_identity_memory": unit_identity_memory,
                "point_unit_weights": assign,
                "assignment_bound_residual": assignment_bound_residual,
                "unit_semantic_residual": assignment_bound_residual,
                "unit_identity_residual": assignment_bound_identity,
                "semantic_residual_gate": sem_gate,
                "identity_residual_gate": id_gate,
                "final_semantic_belief": final_sem,
                "future_semantic_belief": final_sem,
                "future_identity_belief": final_identity,
                "identity_embedding": final_embed,
                "semantic_uncertainty": sem_unc,
                "point_to_unit_assignment": assign,
                "trace_unit_features": tok["trace_unit_features"],
                "semantic_unit_features": tok["semantic_unit_features"],
                "z_dyn": z_dyn,
                "z_sem": z_sem,
                "identity_key": identity_key,
                "unit_confidence": unit_conf,
                "assignment_confidence": assignment_confidence,
                "unit_purity_proxy": unit_purity_proxy,
                "teacher_as_method": False,
                "outputs_future_trace_field": True,
                "outputs_future_semantic_field": True,
            }
        )
        return out
