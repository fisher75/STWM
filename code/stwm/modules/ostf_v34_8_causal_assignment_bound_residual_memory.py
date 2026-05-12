from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v34_3_pointwise_unit_residual_world_model import PointwiseUnitResidualWorldModelV343


class CausalAssignmentBoundResidualMemoryV348(PointwiseUnitResidualWorldModelV343):
    """Causal assignment-bound residual memory.

    The pointwise path remains the frozen base. The residual path cannot emit a
    point residual directly: semantic measurements are first compressed into
    unit semantic state, unit memory is produced at unit level, and points read
    it only through point-to-unit assignment.
    """

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
        self.causal_unit_memory_head = nn.Sequential(
            nn.LayerNorm(hidden * 2 + identity_dim + 1),
            nn.Linear(hidden * 2 + identity_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, teacher_embedding_dim),
        )
        self.causal_unit_identity_head = nn.Sequential(
            nn.LayerNorm(hidden * 2 + identity_dim + 1),
            nn.Linear(hidden * 2 + identity_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.assignment_usage_head = nn.Sequential(
            nn.LayerNorm(hidden * 2 + identity_dim + 1),
            nn.Linear(hidden * 2 + identity_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.semantic_measurement_usage_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        for p in self.v30.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _maybe_shuffle_points(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randperm(x.shape[1], device=x.device)
        return x[:, idx], mask[:, idx]

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
        base, aux, _ = self._pointwise_base(
            obs_points,
            obs_vis,
            obs_conf,
            obs_semantic_measurements,
            obs_semantic_measurement_mask,
            semantic_id,
        )
        sem_obs = obs_semantic_measurements
        sem_mask = obs_semantic_measurement_mask
        if intervention in {"zero_semantic_measurements", "zero_observed_semantic_measurements"}:
            sem_obs = torch.zeros_like(sem_obs)
            sem_mask = torch.zeros_like(sem_mask)
        elif intervention == "shuffle_semantic_measurements_across_points":
            sem_obs, sem_mask = self._maybe_shuffle_points(sem_obs, sem_mask)
        elif intervention == "shuffle_semantic_measurements_across_samples" and sem_obs.shape[0] > 1:
            idx = torch.randperm(sem_obs.shape[0], device=sem_obs.device)
            sem_obs = sem_obs[idx]
            sem_mask = sem_mask[idx]

        tok = self.tokenizer(aux["point_token"], obs_points, obs_vis, obs_conf, sem_obs, sem_mask)
        assign = tok["point_to_unit_assignment"]
        if intervention in {"shuffle_assignment", "permute_unit_assignment"}:
            assign = assign[..., torch.randperm(assign.shape[-1], device=assign.device)]
        elif intervention == "uniform_unit_assignment":
            assign = torch.full_like(assign, 1.0 / assign.shape[-1])
        elif intervention == "detach_assignment":
            assign = assign.detach()

        state = self.factorized_state(tok["trace_unit_features"], tok["semantic_unit_features"])
        z_dyn = state["z_dyn"]
        z_sem = state["z_sem"]
        identity_key = state["identity_key"]
        unit_conf = tok["unit_confidence"]
        if intervention == "drop_z_dyn":
            z_dyn = torch.zeros_like(z_dyn)
        elif intervention == "drop_z_sem":
            z_sem = torch.zeros_like(z_sem)
        elif intervention == "drop_identity_key":
            identity_key = torch.zeros_like(identity_key)

        horizon = base["point_pred"].shape[2]
        unit_dyn_h, unit_sem_h = self.unit_rollout(z_dyn, z_sem, horizon)
        id_h = identity_key[:, :, None, :].expand(-1, -1, horizon, -1)
        conf_h = unit_conf[:, :, None, None].expand(-1, -1, horizon, 1)
        memory_input = torch.cat([unit_dyn_h, unit_sem_h, id_h, conf_h], dim=-1)
        unit_memory = F.normalize(self.causal_unit_memory_head(memory_input), dim=-1)
        unit_identity_memory = self.causal_unit_identity_head(memory_input).squeeze(-1)
        if intervention in {"zero_unit_memory", "zero_unit_residual"}:
            unit_memory = torch.zeros_like(unit_memory)
            unit_identity_memory = torch.zeros_like(unit_identity_memory)
        elif intervention in {"shuffle_unit_memory", "shuffle_unit_residual"}:
            idx = torch.randperm(unit_memory.shape[1], device=unit_memory.device)
            unit_memory = unit_memory[:, idx]
            unit_identity_memory = unit_identity_memory[:, idx]

        assignment_bound_residual = torch.einsum("bmu,buhd->bmhd", assign, unit_memory)
        assignment_bound_identity = torch.einsum("bmu,buh->bmh", assign, unit_identity_memory)
        assignment_confidence = assign.max(dim=-1).values
        causal_semantic_state = torch.einsum("bmu,bud->bmd", assign, z_sem)
        assignment_usage_score = torch.sigmoid(self.assignment_usage_head(memory_input)).squeeze(-1)
        semantic_measurement_usage_score = torch.sigmoid(self.semantic_measurement_usage_head(tok["semantic_unit_features"])).squeeze(-1)

        gate = torch.zeros_like(base["pointwise_identity_belief"])
        if intervention == "force_gate_one":
            gate = torch.ones_like(gate)
        final_sem = F.normalize(base["pointwise_semantic_belief"] + gate[..., None] * assignment_bound_residual, dim=-1)
        final_identity = base["pointwise_identity_belief"] + gate * assignment_bound_identity
        out = dict(base)
        out.update(
            {
                "unit_memory": unit_memory,
                "unit_residual_memory": unit_memory,
                "unit_identity_memory": unit_identity_memory,
                "assignment_weights": assign,
                "point_unit_weights": assign,
                "assignment_bound_residual": assignment_bound_residual,
                "unit_semantic_residual": assignment_bound_residual,
                "unit_identity_residual": assignment_bound_identity,
                "semantic_residual_gate": gate,
                "identity_residual_gate": gate,
                "final_semantic_belief": final_sem,
                "future_semantic_belief": final_sem,
                "future_identity_belief": final_identity,
                "identity_embedding": base["pointwise_identity_embedding"],
                "semantic_uncertainty": base["pointwise_semantic_uncertainty"],
                "point_to_unit_assignment": assign,
                "trace_unit_features": tok["trace_unit_features"],
                "semantic_unit_features": tok["semantic_unit_features"],
                "z_dyn": z_dyn,
                "z_sem": z_sem,
                "identity_key": identity_key,
                "unit_confidence": unit_conf,
                "assignment_confidence": assignment_confidence,
                "causal_semantic_state": causal_semantic_state,
                "semantic_measurement_usage_score": semantic_measurement_usage_score,
                "assignment_usage_score": assignment_usage_score,
                "shortcut_diagnostics": {
                    "pointwise_base_detached": True,
                    "residual_reads_pointwise_semantic_belief": False,
                    "residual_reads_global_pooled_semantic_feature": False,
                    "point_residual_from_assignment_weighted_unit_memory": True,
                },
                "teacher_as_method": False,
                "outputs_future_trace_field": True,
                "outputs_future_semantic_field": True,
            }
        )
        return out
