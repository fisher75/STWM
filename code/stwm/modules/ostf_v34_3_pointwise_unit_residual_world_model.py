from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v34_2_dual_source_semantic_trace_units import (
    DualSourceTraceSemanticTokenizer,
    DualSourceFactorizedState,
    TemporalUnitRolloutV342,
)
from stwm.modules.ostf_v34_2_pointwise_no_unit_baseline import PointwiseNoUnitBaselineV342


class PointwiseUnitResidualWorldModelV343(PointwiseNoUnitBaselineV342):
    """Pointwise-preserving semantic/identity model with gated unit memory residuals."""

    def __init__(
        self,
        v30_checkpoint_path: str | Path,
        *,
        teacher_embedding_dim: int = 768,
        identity_dim: int = 64,
        units: int = 16,
        horizon: int = 32,
    ) -> None:
        super().__init__(v30_checkpoint_path, teacher_embedding_dim=teacher_embedding_dim, identity_dim=identity_dim)
        hidden = int(self.v30.cfg.hidden_dim)
        self.units = int(units)
        self.tokenizer = DualSourceTraceSemanticTokenizer(hidden, teacher_embedding_dim, units)
        self.factorized_state = DualSourceFactorizedState(hidden, identity_dim)
        self.unit_rollout = TemporalUnitRolloutV342(hidden, horizon=horizon)
        self.identity_to_hidden = nn.Linear(identity_dim, hidden)
        self.unit_memory = nn.Sequential(nn.LayerNorm(hidden * 5 + 3), nn.Linear(hidden * 5 + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.semantic_residual_head = nn.Linear(hidden, teacher_embedding_dim)
        self.identity_residual_head = nn.Linear(hidden, 1)
        self.semantic_gate = nn.Sequential(nn.LayerNorm(hidden * 2 + 4), nn.Linear(hidden * 2 + 4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.identity_gate = nn.Sequential(nn.LayerNorm(hidden * 2 + 4), nn.Linear(hidden * 2 + 4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.semantic_uncertainty_head = nn.Sequential(nn.LayerNorm(hidden + 2), nn.Linear(hidden + 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.semantic_gate[-1].bias, -2.0)
        nn.init.constant_(self.identity_gate[-1].bias, -2.0)
        for p in self.v30.parameters():
            p.requires_grad_(False)

    def _pointwise_base(
        self,
        obs_points: torch.Tensor,
        obs_vis: torch.Tensor,
        obs_conf: torch.Tensor,
        obs_semantic_measurements: torch.Tensor,
        obs_semantic_measurement_mask: torch.Tensor,
        semantic_id: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        with torch.no_grad():
            v30_out = self.v30(obs_points=obs_points, obs_vis=obs_vis, obs_conf=obs_conf, semantic_id=semantic_id)
            features, _ = self._v30_features(obs_points=obs_points, obs_vis=obs_vis, obs_conf=obs_conf, semantic_id=semantic_id)
        point_token = features["point_token"].detach()
        step_hidden = features["step_hidden"].detach()
        point_pred = v30_out["point_pred"].detach()
        rel_pred = ((point_pred - features["last_visible"][:, :, None, :]) / features["spread"].squeeze(2)[:, :, None, :]).clamp(-16.0, 16.0)
        b, m, h, _ = point_pred.shape
        visual_token = self._visual_context(obs_semantic_measurements, obs_semantic_measurement_mask)
        chunks = [
            point_token[:, :, None, :].expand(-1, -1, h, -1),
            step_hidden[:, None, :, :].expand(-1, m, -1, -1),
            visual_token[:, :, None, :].expand(-1, -1, h, -1),
            rel_pred,
            v30_out["visibility_logits"].detach().sigmoid()[:, :, :, None],
            obs_vis.float()[:, :, -1:, None].expand(-1, -1, h, -1),
        ]
        base_hidden = self.identity_trunk(torch.cat(chunks, dim=-1))
        base_sem = F.normalize(self.semantic_embedding_head(base_hidden), dim=-1)
        base_identity = self.same_instance_head(base_hidden).squeeze(-1)
        base_embed = F.normalize(self.identity_embedding_head(base_hidden), dim=-1)
        base_unc = F.softplus(self.semantic_proto_uncertainty_head(base_hidden)).squeeze(-1)
        base = {
            "point_hypotheses": v30_out["point_hypotheses"].detach(),
            "point_pred": point_pred,
            "top1_point_pred": v30_out["top1_point_pred"].detach(),
            "visibility_logits": v30_out["visibility_logits"].detach(),
            "pointwise_semantic_belief": base_sem,
            "pointwise_identity_belief": base_identity,
            "pointwise_identity_embedding": base_embed,
            "pointwise_semantic_uncertainty": base_unc,
        }
        aux = {
            "point_token": point_token,
            "step_hidden": step_hidden,
            "rel_pred": rel_pred,
            "vis_prob": torch.sigmoid(v30_out["visibility_logits"].detach()),
            "base_hidden": base_hidden,
        }
        return base, aux, visual_token

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
        point_dyn = torch.einsum("bmu,buhd->bmhd", assign, unit_dyn_h)
        point_sem = torch.einsum("bmu,buhd->bmhd", assign, unit_sem_h)
        point_identity_key = torch.einsum("bmu,bud->bmd", assign, identity_key)[:, :, None, :].expand_as(base["pointwise_identity_embedding"])
        point_unit_conf = torch.einsum("bmu,bu->bm", assign, unit_conf)[:, :, None].expand_as(base["pointwise_identity_belief"])
        point_identity_hidden = self.identity_to_hidden(point_identity_key)
        unit_hidden = self.unit_memory(
            torch.cat(
                [
                    aux["base_hidden"],
                    point_dyn,
                    point_sem,
                    point_identity_hidden,
                    aux["step_hidden"][:, None, :, :].expand_as(point_dyn),
                    aux["rel_pred"],
                    aux["vis_prob"][:, :, :, None],
                ],
                dim=-1,
            )
        )
        sem_residual = F.normalize(self.semantic_residual_head(unit_hidden), dim=-1)
        id_residual = self.identity_residual_head(unit_hidden).squeeze(-1)
        base_unc = base["pointwise_semantic_uncertainty"]
        base_ambiguity = (1.0 - torch.sigmoid(base["pointwise_identity_belief"]).sub(0.5).abs() * 2.0).clamp(0.0, 1.0)
        gate_input = torch.cat([aux["base_hidden"], unit_hidden, point_unit_conf[..., None], base_unc[..., None], base_ambiguity[..., None], aux["vis_prob"][:, :, :, None]], dim=-1)
        sem_gate = torch.sigmoid(self.semantic_gate(gate_input)).squeeze(-1)
        id_gate = torch.sigmoid(self.identity_gate(gate_input)).squeeze(-1)
        if intervention == "zero_unit_residual":
            sem_residual = torch.zeros_like(sem_residual)
            id_residual = torch.zeros_like(id_residual)
        elif intervention == "shuffle_unit_residual":
            idx = torch.randperm(sem_residual.shape[1], device=sem_residual.device)
            sem_residual = sem_residual[:, idx]
            id_residual = id_residual[:, idx]
        elif intervention == "force_gate_zero":
            sem_gate = torch.zeros_like(sem_gate)
            id_gate = torch.zeros_like(id_gate)
        elif intervention == "force_gate_one":
            sem_gate = torch.ones_like(sem_gate)
            id_gate = torch.ones_like(id_gate)
        final_sem = F.normalize(base["pointwise_semantic_belief"] + sem_gate[..., None] * sem_residual, dim=-1)
        final_identity = base["pointwise_identity_belief"] + id_gate * id_residual
        final_embed = F.normalize(base["pointwise_identity_embedding"] + id_gate[..., None] * point_identity_key, dim=-1)
        sem_unc = F.softplus(self.semantic_uncertainty_head(torch.cat([unit_hidden, sem_gate[..., None], base_unc[..., None]], dim=-1))).squeeze(-1)
        out = dict(base)
        out.update(
            {
                "unit_semantic_residual": sem_residual,
                "unit_identity_residual": id_residual,
                "semantic_residual_gate": sem_gate,
                "identity_residual_gate": id_gate,
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
                "unit_residual_diagnostics": {
                    "semantic_residual_gate": sem_gate,
                    "identity_residual_gate": id_gate,
                    "point_unit_confidence": point_unit_conf,
                    "base_ambiguity": base_ambiguity,
                },
                "teacher_as_method": False,
                "outputs_future_trace_field": True,
                "outputs_future_semantic_field": True,
            }
        )
        return out
