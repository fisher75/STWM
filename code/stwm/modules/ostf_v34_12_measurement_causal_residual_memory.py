from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v34_8_causal_assignment_bound_residual_memory import CausalAssignmentBoundResidualMemoryV348
from stwm.modules.ostf_v34_12_local_semantic_evidence_encoder import LocalSemanticEvidenceEncoderV3412


class MeasurementCausalResidualMemoryV3412(CausalAssignmentBoundResidualMemoryV348):
    """V34.12 局部 semantic evidence residual memory。

    pointwise base 保持主路径；unit residual 只作为 oracle mask 下的 hard-case
    correction。semantic measurement 先经过 local temporal attention，再进入
    unit memory，并通过 assignment 回读到 point residual。
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
        self.local_semantic_evidence_encoder = LocalSemanticEvidenceEncoderV3412(hidden, teacher_embedding_dim, horizon=horizon)
        self.local_unit_memory_head = nn.Sequential(
            nn.LayerNorm(hidden * 3 + identity_dim + 1),
            nn.Linear(hidden * 3 + identity_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, teacher_embedding_dim),
        )
        self.local_assignment_usage_head = nn.Sequential(
            nn.LayerNorm(hidden * 3 + identity_dim + 1),
            nn.Linear(hidden * 3 + identity_dim + 1, hidden),
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
        obs_measurement_confidence: torch.Tensor | None = None,
        teacher_agreement_score: torch.Tensor | None = None,
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
        sem_conf = obs_measurement_confidence
        sem_agree = teacher_agreement_score
        if intervention in {"zero_semantic_measurements", "zero_observed_semantic_measurements"}:
            sem_obs = torch.zeros_like(sem_obs)
            sem_mask = torch.zeros_like(sem_mask)
            sem_conf = torch.zeros_like(obs_measurement_confidence) if obs_measurement_confidence is not None else None
            sem_agree = torch.zeros_like(teacher_agreement_score) if teacher_agreement_score is not None else None
        elif intervention == "shuffle_semantic_measurements_across_points":
            idx = torch.randperm(sem_obs.shape[1], device=sem_obs.device)
            sem_obs = sem_obs[:, idx]
            sem_mask = sem_mask[:, idx]
            sem_conf = sem_conf[:, idx] if sem_conf is not None else None
            sem_agree = sem_agree[:, idx] if sem_agree is not None else None
        elif intervention == "shuffle_semantic_measurements_across_samples" and sem_obs.shape[0] > 1:
            idx = torch.randperm(sem_obs.shape[0], device=sem_obs.device)
            sem_obs = sem_obs[idx]
            sem_mask = sem_mask[idx]
            sem_conf = sem_conf[idx] if sem_conf is not None else None
            sem_agree = sem_agree[idx] if sem_agree is not None else None

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
        horizon = base["point_pred"].shape[2]
        unit_dyn_h, unit_sem_h = self.unit_rollout(z_dyn, z_sem, horizon)
        evidence = self.local_semantic_evidence_encoder(
            point_token=aux["point_token"],
            step_hidden=aux["step_hidden"],
            obs_semantic_measurements=sem_obs,
            obs_semantic_measurement_mask=sem_mask,
            obs_measurement_confidence=sem_conf,
            teacher_agreement_score=sem_agree,
            obs_vis=obs_vis,
            point_to_unit_assignment=assign,
        )
        id_h = identity_key[:, :, None, :].expand(-1, -1, horizon, -1)
        conf_h = unit_conf[:, :, None, None].expand(-1, -1, horizon, 1)
        memory_input = torch.cat([unit_dyn_h, unit_sem_h, evidence["unit_semantic_evidence"], id_h, conf_h], dim=-1)
        unit_memory = F.normalize(self.local_unit_memory_head(memory_input), dim=-1)
        assignment_usage_score = torch.sigmoid(self.local_assignment_usage_head(memory_input)).squeeze(-1)
        if intervention in {"zero_unit_memory", "zero_unit_residual"}:
            unit_memory = torch.zeros_like(unit_memory)
            assignment_usage_score = torch.zeros_like(assignment_usage_score)
        elif intervention in {"shuffle_unit_memory", "shuffle_unit_residual"}:
            idx = torch.randperm(unit_memory.shape[1], device=unit_memory.device)
            unit_memory = unit_memory[:, idx]
            assignment_usage_score = assignment_usage_score[:, idx]
        assignment_bound_residual = torch.einsum("bmu,buhd->bmhd", assign, unit_memory)
        point_assignment_usage = torch.einsum("bmu,buh->bmh", assign, assignment_usage_score)
        point_semantic_usage = evidence["semantic_measurement_usage_score"] * point_assignment_usage.clamp(0.0, 1.0)
        gate = torch.zeros_like(base["pointwise_identity_belief"])
        if intervention == "force_gate_one":
            gate = torch.ones_like(gate)
        final_sem = F.normalize(base["pointwise_semantic_belief"] + gate[..., None] * point_semantic_usage[..., None] * assignment_bound_residual, dim=-1)
        out = dict(base)
        out.update(
            {
                "unit_memory": unit_memory,
                "unit_residual_memory": unit_memory,
                "assignment_weights": assign,
                "point_unit_weights": assign,
                "assignment_bound_residual": assignment_bound_residual,
                "unit_semantic_residual": assignment_bound_residual,
                "semantic_residual_gate": gate,
                "identity_residual_gate": gate,
                "final_semantic_belief": final_sem,
                "future_semantic_belief": final_sem,
                "future_identity_belief": base["pointwise_identity_belief"],
                "identity_embedding": base["pointwise_identity_embedding"],
                "semantic_uncertainty": base["pointwise_semantic_uncertainty"],
                "point_to_unit_assignment": assign,
                "trace_unit_features": tok["trace_unit_features"],
                "semantic_unit_features": tok["semantic_unit_features"],
                "z_dyn": z_dyn,
                "z_sem": z_sem,
                "identity_key": identity_key,
                "unit_confidence": unit_conf,
                "semantic_measurement_usage_score": point_semantic_usage.clamp(0.0, 1.0),
                "assignment_usage_score": assignment_usage_score,
                "local_semantic_evidence": evidence["local_semantic_evidence"],
                "local_semantic_evidence_embedding": evidence["local_semantic_evidence_embedding"],
                "semantic_evidence_attention_weights": evidence["semantic_evidence_attention_weights"],
                "attention_temporal_entropy": evidence["attention_temporal_entropy"],
                "attention_max_weight": evidence["attention_max_weight"],
                "measurement_causal_diagnostics": {
                    "semantic_measurement_usage_score": point_semantic_usage,
                    "assignment_usage_score": assignment_usage_score,
                    "point_residual_from_assignment_weighted_unit_memory": True,
                    "raw_temporal_measurement_attention": True,
                    "future_teacher_embedding_input_allowed": False,
                },
                "teacher_as_method": False,
                "outputs_future_trace_field": True,
                "outputs_future_semantic_field": True,
            }
        )
        return out
