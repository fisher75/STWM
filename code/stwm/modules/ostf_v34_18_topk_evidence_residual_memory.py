from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from stwm.modules.ostf_v34_12_measurement_causal_residual_memory import MeasurementCausalResidualMemoryV3412
from stwm.modules.ostf_v34_14_horizon_conditioned_measurement_selector import HorizonConditionedMeasurementSelectorV3414


class TopKEvidenceSetEncoderV3418(nn.Module):
    """把 selector top-k raw semantic evidence set 编码成 point/horizon evidence。"""

    def __init__(self, semantic_dim: int, hidden_dim: int, topk: int = 8) -> None:
        super().__init__()
        self.topk = int(topk)
        self.value = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.gate = nn.Sequential(nn.LayerNorm(hidden_dim + 4), nn.Linear(hidden_dim + 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.to_semantic = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, semantic_dim))

    def forward(
        self,
        *,
        obs_semantic_measurements: torch.Tensor,
        obs_measurement_confidence: torch.Tensor | None,
        teacher_agreement_score: torch.Tensor | None,
        selector_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sem = torch.nan_to_num(obs_semantic_measurements.float(), nan=0.0, posinf=0.0, neginf=0.0)
        b, m, t, d = sem.shape
        h = selector_weight.shape[2]
        k = min(self.topk, t)
        conf = obs_measurement_confidence.float().clamp(0.0, 1.0) if obs_measurement_confidence is not None else torch.ones((b, m, t), device=sem.device, dtype=sem.dtype)
        agree = teacher_agreement_score.float().clamp(0.0, 1.0) if teacher_agreement_score is not None else conf
        if conf.dim() == 2:
            conf = conf[:, :, None].expand(-1, -1, t)
        if agree.dim() == 2:
            agree = agree[:, :, None].expand(-1, -1, t)
        vals, idx = torch.topk(selector_weight, k=k, dim=-1)
        sem_exp = sem[:, :, None, :, :].expand(-1, -1, h, -1, -1)
        topk_sem = torch.gather(sem_exp, 3, idx[..., None].expand(-1, -1, -1, -1, d))
        conf_exp = conf[:, :, None, :].expand(-1, -1, h, -1)
        agree_exp = agree[:, :, None, :].expand(-1, -1, h, -1)
        topk_conf = torch.gather(conf_exp, 3, idx)
        topk_agree = torch.gather(agree_exp, 3, idx)
        weights = vals * topk_conf.clamp(0.05, 1.0) * topk_agree.clamp(0.05, 1.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        topk_hidden = self.value(topk_sem)
        evidence = (topk_hidden * weights[..., None]).sum(dim=3)
        maxw = weights.max(dim=-1).values
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1) / torch.log(torch.tensor(float(max(k, 2)), device=sem.device))
        conf_mean = (weights * topk_conf).sum(dim=-1)
        agree_mean = (weights * topk_agree).sum(dim=-1)
        gate_in = torch.cat([evidence, maxw[..., None], entropy[..., None], conf_mean[..., None], agree_mean[..., None]], dim=-1)
        usage = torch.sigmoid(self.gate(gate_in)).squeeze(-1)
        semantic = F.normalize(self.to_semantic(evidence), dim=-1)
        raw_semantic = F.normalize((topk_sem * weights[..., None]).sum(dim=3), dim=-1)
        return {
            "topk_indices": idx,
            "topk_weights": weights,
            "topk_raw_evidence": topk_sem,
            "topk_evidence_hidden": evidence,
            "topk_evidence_embedding": semantic,
            "topk_raw_evidence_embedding": raw_semantic,
            "topk_usage_score": usage,
            "topk_entropy": entropy,
            "topk_max_weight": maxw,
        }


class TopKEvidenceResidualMemoryV3418(MeasurementCausalResidualMemoryV3412):
    """V34.18 top-k evidence-conditioned assignment-bound residual memory。"""

    def __init__(
        self,
        v30_checkpoint_path: str | Path,
        *,
        teacher_embedding_dim: int = 768,
        identity_dim: int = 64,
        units: int = 16,
        horizon: int = 32,
        selector_hidden_dim: int = 256,
        topk: int = 8,
    ) -> None:
        super().__init__(
            v30_checkpoint_path,
            teacher_embedding_dim=teacher_embedding_dim,
            identity_dim=identity_dim,
            units=units,
            horizon=horizon,
        )
        hidden = int(self.v30.cfg.hidden_dim)
        self.measurement_selector = HorizonConditionedMeasurementSelectorV3414(hidden, teacher_embedding_dim, selector_hidden_dim)
        self.topk_evidence_encoder = TopKEvidenceSetEncoderV3418(teacher_embedding_dim, hidden, topk=topk)
        self.topk = int(topk)
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
        base, aux, _ = self._pointwise_base(obs_points, obs_vis, obs_conf, obs_semantic_measurements, obs_semantic_measurement_mask, semantic_id)
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

        tok = self.tokenizer(aux["point_token"], obs_points, obs_vis, obs_conf, sem_obs, sem_mask)
        assign = tok["point_to_unit_assignment"]
        if intervention in {"shuffle_assignment", "permute_unit_assignment"}:
            assign = assign[..., torch.randperm(assign.shape[-1], device=assign.device)]
        elif intervention == "uniform_unit_assignment":
            assign = torch.full_like(assign, 1.0 / assign.shape[-1])
        state = self.factorized_state(tok["trace_unit_features"], tok["semantic_unit_features"])
        z_dyn = state["z_dyn"]
        z_sem = state["z_sem"]
        identity_key = state["identity_key"]
        unit_conf = tok["unit_confidence"]
        horizon = base["point_pred"].shape[2]
        unit_dyn_h, unit_sem_h = self.unit_rollout(z_dyn, z_sem, horizon)
        future_trace_hidden = aux["point_token"][:, :, None, :] + aux["step_hidden"][:, None, :, :]
        selector = self.measurement_selector(
            future_trace_hidden=future_trace_hidden,
            rel_pred=aux["rel_pred"],
            future_visibility_prob=aux["vis_prob"],
            obs_semantic_measurements=sem_obs,
            obs_semantic_measurement_mask=sem_mask,
            obs_measurement_confidence=sem_conf if sem_conf is not None else sem_mask.float(),
            teacher_agreement_score=sem_agree if sem_agree is not None else sem_mask.float(),
            obs_vis=obs_vis,
            obs_conf=obs_conf,
        )
        if intervention in {"uniform_selector", "selector_ablation"}:
            uniform = sem_mask.float()[:, :, None, :] / sem_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)[:, :, None, :]
            selector = dict(selector)
            selector["measurement_weight"] = uniform.expand(-1, -1, horizon, -1)
            selector["selector_confidence"] = torch.ones_like(selector["selector_confidence"]) * 0.5
        evidence = self.topk_evidence_encoder(
            obs_semantic_measurements=sem_obs,
            obs_measurement_confidence=sem_conf,
            teacher_agreement_score=sem_agree,
            selector_weight=selector["measurement_weight"],
        )
        unit_evidence = torch.einsum("bmu,bmhd->buhd", assign, evidence["topk_evidence_hidden"]) / assign.sum(dim=1).clamp_min(1e-6)[:, :, None, None]
        id_h = identity_key[:, :, None, :].expand(-1, -1, horizon, -1)
        conf_h = unit_conf[:, :, None, None].expand(-1, -1, horizon, 1)
        memory_input = torch.cat([unit_dyn_h, unit_sem_h, unit_evidence, id_h, conf_h], dim=-1)
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
        point_semantic_usage = evidence["topk_usage_score"] * point_assignment_usage.clamp(0.0, 1.0) * selector["selector_confidence"].clamp(0.0, 1.0)
        gate = torch.zeros_like(base["pointwise_identity_belief"])
        if intervention == "force_gate_one":
            gate = torch.ones_like(gate)
        final_sem = F.normalize(base["pointwise_semantic_belief"] + gate[..., None] * point_semantic_usage[..., None] * assignment_bound_residual, dim=-1)
        out = dict(base)
        out.update(
            {
                "unit_memory": unit_memory,
                "unit_residual_memory": unit_memory,
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
                "assignment_weights": assign,
                "point_unit_weights": assign,
                "z_dyn": z_dyn,
                "z_sem": z_sem,
                "identity_key": identity_key,
                "unit_confidence": unit_conf,
                "selector_measurement_weight": selector["measurement_weight"],
                "selector_confidence": selector["selector_confidence"],
                "selector_entropy": selector["selector_entropy"],
                "future_trace_hidden": future_trace_hidden,
                "topk_indices": evidence["topk_indices"],
                "topk_weights": evidence["topk_weights"],
                "topk_raw_evidence": evidence["topk_raw_evidence"],
                "topk_evidence_embedding": evidence["topk_evidence_embedding"],
                "topk_raw_evidence_embedding": evidence["topk_raw_evidence_embedding"],
                "local_semantic_evidence_embedding": evidence["topk_evidence_embedding"],
                "semantic_measurement_usage_score": point_semantic_usage.clamp(0.0, 1.0),
                "assignment_usage_score": assignment_usage_score,
                "attention_temporal_entropy": evidence["topk_entropy"],
                "attention_max_weight": evidence["topk_max_weight"],
                "measurement_causal_diagnostics": {
                    "topk_evidence_conditioned": True,
                    "topk": self.topk,
                    "future_teacher_embedding_input_allowed": False,
                    "learned_gate_training_ran": False,
                },
                "teacher_as_method": False,
                "outputs_future_trace_field": True,
                "outputs_future_semantic_field": True,
            }
        )
        return out
