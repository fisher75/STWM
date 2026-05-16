from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SemanticStateWorldModelV35(nn.Module):
    """V35 semantic state head on top of frozen trace state features.

    The model predicts low-dimensional/discrete semantic state variables rather
    than continuous teacher embedding deltas. It does not own or update the V30
    trajectory backbone; trace predictions are treated as frozen external fields.
    """

    def __init__(
        self,
        *,
        point_feature_dim: int,
        semantic_clusters: int = 64,
        evidence_families: int = 5,
        units: int = 12,
        hidden_dim: int = 160,
        identity_dim: int = 64,
        semantic_feature_dim: int | None = None,
        horizon: int = 32,
        copy_prior_strength: float = 0.0,
        assignment_bound_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.semantic_clusters = semantic_clusters
        self.evidence_families = evidence_families
        self.units = units
        self.hidden_dim = hidden_dim
        self.identity_dim = identity_dim
        self.semantic_feature_dim = int(semantic_feature_dim or point_feature_dim)
        self.horizon = horizon
        self.copy_prior_strength = float(copy_prior_strength)
        self.assignment_bound_decoder = bool(assignment_bound_decoder)
        self.semantic_prefix_dim = 2 * semantic_clusters + 1

        self.point_encoder = nn.Sequential(
            nn.LayerNorm(self.semantic_feature_dim),
            nn.Linear(self.semantic_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.identity_point_encoder = nn.Sequential(
            nn.LayerNorm(point_feature_dim),
            nn.Linear(point_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.horizon_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.assignment_head = nn.Linear(hidden_dim, units)
        self.unit_memory = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.unit_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.cluster_head = nn.Linear(hidden_dim, semantic_clusters)
        self.unit_cluster_head = nn.Linear(hidden_dim, semantic_clusters)
        self.change_head = nn.Linear(hidden_dim, 1)
        self.family_head = nn.Linear(hidden_dim, evidence_families)
        self.same_instance_head = nn.Linear(hidden_dim, 1)
        self.identity_embedding_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, identity_dim),
        )
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _shuffle_points(x: torch.Tensor) -> torch.Tensor:
        idx = torch.randperm(x.shape[1], device=x.device)
        return x[:, idx]

    def _apply_intervention(self, point_features: torch.Tensor, intervention: str | None) -> torch.Tensor:
        feat = point_features
        if intervention == "zero_semantic_measurements":
            feat = feat.clone()
            feat[..., : self.semantic_prefix_dim] = 0.0
            if feat.shape[-1] > self.semantic_feature_dim:
                feat[..., self.semantic_feature_dim :] = 0.0
        elif intervention == "shuffle_semantic_measurements":
            feat = feat.clone()
            sem = self._shuffle_points(feat[..., : self.semantic_prefix_dim])
            feat[..., : self.semantic_prefix_dim] = sem
            if feat.shape[-1] > self.semantic_feature_dim:
                feat[..., self.semantic_feature_dim :] = self._shuffle_points(feat[..., self.semantic_feature_dim :])
        return feat

    def forward(
        self,
        point_features: torch.Tensor,
        *,
        horizon: int | None = None,
        intervention: str | None = None,
    ) -> dict[str, torch.Tensor]:
        h = int(horizon or self.horizon)
        feat = self._apply_intervention(point_features, intervention)
        semantic_feat = feat[..., : self.semantic_feature_dim]
        point_h = self.point_encoder(semantic_feat)
        assign = torch.softmax(self.assignment_head(point_h), dim=-1)
        readout_assign = assign
        if intervention == "shuffle_assignment":
            readout_assign = assign[..., torch.randperm(assign.shape[-1], device=assign.device)]
        elif intervention == "uniform_assignment":
            readout_assign = torch.full_like(assign, 1.0 / assign.shape[-1])

        denom = assign.sum(dim=1).clamp_min(1e-6)
        unit_state = torch.einsum("bmu,bmd->bud", assign, point_h) / denom[..., None]
        unit_mem = self.unit_memory(unit_state)
        if intervention == "zero_unit_memory":
            unit_mem = torch.zeros_like(unit_mem)
        elif intervention == "shuffle_unit_memory":
            unit_mem = unit_mem[:, torch.randperm(unit_mem.shape[1], device=unit_mem.device)]
        unit_context = torch.einsum("bmu,bud->bmd", readout_assign, unit_mem)
        gate = torch.sigmoid(self.unit_gate(torch.cat([point_h, unit_context], dim=-1))).squeeze(-1)

        t = torch.linspace(0.0, 1.0, h, device=point_features.device, dtype=point_features.dtype)
        horizon_h = self.horizon_encoder(t[:, None])[None, None, :, :]
        point_context = point_h + gate[..., None] * unit_context
        base_h = point_context[:, :, None, :] + horizon_h
        hidden = self.fuse(torch.cat([base_h, horizon_h.expand(base_h.shape[0], base_h.shape[1], -1, -1)], dim=-1))

        change_logits = self.change_head(hidden).squeeze(-1)
        unit_hidden_h = unit_mem[:, :, None, :] + horizon_h
        unit_cluster_logits = self.unit_cluster_head(unit_hidden_h)
        assignment_cluster_logits = torch.einsum("bmu,buhk->bmhk", readout_assign, unit_cluster_logits)
        raw_cluster_logits = assignment_cluster_logits if self.assignment_bound_decoder else self.cluster_head(hidden)
        if self.copy_prior_strength > 0.0:
            last_onehot = feat[..., : self.semantic_clusters].clamp(0.0, 1.0)
            copy_logits = (last_onehot[:, :, None, :] * 2.0 - 1.0) * self.copy_prior_strength
            change_prob = torch.sigmoid(change_logits)[..., None]
            cluster_logits = (1.0 - change_prob) * copy_logits + change_prob * raw_cluster_logits
        else:
            copy_logits = torch.zeros_like(raw_cluster_logits)
            cluster_logits = raw_cluster_logits
        family_logits = self.family_head(hidden)
        same_logits = self.same_instance_head(hidden).squeeze(-1)
        identity_point_h = self.identity_point_encoder(feat)
        identity_point_embedding = F.normalize(self.identity_embedding_head(identity_point_h), dim=-1)
        identity_embedding = identity_point_embedding[:, :, None, :].expand(-1, -1, h, -1)
        uncertainty = torch.sigmoid(self.uncertainty_head(hidden).squeeze(-1))
        assignment_entropy = -(assign.clamp_min(1e-8) * assign.clamp_min(1e-8).log()).sum(dim=-1)
        return {
            "semantic_cluster_logits": cluster_logits,
            "raw_semantic_cluster_logits": raw_cluster_logits,
            "assignment_bound_cluster_logits": assignment_cluster_logits,
            "unit_cluster_logits": unit_cluster_logits,
            "copy_prior_logits": copy_logits,
            "semantic_change_logits": change_logits,
            "evidence_anchor_family_logits": family_logits,
            "same_instance_logits": same_logits,
            "identity_embedding": identity_embedding,
            "semantic_uncertainty": uncertainty,
            "point_to_unit_assignment": readout_assign,
            "unit_pool_assignment": assign,
            "unit_state": unit_state,
            "unit_memory": unit_mem,
            "unit_gate": gate,
            "assignment_entropy": assignment_entropy,
            "outputs_future_trace_field": True,
            "outputs_future_semantic_field": True,
            "teacher_as_method": False,
        }
