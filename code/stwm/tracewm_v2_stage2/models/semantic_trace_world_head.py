from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from stwm.tracewm_v2_stage2.models.future_semantic_trace_state import FutureSemanticTraceState


@dataclass
class SemanticTraceStateHeadConfig:
    hidden_dim: int
    semantic_embedding_dim: int = 256
    identity_embedding_dim: int = 256
    semantic_logit_dim: int = 0
    hypothesis_count: int = 1
    enable_extent_head: bool = False
    enable_multi_hypothesis_head: bool = False
    dropout: float = 0.0


class FutureExtentHead(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(int(hidden_dim)),
            torch.nn.Linear(int(hidden_dim), 4),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).sigmoid()


class MultiHypothesisTraceHead(torch.nn.Module):
    def __init__(self, hidden_dim: int, hypothesis_count: int) -> None:
        super().__init__()
        self.hypothesis_count = max(int(hypothesis_count), 1)
        self.trace_delta = torch.nn.Sequential(
            torch.nn.LayerNorm(int(hidden_dim)),
            torch.nn.Linear(int(hidden_dim), self.hypothesis_count * 2),
        )
        self.logit_head = torch.nn.Sequential(
            torch.nn.LayerNorm(int(hidden_dim)),
            torch.nn.Linear(int(hidden_dim), self.hypothesis_count),
        )

    def forward(self, hidden: torch.Tensor, base_coord: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, horizon, slots, _ = hidden.shape
        pooled = hidden.mean(dim=(1, 2))
        logits = self.logit_head(pooled)
        delta = self.trace_delta(hidden).view(bsz, horizon, slots, self.hypothesis_count, 2)
        delta = delta.permute(0, 3, 1, 2, 4)
        coord = (base_coord[:, None] + 0.05 * torch.tanh(delta)).clamp(0.0, 1.0)
        return logits, coord


class SemanticTraceStateHead(torch.nn.Module):
    """Optional world-model head for future semantic trajectory fields."""

    def __init__(self, cfg: SemanticTraceStateHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden_dim = int(cfg.hidden_dim)
        self.semantic_embedding_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(float(cfg.dropout)),
            torch.nn.Linear(hidden_dim, int(cfg.semantic_embedding_dim)),
        )
        self.identity_embedding_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(float(cfg.dropout)),
            torch.nn.Linear(hidden_dim, int(cfg.identity_embedding_dim)),
        )
        self.visibility_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.uncertainty_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.semantic_logit_head: torch.nn.Module | None = None
        if int(cfg.semantic_logit_dim) > 0:
            self.semantic_logit_head = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, int(cfg.semantic_logit_dim)),
            )
        self.extent_head: FutureExtentHead | None = FutureExtentHead(hidden_dim) if bool(cfg.enable_extent_head) else None
        self.multi_hypothesis_head: MultiHypothesisTraceHead | None = None
        if bool(cfg.enable_multi_hypothesis_head) or int(cfg.hypothesis_count) > 1:
            self.multi_hypothesis_head = MultiHypothesisTraceHead(hidden_dim, max(int(cfg.hypothesis_count), 1))

    def forward(self, future_hidden: torch.Tensor, *, future_trace_coord: torch.Tensor) -> FutureSemanticTraceState:
        semantic_embedding = self.semantic_embedding_head(future_hidden)
        identity_embedding = self.identity_embedding_head(future_hidden)
        visibility_logit = self.visibility_head(future_hidden).squeeze(-1)
        uncertainty = self.uncertainty_head(future_hidden).squeeze(-1)
        semantic_logits = self.semantic_logit_head(future_hidden) if self.semantic_logit_head is not None else None
        extent_box = self.extent_head(future_hidden) if self.extent_head is not None else None
        hypothesis_logits = None
        hypothesis_trace_coord = None
        if self.multi_hypothesis_head is not None:
            hypothesis_logits, hypothesis_trace_coord = self.multi_hypothesis_head(future_hidden, future_trace_coord)
        state = FutureSemanticTraceState(
            future_trace_coord=future_trace_coord,
            future_visibility_logit=visibility_logit,
            future_semantic_embedding=semantic_embedding,
            future_semantic_logits=semantic_logits,
            future_identity_embedding=identity_embedding,
            future_extent_box=extent_box,
            future_uncertainty=uncertainty,
            future_hypothesis_logits=hypothesis_logits,
            future_hypothesis_trace_coord=hypothesis_trace_coord,
        )
        state.validate(strict=True)
        return state


@dataclass
class FutureSemanticStateLossConfig:
    semantic_loss_weight: float = 0.0
    visibility_loss_weight: float = 0.0
    identity_belief_loss_weight: float = 0.0
    uncertainty_loss_weight: float = 0.0
    hypothesis_loss_weight: float = 0.0
    instance_conf_threshold: float = 0.6


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=value.dtype)
    while mask_f.ndim < value.ndim:
        mask_f = mask_f.unsqueeze(-1)
    return (value * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def _align_last_dim(target: torch.Tensor, dim: int) -> torch.Tensor:
    if int(target.shape[-1]) == int(dim):
        return target
    out = torch.zeros((*target.shape[:-1], int(dim)), device=target.device, dtype=target.dtype)
    copy_dim = min(int(target.shape[-1]), int(dim))
    out[..., :copy_dim] = target[..., :copy_dim]
    return out


def compute_future_semantic_state_losses(
    *,
    state: FutureSemanticTraceState,
    semantic_target: torch.Tensor,
    identity_target: torch.Tensor,
    visibility_target: torch.Tensor,
    valid_mask: torch.Tensor,
    coord_error: torch.Tensor | None,
    instance_confidence: torch.Tensor | None,
    cfg: FutureSemanticStateLossConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute optional future-state losses; all weights default to zero."""

    device = state.future_trace_coord.device
    zero = state.future_trace_coord.sum() * 0.0
    total = zero
    semantic_target = _align_last_dim(semantic_target, int(state.future_semantic_embedding.shape[-1]))
    identity_target = _align_last_dim(identity_target, int(state.future_identity_embedding.shape[-1]))
    semantic_target_f = semantic_target[:, None].expand_as(state.future_semantic_embedding)
    identity_target_f = identity_target[:, None].expand_as(state.future_identity_embedding)
    visibility_target_f = visibility_target.to(device=device, dtype=torch.float32)
    valid = valid_mask.to(device=device, dtype=torch.bool)

    semantic_loss = zero
    if float(cfg.semantic_loss_weight) > 0.0:
        semantic_loss = _masked_mean(1.0 - F.cosine_similarity(state.future_semantic_embedding, semantic_target_f, dim=-1), valid)
        total = total + float(cfg.semantic_loss_weight) * semantic_loss

    visibility_loss = zero
    if float(cfg.visibility_loss_weight) > 0.0:
        bce = F.binary_cross_entropy_with_logits(state.future_visibility_logit, visibility_target_f, reduction="none")
        visibility_loss = _masked_mean(bce, valid)
        total = total + float(cfg.visibility_loss_weight) * visibility_loss

    identity_loss = zero
    id_valid = valid
    if instance_confidence is not None:
        id_valid = id_valid & (instance_confidence[:, None].to(device=device) >= float(cfg.instance_conf_threshold))
    if float(cfg.identity_belief_loss_weight) > 0.0:
        identity_loss = _masked_mean(1.0 - F.cosine_similarity(state.future_identity_embedding, identity_target_f, dim=-1), id_valid)
        total = total + float(cfg.identity_belief_loss_weight) * identity_loss

    uncertainty_loss = zero
    if coord_error is not None and float(cfg.uncertainty_loss_weight) > 0.0:
        target_uncertainty = coord_error.detach().to(device=device, dtype=torch.float32)
        uncertainty_loss = _masked_mean(F.smooth_l1_loss(state.future_uncertainty, target_uncertainty, reduction="none"), valid)
        total = total + float(cfg.uncertainty_loss_weight) * uncertainty_loss

    hypothesis_loss = zero
    if state.future_hypothesis_logits is not None and float(cfg.hypothesis_loss_weight) > 0.0:
        target = torch.zeros((state.future_hypothesis_logits.shape[0],), device=device, dtype=torch.long)
        hypothesis_loss = F.cross_entropy(state.future_hypothesis_logits, target)
        total = total + float(cfg.hypothesis_loss_weight) * hypothesis_loss

    info = {
        "future_trace_coord_loss": 0.0,
        "future_visibility_loss": float(visibility_loss.detach().cpu().item()),
        "future_semantic_embedding_loss": float(semantic_loss.detach().cpu().item()),
        "future_identity_belief_loss": float(identity_loss.detach().cpu().item()),
        "future_uncertainty_loss": float(uncertainty_loss.detach().cpu().item()),
        "future_hypothesis_loss": float(hypothesis_loss.detach().cpu().item()),
        "future_semantic_state_loss": float(total.detach().cpu().item()),
        "future_semantic_loss_weight": float(cfg.semantic_loss_weight),
        "future_visibility_loss_weight": float(cfg.visibility_loss_weight),
        "future_identity_belief_loss_weight": float(cfg.identity_belief_loss_weight),
        "future_uncertainty_loss_weight": float(cfg.uncertainty_loss_weight),
        "future_hypothesis_loss_weight": float(cfg.hypothesis_loss_weight),
        "future_semantic_state_output_valid": bool(state.validate(strict=False)["valid"]),
        "future_semantic_state_shapes": {k: list(v) if v is not None else None for k, v in state.shape_dict().items()},
    }
    return total, info
