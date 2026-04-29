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
    measurement_feature_dim: int = 0
    semantic_proto_count: int = 0
    enable_semantic_proto_head: bool = False
    observed_semantic_proto_count: int = 0
    semantic_proto_memory_dim: int = 128
    semantic_proto_memory_injection: str = "none"
    semantic_proto_prediction_mode: str = "direct_logits"
    semantic_proto_residual_scale: float = 0.1
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
    def __init__(self, hidden_dim: int, hypothesis_count: int, max_coord_dim: int = 3) -> None:
        super().__init__()
        self.hypothesis_count = max(int(hypothesis_count), 1)
        self.max_coord_dim = max(int(max_coord_dim), 2)
        self.trace_delta = torch.nn.Sequential(
            torch.nn.LayerNorm(int(hidden_dim)),
            torch.nn.Linear(int(hidden_dim), self.hypothesis_count * self.max_coord_dim),
        )
        self.logit_head = torch.nn.Sequential(
            torch.nn.LayerNorm(int(hidden_dim)),
            torch.nn.Linear(int(hidden_dim), self.hypothesis_count),
        )

    def forward(self, hidden: torch.Tensor, base_coord: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, horizon, slots, _ = hidden.shape
        coord_dim = int(base_coord.shape[-1])
        if coord_dim < 2 or coord_dim > self.max_coord_dim:
            raise ValueError(f"coord_dim must be in [2,{self.max_coord_dim}], got {coord_dim}")
        pooled = hidden.mean(dim=(1, 2))
        logits = self.logit_head(pooled)
        raw_delta = self.trace_delta(hidden)[..., : self.hypothesis_count * coord_dim]
        delta = raw_delta.view(bsz, horizon, slots, self.hypothesis_count, coord_dim)
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
        self.measurement_feature_head: torch.nn.Module | None = None
        if int(cfg.measurement_feature_dim) > 0:
            self.measurement_feature_head = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Dropout(float(cfg.dropout)),
                torch.nn.Linear(hidden_dim, int(cfg.measurement_feature_dim)),
            )
        self.semantic_proto_head: torch.nn.Module | None = None
        if bool(cfg.enable_semantic_proto_head) and int(cfg.semantic_proto_count) > 1:
            self.semantic_proto_head = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Dropout(float(cfg.dropout)),
                torch.nn.Linear(hidden_dim, int(cfg.semantic_proto_count)),
            )
        self.semantic_proto_memory_embed: torch.nn.Embedding | None = None
        self.semantic_proto_memory_proj: torch.nn.Module | None = None
        if int(cfg.observed_semantic_proto_count) > 1 and int(cfg.semantic_proto_memory_dim) > 0:
            self.semantic_proto_memory_embed = torch.nn.Embedding(
                int(cfg.observed_semantic_proto_count),
                int(cfg.semantic_proto_memory_dim),
            )
            self.semantic_proto_memory_proj = torch.nn.Sequential(
                torch.nn.LayerNorm(int(cfg.semantic_proto_memory_dim)),
                torch.nn.Linear(int(cfg.semantic_proto_memory_dim), hidden_dim),
            )
        self.visibility_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.reappearance_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.reappearance_event_head = torch.nn.Sequential(
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

    def _observed_proto_memory(
        self,
        *,
        observed_semantic_proto_target: torch.Tensor | None,
        observed_semantic_proto_distribution: torch.Tensor | None,
        observed_semantic_proto_mask: torch.Tensor | None,
        horizon: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.semantic_proto_memory_embed is None or self.semantic_proto_memory_proj is None:
            return None, None
        embed_weight = self.semantic_proto_memory_embed.weight
        memory: torch.Tensor | None = None
        base_logits: torch.Tensor | None = None
        if observed_semantic_proto_distribution is not None:
            dist = observed_semantic_proto_distribution.to(device=embed_weight.device, dtype=embed_weight.dtype)
            if dist.shape[-1] != embed_weight.shape[0]:
                dist = _align_last_dim(dist, int(embed_weight.shape[0]))
                dist = dist.clamp_min(0.0)
                dist = dist / dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            memory = torch.matmul(dist, embed_weight)
            base_logits = torch.log(dist.clamp_min(1e-6))
        elif observed_semantic_proto_target is not None:
            target = observed_semantic_proto_target.to(device=embed_weight.device, dtype=torch.long).clamp_min(0)
            target = target.clamp_max(int(embed_weight.shape[0]) - 1)
            memory = self.semantic_proto_memory_embed(target)
            base_logits = F.one_hot(target, num_classes=int(embed_weight.shape[0])).to(dtype=embed_weight.dtype)
            base_logits = torch.log(base_logits * 0.99 + (1.0 - base_logits) * (0.01 / max(int(embed_weight.shape[0]) - 1, 1)))
        if memory is None:
            return None, None
        if observed_semantic_proto_mask is not None:
            mask = observed_semantic_proto_mask.to(device=memory.device, dtype=memory.dtype)[..., None]
            memory = memory * mask
            if base_logits is not None:
                base_logits = base_logits * mask
        memory_hidden = self.semantic_proto_memory_proj(memory)
        memory_hidden = memory_hidden[:, None].expand(-1, int(horizon), -1, -1)
        if base_logits is not None:
            base_logits = base_logits[:, None].expand(-1, int(horizon), -1, -1)
        return memory_hidden, base_logits

    def forward(
        self,
        future_hidden: torch.Tensor,
        *,
        future_trace_coord: torch.Tensor,
        observed_semantic_proto_target: torch.Tensor | None = None,
        observed_semantic_proto_distribution: torch.Tensor | None = None,
        observed_semantic_proto_mask: torch.Tensor | None = None,
    ) -> FutureSemanticTraceState:
        horizon = int(future_hidden.shape[1])
        memory_hidden, memory_base_logits = self._observed_proto_memory(
            observed_semantic_proto_target=observed_semantic_proto_target,
            observed_semantic_proto_distribution=observed_semantic_proto_distribution,
            observed_semantic_proto_mask=observed_semantic_proto_mask,
            horizon=horizon,
        )
        memory_mode = str(self.cfg.semantic_proto_memory_injection).strip().lower()
        conditioned_hidden = future_hidden
        if memory_hidden is not None and memory_mode in {"future_head_condition", "all"}:
            conditioned_hidden = future_hidden + memory_hidden
        semantic_embedding = self.semantic_embedding_head(conditioned_hidden)
        identity_embedding = self.identity_embedding_head(conditioned_hidden)
        measurement_feature_pred = (
            self.measurement_feature_head(conditioned_hidden) if self.measurement_feature_head is not None else None
        )
        semantic_proto_logits = self.semantic_proto_head(conditioned_hidden) if self.semantic_proto_head is not None else None
        prediction_mode = str(self.cfg.semantic_proto_prediction_mode).strip().lower()
        if semantic_proto_logits is not None and memory_base_logits is not None and prediction_mode == "memory_residual_logits":
            base = _align_last_dim(memory_base_logits, int(semantic_proto_logits.shape[-1]))
            semantic_proto_logits = base + float(self.cfg.semantic_proto_residual_scale) * semantic_proto_logits
        elif semantic_proto_logits is not None and memory_base_logits is not None and prediction_mode == "copy_only":
            semantic_proto_logits = _align_last_dim(memory_base_logits, int(semantic_proto_logits.shape[-1]))
        visibility_logit = self.visibility_head(conditioned_hidden).squeeze(-1)
        reappearance_logit = self.reappearance_head(conditioned_hidden).squeeze(-1)
        reappearance_event_logit = self.reappearance_event_head(conditioned_hidden.mean(dim=1)).squeeze(-1)
        uncertainty = self.uncertainty_head(conditioned_hidden).squeeze(-1)
        semantic_logits = self.semantic_logit_head(conditioned_hidden) if self.semantic_logit_head is not None else None
        extent_box = self.extent_head(conditioned_hidden) if self.extent_head is not None else None
        hypothesis_logits = None
        hypothesis_trace_coord = None
        if self.multi_hypothesis_head is not None:
            hypothesis_logits, hypothesis_trace_coord = self.multi_hypothesis_head(future_hidden, future_trace_coord)
        state = FutureSemanticTraceState(
            future_trace_coord=future_trace_coord,
            future_visibility_logit=visibility_logit,
            future_reappearance_logit=reappearance_logit,
            future_reappearance_event_logit=reappearance_event_logit,
            future_semantic_embedding=semantic_embedding,
            future_measurement_feature_pred=measurement_feature_pred,
            future_semantic_proto_logits=semantic_proto_logits,
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
    measurement_feature_loss_weight: float = 0.0
    semantic_proto_loss_weight: float = 0.0
    semantic_proto_soft_loss_weight: float = 0.0
    visibility_loss_weight: float = 0.0
    reappearance_loss_weight: float = 0.0
    reappearance_event_loss_weight: float = 0.0
    reappearance_pos_weight: float | str = "auto"
    reappearance_pos_weight_max: float = 50.0
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


def _expand_target_to_state(target: torch.Tensor, state_tensor: torch.Tensor) -> torch.Tensor:
    if target.ndim == 3:
        target = target[:, None]
    if target.ndim != 4:
        raise ValueError(f"target must be [B,K,D] or [B,H,K,D], got {tuple(target.shape)}")
    target = _align_last_dim(target, int(state_tensor.shape[-1]))
    if target.shape[1] == 1:
        return target.expand_as(state_tensor)
    if tuple(target.shape[:3]) != tuple(state_tensor.shape[:3]):
        raise ValueError(f"target shape {tuple(target.shape)} cannot align with state {tuple(state_tensor.shape)}")
    return target


def _broadcast_instance_confidence(instance_confidence: torch.Tensor | None, valid: torch.Tensor, threshold: float) -> tuple[torch.Tensor, str]:
    if instance_confidence is None:
        return valid, "none"
    conf = instance_confidence.to(device=valid.device, dtype=torch.float32)
    if conf.ndim == 1:
        conf = conf[:, None, None]
        source = "batch_confidence_[B]"
    elif conf.ndim == 2:
        conf = conf[:, None, :]
        source = "slot_confidence_[B,K]"
    elif conf.ndim == 3:
        source = "temporal_slot_confidence_[B,H,K]"
    else:
        raise ValueError(f"instance_confidence must be [B], [B,K], or [B,H,K], got {tuple(conf.shape)}")
    return valid & (conf.expand_as(valid.to(dtype=torch.float32)) >= float(threshold)), source


def _resolve_reappearance_pos_weight(
    *,
    target: torch.Tensor,
    mask: torch.Tensor,
    setting: float | str,
    max_value: float,
) -> tuple[torch.Tensor | None, float | None, float | None]:
    if not bool(mask.any().item()):
        return None, None, None
    target_valid = target[mask].to(dtype=torch.float32)
    positive_rate = float(target_valid.mean().detach().cpu().item())
    if isinstance(setting, str) and setting.strip().lower() == "auto":
        pos = target_valid.sum()
        neg = target_valid.numel() - pos
        if float(pos.detach().cpu().item()) <= 0.0:
            weight_value = float(max_value)
        else:
            weight_value = float((neg / pos).detach().cpu().item())
        weight_value = max(1.0, min(float(max_value), weight_value))
    else:
        weight_value = float(setting)
        weight_value = max(1.0, min(float(max_value), weight_value))
    return torch.tensor(weight_value, device=target.device, dtype=torch.float32), weight_value, positive_rate


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
    identity_target_source: str = "semantic_token_surrogate",
    visibility_mask: torch.Tensor | None = None,
    measurement_feature_target: torch.Tensor | None = None,
    measurement_feature_mask: torch.Tensor | None = None,
    measurement_feature_info: dict[str, Any] | None = None,
    semantic_proto_target: torch.Tensor | None = None,
    semantic_proto_distribution: torch.Tensor | None = None,
    semantic_proto_mask: torch.Tensor | None = None,
    semantic_proto_info: dict[str, Any] | None = None,
    reappearance_target: torch.Tensor | None = None,
    reappearance_mask: torch.Tensor | None = None,
    reappearance_event_target: torch.Tensor | None = None,
    reappearance_event_mask: torch.Tensor | None = None,
    visibility_target_info: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute optional future-state losses; all weights default to zero."""

    device = state.future_trace_coord.device
    zero = state.future_trace_coord.sum() * 0.0
    total = zero
    semantic_target_f = _expand_target_to_state(semantic_target, state.future_semantic_embedding)
    identity_target_f = _expand_target_to_state(identity_target, state.future_identity_embedding)
    visibility_target_f = visibility_target.to(device=device, dtype=torch.float32)
    valid = valid_mask.to(device=device, dtype=torch.bool)
    visibility_valid = visibility_mask.to(device=device, dtype=torch.bool) if visibility_mask is not None else valid

    semantic_loss = zero
    if float(cfg.semantic_loss_weight) > 0.0:
        semantic_loss = _masked_mean(1.0 - F.cosine_similarity(state.future_semantic_embedding, semantic_target_f, dim=-1), valid)
        total = total + float(cfg.semantic_loss_weight) * semantic_loss

    measurement_feature_loss = zero
    measurement_feature_head_available = state.future_measurement_feature_pred is not None
    measurement_feature_target_available = measurement_feature_target is not None and measurement_feature_mask is not None
    measurement_feature_valid_ratio = 0.0
    measurement_feature_loss_uses_target = False
    if float(cfg.measurement_feature_loss_weight) > 0.0:
        if state.future_measurement_feature_pred is None:
            raise RuntimeError("future_measurement_feature_loss_weight > 0 requires state.future_measurement_feature_pred")
        if measurement_feature_target is None or measurement_feature_mask is None:
            raise RuntimeError("future_measurement_feature_loss_weight > 0 requires measurement feature targets and mask")
        target = measurement_feature_target.to(device=device, dtype=state.future_measurement_feature_pred.dtype)
        mask = measurement_feature_mask.to(device=device, dtype=torch.bool)
        if tuple(target.shape[:3]) != tuple(state.future_measurement_feature_pred.shape[:3]):
            raise ValueError(
                "measurement feature target shape cannot align with prediction: "
                f"target={tuple(target.shape)} pred={tuple(state.future_measurement_feature_pred.shape)}"
            )
        target = _align_last_dim(target, int(state.future_measurement_feature_pred.shape[-1]))
        pred_norm = F.normalize(state.future_measurement_feature_pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        measurement_feature_loss = _masked_mean(1.0 - F.cosine_similarity(pred_norm, target_norm, dim=-1), mask)
        total = total + float(cfg.measurement_feature_loss_weight) * measurement_feature_loss
        measurement_feature_valid_ratio = float(mask.to(dtype=torch.float32).mean().detach().cpu().item())
        measurement_feature_loss_uses_target = True
    elif measurement_feature_mask is not None:
        measurement_feature_valid_ratio = float(
            measurement_feature_mask.to(device=device, dtype=torch.float32).mean().detach().cpu().item()
        )

    semantic_proto_loss = zero
    semantic_proto_soft_loss = zero
    semantic_proto_head_available = state.future_semantic_proto_logits is not None
    semantic_proto_target_available = semantic_proto_target is not None and semantic_proto_mask is not None
    semantic_proto_valid_ratio = 0.0
    semantic_proto_accuracy: float | None = None
    semantic_proto_top5_accuracy: float | None = None
    semantic_proto_loss_uses_target = False
    semantic_proto_count = int(state.future_semantic_proto_logits.shape[-1]) if state.future_semantic_proto_logits is not None else 0
    if float(cfg.semantic_proto_loss_weight) > 0.0 or float(cfg.semantic_proto_soft_loss_weight) > 0.0:
        if state.future_semantic_proto_logits is None:
            raise RuntimeError("future semantic prototype loss requires state.future_semantic_proto_logits")
        if semantic_proto_target is None or semantic_proto_mask is None:
            raise RuntimeError("future semantic prototype loss requires prototype targets and mask")
        proto_logits = state.future_semantic_proto_logits
        proto_target = semantic_proto_target.to(device=device, dtype=torch.long)
        proto_mask = semantic_proto_mask.to(device=device, dtype=torch.bool)
        if tuple(proto_target.shape[:3]) != tuple(proto_logits.shape[:3]):
            raise ValueError(
                "semantic prototype target shape cannot align with logits: "
                f"target={tuple(proto_target.shape)} logits={tuple(proto_logits.shape)}"
            )
        valid_proto = proto_mask & (proto_target >= 0)
        semantic_proto_valid_ratio = float(valid_proto.to(dtype=torch.float32).mean().detach().cpu().item())
        if bool(valid_proto.any().item()):
            flat_logits = proto_logits[valid_proto]
            flat_target = proto_target[valid_proto]
            ce = F.cross_entropy(flat_logits, flat_target, reduction="mean")
            semantic_proto_loss = ce
            if float(cfg.semantic_proto_loss_weight) > 0.0:
                total = total + float(cfg.semantic_proto_loss_weight) * semantic_proto_loss
            pred = flat_logits.argmax(dim=-1)
            semantic_proto_accuracy = float((pred == flat_target).to(dtype=torch.float32).mean().detach().cpu().item())
            topk = min(5, int(flat_logits.shape[-1]))
            semantic_proto_top5_accuracy = float(
                (flat_logits.topk(k=topk, dim=-1).indices == flat_target[:, None])
                .any(dim=-1)
                .to(dtype=torch.float32)
                .mean()
                .detach()
                .cpu()
                .item()
            )
            if (
                semantic_proto_distribution is not None
                and float(cfg.semantic_proto_soft_loss_weight) > 0.0
                and tuple(semantic_proto_distribution.shape[:3]) == tuple(proto_logits.shape[:3])
            ):
                soft_target = semantic_proto_distribution.to(device=device, dtype=torch.float32)
                soft_target = _align_last_dim(soft_target, int(proto_logits.shape[-1])).clamp_min(0.0)
                soft_target = soft_target / soft_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                logp = F.log_softmax(proto_logits, dim=-1)
                kl = F.kl_div(logp[valid_proto], soft_target[valid_proto], reduction="batchmean")
                semantic_proto_soft_loss = kl
                total = total + float(cfg.semantic_proto_soft_loss_weight) * semantic_proto_soft_loss
            semantic_proto_loss_uses_target = True
    elif semantic_proto_mask is not None:
        semantic_proto_valid_ratio = float(
            semantic_proto_mask.to(device=device, dtype=torch.float32).mean().detach().cpu().item()
        )

    visibility_loss = zero
    if float(cfg.visibility_loss_weight) > 0.0:
        bce = F.binary_cross_entropy_with_logits(state.future_visibility_logit, visibility_target_f, reduction="none")
        visibility_loss = _masked_mean(bce, visibility_valid)
        total = total + float(cfg.visibility_loss_weight) * visibility_loss

    reappearance_loss = zero
    reappearance_pos_weight_value: float | None = None
    reappearance_positive_rate: float | None = None
    reappearance_blocked_reason: str | None = None
    reappearance_head_available = state.future_reappearance_logit is not None
    reappearance_uses_independent_logit = bool(reappearance_head_available and reappearance_target is not None)
    if reappearance_target is not None and float(cfg.reappearance_loss_weight) > 0.0:
        if state.future_reappearance_logit is None:
            raise RuntimeError("future_reappearance_loss_weight > 0 requires independent state.future_reappearance_logit")
        rep_target = reappearance_target.to(device=device, dtype=torch.float32)
        rep_valid = reappearance_mask.to(device=device, dtype=torch.bool) if reappearance_mask is not None else visibility_valid
        pos_weight, reappearance_pos_weight_value, reappearance_positive_rate = _resolve_reappearance_pos_weight(
            target=rep_target,
            mask=rep_valid,
            setting=cfg.reappearance_pos_weight,
            max_value=float(cfg.reappearance_pos_weight_max),
        )
        if pos_weight is None:
            reappearance_blocked_reason = "future_reappearance_mask_has_no_valid_entries"
            rep_bce = state.future_reappearance_logit.sum() * 0.0
        else:
            rep_bce = F.binary_cross_entropy_with_logits(
                state.future_reappearance_logit,
                rep_target,
                reduction="none",
                pos_weight=pos_weight,
            )
        reappearance_loss = _masked_mean(rep_bce, rep_valid)
        total = total + float(cfg.reappearance_loss_weight) * reappearance_loss
    elif reappearance_target is not None:
        rep_target = reappearance_target.to(device=device, dtype=torch.float32)
        rep_valid = reappearance_mask.to(device=device, dtype=torch.bool) if reappearance_mask is not None else visibility_valid
        _, reappearance_pos_weight_value, reappearance_positive_rate = _resolve_reappearance_pos_weight(
            target=rep_target,
            mask=rep_valid,
            setting=cfg.reappearance_pos_weight,
            max_value=float(cfg.reappearance_pos_weight_max),
        )
    elif float(cfg.reappearance_loss_weight) > 0.0:
        reappearance_blocked_reason = "future_reappearance_target_missing"

    reappearance_event_loss = zero
    reappearance_event_pos_weight_value: float | None = None
    reappearance_event_positive_rate: float | None = None
    reappearance_event_blocked_reason: str | None = None
    reappearance_event_head_available = state.future_reappearance_event_logit is not None
    reappearance_event_uses_independent_logit = bool(reappearance_event_head_available and reappearance_event_target is not None)
    if reappearance_event_target is not None and float(cfg.reappearance_event_loss_weight) > 0.0:
        if state.future_reappearance_event_logit is None:
            raise RuntimeError("future_reappearance_event_loss_weight > 0 requires state.future_reappearance_event_logit")
        event_target = reappearance_event_target.to(device=device, dtype=torch.float32)
        event_valid = reappearance_event_mask.to(device=device, dtype=torch.bool) if reappearance_event_mask is not None else event_target.new_ones(event_target.shape, dtype=torch.bool)
        event_pos_weight, reappearance_event_pos_weight_value, reappearance_event_positive_rate = _resolve_reappearance_pos_weight(
            target=event_target,
            mask=event_valid,
            setting=cfg.reappearance_pos_weight,
            max_value=float(cfg.reappearance_pos_weight_max),
        )
        if event_pos_weight is None:
            reappearance_event_blocked_reason = "future_reappearance_event_mask_has_no_valid_entries"
            event_bce = state.future_reappearance_event_logit.sum() * 0.0
        else:
            event_bce = F.binary_cross_entropy_with_logits(
                state.future_reappearance_event_logit,
                event_target,
                reduction="none",
                pos_weight=event_pos_weight,
            )
        reappearance_event_loss = _masked_mean(event_bce, event_valid)
        total = total + float(cfg.reappearance_event_loss_weight) * reappearance_event_loss
    elif reappearance_event_target is not None:
        event_target = reappearance_event_target.to(device=device, dtype=torch.float32)
        event_valid = reappearance_event_mask.to(device=device, dtype=torch.bool) if reappearance_event_mask is not None else event_target.new_ones(event_target.shape, dtype=torch.bool)
        _, reappearance_event_pos_weight_value, reappearance_event_positive_rate = _resolve_reappearance_pos_weight(
            target=event_target,
            mask=event_valid,
            setting=cfg.reappearance_pos_weight,
            max_value=float(cfg.reappearance_pos_weight_max),
        )
    elif float(cfg.reappearance_event_loss_weight) > 0.0:
        reappearance_event_blocked_reason = "future_reappearance_event_target_missing"

    identity_loss = zero
    id_valid, instance_confidence_broadcast = _broadcast_instance_confidence(
        instance_confidence,
        valid,
        threshold=float(cfg.instance_conf_threshold),
    )
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
        "future_reappearance_loss": float(reappearance_loss.detach().cpu().item()),
        "future_reappearance_event_loss": float(reappearance_event_loss.detach().cpu().item()),
        "future_semantic_embedding_loss": float(semantic_loss.detach().cpu().item()),
        "future_measurement_feature_loss": float(measurement_feature_loss.detach().cpu().item()),
        "future_semantic_proto_loss": float(semantic_proto_loss.detach().cpu().item()),
        "future_semantic_proto_soft_loss": float(semantic_proto_soft_loss.detach().cpu().item()),
        "future_semantic_proto_loss_uses_target": bool(semantic_proto_loss_uses_target),
        "future_semantic_proto_head_available": bool(semantic_proto_head_available),
        "future_semantic_proto_target_available": bool(semantic_proto_target_available),
        "future_semantic_proto_target_valid_ratio": float(semantic_proto_valid_ratio),
        "future_semantic_proto_accuracy": semantic_proto_accuracy,
        "future_semantic_proto_top5_accuracy": semantic_proto_top5_accuracy,
        "future_semantic_proto_count": int(semantic_proto_count),
        "future_semantic_proto_entropy": float((semantic_proto_info or {}).get("prototype_entropy", 0.0) or 0.0),
        "future_semantic_proto_cache_hit_ratio": float((semantic_proto_info or {}).get("cache_hit_ratio", 0.0) or 0.0),
        "future_semantic_proto_feature_backbone": str((semantic_proto_info or {}).get("feature_backbone", "")),
        "future_semantic_proto_no_candidate_leakage": bool((semantic_proto_info or {}).get("no_candidate_leakage", True)),
        "future_measurement_feature_loss_uses_target": bool(measurement_feature_loss_uses_target),
        "future_measurement_feature_head_available": bool(measurement_feature_head_available),
        "future_measurement_feature_target_available": bool(measurement_feature_target_available),
        "future_measurement_feature_target_valid_ratio": float(measurement_feature_valid_ratio),
        "future_measurement_feature_dim": int(
            state.future_measurement_feature_pred.shape[-1] if state.future_measurement_feature_pred is not None else 0
        ),
        "future_measurement_feature_backbone": str((measurement_feature_info or {}).get("feature_backbone", "")),
        "future_measurement_feature_no_candidate_leakage": bool((measurement_feature_info or {}).get("no_candidate_leakage", True)),
        "future_identity_belief_loss": float(identity_loss.detach().cpu().item()),
        "future_uncertainty_loss": float(uncertainty_loss.detach().cpu().item()),
        "future_hypothesis_loss": float(hypothesis_loss.detach().cpu().item()),
        "future_semantic_state_loss": float(total.detach().cpu().item()),
        "future_semantic_loss_weight": float(cfg.semantic_loss_weight),
        "future_semantic_proto_loss_weight": float(cfg.semantic_proto_loss_weight),
        "future_semantic_proto_soft_loss_weight": float(cfg.semantic_proto_soft_loss_weight),
        "future_visibility_loss_weight": float(cfg.visibility_loss_weight),
        "future_reappearance_loss_weight": float(cfg.reappearance_loss_weight),
        "future_reappearance_event_loss_weight": float(cfg.reappearance_event_loss_weight),
        "future_reappearance_head_available": bool(reappearance_head_available),
        "future_reappearance_event_head_available": bool(reappearance_event_head_available),
        "future_reappearance_pos_weight": reappearance_pos_weight_value,
        "future_reappearance_event_pos_weight": reappearance_event_pos_weight_value,
        "future_reappearance_pos_weight_setting": str(cfg.reappearance_pos_weight),
        "future_reappearance_pos_weight_max": float(cfg.reappearance_pos_weight_max),
        "future_reappearance_positive_rate": reappearance_positive_rate,
        "future_reappearance_event_positive_rate": reappearance_event_positive_rate,
        "future_reappearance_loss_uses_independent_logit": bool(reappearance_uses_independent_logit),
        "future_reappearance_event_loss_uses_independent_logit": bool(reappearance_event_uses_independent_logit),
        "future_reappearance_loss_blocked_reason": reappearance_blocked_reason,
        "future_reappearance_event_loss_blocked_reason": reappearance_event_blocked_reason,
        "future_identity_belief_loss_weight": float(cfg.identity_belief_loss_weight),
        "future_uncertainty_loss_weight": float(cfg.uncertainty_loss_weight),
        "future_hypothesis_loss_weight": float(cfg.hypothesis_loss_weight),
        "identity_target_source": str(identity_target_source),
        "instance_confidence_broadcast": str(instance_confidence_broadcast),
        "future_semantic_state_output_valid": bool(state.validate(strict=False)["valid"]),
        "future_semantic_state_shapes": {k: list(v) if v is not None else None for k, v in state.shape_dict().items()},
    }
    if isinstance(visibility_target_info, dict):
        info.update(visibility_target_info)
    return total, info
