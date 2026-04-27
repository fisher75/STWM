from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FutureVisibilityReappearanceTargets:
    future_visibility_target: torch.Tensor
    future_reappearance_target: torch.Tensor
    future_visibility_mask: torch.Tensor
    future_reappearance_mask: torch.Tensor
    target_source: str
    target_quality: str
    target_reason: str

    def to_loss_info(self) -> dict[str, Any]:
        vis_mask_f = self.future_visibility_mask.to(dtype=torch.float32)
        rep_mask_f = self.future_reappearance_mask.to(dtype=torch.float32)
        vis_denom = vis_mask_f.sum().clamp_min(1.0)
        rep_denom = rep_mask_f.sum().clamp_min(1.0)
        return {
            "future_visibility_target_source": self.target_source,
            "future_visibility_target_quality": self.target_quality,
            "future_visibility_target_reason": self.target_reason,
            "future_visibility_supervised_ratio": float(vis_mask_f.mean().detach().cpu().item()),
            "future_reappearance_supervised_ratio": float(rep_mask_f.mean().detach().cpu().item()),
            "future_visibility_positive_rate": float((self.future_visibility_target.to(dtype=torch.float32) * vis_mask_f).sum().detach().cpu().item() / float(vis_denom.detach().cpu().item())),
            "future_reappearance_positive_rate": float((self.future_reappearance_target.to(dtype=torch.float32) * rep_mask_f).sum().detach().cpu().item() / float(rep_denom.detach().cpu().item())),
        }


def _expand_token_mask(token_mask: torch.Tensor, horizon: int) -> torch.Tensor:
    return token_mask.to(dtype=torch.bool)[:, None, :].expand(-1, int(horizon), -1)


def build_future_visibility_reappearance_targets(
    batch: dict[str, Any],
    out: dict[str, Any] | None,
    obs_len: int,
    fut_len: int,
    slot_count: int,
) -> FutureVisibilityReappearanceTargets:
    """Build per-horizon/per-entity visibility and reappearance targets.

    The preferred target uses Stage2 entity-slot aligned ``fut_valid`` with
    shape [B,H,K]. This is a real per-time target over tracked entity slots,
    not a sample-level "any future candidate exists" label. Reappearance is
    defined as future visibility after the entity is absent at the observed
    endpoint or has at least one missing step in the observed window.
    """

    fut_valid = batch.get("fut_valid")
    obs_valid = batch.get("obs_valid")
    token_mask = batch.get("token_mask")
    if isinstance(fut_valid, torch.Tensor) and fut_valid.ndim == 3:
        device = fut_valid.device
        h = min(int(fut_len), int(fut_valid.shape[1]))
        k = min(int(slot_count), int(fut_valid.shape[2]))
        visibility = fut_valid[:, :h, :k].to(dtype=torch.bool)
        if isinstance(token_mask, torch.Tensor) and token_mask.ndim == 2:
            supervision_mask = _expand_token_mask(token_mask[:, :k].to(device=device), h)
        else:
            supervision_mask = torch.ones_like(visibility, dtype=torch.bool)
        if h < int(fut_len) or k < int(slot_count):
            full_visibility = torch.zeros((int(fut_valid.shape[0]), int(fut_len), int(slot_count)), device=device, dtype=torch.bool)
            full_mask = torch.zeros_like(full_visibility, dtype=torch.bool)
            full_visibility[:, :h, :k] = visibility
            full_mask[:, :h, :k] = supervision_mask
            visibility = full_visibility
            supervision_mask = full_mask
        if isinstance(obs_valid, torch.Tensor) and obs_valid.ndim == 3:
            obs = obs_valid[:, : int(obs_len), : int(slot_count)].to(device=device, dtype=torch.bool)
            obs_seen_any = obs.any(dim=1)
            obs_endpoint_visible = obs[:, -1] if obs.shape[1] > 0 else torch.zeros_like(obs_seen_any)
            obs_occluded = obs_seen_any & (~obs.all(dim=1))
            reappearance_gate = ((~obs_endpoint_visible) | obs_occluded) & obs_seen_any
        else:
            reappearance_gate = torch.ones((visibility.shape[0], visibility.shape[2]), device=device, dtype=torch.bool)
        reappearance = visibility & reappearance_gate[:, None, :]
        return FutureVisibilityReappearanceTargets(
            future_visibility_target=visibility,
            future_reappearance_target=reappearance,
            future_visibility_mask=supervision_mask,
            future_reappearance_mask=supervision_mask,
            target_source="fut_valid_slot_aligned",
            target_quality="strong_slot_aligned",
            target_reason="Stage2 fut_valid is [B,H,K] entity-slot aligned and supervises both visible and invisible future horizon entries.",
        )

    # Conservative fallback: if only valid_mask exists, treat it as medium-quality
    # future-valid broadcast. This path is not calibrated visibility evidence.
    valid_mask = out.get("valid_mask") if isinstance(out, dict) else None
    if isinstance(valid_mask, torch.Tensor) and valid_mask.ndim == 3:
        visibility = valid_mask[:, : int(fut_len), : int(slot_count)].to(dtype=torch.bool)
        supervision_mask = torch.ones_like(visibility, dtype=torch.bool)
        reappearance = torch.zeros_like(visibility, dtype=torch.bool)
        return FutureVisibilityReappearanceTargets(
            future_visibility_target=visibility,
            future_reappearance_target=reappearance,
            future_visibility_mask=supervision_mask,
            future_reappearance_mask=supervision_mask,
            target_source="future_valid_broadcast",
            target_quality="medium_broadcast",
            target_reason="Only future valid mask was available; reappearance is not strongly slot-aligned.",
        )

    device = fut_valid.device if isinstance(fut_valid, torch.Tensor) else torch.device("cpu")
    empty = torch.zeros((1, int(fut_len), int(slot_count)), device=device, dtype=torch.bool)
    return FutureVisibilityReappearanceTargets(
        future_visibility_target=empty,
        future_reappearance_target=empty,
        future_visibility_mask=torch.zeros_like(empty, dtype=torch.bool),
        future_reappearance_mask=torch.zeros_like(empty, dtype=torch.bool),
        target_source="unavailable",
        target_quality="weak_unavailable",
        target_reason="No per-time future valid or slot-aligned visibility field is available.",
    )
