from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class StructuredTraceLossConfig:
    coord_weight: float = 1.0
    visibility_weight: float = 0.5
    residual_weight: float = 0.25
    velocity_weight: float = 0.25
    endpoint_weight: float = 0.1
    enable_visibility: bool = True
    enable_residual: bool = True
    enable_velocity: bool = True
    enable_endpoint: bool = False


class StructuredTraceLoss:
    def __init__(self, config: StructuredTraceLossConfig) -> None:
        self.config = config

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().clamp_min(1.0)
        return (value * mask).sum() / denom

    def __call__(
        self,
        pred: Dict[str, torch.Tensor],
        target_state: torch.Tensor,
        valid_mask: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if target_state.ndim != 4:
            raise ValueError(f"target_state must be [B,T,K,D], got {tuple(target_state.shape)}")

        base_mask = (valid_mask.bool() & token_mask[:, None, :].bool()).float()

        coord_target = target_state[..., 0:2]
        coord_err = ((pred["coord"] - coord_target) ** 2).sum(dim=-1)
        coord_loss = self._masked_mean(coord_err, base_mask)

        vis_target = target_state[..., 3:4]
        vis_bce = F.binary_cross_entropy_with_logits(pred["vis_logit"], vis_target, reduction="none").squeeze(-1)
        vis_loss = self._masked_mean(vis_bce, base_mask)

        residual_target = target_state[..., 6:8]
        residual_err = torch.abs(pred["residual"] - residual_target).sum(dim=-1)
        residual_loss = self._masked_mean(residual_err, base_mask)

        velocity_target = target_state[..., 4:6]
        velocity_err = torch.abs(pred["velocity"] - velocity_target).sum(dim=-1)
        velocity_loss = self._masked_mean(velocity_err, base_mask)

        endpoint_loss = coord_loss * 0.0
        if self.config.enable_endpoint and "endpoint" in pred:
            endpoint_target = target_state[:, -1, :, 0:2]
            endpoint_mask = (token_mask.bool() & valid_mask[:, -1, :].bool()).float()
            endpoint_err = ((pred["endpoint"] - endpoint_target) ** 2).sum(dim=-1)
            endpoint_loss = self._masked_mean(endpoint_err, endpoint_mask)

        total = self.config.coord_weight * coord_loss
        if self.config.enable_visibility:
            total = total + self.config.visibility_weight * vis_loss
        if self.config.enable_residual:
            total = total + self.config.residual_weight * residual_loss
        if self.config.enable_velocity:
            total = total + self.config.velocity_weight * velocity_loss
        if self.config.enable_endpoint:
            total = total + self.config.endpoint_weight * endpoint_loss

        return {
            "total_loss": total,
            "coord_loss": coord_loss,
            "visibility_loss": vis_loss,
            "residual_loss": residual_loss,
            "velocity_loss": velocity_loss,
            "endpoint_loss": endpoint_loss,
        }
