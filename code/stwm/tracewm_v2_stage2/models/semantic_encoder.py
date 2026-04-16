from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from stwm.tracewm_v2_stage2.models.semantic_crop_encoder import (
    SemanticCropEncoder,
    SemanticCropEncoderConfig,
)


@dataclass
class SemanticEncoderConfig:
    input_dim: int = 10
    hidden_dim: int = 128
    output_dim: int = 256
    dropout: float = 0.1
    mainline_source: str = "crop_visual_encoder"
    legacy_source: str = "hand_crafted_stats"
    crop_input_channels: int = 4
    crop_base_dim: int = 64
    local_temporal_window: int = 1
    local_temporal_fuse_weight: float = 0.5


class SemanticEncoder(nn.Module):
    def __init__(self, cfg: SemanticEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.legacy_mlp = nn.Sequential(
            nn.Linear(int(cfg.input_dim), int(cfg.hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.hidden_dim), int(cfg.output_dim)),
            nn.LayerNorm(int(cfg.output_dim)),
        )
        self.crop_encoder = SemanticCropEncoder(
            SemanticCropEncoderConfig(
                input_channels=int(cfg.crop_input_channels),
                base_dim=int(cfg.crop_base_dim),
                output_dim=int(cfg.output_dim),
                dropout=float(cfg.dropout),
            )
        )
        self.temporal_mixer = nn.Sequential(
            nn.Linear(int(cfg.output_dim) * 2, int(cfg.output_dim)),
            nn.GELU(),
            nn.Linear(int(cfg.output_dim), int(cfg.output_dim)),
            nn.LayerNorm(int(cfg.output_dim)),
        )

    def _forward_legacy(self, semantic_features: torch.Tensor) -> torch.Tensor:
        if semantic_features.ndim != 3:
            raise ValueError(f"semantic_features must be [B,K,F], got {tuple(semantic_features.shape)}")
        return self.legacy_mlp(semantic_features)

    def _forward_temporal_crop(
        self,
        base_tokens: torch.Tensor,
        *,
        semantic_rgb_crop_temporal: torch.Tensor | None,
        semantic_mask_crop_temporal: torch.Tensor | None,
        semantic_temporal_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        if semantic_rgb_crop_temporal is None or int(self.cfg.local_temporal_window) <= 1:
            return base_tokens
        if semantic_rgb_crop_temporal.ndim != 6:
            raise ValueError(
                "semantic_rgb_crop_temporal must be [B,K,W,3,H,W], "
                f"got {tuple(semantic_rgb_crop_temporal.shape)}"
            )
        bsz, k_len, window, channels, crop_h, crop_w = semantic_rgb_crop_temporal.shape
        if channels != 3:
            raise ValueError(f"semantic_rgb_crop_temporal channel must be 3, got {channels}")
        temporal_rgb = semantic_rgb_crop_temporal.reshape(bsz * k_len, window, channels, crop_h, crop_w)
        temporal_mask = None
        if semantic_mask_crop_temporal is not None:
            if semantic_mask_crop_temporal.shape[:3] != (bsz, k_len, window):
                raise ValueError(
                    "semantic_mask_crop_temporal shape mismatch: "
                    f"rgb={tuple(semantic_rgb_crop_temporal.shape)} mask={tuple(semantic_mask_crop_temporal.shape)}"
                )
            temporal_mask = semantic_mask_crop_temporal.reshape(bsz * k_len, window, 1, crop_h, crop_w)
        temporal_tokens = self.crop_encoder(
            semantic_rgb_crop=temporal_rgb,
            semantic_mask_crop=temporal_mask,
        ).reshape(bsz, k_len, window, int(self.cfg.output_dim))
        if semantic_temporal_valid is None:
            valid = torch.ones((bsz, k_len, window, 1), dtype=temporal_tokens.dtype, device=temporal_tokens.device)
        else:
            if semantic_temporal_valid.shape != (bsz, k_len, window):
                raise ValueError(
                    "semantic_temporal_valid shape mismatch: "
                    f"rgb={tuple(semantic_rgb_crop_temporal.shape)} valid={tuple(semantic_temporal_valid.shape)}"
                )
            valid = semantic_temporal_valid.to(dtype=temporal_tokens.dtype, device=temporal_tokens.device)[..., None]
        pooled = (temporal_tokens * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        mixed = self.temporal_mixer(torch.cat([base_tokens, pooled], dim=-1))
        fuse_weight = float(self.cfg.local_temporal_fuse_weight)
        if fuse_weight <= 0.0:
            return base_tokens
        return (1.0 - fuse_weight) * base_tokens + fuse_weight * mixed

    def forward(
        self,
        semantic_features: torch.Tensor | None = None,
        *,
        semantic_rgb_crop: torch.Tensor | None = None,
        semantic_mask_crop: torch.Tensor | None = None,
        semantic_rgb_crop_temporal: torch.Tensor | None = None,
        semantic_mask_crop_temporal: torch.Tensor | None = None,
        semantic_temporal_valid: torch.Tensor | None = None,
        source_mode: str = "",
    ) -> torch.Tensor:
        mode = str(source_mode).strip().lower() if str(source_mode).strip() else str(self.cfg.mainline_source).strip().lower()

        if mode == "crop_visual_encoder":
            if semantic_rgb_crop is None:
                if semantic_features is not None:
                    return self._forward_legacy(semantic_features)
                raise ValueError("crop_visual_encoder requires semantic_rgb_crop")
            base_tokens = self.crop_encoder(semantic_rgb_crop=semantic_rgb_crop, semantic_mask_crop=semantic_mask_crop)
            return self._forward_temporal_crop(
                base_tokens,
                semantic_rgb_crop_temporal=semantic_rgb_crop_temporal,
                semantic_mask_crop_temporal=semantic_mask_crop_temporal,
                semantic_temporal_valid=semantic_temporal_valid,
            )

        if mode == "hand_crafted_stats":
            if semantic_features is None:
                raise ValueError("hand_crafted_stats requires semantic_features")
            return self._forward_legacy(semantic_features)

        if semantic_rgb_crop is not None:
            return self.crop_encoder(semantic_rgb_crop=semantic_rgb_crop, semantic_mask_crop=semantic_mask_crop)
        if semantic_features is not None:
            return self._forward_legacy(semantic_features)
        raise ValueError(f"unsupported semantic source mode: {source_mode}")
