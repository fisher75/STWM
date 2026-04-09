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

    def _forward_legacy(self, semantic_features: torch.Tensor) -> torch.Tensor:
        if semantic_features.ndim != 3:
            raise ValueError(f"semantic_features must be [B,K,F], got {tuple(semantic_features.shape)}")
        return self.legacy_mlp(semantic_features)

    def forward(
        self,
        semantic_features: torch.Tensor | None = None,
        *,
        semantic_rgb_crop: torch.Tensor | None = None,
        semantic_mask_crop: torch.Tensor | None = None,
        source_mode: str = "",
    ) -> torch.Tensor:
        mode = str(source_mode).strip().lower() if str(source_mode).strip() else str(self.cfg.mainline_source).strip().lower()

        if mode == "crop_visual_encoder":
            if semantic_rgb_crop is None:
                if semantic_features is not None:
                    return self._forward_legacy(semantic_features)
                raise ValueError("crop_visual_encoder requires semantic_rgb_crop")
            return self.crop_encoder(semantic_rgb_crop=semantic_rgb_crop, semantic_mask_crop=semantic_mask_crop)

        if mode == "hand_crafted_stats":
            if semantic_features is None:
                raise ValueError("hand_crafted_stats requires semantic_features")
            return self._forward_legacy(semantic_features)

        if semantic_rgb_crop is not None:
            return self.crop_encoder(semantic_rgb_crop=semantic_rgb_crop, semantic_mask_crop=semantic_mask_crop)
        if semantic_features is not None:
            return self._forward_legacy(semantic_features)
        raise ValueError(f"unsupported semantic source mode: {source_mode}")
