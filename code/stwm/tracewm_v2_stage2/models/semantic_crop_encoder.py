from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class SemanticCropEncoderConfig:
    input_channels: int = 4
    base_dim: int = 64
    output_dim: int = 256
    dropout: float = 0.1


class SemanticCropEncoder(nn.Module):
    def __init__(self, cfg: SemanticCropEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        base = int(cfg.base_dim)
        c_in = int(cfg.input_channels)
        c2 = base * 2
        c3 = base * 4

        self.backbone = nn.Sequential(
            nn.Conv2d(c_in, base, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.GELU(),
            nn.Conv2d(base, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(c3, int(cfg.output_dim)),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.LayerNorm(int(cfg.output_dim)),
        )

    def forward(self, semantic_rgb_crop: torch.Tensor, semantic_mask_crop: torch.Tensor | None = None) -> torch.Tensor:
        if semantic_rgb_crop.ndim != 5:
            raise ValueError(f"semantic_rgb_crop must be [B,K,3,H,W], got {tuple(semantic_rgb_crop.shape)}")

        bsz, k_len, channels, h, w = semantic_rgb_crop.shape
        if channels != 3:
            raise ValueError(f"semantic_rgb_crop channel must be 3, got {channels}")

        if semantic_mask_crop is None:
            mask = torch.zeros((bsz, k_len, 1, h, w), dtype=semantic_rgb_crop.dtype, device=semantic_rgb_crop.device)
        else:
            if semantic_mask_crop.ndim != 5:
                raise ValueError(
                    f"semantic_mask_crop must be [B,K,1,H,W], got {tuple(semantic_mask_crop.shape)}"
                )
            if semantic_mask_crop.shape[0] != bsz or semantic_mask_crop.shape[1] != k_len:
                raise ValueError(
                    "semantic_mask_crop batch/token shape mismatch: "
                    f"rgb={tuple(semantic_rgb_crop.shape)} mask={tuple(semantic_mask_crop.shape)}"
                )
            if semantic_mask_crop.shape[2] != 1:
                raise ValueError(f"semantic_mask_crop channel must be 1, got {semantic_mask_crop.shape[2]}")
            if semantic_mask_crop.shape[-2] != h or semantic_mask_crop.shape[-1] != w:
                raise ValueError(
                    "semantic_mask_crop spatial shape mismatch: "
                    f"rgb={tuple(semantic_rgb_crop.shape)} mask={tuple(semantic_mask_crop.shape)}"
                )
            mask = semantic_mask_crop.to(dtype=semantic_rgb_crop.dtype)

        x = torch.cat([semantic_rgb_crop, mask], dim=2)
        x = x.reshape(bsz * k_len, int(self.cfg.input_channels), h, w)
        x = self.backbone(x)
        x = self.pool(x).reshape(bsz * k_len, -1)
        x = self.proj(x)
        return x.reshape(bsz, k_len, int(self.cfg.output_dim))