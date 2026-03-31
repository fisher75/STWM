from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import torch
from torch import nn


@dataclass
class STWMConfig:
    input_dim: int = 21
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    trajectory_dim: int = 2
    semantic_dim: int = 16
    dropout: float = 0.1


def _default_preset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "model_presets.json"


def load_model_config(
    preset: str,
    input_dim: int,
    preset_path: str | Path | None = None,
) -> STWMConfig:
    """Build model config from a named preset and runtime input_dim."""
    base = STWMConfig(input_dim=input_dim)
    if preset in {"", "debug", "debug_tiny", "default"}:
        return base

    path = Path(preset_path) if preset_path is not None else _default_preset_path()
    if not path.exists():
        raise FileNotFoundError(f"Preset file not found: {path}")

    payload = json.loads(path.read_text())
    if preset not in payload:
        available = ", ".join(sorted(payload.keys()))
        raise KeyError(f"Unknown preset '{preset}'. Available: {available}")

    merged = {
        "input_dim": input_dim,
        "hidden_size": base.hidden_size,
        "num_layers": base.num_layers,
        "num_heads": base.num_heads,
        "trajectory_dim": base.trajectory_dim,
        "semantic_dim": base.semantic_dim,
        "dropout": base.dropout,
    }
    merged.update(payload[preset])
    merged["input_dim"] = input_dim
    return STWMConfig(**merged)


def estimate_transformer_parameter_budget(config: STWMConfig) -> int:
    """Rough parameter estimate used for quick preset sizing.

    Uses $12LH^2$ for transformer blocks and adds lightweight head/proj terms.
    """
    h = config.hidden_size
    l = config.num_layers
    transformer_core = 12 * l * h * h
    io_heads = (
        (config.input_dim * h)
        + (h * config.trajectory_dim)
        + (h * 1)
        + (h * config.semantic_dim)
        + (h * h)
    )
    return int(transformer_core + io_heads)


class STWM1B(nn.Module):
    """A small causal transformer skeleton standing in for the future 1B model.

    TODO:
    - scale config to 1B-class capacity
    - add RoPE and flash attention
    - add object query grounding and mask latent heads
    """

    def __init__(self, config: STWMConfig) -> None:
      super().__init__()
      self.config = config
      encoder_layer = nn.TransformerEncoderLayer(
        d_model=config.hidden_size,
        nhead=config.num_heads,
        dim_feedforward=config.hidden_size * 4,
        dropout=config.dropout,
        batch_first=True,
        activation="gelu",
      )
      self.input_proj = nn.Linear(config.input_dim, config.hidden_size)
      self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
      self.norm = nn.LayerNorm(config.hidden_size)
      self.trajectory_head = nn.Linear(config.hidden_size, config.trajectory_dim)
      self.visibility_head = nn.Linear(config.hidden_size, 1)
      self.semantic_head = nn.Linear(config.hidden_size, config.semantic_dim)
      self.query_head = nn.Linear(config.hidden_size, config.hidden_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
      mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
      return torch.triu(mask, diagonal=1)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
      hidden = self.input_proj(tokens)
      seq_len = hidden.shape[1]
      hidden = self.transformer(hidden, mask=self._causal_mask(seq_len, hidden.device))
      hidden = self.norm(hidden)
      return {
        "trajectory": self.trajectory_head(hidden),
        "visibility": self.visibility_head(hidden),
        "semantic": self.semantic_head(hidden),
        "query": self.query_head(hidden),
      }
