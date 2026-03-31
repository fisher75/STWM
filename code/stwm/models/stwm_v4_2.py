from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torch import nn

from stwm.modules.retrieval_memory_v4_2 import RetrievalMemoryStateV42, RetrievalReconnectMemoryV42
from stwm.modules.state_tokenizer_v4_2 import ObjectBiasedStateTokenizerV42


@dataclass
class STWMV42Config:
    trace_dim: int = 5
    semantic_dim: int = 48
    prior_dim: int = 4
    hidden_size: int = 1024
    num_heads: int = 16
    seq_num_layers: int = 8
    token_num_layers: int = 8
    num_state_tokens: int = 16
    semantic_classes: int = 16
    identity_dim: int = 64
    memory_slots: int = 32
    dropout: float = 0.1


def _default_preset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "model_presets_v4_2.json"


def load_model_config_v4_2(
    preset: str,
    *,
    trace_dim: int,
    semantic_dim: int,
    prior_dim: int,
    preset_path: str | Path | None = None,
) -> STWMV42Config:
    base = STWMV42Config(trace_dim=trace_dim, semantic_dim=semantic_dim, prior_dim=prior_dim)
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
        "trace_dim": trace_dim,
        "semantic_dim": semantic_dim,
        "prior_dim": prior_dim,
        "hidden_size": base.hidden_size,
        "num_heads": base.num_heads,
        "seq_num_layers": base.seq_num_layers,
        "token_num_layers": base.token_num_layers,
        "num_state_tokens": base.num_state_tokens,
        "semantic_classes": base.semantic_classes,
        "identity_dim": base.identity_dim,
        "memory_slots": base.memory_slots,
        "dropout": base.dropout,
    }
    merged.update(payload[preset])
    merged["trace_dim"] = trace_dim
    merged["semantic_dim"] = semantic_dim
    merged["prior_dim"] = prior_dim
    return STWMV42Config(**merged)


def estimate_v4_2_parameter_budget(config: STWMV42Config) -> int:
    h = config.hidden_size
    # Rough transformer budget for two branches.
    core = 12 * h * h * (config.seq_num_layers + config.token_num_layers)
    io = (
        (config.trace_dim + config.semantic_dim + config.prior_dim) * h
        + h * config.semantic_classes
        + h * config.identity_dim
        + h * 3
    )
    return int(core + io)


class STWMV42(nn.Module):
    """STWM V4.2 minimal architecture.

    - Dense sequence branch for motion substrate
    - Object-biased learned tokenizer for state abstraction
    - Single retrieval/reconnect memory for persistence
    - Factorized heads for motion / semantics / identity
    """

    def __init__(self, config: STWMV42Config) -> None:
        super().__init__()
        self.config = config

        self.tokenizer = ObjectBiasedStateTokenizerV42(
            trace_dim=config.trace_dim,
            semantic_dim=config.semantic_dim,
            prior_dim=config.prior_dim,
            hidden_size=config.hidden_size,
            num_tokens=config.num_state_tokens,
            dropout=config.dropout,
        )

        input_dim = config.trace_dim + config.semantic_dim + config.prior_dim
        self.seq_input_proj = nn.Linear(input_dim, config.hidden_size)
        self.seq_input_norm = nn.LayerNorm(config.hidden_size)

        seq_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.seq_backbone = nn.TransformerEncoder(seq_layer, num_layers=config.seq_num_layers)

        token_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.token_backbone = nn.TransformerEncoder(token_layer, num_layers=config.token_num_layers)

        self.memory = RetrievalReconnectMemoryV42(
            token_dim=config.hidden_size,
            memory_slots=config.memory_slots,
        )

        self.motion_head = nn.Linear(config.hidden_size, 2)
        self.visibility_head = nn.Linear(config.hidden_size, 1)
        self.semantic_head = nn.Linear(config.hidden_size, config.semantic_classes)
        self.identity_head = nn.Linear(config.hidden_size, config.identity_dim)
        self.query_head = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        trace_features: torch.Tensor,
        semantic_features: torch.Tensor,
        *,
        prior_features: torch.Tensor,
        teacher_objectness: torch.Tensor | None = None,
        memory_state: RetrievalMemoryStateV42 | None = None,
        use_memory: bool = True,
        update_memory: bool = True,
    ) -> dict[str, torch.Tensor | dict[str, float] | RetrievalMemoryStateV42]:
        if trace_features.ndim != 3:
            raise ValueError("trace_features must be [B, T, D]")

        seq_input = torch.cat([trace_features, semantic_features, prior_features], dim=-1)
        seq_hidden = self.seq_input_norm(self.seq_input_proj(seq_input))
        seq_hidden = self.seq_backbone(seq_hidden)

        tokenizer_out = self.tokenizer(
            trace_features,
            semantic_features,
            prior_features=prior_features,
            teacher_objectness=teacher_objectness,
        )

        token_context = torch.einsum(
            "bnt,bth->bnh",
            tokenizer_out.token_time_attention,
            seq_hidden,
        )
        token_hidden = tokenizer_out.state_tokens + token_context
        token_hidden = self.token_backbone(token_hidden)

        memory_diag: dict[str, float] = {
            "memory_gate_mean": 0.0,
            "memory_gate_std": 0.0,
            "retrieval_entropy": 0.0,
            "valid_slots_mean": 0.0,
        }
        next_memory_state = memory_state
        if use_memory:
            token_hidden, next_memory_state, memory_diag = self.memory(
                token_hidden,
                memory_state=memory_state,
                update_memory=update_memory,
            )

        trajectory = self.motion_head(seq_hidden)
        visibility = self.visibility_head(seq_hidden)
        semantic_logits = self.semantic_head(token_hidden)
        identity_embeddings = F.normalize(self.identity_head(token_hidden), dim=-1)
        query_token_logits = self.query_head(token_hidden).squeeze(-1)

        return {
            "trajectory": trajectory,
            "visibility": visibility,
            "semantic_logits": semantic_logits,
            "identity_embeddings": identity_embeddings,
            "query_token_logits": query_token_logits,
            "token_time_attention": tokenizer_out.token_time_attention,
            "objectness": tokenizer_out.objectness,
            "tokenizer_diagnostics": tokenizer_out.diagnostics,
            "memory_diagnostics": memory_diag,
            "memory_state": next_memory_state,
        }
