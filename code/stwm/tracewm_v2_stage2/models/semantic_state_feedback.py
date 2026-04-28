from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from stwm.tracewm_v2_stage2.models.future_semantic_trace_state import FutureSemanticTraceState


@dataclass
class SemanticStateFeedbackConfig:
    hidden_dim: int
    semantic_embedding_dim: int = 256
    identity_embedding_dim: int = 256
    alpha: float = 0.05
    gate_bias_init: float = -4.0
    dropout: float = 0.0


@dataclass
class SemanticStateFeedbackOutput:
    feedback_hidden_delta: torch.Tensor
    feedback_gate: torch.Tensor
    enhanced_future_hidden: torch.Tensor
    feedback_info: dict[str, Any]


class SemanticStateFeedbackAdapter(torch.nn.Module):
    """Small gated residual adapter from predicted semantic state to future hidden.

    This module is intentionally lightweight and default-off. It consumes the
    FutureSemanticTraceState readout and produces a small residual update for the
    future hidden/readout path; it does not implement autoregressive feedback.
    """

    def __init__(self, cfg: SemanticStateFeedbackConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden_dim = int(cfg.hidden_dim)
        semantic_dim = int(cfg.semantic_embedding_dim)
        identity_dim = int(cfg.identity_embedding_dim)
        scalar_dim = 4
        in_dim = semantic_dim + identity_dim + scalar_dim
        bottleneck = min(64, max(16, hidden_dim // 16))
        self.input_norm = torch.nn.LayerNorm(in_dim)
        self.delta_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, bottleneck),
            torch.nn.GELU(),
            torch.nn.Dropout(float(cfg.dropout)),
            torch.nn.Linear(bottleneck, hidden_dim),
        )
        self.gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, bottleneck),
            torch.nn.GELU(),
            torch.nn.Linear(bottleneck, 1),
        )
        torch.nn.init.zeros_(self.delta_mlp[-1].weight)
        torch.nn.init.zeros_(self.delta_mlp[-1].bias)
        torch.nn.init.zeros_(self.gate_mlp[-1].weight)
        torch.nn.init.constant_(self.gate_mlp[-1].bias, float(cfg.gate_bias_init))

    def forward(
        self,
        future_hidden: torch.Tensor,
        state: FutureSemanticTraceState,
        *,
        alpha: float | None = None,
        stopgrad_state: bool = True,
    ) -> SemanticStateFeedbackOutput:
        sem = state.future_semantic_embedding
        ident = state.future_identity_embedding
        scalars = [
            state.future_visibility_logit.unsqueeze(-1),
            state.future_uncertainty.unsqueeze(-1),
        ]
        if state.future_reappearance_logit is None:
            scalars.append(torch.zeros_like(state.future_visibility_logit).unsqueeze(-1))
        else:
            scalars.append(state.future_reappearance_logit.unsqueeze(-1))
        if state.future_reappearance_event_logit is None:
            event = torch.zeros_like(state.future_visibility_logit)
        else:
            event = state.future_reappearance_event_logit[:, None, :].expand_as(state.future_visibility_logit)
        scalars.append(event.unsqueeze(-1))
        feedback_features = torch.cat([sem, ident, *scalars], dim=-1)
        if bool(stopgrad_state):
            feedback_features = feedback_features.detach()
        feedback_features = self.input_norm(feedback_features)
        delta = torch.tanh(self.delta_mlp(feedback_features))
        gate = torch.sigmoid(self.gate_mlp(feedback_features))
        alpha_value = float(self.cfg.alpha if alpha is None else alpha)
        enhanced = future_hidden + float(alpha_value) * gate * delta
        gate_detached = gate.detach()
        delta_detached = delta.detach()
        saturation = ((gate_detached < 1e-3) | (gate_detached > 1.0 - 1e-3)).to(torch.float32).mean()
        info = {
            "semantic_state_feedback_enabled": True,
            "semantic_state_feedback_alpha": alpha_value,
            "semantic_state_feedback_stopgrad_state": bool(stopgrad_state),
            "feedback_gate_mean": float(gate_detached.mean().cpu().item()),
            "feedback_gate_std": float(gate_detached.std(unbiased=False).cpu().item()),
            "feedback_gate_min": float(gate_detached.min().cpu().item()),
            "feedback_gate_max": float(gate_detached.max().cpu().item()),
            "feedback_gate_saturation_ratio": float(saturation.cpu().item()),
            "feedback_delta_norm": float(delta_detached.norm(dim=-1).mean().cpu().item()),
        }
        return SemanticStateFeedbackOutput(
            feedback_hidden_delta=delta,
            feedback_gate=gate,
            enhanced_future_hidden=enhanced,
            feedback_info=info,
        )

