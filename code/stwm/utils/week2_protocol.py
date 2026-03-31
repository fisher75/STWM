from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from stwm.modules.semantic_adapter import SemanticAdapter, SemanticSummary
from stwm.modules.tokenizer import SemanticTrajectoryTokenizer
from stwm.modules.trace_adapter import TraceAdapter, TraceSummary


@dataclass
class AblationConfig:
    disable_semantics: bool = False
    disable_trajectory: bool = False
    disable_identity_memory: bool = False
    identity_memory_dim: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenBuildResult:
    tokens: torch.Tensor
    trace_summary: TraceSummary
    semantic_summary: SemanticSummary
    token_layout: dict[str, list[int]]


def ablation_from_args(args: Any) -> AblationConfig:
    return AblationConfig(
        disable_semantics=bool(getattr(args, "disable_semantics", False)),
        disable_trajectory=bool(getattr(args, "disable_trajectory", False)),
        disable_identity_memory=bool(getattr(args, "disable_identity_memory", False)),
        identity_memory_dim=int(getattr(args, "identity_memory_dim", 8)),
    )


def build_tokens_for_sample(
    sample: Any,
    trace_adapter: TraceAdapter,
    semantic_adapter: SemanticAdapter,
    tokenizer: SemanticTrajectoryTokenizer,
    ablation: AblationConfig,
    device: torch.device,
) -> TokenBuildResult:
    trace_summary = trace_adapter.encode(sample.frame_paths, metadata=sample.metadata, clip_id=sample.clip_id)
    semantic_summary = semantic_adapter.encode(
        sample.text_labels,
        len(sample.frame_paths),
        metadata=sample.metadata,
        clip_id=sample.clip_id,
    )
    token_batch = tokenizer.encode(trace_summary, semantic_summary)
    tokens = token_batch.tokens.to(device=device, dtype=torch.float32)

    token_dim = tokens.shape[-1]
    center_slice = slice(0, 2)
    velocity_slice = slice(2, 4)
    visibility_slice = slice(4, 5)
    semantic_slice = slice(5, token_dim)

    if ablation.disable_trajectory:
        tokens[..., center_slice] = 0.0
        tokens[..., velocity_slice] = 0.0

    if ablation.disable_semantics and semantic_slice.start < semantic_slice.stop:
        tokens[..., semantic_slice] = 0.0

    identity_added_dim = 0
    if (not ablation.disable_identity_memory) and ablation.identity_memory_dim > 0:
        identity_added_dim = int(ablation.identity_memory_dim)
        seq_len = tokens.shape[1]
        positions = torch.linspace(0.0, 1.0, steps=seq_len, dtype=tokens.dtype, device=tokens.device)
        identity_feats = []
        for idx in range(identity_added_dim):
            scale = float(idx + 1)
            if idx % 2 == 0:
                identity_feats.append(torch.sin(positions * 3.14159265 * scale))
            else:
                identity_feats.append(torch.cos(positions * 3.14159265 * scale))
        identity_tensor = torch.stack(identity_feats, dim=-1).unsqueeze(0)
        tokens = torch.cat([tokens, identity_tensor], dim=-1)

    token_layout = {
        "center": [center_slice.start, center_slice.stop],
        "velocity": [velocity_slice.start, velocity_slice.stop],
        "visibility": [visibility_slice.start, visibility_slice.stop],
        "semantics": [semantic_slice.start, semantic_slice.stop],
        "base_token_dim": [0, token_dim],
        "identity_memory": [token_dim, token_dim + identity_added_dim],
    }

    return TokenBuildResult(
        tokens=tokens,
        trace_summary=trace_summary,
        semantic_summary=semantic_summary,
        token_layout=token_layout,
    )


def build_supervision_targets(
    outputs: dict[str, torch.Tensor],
    trace_summary: TraceSummary,
    semantic_summary: SemanticSummary,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    target_trajectory = trace_summary.centers.to(device=device, dtype=torch.float32).unsqueeze(0)
    target_visibility = trace_summary.visibility.to(device=device, dtype=torch.float32).unsqueeze(0)

    # Semantic adapter text embeddings are [T, N, D_text]. We pool labels and pad/truncate
    # to match model semantic head dimension.
    pooled_semantics = semantic_summary.text_embeddings.mean(dim=1).to(device=device, dtype=torch.float32)
    semantic_dim = int(outputs["semantic"].shape[-1])
    if pooled_semantics.shape[-1] < semantic_dim:
        pad = semantic_dim - pooled_semantics.shape[-1]
        pooled_semantics = nn.functional.pad(pooled_semantics, (0, pad), mode="constant", value=0.0)
    elif pooled_semantics.shape[-1] > semantic_dim:
        pooled_semantics = pooled_semantics[:, :semantic_dim]
    target_semantic = pooled_semantics.unsqueeze(0)

    return {
        "trajectory": target_trajectory,
        "visibility": target_visibility,
        "semantic": target_semantic,
    }


def compute_training_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_trajectory = torch.sigmoid(outputs["trajectory"])
    target_trajectory = targets["trajectory"]

    trajectory_loss = nn.functional.smooth_l1_loss(pred_trajectory, target_trajectory)
    visibility_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["visibility"],
        targets["visibility"],
    )
    semantic_loss = nn.functional.mse_loss(outputs["semantic"], targets["semantic"])

    if pred_trajectory.shape[1] > 1:
        pred_delta = pred_trajectory[:, 1:] - pred_trajectory[:, :-1]
        target_delta = target_trajectory[:, 1:] - target_trajectory[:, :-1]
        temporal_consistency_loss = nn.functional.smooth_l1_loss(pred_delta, target_delta)
    else:
        temporal_consistency_loss = torch.zeros((), dtype=pred_trajectory.dtype, device=pred_trajectory.device)

    total_loss = trajectory_loss + 0.5 * visibility_loss + 0.2 * semantic_loss + 0.1 * temporal_consistency_loss
    loss_items = {
        "total_loss": float(total_loss.detach().cpu()),
        "trajectory_loss": float(trajectory_loss.detach().cpu()),
        "visibility_loss": float(visibility_loss.detach().cpu()),
        "semantic_loss": float(semantic_loss.detach().cpu()),
        "temporal_consistency_loss": float(temporal_consistency_loss.detach().cpu()),
    }
    return total_loss, loss_items
