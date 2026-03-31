from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import torch
from torch import nn

from stwm.datasets.stwm_dataset import STWMDataset
from stwm.models.stwm_1b import STWM1B, estimate_transformer_parameter_budget, load_model_config
from stwm.modules.semantic_adapter import SemanticAdapter
from stwm.modules.tokenizer import SemanticTrajectoryTokenizer
from stwm.modules.trace_adapter import TraceAdapter


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Minimal STWM training skeleton")
    parser.add_argument("--data-root", default="/home/chen034/workspace/stwm/data/external")
    parser.add_argument("--manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week1_mini.json")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--model-preset", default="debug_tiny")
    parser.add_argument("--preset-file", default="/home/chen034/workspace/stwm/code/stwm/configs/model_presets.json")
    parser.add_argument("--output", default="/home/chen034/workspace/stwm/outputs/training/minimal_train_step.json")
    parser.add_argument("--disable-semantics", action="store_true")
    parser.add_argument("--disable-trajectory", action="store_true")
    parser.add_argument("--disable-identity-memory", action="store_true")
    parser.add_argument("--identity-memory-dim", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = STWMDataset(args.data_root, manifest=args.manifest, limit=args.limit)
    sample = dataset[0]

    trace_adapter = TraceAdapter()
    semantic_adapter = SemanticAdapter()
    tokenizer = SemanticTrajectoryTokenizer()
    trace_summary = trace_adapter.encode(sample.frame_paths, metadata=sample.metadata, clip_id=sample.clip_id)
    semantic_summary = semantic_adapter.encode(
            sample.text_labels,
            len(sample.frame_paths),
            metadata=sample.metadata,
            clip_id=sample.clip_id,
    )
    token_batch = tokenizer.encode(trace_summary, semantic_summary)

    # Build minimal ablations directly on token features.
    tokens = token_batch.tokens.clone()
    token_dim = tokens.shape[-1]
    # Layout in SemanticTrajectoryTokenizer: [center(2), velocity(2), visibility(1), semantics(text_dim=32)]
    center_slice = slice(0, 2)
    velocity_slice = slice(2, 4)
    visibility_slice = slice(4, 5)
    semantic_slice = slice(5, token_dim)

    if args.disable_trajectory:
        tokens[..., center_slice] = 0.0
        tokens[..., velocity_slice] = 0.0

    if args.disable_semantics and semantic_slice.start < semantic_slice.stop:
        tokens[..., semantic_slice] = 0.0

    if not args.disable_identity_memory and args.identity_memory_dim > 0:
        # Deterministic tiny identity memory: normalized frame index Fourier features.
        seq_len = tokens.shape[1]
        positions = torch.linspace(0.0, 1.0, steps=seq_len, dtype=tokens.dtype, device=tokens.device)
        identity_feats = []
        for idx in range(args.identity_memory_dim):
            scale = float(idx + 1)
            if idx % 2 == 0:
                identity_feats.append(torch.sin(positions * 3.14159265 * scale))
            else:
                identity_feats.append(torch.cos(positions * 3.14159265 * scale))
        identity_tensor = torch.stack(identity_feats, dim=-1).unsqueeze(0)
        tokens = torch.cat([tokens, identity_tensor], dim=-1)

    model_config = load_model_config(
        preset=args.model_preset,
        input_dim=tokens.shape[-1],
        preset_path=args.preset_file,
    )
    model = STWM1B(model_config)
    outputs = model(tokens)
    parameter_count = sum(param.numel() for param in model.parameters())
    rough_parameter_budget = estimate_transformer_parameter_budget(model_config)

    target = torch.zeros_like(outputs["trajectory"])
    loss = nn.functional.smooth_l1_loss(outputs["trajectory"], target)
    loss.backward()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "clip_id": sample.clip_id,
                "dataset": sample.metadata.get("dataset", "unknown"),
                "frame_count": len(sample.frame_paths),
                "model_preset": args.model_preset,
                "model_config": {
                    "input_dim": model_config.input_dim,
                    "hidden_size": model_config.hidden_size,
                    "num_layers": model_config.num_layers,
                    "num_heads": model_config.num_heads,
                    "semantic_dim": model_config.semantic_dim,
                },
                "model_parameters": int(parameter_count),
                "rough_parameter_budget": int(rough_parameter_budget),
                "loss": float(loss.detach().cpu()),
                "token_shape": list(tokens.shape),
                "ablation": {
                    "disable_semantics": bool(args.disable_semantics),
                    "disable_trajectory": bool(args.disable_trajectory),
                    "disable_identity_memory": bool(args.disable_identity_memory),
                    "identity_memory_dim": int(args.identity_memory_dim),
                    "semantic_token_dim": int(max(0, semantic_slice.stop - semantic_slice.start)),
                },
                "token_layout": {
                    "center": [center_slice.start, center_slice.stop],
                    "velocity": [velocity_slice.start, velocity_slice.stop],
                    "visibility": [visibility_slice.start, visibility_slice.stop],
                    "semantics": [semantic_slice.start, semantic_slice.stop],
                },
                "trajectory_shape": list(outputs["trajectory"].shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
