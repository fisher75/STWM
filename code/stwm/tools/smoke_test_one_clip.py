from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

from stwm.datasets.stwm_dataset import STWMDataset
from stwm.models.stwm_1b import STWM1B, STWMConfig
from stwm.modules.semantic_adapter import SemanticAdapter
from stwm.modules.tokenizer import SemanticTrajectoryTokenizer
from stwm.modules.trace_adapter import TraceAdapter


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run a minimal STWM smoke test on one discovered clip")
    parser.add_argument("--data-root", default="/home/chen034/workspace/stwm/data/external")
    parser.add_argument("--manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week1_mini.json")
    parser.add_argument("--output", default="/home/chen034/workspace/stwm/outputs/smoke_tests/smoke_test_one_clip.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = STWMDataset(args.data_root, manifest=args.manifest, limit=1)
    sample = dataset[0]

    trace_summary = TraceAdapter().encode(sample.frame_paths, metadata=sample.metadata, clip_id=sample.clip_id)
    semantic_summary = SemanticAdapter().encode(
        sample.text_labels,
        len(sample.frame_paths),
        metadata=sample.metadata,
        clip_id=sample.clip_id,
    )
    token_batch = SemanticTrajectoryTokenizer().encode(trace_summary, semantic_summary)
    model = STWM1B(STWMConfig(input_dim=token_batch.tokens.shape[-1]))
    outputs = model(token_batch.tokens)

    report = {
        "clip_id": sample.clip_id,
        "dataset": sample.metadata.get("dataset", "unknown"),
        "frame_count": len(sample.frame_paths),
        "mask_count": len(sample.metadata.get("mask_paths", [])),
        "text_labels": sample.text_labels,
        "token_shape": list(token_batch.tokens.shape),
        "trajectory_shape": list(outputs["trajectory"].shape),
        "visibility_shape": list(outputs["visibility"].shape),
        "semantic_shape": list(outputs["semantic"].shape),
        "trace_metadata": trace_summary.metadata,
        "semantic_metadata": semantic_summary.metadata,
        "todo": [
            "replace proxy adapters with TraceAnything and open-vocab teachers",
            "save visualization overlays",
            "scale to multi-object tokens",
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
