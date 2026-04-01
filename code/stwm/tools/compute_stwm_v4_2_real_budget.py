from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import math


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Compute STWM V4.2 real-run budget from manifest and effective batch")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--effective-batch", type=int, required=True)
    parser.add_argument("--target-epochs", type=float, default=3.0)
    parser.add_argument("--min-optimizer-steps", type=int, default=5000)
    parser.add_argument("--max-optimizer-steps", type=int, default=8000)
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, list):
        raise RuntimeError("manifest must be a JSON list")

    sample_count = len(payload)
    if sample_count <= 0:
        raise RuntimeError("manifest has zero samples")

    effective_batch = max(1, int(args.effective_batch))
    steps_per_epoch = int(math.ceil(sample_count / float(effective_batch)))
    steps_for_target_epochs = int(math.ceil(float(args.target_epochs) * float(steps_per_epoch)))

    min_steps = max(int(args.min_optimizer_steps), int(steps_for_target_epochs))
    if int(args.max_optimizer_steps) > 0:
        resolved_steps = min(min_steps, int(args.max_optimizer_steps))
    else:
        resolved_steps = min_steps

    if resolved_steps < int(args.min_optimizer_steps):
        raise RuntimeError("resolved steps violates min optimizer steps")

    out = {
        "manifest": str(manifest_path),
        "sample_count": int(sample_count),
        "effective_batch": int(effective_batch),
        "steps_per_epoch": int(steps_per_epoch),
        "target_epochs": float(args.target_epochs),
        "steps_for_target_epochs": int(steps_for_target_epochs),
        "min_optimizer_steps": int(args.min_optimizer_steps),
        "max_optimizer_steps": int(args.max_optimizer_steps),
        "resolved_optimizer_steps": int(resolved_steps),
        "resolved_rule": "max(min_optimizer_steps, steps_for_target_epochs), then cap by max_optimizer_steps",
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
