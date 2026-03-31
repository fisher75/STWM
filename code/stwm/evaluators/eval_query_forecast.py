from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Placeholder text-conditioned future grounding evaluator")
    parser.add_argument("--prediction", required=True, help="Path to a prediction JSON file")
    parser.add_argument("--query", default="object")
    parser.add_argument("--output", default="/home/chen034/workspace/stwm/outputs/baselines/query_forecast_eval.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prediction = json.loads(Path(args.prediction).read_text())
    report = {
        "status": "placeholder",
        "query": args.query,
        "prediction_keys": sorted(prediction.keys()),
        "todo": ["add R@1 and R@5", "add seen/unseen split", "add future mask IoU"],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
