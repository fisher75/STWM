from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any

import torch


RULE_VERSION = "protocol_best_rule_v2"
PRIMARY_METRIC = "query_localization_error"  # lower is better
TIEBREAK_1 = "query_top1_acc"  # higher is better
TIEBREAK_2 = "future_trajectory_l1"  # lower is better


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_metric_triplet(eval_summary: dict[str, Any]) -> tuple[float, float, float]:
    metrics = eval_summary.get("metrics", {})
    if not isinstance(metrics, dict):
        raise RuntimeError("eval summary has no metrics dict")

    try:
        primary = float(metrics[PRIMARY_METRIC])
        tie1 = float(metrics[TIEBREAK_1])
        tie2 = float(metrics[TIEBREAK_2])
    except KeyError as exc:
        raise RuntimeError(f"missing required metric in eval summary: {exc}") from exc

    return primary, tie1, tie2


def _is_better(candidate: tuple[float, float, float], incumbent: tuple[float, float, float], eps: float = 1e-12) -> bool:
    c_primary, c_tie1, c_tie2 = candidate
    i_primary, i_tie1, i_tie2 = incumbent

    if c_primary < i_primary - eps:
        return True
    if c_primary > i_primary + eps:
        return False

    if c_tie1 > i_tie1 + eps:
        return True
    if c_tie1 < i_tie1 - eps:
        return False

    if c_tie2 < i_tie2 - eps:
        return True
    return False


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Update best_protocol_main.pt from detached eval summary")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--candidate-checkpoint", default="latest.pt")
    parser.add_argument("--eval-summary", required=True)
    parser.add_argument("--output-checkpoint", default="best_protocol_main.pt")
    parser.add_argument("--selection-sidecar", default="best_protocol_main_selection.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    eval_summary_path = Path(args.eval_summary)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")
    if not eval_summary_path.exists():
        raise FileNotFoundError(f"eval_summary not found: {eval_summary_path}")

    candidate_checkpoint_path = Path(args.candidate_checkpoint)
    if not candidate_checkpoint_path.is_absolute():
        candidate_checkpoint_path = checkpoint_dir / candidate_checkpoint_path
    if not candidate_checkpoint_path.exists():
        raise FileNotFoundError(f"candidate checkpoint not found: {candidate_checkpoint_path}")

    output_checkpoint_path = Path(args.output_checkpoint)
    if not output_checkpoint_path.is_absolute():
        output_checkpoint_path = checkpoint_dir / output_checkpoint_path

    sidecar_path = Path(args.selection_sidecar)
    if not sidecar_path.is_absolute():
        sidecar_path = checkpoint_dir / sidecar_path

    eval_summary = _load_json(eval_summary_path)
    candidate_triplet = _load_metric_triplet(eval_summary)

    incumbent_triplet = (float("inf"), float("-inf"), float("inf"))
    incumbent_meta: dict[str, Any] | None = None

    if sidecar_path.exists():
        try:
            incumbent_meta = _load_json(sidecar_path)
            incumbent_triplet = (
                float(incumbent_meta["metrics"][PRIMARY_METRIC]),
                float(incumbent_meta["metrics"][TIEBREAK_1]),
                float(incumbent_meta["metrics"][TIEBREAK_2]),
            )
        except Exception:
            incumbent_meta = None

    improved = _is_better(candidate_triplet, incumbent_triplet)

    result = {
        "rule_version": RULE_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "tie_break_1": TIEBREAK_1,
        "tie_break_2": TIEBREAK_2,
        "candidate_checkpoint": str(candidate_checkpoint_path),
        "output_checkpoint": str(output_checkpoint_path),
        "eval_summary": str(eval_summary_path),
        "metrics": {
            PRIMARY_METRIC: float(candidate_triplet[0]),
            TIEBREAK_1: float(candidate_triplet[1]),
            TIEBREAK_2: float(candidate_triplet[2]),
        },
        "improved": bool(improved),
    }

    if improved or not output_checkpoint_path.exists():
        payload = torch.load(candidate_checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError("candidate checkpoint payload must be dict")

        payload["protocol_best"] = {
            "rule_version": RULE_VERSION,
            "primary_metric": PRIMARY_METRIC,
            "tie_break_1": TIEBREAK_1,
            "tie_break_2": TIEBREAK_2,
            "metric_values": result["metrics"],
            "source_eval_summary": str(eval_summary_path),
            "source_candidate_checkpoint": str(candidate_checkpoint_path),
        }
        torch.save(payload, output_checkpoint_path)
        result["action"] = "updated"
    else:
        result["action"] = "kept_existing"

    if incumbent_meta is not None:
        result["incumbent_metrics"] = {
            PRIMARY_METRIC: float(incumbent_triplet[0]),
            TIEBREAK_1: float(incumbent_triplet[1]),
            TIEBREAK_2: float(incumbent_triplet[2]),
        }

    sidecar_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
