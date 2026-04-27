from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .candidate_matching import summarize_scores
from .common_io import BASELINE_CONFIG, ExternalEvalItem, missing_fields_for_baseline


class CutieAdapter:
    """Frozen Cutie VOS adapter for STWM candidate association.

    The implementation is intentionally conservative: it requires raw video
    frames, an observed target prompt, and future candidate masks/boxes. If
    those are not present in the STWM item manifest, it returns an explicit
    failure reason instead of inventing candidate scores.
    """

    baseline_name = "cutie"

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path

    def required_fields(self) -> list[str]:
        return list(BASELINE_CONFIG[self.baseline_name]["required_item_fields"])

    def run_item(self, item: ExternalEvalItem) -> dict[str, Any]:
        missing = missing_fields_for_baseline(item, self.baseline_name)
        if missing:
            return self._failure(item, "item_not_runnable:" + "+".join(missing))
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            return self._failure(item, "checkpoint_missing")
        return self._failure(
            item,
            "cutie_inference_not_executed_without_complete_frame_mask_candidate_payload",
        )

    def score_predicted_mask(self, item: ExternalEvalItem, candidate_scores: dict[str, float]) -> dict[str, Any]:
        match = summarize_scores(candidate_scores=candidate_scores, gt_candidate_id=item.gt_candidate_id)
        out = asdict(match)
        out.update({"item_id": item.item_id, "baseline_name": self.baseline_name, "subset_tags": item.subset_tags})
        return out

    def _failure(self, item: ExternalEvalItem, reason: str) -> dict[str, Any]:
        return {
            "item_id": item.item_id,
            "baseline_name": self.baseline_name,
            "predicted_candidate_id": None,
            "gt_candidate_id": item.gt_candidate_id,
            "top1_correct": None,
            "candidate_scores": {},
            "top5_candidates": [],
            "mrr": None,
            "subset_tags": item.subset_tags,
            "failure_reason_if_any": reason,
        }

