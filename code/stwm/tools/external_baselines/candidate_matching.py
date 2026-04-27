from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateMatchResult:
    predicted_candidate_id: str | None
    gt_candidate_id: str | None
    top1_correct: bool | None
    candidate_scores: dict[str, float]
    top5_candidates: list[str]
    mrr: float | None
    false_confuser_rate: float | None
    false_reacquisition_rate: float | None


def box_iou(box_a: list[float] | tuple[float, ...], box_b: list[float] | tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in box_a[:4]]
    bx1, by1, bx2, by2 = [float(x) for x in box_b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def mask_iou(mask_a: Any, mask_b: Any) -> float:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"mask_shape_mismatch:{a.shape}!={b.shape}")
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def point_inside_box(point: tuple[float, float], box: list[float] | tuple[float, ...]) -> bool:
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return x1 <= x <= x2 and y1 <= y <= y2


def summarize_scores(
    *,
    candidate_scores: dict[str, float],
    gt_candidate_id: str | None,
) -> CandidateMatchResult:
    ranked = sorted(candidate_scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
    top_ids = [str(k) for k, _ in ranked]
    predicted = top_ids[0] if top_ids else None
    top1_correct = None
    mrr = None
    false_confuser_rate = None
    false_reacquisition_rate = None
    if gt_candidate_id is not None and top_ids:
        gt = str(gt_candidate_id)
        top1_correct = predicted == gt
        if gt in top_ids:
            mrr = 1.0 / float(top_ids.index(gt) + 1)
        else:
            mrr = 0.0
        false_confuser_rate = 0.0 if top1_correct else 1.0
        false_reacquisition_rate = 0.0 if top1_correct else 1.0
    return CandidateMatchResult(
        predicted_candidate_id=predicted,
        gt_candidate_id=str(gt_candidate_id) if gt_candidate_id is not None else None,
        top1_correct=top1_correct,
        candidate_scores={str(k): float(v) for k, v in candidate_scores.items()},
        top5_candidates=top_ids[:5],
        mrr=mrr,
        false_confuser_rate=false_confuser_rate,
        false_reacquisition_rate=false_reacquisition_rate,
    )

