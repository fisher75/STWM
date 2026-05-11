#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


SOURCE_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_4_residual_utility_targets/pointodyssey"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_5_strict_residual_utility_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_5_strict_residual_utility_target_build_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_5_STRICT_RESIDUAL_UTILITY_TARGET_BUILD_20260511.md"


def _q(values: np.ndarray, q: float, default: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return default
    return float(np.quantile(values, q))


def gather_thresholds(split: str, args: argparse.Namespace) -> dict[str, float]:
    confs: list[np.ndarray] = []
    errs: list[np.ndarray] = []
    stable_errs: list[np.ndarray] = []
    id_errs: list[np.ndarray] = []
    for path in sorted((SOURCE_ROOT / split).glob("*.npz")):
        z = np.load(path, allow_pickle=True)
        conf = np.asarray(z["semantic_target_confidence"], dtype=np.float32)
        err = 1.0 - np.asarray(z["pointwise_semantic_cosine"], dtype=np.float32)
        candidate = (np.asarray(z["semantic_hard_mask"]).astype(bool) | np.asarray(z["changed_mask"]).astype(bool)) & (conf > 0)
        if candidate.any():
            confs.append(conf[candidate])
        stable = np.asarray(z["stable_mask"]).astype(bool) & (conf > 0)
        if stable.any():
            stable_errs.append(err[stable])
        id_mask = np.asarray(z["identity_hard_mask"]).astype(bool)
        if id_mask.any():
            id_errs.append(np.asarray(z["pointwise_identity_error"], dtype=np.float32)[id_mask])
    all_conf = np.concatenate(confs) if confs else np.array([], dtype=np.float32)
    conf_thr = _q(all_conf, args.confidence_quantile, args.min_teacher_confidence)
    high_errs = []
    for path in sorted((SOURCE_ROOT / split).glob("*.npz")):
        z = np.load(path, allow_pickle=True)
        conf = np.asarray(z["semantic_target_confidence"], dtype=np.float32)
        err = 1.0 - np.asarray(z["pointwise_semantic_cosine"], dtype=np.float32)
        candidate = (np.asarray(z["semantic_hard_mask"]).astype(bool) | np.asarray(z["changed_mask"]).astype(bool)) & (conf >= conf_thr)
        if candidate.any():
            high_errs.append(err[candidate])
    all_err = np.concatenate(high_errs) if high_errs else np.array([], dtype=np.float32)
    err_thr = _q(all_err, args.error_quantile, 0.35)
    stable_err_thr = _q(np.concatenate(stable_errs) if stable_errs else np.array([], dtype=np.float32), args.stable_error_quantile, 0.15)
    id_err_thr = _q(np.concatenate(id_errs) if id_errs else np.array([], dtype=np.float32), args.identity_error_quantile, 0.35)
    return {
        "teacher_confidence_threshold": max(float(conf_thr), args.min_teacher_confidence),
        "pointwise_error_threshold": float(err_thr),
        "stable_good_error_threshold": float(stable_err_thr),
        "identity_error_threshold": float(id_err_thr),
    }


def process_split(split: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, float]]:
    th = gather_thresholds(split, args)
    out_dir = OUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = {
        "samples": 0,
        "valid_sem": 0,
        "sem_pos": 0,
        "id_valid": 0,
        "id_pos": 0,
        "stable": 0,
        "stable_suppress": 0,
        "gate_available": 0,
        "gate_pos": 0,
    }
    for path in sorted((SOURCE_ROOT / split).glob("*.npz")):
        z = np.load(path, allow_pickle=True)
        uid = str(np.asarray(z["sample_uid"]).item())
        conf = np.asarray(z["semantic_target_confidence"], dtype=np.float32)
        point_cos = np.asarray(z["pointwise_semantic_cosine"], dtype=np.float32)
        err = 1.0 - point_cos
        semantic_hard = np.asarray(z["semantic_hard_mask"]).astype(bool)
        changed = np.asarray(z["changed_mask"]).astype(bool)
        stable = np.asarray(z["stable_mask"]).astype(bool)
        id_hard = np.asarray(z["identity_hard_mask"]).astype(bool)
        id_err = np.asarray(z["pointwise_identity_error"], dtype=np.float32)
        valid_sem = conf > 0
        sem_pos = (
            valid_sem
            & (conf >= th["teacher_confidence_threshold"])
            & (err >= th["pointwise_error_threshold"])
            & (semantic_hard | changed)
        )
        if sem_pos.sum() > args.max_semantic_positive_ratio * max(valid_sem.sum(), 1):
            # Keep the highest-error high-confidence corrections only.
            scores = np.where(sem_pos, err + 0.05 * conf, -np.inf).reshape(-1)
            keep = int(args.max_semantic_positive_ratio * max(valid_sem.sum(), 1))
            keep = max(1, keep)
            top = np.argpartition(-scores, min(keep, scores.size - 1))[:keep]
            strict = np.zeros_like(scores, dtype=bool)
            strict[top] = np.isfinite(scores[top])
            sem_pos = strict.reshape(sem_pos.shape)
        stable_suppress = (
            stable
            & valid_sem
            & (conf >= th["teacher_confidence_threshold"])
            & (err <= th["stable_good_error_threshold"])
        )
        id_pos = id_hard & (id_err >= th["identity_error_threshold"])
        gate_target = (sem_pos | id_pos).astype(np.float32)
        gate_available = sem_pos | id_pos | stable_suppress
        np.savez_compressed(
            out_dir / f"{uid}.npz",
            sample_uid=uid,
            point_id=np.asarray(z["point_id"]),
            pointwise_semantic_cosine=point_cos.astype(np.float32),
            pointwise_semantic_margin=err.astype(np.float32),
            pointwise_identity_error=id_err.astype(np.float32),
            teacher_confidence=conf.astype(np.float32),
            semantic_hard_mask=semantic_hard,
            changed_mask=changed,
            stable_mask=stable,
            identity_hard_mask=id_hard,
            strict_residual_semantic_utility_mask=sem_pos,
            strict_residual_identity_utility_mask=id_pos,
            strict_stable_suppress_mask=stable_suppress,
            strict_residual_gate_target=gate_target,
            strict_residual_gate_available_mask=gate_available,
            leakage_safe=True,
            future_labels_supervision_only=True,
        )
        totals["samples"] += 1
        totals["valid_sem"] += int(valid_sem.sum())
        totals["sem_pos"] += int(sem_pos.sum())
        totals["id_valid"] += int(id_hard.sum())
        totals["id_pos"] += int(id_pos.sum())
        totals["stable"] += int(stable.sum())
        totals["stable_suppress"] += int(stable_suppress.sum())
        totals["gate_available"] += int(gate_available.sum())
        totals["gate_pos"] += int(gate_target[gate_available].sum()) if gate_available.any() else 0
    summary = {
        "sample_count": totals["samples"],
        "strict_residual_semantic_positive_count": totals["sem_pos"],
        "strict_residual_identity_positive_count": totals["id_pos"],
        "strict_stable_suppress_count": totals["stable_suppress"],
        "valid_semantic_count": totals["valid_sem"],
        "identity_hard_count": totals["id_valid"],
        "strict_residual_semantic_positive_ratio": float(totals["sem_pos"] / max(totals["valid_sem"], 1)),
        "strict_residual_identity_positive_ratio": float(totals["id_pos"] / max(totals["id_valid"], 1)),
        "strict_stable_suppress_ratio": float(totals["stable_suppress"] / max(totals["stable"], 1)),
        "strict_gate_positive_ratio": float(totals["gate_pos"] / max(totals["gate_available"], 1)),
    }
    return summary, th


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--confidence-quantile", type=float, default=0.60)
    p.add_argument("--error-quantile", type=float, default=0.75)
    p.add_argument("--stable-error-quantile", type=float, default=0.25)
    p.add_argument("--identity-error-quantile", type=float, default=0.75)
    p.add_argument("--min-teacher-confidence", type=float, default=0.60)
    p.add_argument("--max-semantic-positive-ratio", type=float, default=0.25)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    split_summaries: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    blockers: dict[str, str] = {}
    for split in ("train", "val", "test"):
        summary, th = process_split(split, args)
        split_summaries[split] = summary
        thresholds[split] = th
        if summary["strict_residual_semantic_positive_count"] <= 0:
            blockers[split] = "no strict residual semantic positives"
        elif summary["strict_residual_semantic_positive_ratio"] > args.max_semantic_positive_ratio + 1e-6:
            blockers[split] = "strict positives exceed 25 percent cap"
        elif summary["strict_stable_suppress_count"] <= 0:
            blockers[split] = "no strict stable suppress negatives"
    ready = not blockers
    payload = {
        "generated_at_utc": utc_now(),
        "source_root": str(SOURCE_ROOT.relative_to(ROOT)),
        "output_root": str(OUT_ROOT.relative_to(ROOT)),
        "strict_residual_semantic_positive_ratio_by_split": {k: v["strict_residual_semantic_positive_ratio"] for k, v in split_summaries.items()},
        "strict_residual_identity_positive_ratio_by_split": {k: v["strict_residual_identity_positive_ratio"] for k, v in split_summaries.items()},
        "strict_stable_suppress_ratio_by_split": {k: v["strict_stable_suppress_ratio"] for k, v in split_summaries.items()},
        "thresholds_by_split": thresholds,
        "teacher_confidence_threshold_by_split": {k: v["teacher_confidence_threshold"] for k, v in thresholds.items()},
        "pointwise_error_threshold_by_split": {k: v["pointwise_error_threshold"] for k, v in thresholds.items()},
        "split_summaries": split_summaries,
        "strict_residual_utility_target_ready": bool(ready),
        "exact_blockers": blockers,
        "leakage_safe": True,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.5 Strict Residual Utility Target Build",
        payload,
        [
            "strict_residual_semantic_positive_ratio_by_split",
            "strict_residual_identity_positive_ratio_by_split",
            "strict_stable_suppress_ratio_by_split",
            "thresholds_by_split",
            "teacher_confidence_threshold_by_split",
            "pointwise_error_threshold_by_split",
            "strict_residual_utility_target_ready",
            "exact_blockers",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
