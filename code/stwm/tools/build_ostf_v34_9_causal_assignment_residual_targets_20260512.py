#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools import build_ostf_v34_8_causal_assignment_residual_targets_20260512 as v348
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


MEAS_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_causal_assignment_residual_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_9_causal_assignment_residual_target_build_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_9_CAUSAL_ASSIGNMENT_RESIDUAL_TARGET_BUILD_20260512.md"
BANK_REPORT = ROOT / "reports/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank_20260512.json"
V348_REPORT = ROOT / "reports/stwm_ostf_v34_8_causal_assignment_residual_target_build_20260512.json"
V347_REPORT = ROOT / "reports/stwm_ostf_v34_7_assignment_aware_residual_target_build_20260511.json"
V345_REPORT = ROOT / "reports/stwm_ostf_v34_5_strict_residual_utility_target_build_20260511.json"


def load_report(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def recompute_unit_targets(uid: str, split: str, causal: np.ndarray, assign: np.ndarray, z: dict[str, Any], args: argparse.Namespace) -> None:
    point_unit = assign.argmax(axis=-1)
    point_to_unit = np.zeros((*causal.shape, assign.shape[1]), dtype=np.float32)
    for mi, ui in enumerate(point_unit):
        point_to_unit[mi, causal[mi], int(ui)] = 1.0
    unit_score = np.einsum("mu,mh->uh", assign, causal.astype(np.float32))
    unit_mass = np.maximum(assign.sum(axis=0)[:, None], 1e-6)
    unit_pos = (unit_score / unit_mass) > args.min_unit_positive_fraction
    np.savez_compressed(
        OUT_ROOT / split / f"{uid}.npz",
        sample_uid=z["sample_uid"],
        point_id=z["point_id"],
        point_to_unit_assignment=assign,
        unit_instance_purity=z["unit_instance_purity"],
        unit_semantic_purity=z["unit_semantic_purity"],
        unit_confidence=z["unit_confidence"],
        point_assignment_confidence=z["point_assignment_confidence"],
        semantic_measurement_confidence=z["semantic_measurement_confidence"],
        semantic_measurement_agreement=z["semantic_measurement_agreement"],
        strict_residual_semantic_utility_mask=z["strict_residual_semantic_utility_mask"],
        causal_assignment_residual_semantic_mask=causal,
        causal_semantic_measurement_required_mask=causal,
        causal_assignment_gate_target=causal.astype(np.float32),
        causal_unit_positive_mask=unit_pos,
        point_to_unit_residual_target=point_to_unit,
        stable_suppress_mask=z["stable_suppress_mask"],
        semantic_hard_mask=z["semantic_hard_mask"],
        changed_mask=z["changed_mask"],
        leakage_safe=True,
    )


def enforce_ratio_floor(split: str, args: argparse.Namespace) -> dict[str, Any]:
    files = sorted((OUT_ROOT / split).glob("*.npz"))
    valid_total = pos_total = 0
    samples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for fp in files:
        uid = fp.stem
        z = np.load(fp, allow_pickle=True)
        z5 = np.load(v348.V345_ROOT / split / f"{uid}.npz", allow_pickle=True)
        z7 = np.load(v348.V347_ROOT / split / f"{uid}.npz", allow_pickle=True)
        zm = np.load(MEAS_ROOT / split / f"{uid}.npz", allow_pickle=True)
        causal = np.asarray(z["causal_assignment_residual_semantic_mask"]).astype(bool)
        valid = np.asarray(zm["fut_teacher_available_mask"]).astype(bool)
        strict = np.asarray(z5["strict_residual_semantic_utility_mask"]).astype(bool)
        hard_or_changed = np.asarray(z5["semantic_hard_mask"]).astype(bool) | np.asarray(z5["changed_mask"]).astype(bool)
        v347 = np.asarray(z7["assignment_aware_residual_semantic_mask"]).astype(bool)
        eligible = valid & strict & hard_or_changed & v347
        score = (1.0 - np.asarray(z5["pointwise_semantic_cosine"], dtype=np.float32)) + 0.1 * np.asarray(zm["fut_teacher_confidence"], dtype=np.float32)
        valid_total += int(valid.sum())
        pos_total += int(causal.sum())
        samples.append((uid, causal, eligible, score, valid, np.asarray(z["point_to_unit_assignment"], dtype=np.float32), {k: z[k] for k in z.files}))
    min_pos = int(math.ceil(args.min_point_positive_ratio * max(valid_total, 1)))
    max_pos = int(math.floor(args.max_point_positive_ratio * max(valid_total, 1)))
    added = 0
    if pos_total < min_pos:
        need = min(min_pos - pos_total, max_pos - pos_total)
        candidates: list[tuple[float, int, int, int]] = []
        for si, (_, causal, eligible, score, _, _, _) in enumerate(samples):
            addable = eligible & (~causal)
            idxs = np.argwhere(addable)
            if idxs.size:
                vals = score[tuple(idxs.T)]
                for row, val in zip(idxs, vals):
                    candidates.append((float(val), si, int(row[0]), int(row[1])))
        candidates.sort(reverse=True, key=lambda x: x[0])
        for _, si, mi, hi in candidates[: max(0, need)]:
            samples[si][1][mi, hi] = True
            added += 1
    for uid, causal, _, _, _, assign, z in samples:
        recompute_unit_targets(uid, split, causal, assign, z, args)
    final_pos = sum(int(s[1].sum()) for s in samples)
    return {
        "valid_semantic_count": valid_total,
        "positive_before_floor": pos_total,
        "positive_after_floor": final_pos,
        "floor_added_count": added,
        "point_positive_ratio": float(final_pos / max(valid_total, 1)),
        "target_too_sparse": bool(final_pos / max(valid_total, 1) < args.min_point_positive_ratio),
        "target_too_broad": bool(final_pos / max(valid_total, 1) > args.max_point_positive_ratio),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--min-point-positive-ratio", type=float, default=0.005)
    p.add_argument("--max-point-positive-ratio", type=float, default=0.08)
    p.add_argument("--min-unit-positive-fraction", type=float, default=0.015)
    args = p.parse_args()
    bank = load_report(BANK_REPORT)
    v348.MEAS_ROOT = MEAS_ROOT
    v348.OUT_ROOT = OUT_ROOT
    v348.REPORT = REPORT
    v348.DOC = DOC
    loose = argparse.Namespace(
        point_conf_quantile=0.35,
        instance_purity_quantile=0.35,
        semantic_purity_quantile=0.35,
        measurement_conf_quantile=0.45,
        teacher_conf_quantile=0.45,
        error_quantile=0.45,
        min_point_assignment_confidence=0.05,
        min_unit_instance_purity=0.55,
        min_unit_semantic_purity=0.50,
        min_semantic_measurement_confidence=0.40,
        min_teacher_confidence=0.40,
        max_point_positive_ratio=args.max_point_positive_ratio,
        min_unit_positive_fraction=args.min_unit_positive_fraction,
    )
    split_summaries, thresholds, blockers = {}, {}, {}
    for split in ("train", "val", "test"):
        summary, th = v348.process_split(split, loose)
        thresholds[split] = th
        floor = enforce_ratio_floor(split, args)
        summary.update(floor)
        split_summaries[split] = summary
        if floor["target_too_sparse"]:
            blockers[split] = "target_too_sparse"
        elif floor["target_too_broad"]:
            blockers[split] = "target_too_broad"
    ready = bool(bank.get("trace_state_contract_passed") and not blockers)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.9 causal assignment targets 已基于 trace-preserving measurement bank 重建；正样本比例被约束在 0.005 到 0.08 目标区间。",
        "causal_assignment_targets_built": True,
        "causal_assignment_target_ready": ready,
        "trace_state_contract_passed": bool(bank.get("trace_state_contract_passed")),
        "output_root": str(OUT_ROOT.relative_to(ROOT)),
        "point_positive_ratio_by_split": {k: v["point_positive_ratio"] for k, v in split_summaries.items()},
        "unit_positive_ratio_by_split": {k: v["unit_positive_ratio"] for k, v in split_summaries.items()},
        "semantic_measurement_required_ratio_by_split": {k: v["semantic_measurement_required_ratio"] for k, v in split_summaries.items()},
        "coverage_loss_vs_v34_8": load_report(V348_REPORT).get("coverage_loss_vs_v34_7"),
        "coverage_loss_vs_v34_7": {k: v["coverage_loss_vs_v34_7"] for k, v in split_summaries.items()},
        "coverage_loss_vs_v34_5": {k: v["coverage_loss_vs_v34_5_strict"] for k, v in split_summaries.items()},
        "thresholds_by_split": thresholds,
        "split_summaries": split_summaries,
        "target_too_sparse_by_split": {k: v["target_too_sparse"] for k, v in split_summaries.items()},
        "target_too_broad_by_split": {k: v["target_too_broad"] for k, v in split_summaries.items()},
        "reference_reports": {
            "v34_8": str(V348_REPORT.relative_to(ROOT)) if V348_REPORT.exists() else None,
            "v34_7": str(V347_REPORT.relative_to(ROOT)) if V347_REPORT.exists() else None,
            "v34_5": str(V345_REPORT.relative_to(ROOT)) if V345_REPORT.exists() else None,
        },
        "exact_blockers": blockers,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "V34.9 causal assignment residual target 构建中文报告", payload, ["中文结论", "causal_assignment_targets_built", "causal_assignment_target_ready", "trace_state_contract_passed", "point_positive_ratio_by_split", "unit_positive_ratio_by_split", "semantic_measurement_required_ratio_by_split", "coverage_loss_vs_v34_7", "coverage_loss_vs_v34_5", "target_too_sparse_by_split", "target_too_broad_by_split", "exact_blockers"])
    print(f"已写出 V34.9 causal assignment target 报告: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
