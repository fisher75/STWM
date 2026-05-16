#!/usr/bin/env python3
"""V35.21 构建 domain-normalized video semantic state targets。"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

SRC_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_19_boundary_risk_video_semantic_state_targets/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_21_domain_normalized_video_semantic_state_targets/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_21_domain_normalized_video_semantic_state_target_build_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_21_DOMAIN_NORMALIZED_VIDEO_SEMANTIC_STATE_TARGET_BUILD_20260515.md"

FAMILY_NAMES = ["copy_last_visible", "copy_instance_pooled", "copy_max_confidence", "changed_transition", "uncertain_abstain"]
COPY_LAST = 0
COPY_INSTANCE = 1
COPY_MAX = 2
CHANGED = 3
UNCERTAIN = 4


def jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def percentile_rank(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(score, dtype=np.float32)
    vals = score[valid]
    if vals.size == 0:
        return out
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(vals), dtype=np.float32)
    out[valid] = ranks
    return out


def postprocess_payload(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    valid = np.asarray(payload["target_semantic_cluster_available_mask"], dtype=bool)
    target = np.asarray(payload["target_semantic_cluster_id"], dtype=np.int64)
    obs_last = np.asarray(payload["obs_semantic_last_id"], dtype=np.int64)
    future_conf = np.asarray(payload["future_conf"], dtype=np.float32)
    future_vis = np.asarray(payload["future_vis"], dtype=bool)
    obs_points = np.asarray(payload["obs_points"], dtype=np.float32)
    future_points = np.asarray(payload["future_points"], dtype=np.float32)
    boundary = np.asarray(payload["mask_local_boundary_future"], dtype=bool) if "mask_local_boundary_future" in payload else np.zeros_like(valid)
    old_risk = np.asarray(payload["visibility_conditioned_semantic_risk"], dtype=np.float32)
    label_transition = np.asarray(payload["mask_label_transition_mask"], dtype=bool) if "mask_label_transition_mask" in payload else (valid & (target != obs_last[:, None]))
    motion = np.linalg.norm(future_points - obs_points[:, -1:, :], axis=-1).astype(np.float32)
    motion_valid = motion[valid]
    motion_rank = percentile_rank(motion, valid)
    conf_risk = np.clip(1.0 - future_conf, 0.0, 1.0)
    raw_score = np.clip(0.38 * old_risk + 0.25 * boundary.astype(np.float32) + 0.22 * conf_risk + 0.15 * motion_rank, 0.0, 1.0)
    # Per-video percentile calibration: this removes dataset-specific mask density / confidence scale
    # while preserving within-video high-risk semantic state events.
    risk_rank = percentile_rank(raw_score, valid)
    low_vis = valid & (~future_vis)
    domain_changed = valid & ((risk_rank >= 0.90) | label_transition | (boundary & (risk_rank >= 0.78)))
    domain_hard = valid & ((risk_rank >= 0.82) | domain_changed | low_vis)
    uncertainty = np.clip(0.70 * risk_rank + 0.30 * low_vis.astype(np.float32), 0.0, 1.0)
    family = np.full(target.shape, COPY_INSTANCE, dtype=np.int64)
    stable = valid & ~domain_changed
    family[stable & (risk_rank < 0.55)] = COPY_MAX
    family[stable & (risk_rank >= 0.55)] = COPY_LAST
    family[domain_changed] = CHANGED
    family[(risk_rank >= 0.92) | (~valid)] = UNCERTAIN
    payload["pre_domain_normalized_semantic_changed_mask"] = np.asarray(payload["semantic_changed_mask"], dtype=bool)
    payload["pre_domain_normalized_semantic_hard_mask"] = np.asarray(payload["semantic_hard_mask"], dtype=bool)
    payload["pre_domain_normalized_semantic_uncertainty_target"] = np.asarray(payload["semantic_uncertainty_target"], dtype=np.float32)
    payload["domain_normalized_raw_risk_score"] = raw_score.astype(np.float32)
    payload["domain_normalized_risk_percentile"] = risk_rank.astype(np.float32)
    payload["semantic_changed_mask"] = domain_changed.astype(bool)
    payload["semantic_cluster_changed_mask"] = domain_changed.astype(bool)
    payload["semantic_hard_mask"] = domain_hard.astype(bool)
    payload["semantic_uncertainty_target"] = uncertainty.astype(np.float32)
    payload["target_confidence"] = np.clip(1.0 - uncertainty, 0.0, 1.0).astype(np.float32)
    payload["evidence_anchor_family_target"] = family.astype(np.int64)
    payload["evidence_anchor_family_available_mask"] = valid.astype(bool)
    payload["video_semantic_target_source"] = "domain_normalized_mask_boundary_visibility_risk"
    payload["domain_normalized_risk_calibrated"] = np.asarray(True)
    payload["future_teacher_embedding_input_allowed"] = np.asarray(False)
    payload["leakage_safe"] = np.asarray(True)
    return payload


def main() -> int:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    split_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    family_counts: Counter[int] = Counter()
    blockers: list[str] = []
    for src in sorted(SRC_ROOT.glob("*/*.npz")):
        try:
            z = np.load(src, allow_pickle=True)
            payload = {k: z[k] for k in z.files}
            payload = postprocess_payload(payload)
            split = str(np.asarray(payload["split"]).item())
            dataset = str(np.asarray(payload["dataset"]).item())
            out_dir = OUT_ROOT / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / src.name
            np.savez_compressed(out, **payload)
            valid = np.asarray(payload["target_semantic_cluster_available_mask"], dtype=bool)
            changed = np.asarray(payload["semantic_changed_mask"], dtype=bool) & valid
            hard = np.asarray(payload["semantic_hard_mask"], dtype=bool) & valid
            uncertainty_high = (np.asarray(payload["semantic_uncertainty_target"], dtype=np.float32) > 0.5) & valid
            family = np.asarray(payload["evidence_anchor_family_target"], dtype=np.int64)
            key = f"{split}:{dataset}"
            split_dataset[key]["samples"] += 1
            split_dataset[key]["tokens"] += int(valid.size)
            split_dataset[key]["valid"] += int(valid.sum())
            split_dataset[key]["changed"] += int(changed.sum())
            split_dataset[key]["hard"] += int(hard.sum())
            split_dataset[key]["uncertainty_high"] += int(uncertainty_high.sum())
            family_counts.update([int(v) for v in family[valid].reshape(-1)])
            rows.append(
                {
                    "source_path": rel(src),
                    "output_path": rel(out),
                    "split": split,
                    "dataset": dataset,
                    "valid_ratio": float(valid.mean()),
                    "domain_normalized_changed_ratio": float(changed[valid].mean()) if valid.any() else 0.0,
                    "domain_normalized_hard_ratio": float(hard[valid].mean()) if valid.any() else 0.0,
                    "domain_normalized_uncertainty_high_ratio": float(uncertainty_high[valid].mean()) if valid.any() else 0.0,
                }
            )
        except Exception as exc:  # pragma: no cover
            blockers.append(f"{src}: {type(exc).__name__}: {exc}")
    split_dataset_report: dict[str, Any] = {}
    for key, c in sorted(split_dataset.items()):
        split_dataset_report[key] = {
            "samples": int(c["samples"]),
            "valid_ratio": float(c["valid"] / max(c["tokens"], 1)),
            "changed_ratio": float(c["changed"] / max(c["valid"], 1)),
            "semantic_hard_ratio": float(c["hard"] / max(c["valid"], 1)),
            "uncertainty_high_ratio": float(c["uncertainty_high"] / max(c["valid"], 1)),
        }
    ratios = [v["changed_ratio"] for v in split_dataset_report.values()]
    target_split_balanced = bool(ratios and max(ratios) - min(ratios) <= 0.08)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "domain_normalized_video_semantic_state_targets_built": bool(rows),
        "source_root": rel(SRC_ROOT),
        "out_root": rel(OUT_ROOT),
        "sample_count": len(rows),
        "domain_normalization_scope": "per_video_percentile_rank",
        "split_dataset_coverage": split_dataset_report,
        "target_split_balanced_after_normalization": target_split_balanced,
        "evidence_anchor_family_distribution": {FAMILY_NAMES[k]: int(v) for k, v in sorted(family_counts.items()) if 0 <= k < len(FAMILY_NAMES)},
        "video_semantic_target_source": "domain_normalized_mask_boundary_visibility_risk",
        "future_teacher_embedding_input_allowed": False,
        "leakage_safe": True,
        "rows": rows,
        "exact_blockers": blockers[:30],
        "recommended_next_step": "eval_domain_normalized_video_semantic_predictability",
        "中文结论": (
            "V35.21 已将 mask-boundary / visibility risk 做 per-video percentile calibration，"
            "避免 VSPW held-out 因 mask 密度/置信度尺度不同而被全局阈值压成稀疏正例。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.21 Domain-Normalized Video Semantic State Target Build\n\n"
        f"- domain_normalized_video_semantic_state_targets_built: {bool(rows)}\n"
        f"- sample_count: {len(rows)}\n"
        f"- domain_normalization_scope: {report['domain_normalization_scope']}\n"
        f"- target_split_balanced_after_normalization: {target_split_balanced}\n"
        f"- future_teacher_embedding_input_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"样本数": len(rows), "target_split_balanced_after_normalization": target_split_balanced, "报告": rel(REPORT)}, ensure_ascii=False), flush=True)
    return 0 if rows and not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
