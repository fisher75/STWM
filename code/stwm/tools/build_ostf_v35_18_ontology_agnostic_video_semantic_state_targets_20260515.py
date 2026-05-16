#!/usr/bin/env python3
"""构建 V35.18 ontology-agnostic video semantic state targets。"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools import build_ostf_v35_14_mask_derived_video_semantic_state_targets_20260515 as base
from stwm.tools.ostf_v17_common_20260502 import ROOT

OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_18_ontology_agnostic_video_semantic_state_targets/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_18_ontology_agnostic_video_semantic_state_target_build_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_18_ONTOLOGY_AGNOSTIC_VIDEO_SEMANTIC_STATE_TARGET_BUILD_20260515.md"

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


def _transition_masks(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    target = np.asarray(payload["target_semantic_cluster_id"], dtype=np.int64)
    available = np.asarray(payload["target_semantic_cluster_available_mask"], dtype=bool)
    obs_last = np.asarray(payload["obs_semantic_last_id"], dtype=np.int64)
    future_vis = np.asarray(payload["future_vis"], dtype=bool)
    future_conf = np.asarray(payload["future_conf"], dtype=np.float32)
    obs_points = np.asarray(payload["obs_points"], dtype=np.float32)
    future_points = np.asarray(payload["future_points"], dtype=np.float32)
    m, h = target.shape

    prev = np.concatenate([obs_last[:, None], target[:, :-1]], axis=1)
    label_transition = available & (prev >= 0) & (target >= 0) & (target != prev)
    obs_to_future_transition = available & (obs_last[:, None] >= 0) & (target != obs_last[:, None])
    motion = np.linalg.norm(future_points - obs_points[:, -1:, :], axis=-1).astype(np.float32)
    valid_motion = motion[available]
    motion_hi = float(np.quantile(valid_motion, 0.70)) if valid_motion.size else 0.0
    motion_boundary_risk = available & (motion >= motion_hi) & label_transition
    visibility_risk = available & ((~future_vis) | (future_conf < 0.35))
    # 这里的 changed 不依赖跨数据集 ontology 名称，只看同一 video 内 mask id 是否发生状态迁移；
    # visibility risk 单独作为 hard/risk，而不是把所有低置信点都当 changed positive。
    ontology_changed = available & (label_transition | motion_boundary_risk)
    risk = np.clip(
        0.52 * visibility_risk.astype(np.float32)
        + 0.33 * label_transition.astype(np.float32)
        + 0.15 * np.clip(motion / max(motion_hi, 1.0), 0.0, 1.0),
        0.0,
        1.0,
    ).astype(np.float32)
    ontology_hard = available & (ontology_changed | (risk > 0.58))
    family = np.full((m, h), COPY_INSTANCE, dtype=np.int64)
    stable = available & ~ontology_changed
    family[stable & (future_conf > 0.82)] = COPY_MAX
    family[stable & (future_conf <= 0.82)] = COPY_INSTANCE
    family[stable & visibility_risk] = COPY_LAST
    family[ontology_changed] = CHANGED
    family[(risk > 0.74) | (~available)] = UNCERTAIN
    return {
        "mask_label_transition_mask": label_transition.astype(bool),
        "obs_to_future_mask_transition_mask": obs_to_future_transition.astype(bool),
        "motion_boundary_risk_mask": motion_boundary_risk.astype(bool),
        "visibility_conditioned_semantic_risk": risk.astype(np.float32),
        "ontology_agnostic_semantic_changed_mask": ontology_changed.astype(bool),
        "ontology_agnostic_semantic_hard_mask": ontology_hard.astype(bool),
        "ontology_agnostic_evidence_anchor_family_target": family.astype(np.int64),
    }


def _rewrite_outputs(out_root: Path) -> tuple[list[dict[str, Any]], dict[str, Counter[str]], Counter[int]]:
    rows: list[dict[str, Any]] = []
    by_split_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    family_counts: Counter[int] = Counter()
    for p in sorted(out_root.glob("*/*.npz")):
        z = np.load(p, allow_pickle=True)
        payload = {k: z[k] for k in z.files}
        extra = _transition_masks(payload)
        payload.update(extra)
        # 让 V35.18 默认 target 就是 ontology-agnostic changed/hard/family，同时保留原字段副本。
        payload["original_semantic_changed_mask"] = np.asarray(payload["semantic_changed_mask"], dtype=bool)
        payload["original_semantic_hard_mask"] = np.asarray(payload["semantic_hard_mask"], dtype=bool)
        payload["original_evidence_anchor_family_target"] = np.asarray(payload["evidence_anchor_family_target"], dtype=np.int64)
        payload["semantic_changed_mask"] = extra["ontology_agnostic_semantic_changed_mask"]
        payload["semantic_hard_mask"] = extra["ontology_agnostic_semantic_hard_mask"]
        payload["evidence_anchor_family_target"] = extra["ontology_agnostic_evidence_anchor_family_target"]
        payload["semantic_changed_is_real_video_state"] = True
        payload["video_semantic_target_source"] = "mask_transition / panoptic_instance_transition / visibility_conditioned_risk"
        payload["future_teacher_embedding_input_allowed"] = False
        payload["leakage_safe"] = True
        np.savez_compressed(p, **payload)

        split = str(np.asarray(payload["split"]).item())
        dataset = str(np.asarray(payload["dataset"]).item())
        available = np.asarray(payload["target_semantic_cluster_available_mask"], dtype=bool)
        changed = np.asarray(payload["semantic_changed_mask"], dtype=bool) & available
        hard = np.asarray(payload["semantic_hard_mask"], dtype=bool) & available
        family = np.asarray(payload["evidence_anchor_family_target"], dtype=np.int64)
        by_split_dataset[f"{split}:{dataset}"]["samples"] += 1
        by_split_dataset[f"{split}:{dataset}"]["valid"] += int(available.sum())
        by_split_dataset[f"{split}:{dataset}"]["tokens"] += int(available.size)
        by_split_dataset[f"{split}:{dataset}"]["changed"] += int(changed.sum())
        by_split_dataset[f"{split}:{dataset}"]["hard"] += int(hard.sum())
        family_counts.update([int(v) for v in family[available].reshape(-1)])
        rows.append(
            {
                "path": rel(p),
                "split": split,
                "dataset": dataset,
                "valid_ratio": float(available.mean()),
                "ontology_agnostic_changed_ratio": float(changed[available].mean()) if available.any() else 0.0,
                "ontology_agnostic_hard_ratio": float(hard[available].mean()) if available.any() else 0.0,
            }
        )
    return rows, by_split_dataset, family_counts


def main() -> int:
    base.OUT_ROOT = OUT_ROOT
    base.REPORT = REPORT
    base.DOC = DOC
    rc = base.main()
    rows, by_split_dataset, family_counts = _rewrite_outputs(OUT_ROOT)

    split_dataset_report: dict[str, Any] = {}
    for key, c in sorted(by_split_dataset.items()):
        split_dataset_report[key] = {
            "samples": int(c["samples"]),
            "valid_ratio": float(c["valid"] / max(c["tokens"], 1)),
            "changed_ratio": float(c["changed"] / max(c["valid"], 1)),
            "semantic_hard_ratio": float(c["hard"] / max(c["valid"], 1)),
        }
    vipseg_train = split_dataset_report.get("train:VIPSEG", {}).get("samples", 0)
    vipseg_val = split_dataset_report.get("val:VIPSEG", {}).get("samples", 0)
    vspw_test_changed = split_dataset_report.get("test:VSPW", {}).get("changed_ratio", 0.0)
    target_split_still_sparse = bool(vspw_test_changed < 0.06)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ontology_agnostic_video_semantic_state_targets_built": bool(rows),
        "base_builder_return_code": rc,
        "out_root": rel(OUT_ROOT),
        "sample_count": len(rows),
        "split_dataset_coverage": split_dataset_report,
        "vipseg_source_train_val_expanded": bool(vipseg_train >= 50 and vipseg_val >= 12),
        "vspw_test_changed_hard_sparse": target_split_still_sparse,
        "evidence_anchor_family_distribution": {FAMILY_NAMES[k]: int(v) for k, v in sorted(family_counts.items()) if 0 <= k < len(FAMILY_NAMES)},
        "video_semantic_target_source": "mask_transition / panoptic_instance_transition / visibility_conditioned_risk",
        "future_teacher_embedding_input_allowed": False,
        "semantic_changed_is_real_video_state": True,
        "leakage_safe": True,
        "rows": rows,
        "recommended_next_step": "eval_vipseg_to_vspw_domain_shift_with_stratified_target",
        "中文结论": (
            "V35.18 已把 semantic change 从跨数据集 ontology 类名依赖，改成同一 video 内 mask transition "
            "和 visibility-conditioned semantic risk；VSPW test 如果 changed/hard 仍稀疏，后续评估必须做分层诊断。"
        ),
    }
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.18 Ontology-Agnostic Video Semantic Target Build\n\n"
        f"- ontology_agnostic_video_semantic_state_targets_built: {bool(rows)}\n"
        f"- sample_count: {len(rows)}\n"
        f"- video_semantic_target_source: {report['video_semantic_target_source']}\n"
        f"- vipseg_source_train_val_expanded: {report['vipseg_source_train_val_expanded']}\n"
        f"- vspw_test_changed_hard_sparse: {target_split_still_sparse}\n"
        f"- future_teacher_embedding_input_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"样本数": len(rows), "VSPW_test_changed稀疏": target_split_still_sparse, "报告": rel(REPORT)}, ensure_ascii=False), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
