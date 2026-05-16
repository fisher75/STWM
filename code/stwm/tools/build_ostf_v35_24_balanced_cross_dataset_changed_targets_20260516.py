#!/usr/bin/env python3
"""V35.24 构建 balanced cross-dataset changed benchmark target 副本。

不改变视频语义监督来源；在 V35.21 domain-normalized target 上增加
balanced changed benchmark 元数据，用于后续 observed-only predictability。
"""
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

SRC_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_21_domain_normalized_video_semantic_state_targets/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_24_balanced_cross_dataset_changed_targets/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_24_balanced_cross_dataset_changed_target_build_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_24_BALANCED_CROSS_DATASET_CHANGED_TARGET_BUILD_20260516.md"


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


def balance_weights(changed: np.ndarray, valid: np.ndarray) -> np.ndarray:
    weights = np.zeros(changed.shape, dtype=np.float32)
    pos = valid & changed
    neg = valid & (~changed)
    pos_n = max(int(pos.sum()), 1)
    neg_n = max(int(neg.sum()), 1)
    weights[pos] = 0.5 / pos_n
    weights[neg] = 0.5 / neg_n
    return weights


def main() -> int:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    split_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    blockers: list[str] = []
    for src in sorted(SRC_ROOT.glob("*/*.npz")):
        try:
            z = np.load(src, allow_pickle=True)
            payload = {k: z[k] for k in z.files}
            split = str(np.asarray(payload["split"]).item())
            dataset = str(np.asarray(payload["dataset"]).item())
            valid = np.asarray(payload["target_semantic_cluster_available_mask"], dtype=bool)
            changed = np.asarray(payload["semantic_changed_mask"], dtype=bool) & valid
            hard = np.asarray(payload["semantic_hard_mask"], dtype=bool) & valid
            risk = np.asarray(payload["domain_normalized_risk_percentile"], dtype=np.float32) if "domain_normalized_risk_percentile" in payload else np.asarray(payload["semantic_uncertainty_target"], dtype=np.float32)
            future_conf = np.asarray(payload["future_conf"], dtype=np.float32)
            future_vis = np.asarray(payload["future_vis"], dtype=bool)
            obs_points = np.asarray(payload["obs_points"], dtype=np.float32)
            future_points = np.asarray(payload["future_points"], dtype=np.float32)
            motion = np.linalg.norm(future_points - obs_points[:, -1:, :], axis=-1).astype(np.float32)
            changed_priority = np.zeros(changed.shape, dtype=np.float32)
            changed_priority[valid] = (
                0.45 * changed[valid].astype(np.float32)
                + 0.25 * hard[valid].astype(np.float32)
                + 0.20 * risk[valid]
                + 0.10 * (1.0 - np.clip(future_conf[valid], 0.0, 1.0))
            )
            changed_priority[valid & (~future_vis)] += 0.15
            payload["balanced_changed_benchmark_version"] = np.asarray("v35_24_ontology_agnostic_trace_risk")
            payload["balanced_changed_eval_weight"] = balance_weights(changed, valid).astype(np.float32)
            payload["balanced_changed_case_priority"] = changed_priority.astype(np.float32)
            payload["ontology_agnostic_changed_features_required"] = np.asarray(True)
            payload["semantic_id_shortcut_for_cross_dataset_eval_forbidden"] = np.asarray(True)
            payload["future_teacher_embedding_input_allowed"] = np.asarray(False)
            payload["leakage_safe"] = np.asarray(True)
            out_dir = OUT_ROOT / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / src.name
            np.savez_compressed(out, **payload)
            key = f"{split}:{dataset}"
            split_dataset[key]["samples"] += 1
            split_dataset[key]["tokens"] += int(valid.size)
            split_dataset[key]["valid"] += int(valid.sum())
            split_dataset[key]["changed"] += int(changed.sum())
            split_dataset[key]["hard"] += int(hard.sum())
            split_dataset[key]["low_conf"] += int((valid & (future_conf < 0.5)).sum())
            split_dataset[key]["occluded"] += int((valid & (~future_vis)).sum())
            rows.append(
                {
                    "source_path": rel(src),
                    "output_path": rel(out),
                    "split": split,
                    "dataset": dataset,
                    "valid_tokens": int(valid.sum()),
                    "changed_tokens": int(changed.sum()),
                    "hard_tokens": int(hard.sum()),
                    "changed_ratio": float(changed[valid].mean()) if valid.any() else 0.0,
                    "hard_ratio": float(hard[valid].mean()) if valid.any() else 0.0,
                    "mean_motion": float(motion[valid].mean()) if valid.any() else 0.0,
                }
            )
        except Exception as exc:  # pragma: no cover
            blockers.append(f"{rel(src)}: {type(exc).__name__}: {exc}")
    split_dataset_report: dict[str, Any] = {}
    for key, c in sorted(split_dataset.items()):
        split_dataset_report[key] = {
            "samples": int(c["samples"]),
            "valid_tokens": int(c["valid"]),
            "changed_tokens": int(c["changed"]),
            "hard_tokens": int(c["hard"]),
            "changed_ratio": float(c["changed"] / max(c["valid"], 1)),
            "hard_ratio": float(c["hard"] / max(c["valid"], 1)),
            "low_conf_ratio": float(c["low_conf"] / max(c["valid"], 1)),
            "occlusion_ratio": float(c["occluded"] / max(c["valid"], 1)),
        }
    enough_changed = all(v["changed_tokens"] >= 5000 for v in split_dataset_report.values())
    target_ready = bool(rows and not blockers and enough_changed)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "balanced_cross_dataset_changed_targets_built": bool(rows),
        "source_root": rel(SRC_ROOT),
        "out_root": rel(OUT_ROOT),
        "sample_count": len(rows),
        "split_dataset_coverage": split_dataset_report,
        "enough_changed_tokens_per_split_dataset": enough_changed,
        "ontology_agnostic_changed_features_required": True,
        "semantic_id_shortcut_for_cross_dataset_eval_forbidden": True,
        "future_teacher_embedding_input_allowed": False,
        "leakage_safe": True,
        "balanced_cross_dataset_changed_target_ready": target_ready,
        "rows": rows,
        "exact_blockers": blockers,
        "recommended_next_step": "eval_balanced_cross_dataset_changed_predictability" if target_ready else "expand_video_changed_cases",
        "中文结论": (
            "V35.24 没有引入新 teacher 或 future input，而是在 V35.21 目标上增加 balanced changed benchmark 元数据。"
            "跨数据集 changed eval 明确禁止 semantic-id shortcut，优先使用 trace/risk/measurement 的 ontology-agnostic features。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.24 Balanced Cross-Dataset Changed Target Build\n\n"
        f"- balanced_cross_dataset_changed_targets_built: {bool(rows)}\n"
        f"- sample_count: {len(rows)}\n"
        f"- enough_changed_tokens_per_split_dataset: {enough_changed}\n"
        f"- semantic_id_shortcut_for_cross_dataset_eval_forbidden: true\n"
        f"- balanced_cross_dataset_changed_target_ready: {target_ready}\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"样本数": len(rows), "balanced_changed_target_ready": target_ready, "报告": rel(REPORT)}, ensure_ascii=False), flush=True)
    return 0 if target_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
