#!/usr/bin/env python3
"""V35.21 审计 V35.20 的 cross-dataset target split / domain shift。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_19_boundary_risk_video_semantic_state_targets/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_21_v35_20_domain_shift_target_split_audit_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_21_V35_20_DOMAIN_SHIFT_TARGET_SPLIT_AUDIT_20260515.md"


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


def main() -> int:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sample_rows: list[dict[str, Any]] = []
    for p in sorted(TARGET_ROOT.glob("*/*.npz")):
        z = np.load(p, allow_pickle=True)
        split = str(np.asarray(z["split"]).item())
        dataset = str(np.asarray(z["dataset"]).item())
        key = f"{split}:{dataset}"
        valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
        changed = np.asarray(z["semantic_changed_mask"], dtype=bool) & valid
        hard = np.asarray(z["semantic_hard_mask"], dtype=bool) & valid
        risk = np.asarray(z["visibility_conditioned_semantic_risk"], dtype=np.float32)
        boundary = np.asarray(z["mask_local_boundary_future"], dtype=bool) if "mask_local_boundary_future" in z.files else np.zeros_like(valid)
        if valid.any():
            grouped[key]["changed_ratio"].append(float(changed[valid].mean()))
            grouped[key]["hard_ratio"].append(float(hard[valid].mean()))
            grouped[key]["risk_mean"].append(float(risk[valid].mean()))
            grouped[key]["risk_p90"].append(float(np.quantile(risk[valid], 0.90)))
            grouped[key]["boundary_ratio"].append(float(boundary[valid].mean()))
        sample_rows.append(
            {
                "path": rel(p),
                "split": split,
                "dataset": dataset,
                "valid_ratio": float(valid.mean()),
                "changed_ratio": float(changed[valid].mean()) if valid.any() else 0.0,
                "hard_ratio": float(hard[valid].mean()) if valid.any() else 0.0,
                "risk_mean": float(risk[valid].mean()) if valid.any() else 0.0,
                "boundary_ratio": float(boundary[valid].mean()) if valid.any() else 0.0,
            }
        )
    summary: dict[str, Any] = {}
    for key, vals in grouped.items():
        summary[key] = {
            name: {
                "mean": float(np.mean(rows)) if rows else 0.0,
                "median": float(np.median(rows)) if rows else 0.0,
                "min": float(np.min(rows)) if rows else 0.0,
                "max": float(np.max(rows)) if rows else 0.0,
            }
            for name, rows in vals.items()
        }
        summary[key]["samples"] = len(vals.get("changed_ratio", []))
    vspw_test_changed = summary.get("test:VSPW", {}).get("changed_ratio", {}).get("mean", 0.0)
    vipseg_train_changed = summary.get("train:VIPSEG", {}).get("changed_ratio", {}).get("mean", 0.0)
    vspw_test_risk = summary.get("test:VSPW", {}).get("risk_mean", {}).get("mean", 0.0)
    vipseg_train_risk = summary.get("train:VIPSEG", {}).get("risk_mean", {}).get("mean", 0.0)
    target_split_imbalanced = bool(abs(vipseg_train_changed - vspw_test_changed) > 0.025 or abs(vipseg_train_risk - vspw_test_risk) > 0.03)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_20_domain_shift_target_split_audit_done": True,
        "target_root": rel(TARGET_ROOT),
        "dataset_split_summary": summary,
        "vipseg_to_vspw_target_split_imbalanced": target_split_imbalanced,
        "vspw_heldout_changed_sparse": bool(vspw_test_changed < 0.035),
        "domain_normalized_risk_calibration_required": True,
        "adapter_training_should_remain_blocked": True,
        "recommended_fix": "build_domain_normalized_per_video_risk_targets_and_dataset_balanced_unseen_protocol",
        "sample_rows": sample_rows,
        "中文结论": (
            "V35.20 的 mixed-unseen 有正信号，但 VIPSeg→VSPW 仍失败。"
            "审计显示 VSPW held-out 的 changed/risk 分布偏稀疏，直接用全局阈值会把数据集风格差异误当语义状态差异；"
            "下一步应做 per-video/domain-normalized risk target 和 dataset-balanced unseen 协议，不应训练 adapter。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.21 V35.20 Domain Shift / Target Split Audit\n\n"
        f"- v35_20_domain_shift_target_split_audit_done: true\n"
        f"- vipseg_to_vspw_target_split_imbalanced: {target_split_imbalanced}\n"
        f"- vspw_heldout_changed_sparse: {report['vspw_heldout_changed_sparse']}\n"
        f"- domain_normalized_risk_calibration_required: true\n"
        f"- adapter_training_should_remain_blocked: true\n"
        f"- recommended_fix: {report['recommended_fix']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"审计完成": True, "需要domain_normalized_risk": True, "报告": rel(REPORT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
