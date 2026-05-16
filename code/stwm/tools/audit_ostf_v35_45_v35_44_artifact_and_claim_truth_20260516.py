#!/usr/bin/env python3
"""V35.45 审计 V35.44 artifact packaging 与 claim truth。"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

REPORT = ROOT / "reports/stwm_ostf_v35_45_v35_44_artifact_and_claim_truth_audit_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_45_V35_44_ARTIFACT_AND_CLAIM_TRUTH_AUDIT_20260516.md"

JSONS = {
    "v35_34": ROOT / "reports/stwm_ostf_v35_34_raw_video_frontend_reproducibility_harness_20260516.json",
    "v35_38": ROOT / "reports/stwm_ostf_v35_38_eval_balanced_raw_video_frontend_rerun_subset_20260516.json",
    "v35_42": ROOT / "reports/stwm_ostf_v35_42_identity_label_provenance_and_valid_claim_20260516.json",
    "v35_43": ROOT / "reports/stwm_ostf_v35_43_raw_video_closure_visualization_manifest_20260516.json",
}
DOCS = {
    "v35_34": ROOT / "docs/STWM_OSTF_V35_34_RAW_VIDEO_FRONTEND_REPRODUCIBILITY_HARNESS_20260516.md",
    "v35_38": ROOT / "docs/STWM_OSTF_V35_38_EVAL_BALANCED_RAW_VIDEO_FRONTEND_RERUN_SUBSET_20260516.md",
    "v35_42": ROOT / "docs/STWM_OSTF_V35_42_IDENTITY_LABEL_PROVENANCE_AND_VALID_CLAIM_20260516.md",
    "v35_43": ROOT / "docs/STWM_OSTF_V35_43_RAW_VIDEO_CLOSURE_VISUALIZATION_20260516.md",
    "v35_44": ROOT / "docs/STWM_OSTF_V35_44_RAW_VIDEO_CLOSURE_FINAL_DECISION_20260516.md",
}
V35_44 = ROOT / "reports/stwm_ostf_v35_44_raw_video_closure_final_decision_20260516.json"
V35_44_WRITER = ROOT / "code/stwm/tools/write_ostf_v35_44_raw_video_closure_final_decision_20260516.py"


def jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def find_code_locations() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if V35_44_WRITER.exists():
        text = V35_44_WRITER.read_text(encoding="utf-8")
        for key, path in JSONS.items():
            needle = path.name
            locs = []
            for i, line in enumerate(text.splitlines(), start=1):
                if needle in line or key.upper() in line:
                    locs.append(f"{rel(V35_44_WRITER)}:{i}: {line.strip()}")
            out[key] = locs
    return out


def md_bool(path: Path, key: str) -> bool | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(rf"{re.escape(key)}:\s*(true|false|True|False)", text)
    if not m:
        return None
    return m.group(1).lower() == "true"


def main() -> int:
    v44 = load_json(V35_44)
    missing = {k: not p.exists() for k, p in JSONS.items()}
    depends = bool(
        v44
        and any(missing.values())
        and (
            v44.get("raw_video_frontend_rerun_done")
            or v44.get("semantic_three_seed_passed_on_eval_balanced_raw_rerun")
            or v44.get("identity_three_seed_passed_on_real_instance_subset")
            or v44.get("visualization_ready")
        )
    )
    harness_ready_json = load_json(JSONS["v35_34"]).get("raw_video_frontend_reproducibility_harness_ready")
    harness_ready_md = md_bool(DOCS["v35_34"], "raw_video_frontend_reproducibility_harness_ready")
    harness_false_due_missing = bool(missing["v35_34"] and harness_ready_md is True)
    bounded_only = bool(
        v44.get("m128_h32_video_system_benchmark_claim_allowed", False)
        and not v44.get("full_cvpr_scale_claim_allowed", True)
        and "bounded" in str(v44.get("m128_h32_video_system_benchmark_claim_boundary", "")).lower()
    )
    vspw_excluded = bool("VSPW" in str(v44.get("m128_h32_video_system_benchmark_claim_boundary", "")) and "诊断" in str(v44.get("m128_h32_video_system_benchmark_claim_boundary", "")))
    artifact_required = any(missing.values())
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_truth_audit_done": True,
        "v35_34_json_missing": missing["v35_34"],
        "v35_38_json_missing": missing["v35_38"],
        "v35_42_json_missing": missing["v35_42"],
        "v35_43_json_missing": missing["v35_43"],
        "v35_44_depends_on_missing_json": depends,
        "artifact_packaging_fixed_required": artifact_required,
        "harness_ready_false_due_missing_json": harness_false_due_missing,
        "v35_34_harness_ready_json_value": harness_ready_json,
        "bounded_m128_h32_claim_only": bounded_only,
        "full_cvpr_scale_claim_allowed": False,
        "vspw_pseudo_identity_excluded_from_claim_gate": vspw_excluded,
        "exact_code_locations": find_code_locations(),
        "recommended_fix": "补齐缺失 JSON artifacts" if artifact_required else "无需补齐；继续扩大 V35.45 raw-video closure benchmark",
        "中文结论": (
            "本地 live repo 中 V35.34/V35.38/V35.42/V35.43 依赖 JSON 均存在；V35.44 是 bounded M128/H32 claim，不是 full CVPR-scale claim。"
            if not artifact_required
            else "发现 V35.44 依赖 JSON 缺失，必须先 rematerialize artifacts。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.45 V35.44 Artifact And Claim Truth Audit\n\n"
        f"- artifact_truth_audit_done: true\n"
        f"- v35_34_json_missing: {missing['v35_34']}\n"
        f"- v35_38_json_missing: {missing['v35_38']}\n"
        f"- v35_42_json_missing: {missing['v35_42']}\n"
        f"- v35_43_json_missing: {missing['v35_43']}\n"
        f"- v35_44_depends_on_missing_json: {depends}\n"
        f"- artifact_packaging_fixed_required: {artifact_required}\n"
        f"- bounded_m128_h32_claim_only: {bounded_only}\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_fix: {report['recommended_fix']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact_truth_audit_done": True, "artifact_packaging_fixed_required": artifact_required}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
