#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


DECISION = ROOT / "reports/stwm_ostf_v34_12_decision_20260513.json"
TRUTH_AUDIT = ROOT / "reports/stwm_ostf_v34_12_v34_11_measurement_truth_audit_20260513.json"
LOCAL_DECISION = ROOT / "reports/stwm_ostf_v34_12_local_evidence_oracle_residual_probe_decision_20260513.json"
NONORACLE_JSON = ROOT / "reports/stwm_ostf_v34_12_nonoracle_measurement_selector_probe_20260513.json"
ART_JSON = ROOT / "reports/stwm_ostf_v34_12_artifact_rematerialization_20260513.json"
VIS_JSON = ROOT / "reports/stwm_ostf_v34_12_local_evidence_visualization_manifest_20260513.json"

DOC_NONORACLE = ROOT / "docs/STWM_OSTF_V34_12_NONORACLE_MEASUREMENT_SELECTOR_PROBE_20260513.md"
DOC_VIS = ROOT / "docs/STWM_OSTF_V34_12_LOCAL_EVIDENCE_VISUALIZATION_20260513.md"
DOC_ART = ROOT / "docs/STWM_OSTF_V34_12_ARTIFACT_REMATERIALIZATION_20260513.md"

SRC_SELECTOR = ROOT / "code/stwm/tools/eval_ostf_v34_12_nonoracle_measurement_selector_probe_20260513.py"
SRC_REMAT = ROOT / "code/stwm/tools/rematerialize_ostf_v34_12_artifacts_20260513.py"
SRC_DECISION = ROOT / "code/stwm/tools/write_ostf_v34_12_decision_20260513.py"
SRC_ENCODER = ROOT / "code/stwm/modules/ostf_v34_12_local_semantic_evidence_encoder.py"
SRC_MODEL = ROOT / "code/stwm/modules/ostf_v34_12_measurement_causal_residual_memory.py"

REPORT = ROOT / "reports/stwm_ostf_v34_13_v34_12_artifact_and_selector_truth_audit_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_13_V34_12_ARTIFACT_AND_SELECTOR_TRUTH_AUDIT_20260513.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def line_hits(path: Path, patterns: list[str]) -> list[dict[str, Any]]:
    src = text(path).splitlines()
    out: list[dict[str, Any]] = []
    for i, line in enumerate(src, start=1):
        for pat in patterns:
            if re.search(pat, line):
                out.append({"path": str(path.relative_to(ROOT)), "line": i, "pattern": pat, "code": line.strip()})
    return out


def main() -> int:
    decision = load(DECISION)
    truth = load(TRUTH_AUDIT)
    local = load(LOCAL_DECISION)
    selector_src = text(SRC_SELECTOR)
    model_src = text(SRC_MODEL)
    decision_src = text(SRC_DECISION)

    nonoracle_missing = not NONORACLE_JSON.exists()
    art_missing = not ART_JSON.exists()
    vis_missing = not VIS_JSON.exists()
    final_depends = bool(
        ("SELECTOR" in decision_src and "stwm_ostf_v34_12_nonoracle_measurement_selector_probe_20260513.json" in decision_src)
        and nonoracle_missing
    )
    teacher_inconsistent = bool(
        truth.get("measurement_teacher_name_inconsistent", False)
        or decision.get("measurement_teacher_name_inconsistent", False)
    )
    selector_fixed = bool(
        "observed_only_selector" in selector_src
        and "torch.optim" not in selector_src
        and "train" not in selector_src.lower().split("def main", 1)[0]
    )
    selector_trained = bool("torch.optim" in selector_src or "checkpoint" in selector_src.lower())
    forward_gate_zero = bool("gate = torch.zeros_like" in model_src and "force_gate_one" in model_src)
    local_oracle_mask = bool(
        "causal_assignment_residual_semantic_mask" in text(ROOT / "code/stwm/tools/train_ostf_v34_12_local_evidence_oracle_residual_probe_20260513.py")
        and "compose(" in text(ROOT / "code/stwm/tools/train_ostf_v34_12_local_evidence_oracle_residual_probe_20260513.py")
    )

    exact = {
        "selector_fixed_heuristic": line_hits(SRC_SELECTOR, [r"def observed_only_selector", r"quantile", r"fixed_"]),
        "forward_gate_zero": line_hits(SRC_MODEL, [r"gate = torch\.zeros_like", r"force_gate_one"]),
        "oracle_mask_compose": line_hits(ROOT / "code/stwm/tools/train_ostf_v34_12_local_evidence_oracle_residual_probe_20260513.py", [r"causal_assignment_residual_semantic_mask", r"def compose"]),
        "decision_dependencies": line_hits(SRC_DECISION, [r"SELECTOR", r"VIS", r"ART"]),
    }
    if nonoracle_missing or art_missing or vis_missing:
        rec = "先补齐 V34.12 缺失 JSON artifacts，再训练真正 observed-only selector。"
    elif selector_fixed or not selector_trained:
        rec = "把 V34.12 固定规则 selector 替换成 V34.13 训练式 non-oracle selector。"
    elif forward_gate_zero or local_oracle_mask:
        rec = "selector 过后再修 selector-conditioned local evidence；当前仍不是 learned semantic field。"
    else:
        rec = "继续执行 selector-conditioned local evidence oracle probe。"

    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.12 artifact/source truth 二次审计完成：重点核验缺失 JSON、selector 是否训练、forward gate 是否默认关闭，以及 local evidence 是否仍靠 oracle mask compose。",
        "nonoracle_selector_json_missing": nonoracle_missing,
        "artifact_rematerialization_json_missing": art_missing,
        "visualization_manifest_json_missing": vis_missing,
        "final_decision_depends_on_missing_selector_json": final_depends,
        "measurement_teacher_name_inconsistent": teacher_inconsistent,
        "selector_is_fixed_heuristic": selector_fixed,
        "selector_was_trained": selector_trained,
        "forward_gate_zero_by_default": forward_gate_zero,
        "local_probe_is_oracle_masked": local_oracle_mask,
        "docs_present": {
            "nonoracle_selector_doc": DOC_NONORACLE.exists(),
            "artifact_rematerialization_doc": DOC_ART.exists(),
            "visualization_doc": DOC_VIS.exists(),
        },
        "v34_12_decision_snapshot": {
            "measurement_selector_nonoracle_passed": decision.get("measurement_selector_nonoracle_passed"),
            "measurement_quality_overestimated_by_oracle": decision.get("measurement_quality_overestimated_by_oracle"),
            "oracle_residual_probe_passed": decision.get("oracle_residual_probe_passed"),
            "semantic_measurements_load_bearing_on_residual": decision.get("semantic_measurements_load_bearing_on_residual"),
        },
        "v34_12_local_decision_snapshot": {
            "zero_semantic_measurements_metric_delta": local.get("zero_semantic_measurements_metric_delta"),
            "shuffle_semantic_measurements_metric_delta": local.get("shuffle_semantic_measurements_metric_delta"),
        },
        "exact_code_locations": exact,
        "recommended_fix": rec,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.13 对 V34.12 artifact/source truth 的中文审计",
        payload,
        [
            "中文结论",
            "nonoracle_selector_json_missing",
            "artifact_rematerialization_json_missing",
            "visualization_manifest_json_missing",
            "final_decision_depends_on_missing_selector_json",
            "measurement_teacher_name_inconsistent",
            "selector_is_fixed_heuristic",
            "selector_was_trained",
            "forward_gate_zero_by_default",
            "local_probe_is_oracle_masked",
            "recommended_fix",
        ],
    )
    print(f"已写出 V34.13 artifact/source truth 审计: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
