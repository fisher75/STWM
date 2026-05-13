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


DECISION = ROOT / "reports/stwm_ostf_v34_11_decision_20260513.json"
AUDIT = ROOT / "reports/stwm_ostf_v34_11_v34_10_semantic_measurement_failure_audit_20260513.json"
LOCAL_TRAIN = ROOT / "reports/stwm_ostf_v34_11_local_semantic_usage_oracle_probe_train_summary_20260513.json"
LOCAL_EVAL = ROOT / "reports/stwm_ostf_v34_11_local_semantic_usage_oracle_probe_eval_summary_20260513.json"
LOCAL_DECISION = ROOT / "reports/stwm_ostf_v34_11_local_semantic_usage_oracle_probe_decision_20260513.json"
QUALITY_JSON = ROOT / "reports/stwm_ostf_v34_11_semantic_measurement_quality_probe_20260513.json"
VIS_JSON = ROOT / "reports/stwm_ostf_v34_11_semantic_measurement_causality_visualization_manifest_20260513.json"
QUALITY_DOC = ROOT / "docs/STWM_OSTF_V34_11_SEMANTIC_MEASUREMENT_QUALITY_PROBE_20260513.md"
VIS_DOC = ROOT / "docs/STWM_OSTF_V34_11_SEMANTIC_MEASUREMENT_CAUSALITY_VISUALIZATION_20260513.md"
QUALITY_SRC = ROOT / "code/stwm/tools/eval_ostf_v34_11_semantic_measurement_quality_probe_20260513.py"
LOCAL_TRAIN_SRC = ROOT / "code/stwm/tools/train_ostf_v34_11_local_semantic_usage_oracle_probe_20260513.py"
LOCAL_EVAL_SRC = ROOT / "code/stwm/tools/eval_ostf_v34_11_local_semantic_usage_oracle_probe_20260513.py"
MODEL_SRC = ROOT / "code/stwm/modules/ostf_v34_8_causal_assignment_bound_residual_memory.py"
REPORT = ROOT / "reports/stwm_ostf_v34_12_v34_11_measurement_truth_audit_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_12_V34_11_MEASUREMENT_TRUTH_AUDIT_20260513.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def locations(path: Path, pats: list[str]) -> list[str]:
    out: list[str] = []
    src = text(path)
    for i, line in enumerate(src.splitlines(), 1):
        if any(p in line for p in pats):
            out.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    return out


def main() -> int:
    dec = load(DECISION)
    audit = load(AUDIT)
    quality = load(QUALITY_JSON)
    local_dec = load(LOCAL_DECISION)
    qsrc = text(QUALITY_SRC)
    train_src = text(LOCAL_TRAIN_SRC)
    model_src = text(MODEL_SRC)
    quality_missing = not QUALITY_JSON.exists()
    vis_missing = not VIS_JSON.exists()
    teacher_name = audit.get("semantic_measurement_bank_teacher_name") or audit.get("semantic_measurement_stats", {}).get("teacher_name")
    best_bank = dec.get("best_measurement_bank")
    inconsistent = bool(teacher_name and best_bank and str(teacher_name) not in str(best_bank))
    uses_oracle_best = bool("np.maximum.reduce" in qsrc and "best_cos" in qsrc and "fut_teacher_embedding" in qsrc)
    gate_zero = bool("gate = torch.zeros_like" in model_src and "final_sem = F.normalize" in model_src)
    local_oracle = bool("causal_assignment_residual_semantic_mask" in train_src and "local_compose" in train_src)
    usage_indirect = bool("semantic_measurement_usage_score" in train_src and "local_compose" in train_src and "semantic_measurement_usage_score" not in re.sub(r"def local_compose.*?return", "", train_src, flags=re.S))
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.12 truth audit 确认：V34.11 quality/visual JSON 在当前 repo 存在；quality probe 使用 future target 做逐 token oracle best 上界，且 teacher source 名称与 final decision 的 best_measurement_bank 不一致。",
        "quality_probe_json_missing": quality_missing,
        "visualization_json_missing": vis_missing,
        "final_decision_depends_on_missing_quality_json": bool((dec.get("semantic_measurement_quality_passed") is not None or dec.get("measurement_beats_random") is not None) and quality_missing),
        "quality_probe_uses_oracle_best_measurement": uses_oracle_best,
        "quality_probe_may_overestimate_measurement_quality": uses_oracle_best,
        "measurement_teacher_name": teacher_name,
        "final_decision_best_measurement_bank": best_bank,
        "measurement_teacher_name_inconsistent": inconsistent,
        "local_probe_is_oracle_masked_residual": local_oracle,
        "model_forward_gate_zero_by_default": gate_zero,
        "semantic_usage_score_only_indirectly_used": usage_indirect,
        "quality_probe_passed": quality.get("semantic_measurement_quality_passed"),
        "local_probe_passed": local_dec.get("local_semantic_usage_probe_passed"),
        "exact_code_locations": {
            "quality_oracle_best": locations(QUALITY_SRC, ["best_cos", "np.maximum.reduce", "fut_teacher_embedding"]),
            "local_oracle_mask_compose": locations(LOCAL_TRAIN_SRC, ["causal_assignment_residual_semantic_mask", "local_compose", "semantic_measurement_usage_score"]),
            "model_gate_zero": locations(MODEL_SRC, ["gate = torch.zeros_like", "final_sem = F.normalize", "semantic_measurement_usage_score"]),
            "local_eval": locations(LOCAL_EVAL_SRC, ["local_compose", "semantic_measurements_load_bearing_on_residual", "zero_semantic_measurements_metric_delta"]),
        },
        "docs_checked": [str(QUALITY_DOC.relative_to(ROOT)), str(VIS_DOC.relative_to(ROOT))],
        "recommended_fix": "先用 non-oracle selector 重估 measurement quality；再实现 raw temporal semantic evidence encoder，避免 pooled-vector repeat 和 oracle best 过乐观。",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.12 V34.11 measurement truth audit 中文报告",
        payload,
        [
            "中文结论",
            "quality_probe_json_missing",
            "visualization_json_missing",
            "final_decision_depends_on_missing_quality_json",
            "quality_probe_uses_oracle_best_measurement",
            "measurement_teacher_name_inconsistent",
            "local_probe_is_oracle_masked_residual",
            "model_forward_gate_zero_by_default",
            "semantic_usage_score_only_indirectly_used",
            "recommended_fix",
        ],
    )
    print(f"已写出 V34.12 measurement truth audit: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
