#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


BANK_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_semantic_measurement_bank/pointodyssey"
BUILDER = ROOT / "code/stwm/tools/build_ostf_v34_semantic_measurement_bank_20260510.py"
MODEL = ROOT / "code/stwm/modules/ostf_v34_8_causal_assignment_bound_residual_memory.py"
TRAIN = ROOT / "code/stwm/tools/train_ostf_v34_8_causal_assignment_oracle_residual_probe_20260512.py"
EVAL = ROOT / "code/stwm/tools/eval_ostf_v34_8_causal_assignment_oracle_residual_probe_20260512.py"
DECISION = ROOT / "reports/stwm_ostf_v34_8_decision_20260512.json"
ORACLE = ROOT / "reports/stwm_ostf_v34_8_causal_assignment_oracle_residual_probe_decision_20260512.json"
AUDIT8 = ROOT / "reports/stwm_ostf_v34_8_v34_7_assignment_causal_path_audit_20260512.json"
BANK_DOC = ROOT / "docs/STWM_OSTF_V34_SEMANTIC_MEASUREMENT_BANK_20260510.md"
REQUIRED_JSON = [
    ROOT / "reports/stwm_ostf_v34_8_causal_assignment_residual_target_build_20260512.json",
    ROOT / "reports/stwm_ostf_v34_8_artifact_rematerialization_20260512.json",
    ROOT / "reports/stwm_ostf_v34_8_causal_assignment_residual_visualization_manifest_20260512.json",
]
OUT = ROOT / "reports/stwm_ostf_v34_9_v34_8_state_contract_audit_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_9_V34_8_STATE_CONTRACT_AUDIT_20260512.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def sample_bank_stats() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    zero_flags = []
    vis_equals_mask = []
    conf_equals_sem_conf = []
    for split in ("train", "val", "test"):
        files = sorted((BANK_ROOT / split).glob("*.npz"))
        split_rows = []
        for fp in files[: min(8, len(files))]:
            z = np.load(fp, allow_pickle=True)
            obs_points = np.asarray(z["obs_points"], dtype=np.float32)
            obs_vis = np.asarray(z["obs_vis"]).astype(bool)
            obs_mask = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
            obs_conf = np.asarray(z.get("obs_conf", z["obs_measurement_confidence"]), dtype=np.float32)
            sem_conf = np.asarray(z["obs_measurement_confidence"], dtype=np.float32)
            zero_ratio = float(np.mean(np.abs(obs_points) < 1e-8))
            same_vis = bool(obs_vis.shape == obs_mask.shape and np.array_equal(obs_vis, obs_mask))
            same_conf = bool(obs_conf.shape == sem_conf.shape and np.allclose(obs_conf, sem_conf, atol=1e-6))
            zero_flags.append(zero_ratio > 0.999)
            vis_equals_mask.append(same_vis)
            conf_equals_sem_conf.append(same_conf)
            split_rows.append({"sample": fp.name, "obs_points_zero_ratio": zero_ratio, "obs_vis_equals_semantic_mask": same_vis, "obs_conf_equals_semantic_confidence": same_conf})
        rows[split] = split_rows
    return {
        "sample_rows": rows,
        "obs_points_zero_detected": bool(zero_flags and all(zero_flags)),
        "obs_vis_from_semantic_mask_detected": bool(vis_equals_mask and all(vis_equals_mask)),
        "obs_conf_from_semantic_confidence_detected": bool(conf_equals_sem_conf and all(conf_equals_sem_conf)),
    }


def loss_inactive(key: str) -> bool:
    train_summary = ROOT / "reports/stwm_ostf_v34_8_causal_assignment_oracle_residual_probe_train_summary_20260512.json"
    payload = load(train_summary)
    trace = payload.get("loss_trace") or []
    vals = [float(row.get(key, 0.0)) for row in trace if row.get(key) is not None]
    return bool(vals and max(abs(v) for v in vals) < 1e-9)


def main() -> int:
    builder_src = read(BUILDER)
    train_src = read(TRAIN)
    eval_src = read(EVAL)
    stats = sample_bank_stats()
    artifact_fixed = all(p.exists() for p in REQUIRED_JSON)
    uses_trace = not stats["obs_points_zero_detected"] and not stats["obs_vis_from_semantic_mask_detected"]
    train_uses_bank_trace = bool(re.search(r"obs_points|obs_vis|obs_conf", train_src) and "semantic_measurement_bank_root" in train_src)
    eval_uses_bank_trace = bool(re.search(r"obs_points|obs_vis|obs_conf", eval_src) and "CausalAssignmentResidualDataset" in eval_src)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.8 使用的 V34 semantic measurement bank 违反 trace contract：obs_points 为零，obs_vis 来自 semantic measurement mask，训练/评估又把这些字段送入 frozen V30。",
        "obs_points_zero_detected": stats["obs_points_zero_detected"],
        "obs_vis_from_semantic_mask_detected": stats["obs_vis_from_semantic_mask_detected"],
        "obs_conf_from_semantic_confidence_detected": stats["obs_conf_from_semantic_confidence_detected"],
        "v34_8_uses_real_trace_input": bool(uses_trace and train_uses_bank_trace and eval_uses_bank_trace),
        "trace_conditioned_contract_broken": bool(not uses_trace),
        "artifact_packaging_truly_fixed": artifact_fixed,
        "missing_artifact_json": [str(p.relative_to(ROOT)) for p in REQUIRED_JSON if not p.exists()],
        "semantic_usage_loss_inactive": loss_inactive("semantic_measurement_usage_loss"),
        "assignment_contrast_loss_inactive": loss_inactive("assignment_contrastive_loss"),
        "bank_sample_stats": stats["sample_rows"],
        "exact_code_locations": {
            "zero_obs_points": "code/stwm/tools/build_ostf_v34_semantic_measurement_bank_20260510.py: obs_points=np.zeros(...)",
            "obs_vis_from_mask": "code/stwm/tools/build_ostf_v34_semantic_measurement_bank_20260510.py: obs_vis=obs_mask.astype(bool)",
            "train_uses_bank_trace": "code/stwm/tools/train_ostf_v34_8_causal_assignment_oracle_residual_probe_20260512.py: CausalAssignmentResidualDataset 读取 measurement bank obs_points/obs_vis/obs_conf",
            "eval_uses_bank_trace": "code/stwm/tools/eval_ostf_v34_8_causal_assignment_oracle_residual_probe_20260512.py: model 输入 obs_points/obs_vis/obs_conf",
        },
        "checked_files": [str(p.relative_to(ROOT)) for p in [BUILDER, MODEL, TRAIN, EVAL, DECISION, ORACLE, AUDIT8, BANK_DOC]],
        "recommended_fix": "重建 trace-preserving semantic measurement bank：obs_points/obs_vis/obs_conf 必须来自真实 V30 external-GT trace sidecar；然后重建 causal targets 并重跑 oracle probe，不训练 learned gate。",
    }
    dump_json(OUT, payload)
    write_doc(DOC, "V34.9 对 V34.8 trace state contract 的中文审计", payload, ["中文结论", "obs_points_zero_detected", "obs_vis_from_semantic_mask_detected", "obs_conf_from_semantic_confidence_detected", "v34_8_uses_real_trace_input", "trace_conditioned_contract_broken", "artifact_packaging_truly_fixed", "semantic_usage_loss_inactive", "assignment_contrast_loss_inactive", "recommended_fix"])
    print(f"已写出 V34.9 state contract 审计报告: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
