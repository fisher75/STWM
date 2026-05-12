#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


FILES = {
    "v34_9_bank_builder": ROOT / "code/stwm/tools/build_ostf_v34_9_trace_preserving_semantic_measurement_bank_20260512.py",
    "v34_9_target_builder": ROOT / "code/stwm/tools/build_ostf_v34_9_causal_assignment_residual_targets_20260512.py",
    "v34_9_train": ROOT / "code/stwm/tools/train_ostf_v34_9_trace_fixed_causal_assignment_oracle_residual_probe_20260512.py",
    "v34_9_eval": ROOT / "code/stwm/tools/eval_ostf_v34_9_trace_fixed_causal_assignment_oracle_residual_probe_20260512.py",
    "v34_8_train": ROOT / "code/stwm/tools/train_ostf_v34_8_causal_assignment_oracle_residual_probe_20260512.py",
    "v34_8_eval": ROOT / "code/stwm/tools/eval_ostf_v34_8_causal_assignment_oracle_residual_probe_20260512.py",
    "model": ROOT / "code/stwm/modules/ostf_v34_8_causal_assignment_bound_residual_memory.py",
}
BANK_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey"
BANK_REPORT = ROOT / "reports/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank_20260512.json"
TARGET_REPORT = ROOT / "reports/stwm_ostf_v34_9_causal_assignment_residual_target_build_20260512.json"
TRAIN_REPORT = ROOT / "reports/stwm_ostf_v34_9_trace_fixed_oracle_residual_probe_train_summary_20260512.json"
DECISION = ROOT / "reports/stwm_ostf_v34_9_decision_20260512.json"
OUT = ROOT / "reports/stwm_ostf_v34_10_v34_9_trace_contract_and_semantic_path_audit_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_10_V34_9_TRACE_CONTRACT_AND_SEMANTIC_PATH_AUDIT_20260512.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def bank_real_fields() -> dict[str, bool]:
    report = load(BANK_REPORT)
    builder_src = read(FILES["v34_9_bank_builder"])
    flags = {"points": [], "vis": [], "conf": []}
    for split in ("train", "val", "test"):
        for fp in sorted((BANK_ROOT / split).glob("*.npz"))[:8]:
            z = np.load(fp, allow_pickle=True)
            pts = np.asarray(z["obs_points"], dtype=np.float32)
            vis = np.asarray(z["obs_vis"]).astype(bool)
            mask = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
            conf = np.asarray(z["obs_conf"], dtype=np.float32)
            sem_conf = np.asarray(z["obs_measurement_confidence"], dtype=np.float32)
            flags["points"].append(float(np.mean(np.abs(pts) < 1e-8)) < 0.999)
            flags["vis"].append(not np.array_equal(vis, mask) or bool(np.any(vis != mask)))
            flags["conf"].append(not np.allclose(conf, sem_conf, atol=1e-6))
    value_based = {k: bool(v and all(v)) for k, v in flags.items()}
    source_based_vis = "obs_vis=np.asarray(tr[\"obs_vis\"])" in builder_src or str(report.get("obs_vis_source", "")).startswith("V30 external-GT")
    source_based_conf = "obs_conf=np.asarray(tr[\"obs_conf\"])" in builder_src or str(report.get("obs_conf_source", "")).startswith("V30 external-GT")
    value_based["vis"] = bool(source_based_vis)
    value_based["conf"] = bool(source_based_conf)
    return value_based


def loss_active_stats() -> dict[str, Any]:
    tr = load(TRAIN_REPORT)
    trace = tr.get("loss_trace") or []
    out: dict[str, Any] = {}
    for key in ["semantic_measurement_usage_loss", "assignment_contrastive_loss"]:
        vals = [float(x.get(key, 0.0)) for x in trace if x.get(key) is not None]
        out[key] = {
            "first": vals[0] if vals else None,
            "last": vals[-1] if vals else None,
            "mean": float(np.mean(vals)) if vals else None,
            "inactive": bool(vals and max(abs(v) for v in vals) < 1e-9),
        }
    return out


def main() -> int:
    real = bank_real_fields()
    train_src = read(FILES["v34_8_train"]) + "\n" + read(FILES["v34_9_train"])
    substitution = "obs_conf = np.asarray(zm[\"obs_measurement_confidence\"" in train_src
    losses = loss_active_stats()
    train_uses_real = bool(real["conf"] and not substitution)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.9 bank 已保存真实 trace state，但 V34.9 probe 仍复用 V34.8 dataset，实际训练把 semantic confidence 当成 obs_conf，且 usage/assignment contrast loss 未激活。",
        "measurement_bank_obs_points_real": real["points"],
        "measurement_bank_obs_vis_real": real["vis"],
        "measurement_bank_obs_conf_real": real["conf"],
        "train_dataset_uses_real_obs_conf": train_uses_real,
        "obs_conf_semantic_confidence_substitution_detected": substitution,
        "trace_state_contract_fully_passed": bool(real["points"] and real["vis"] and real["conf"] and train_uses_real),
        "semantic_usage_loss_inactive": bool(losses["semantic_measurement_usage_loss"]["inactive"]),
        "assignment_contrast_loss_inactive": bool(losses["assignment_contrastive_loss"]["inactive"]),
        "semantic_measurement_usage_loss_stats": losses["semantic_measurement_usage_loss"],
        "assignment_contrastive_loss_stats": losses["assignment_contrastive_loss"],
        "v34_9_measurement_report_json_missing": not BANK_REPORT.exists(),
        "v34_9_target_report_json_missing": not TARGET_REPORT.exists(),
        "exact_code_locations": {
            "real_obs_conf_saved": "code/stwm/tools/build_ostf_v34_9_trace_preserving_semantic_measurement_bank_20260512.py: obs_conf=np.asarray(tr['obs_conf'])",
            "bad_dataset_substitution": "code/stwm/tools/train_ostf_v34_8_causal_assignment_oracle_residual_probe_20260512.py: obs_conf = np.asarray(zm['obs_measurement_confidence'])",
            "v34_9_reuses_v34_8_dataset": "code/stwm/tools/train_ostf_v34_9_trace_fixed_causal_assignment_oracle_residual_probe_20260512.py imports train_ostf_v34_8... as v348",
            "normal_cos_detached": "code/stwm/tools/train_ostf_v34_8_causal_assignment_oracle_residual_probe_20260512.py: normal_cos = ...detach()",
        },
        "checked_files": {k: str(v.relative_to(ROOT)) for k, v in FILES.items()},
        "recommended_fix": "新增 V34.10 dataset，明确读取 zm['obs_conf'] 为 trace_obs_conf；保留 semantic_measurement_confidence 独立字段；usage/assignment contrast loss 只 detach 对照分支，不 detach normal path。",
    }
    dump_json(OUT, payload)
    write_doc(DOC, "V34.10 对 V34.9 trace contract 与 semantic path 的中文审计", payload, ["中文结论", "measurement_bank_obs_points_real", "measurement_bank_obs_vis_real", "measurement_bank_obs_conf_real", "train_dataset_uses_real_obs_conf", "obs_conf_semantic_confidence_substitution_detected", "trace_state_contract_fully_passed", "semantic_usage_loss_inactive", "assignment_contrast_loss_inactive", "v34_9_measurement_report_json_missing", "v34_9_target_report_json_missing", "recommended_fix"])
    print(f"已写出 V34.10 trace contract 二次审计: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
