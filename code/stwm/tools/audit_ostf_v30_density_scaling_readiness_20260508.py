#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.datasets.ostf_v30_external_gt_dataset_20260508 import OSTFExternalGTDataset
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v30_density_scaling_readiness_audit_20260508.json"
DOC = ROOT / "docs/STWM_OSTF_V30_DENSITY_SCALING_READINESS_AUDIT_20260508.md"
RUN_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"
HORIZONS = [32, 64, 96]
M_VALUES = [128, 512, 1024]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_entries(split: str, h: int, m: int) -> list[dict[str, Any]]:
    payload = read_json(ROOT / "manifests/ostf_v30_external_gt" / f"{split}.json")
    return [e for e in payload.get("entries", []) if int(e.get("H", -1)) == h and int(e.get("M", -1)) == m]


def run_payload(name: str) -> dict[str, Any] | None:
    path = RUN_DIR / f"{name}.json"
    if not path.exists():
        return None
    payload = read_json(path)
    payload["_report_path"] = str(path.relative_to(ROOT))
    return payload


def metric_slice(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    allm = payload.get("test_metrics", {}).get("all", {})
    motion = payload.get("test_metrics", {}).get("subsets", {}).get("motion", {})
    return {
        "completed": bool(payload.get("completed")),
        "report_path": payload.get("_report_path"),
        "steps": payload.get("steps"),
        "batch_size": payload.get("batch_size"),
        "eval_interval": payload.get("eval_interval"),
        "grad_accum_steps": payload.get("grad_accum_steps"),
        "effective_batch_size": payload.get("effective_batch_size"),
        "train_loss_decreased": payload.get("train_loss_decreased"),
        "duration_seconds": payload.get("duration_seconds"),
        "minFDE_K": allm.get("minFDE_K"),
        "motion_minFDE_K": motion.get("minFDE_K"),
        "threshold_auc_endpoint_16_32_64_128": allm.get("threshold_auc_endpoint_16_32_64_128"),
        "motion_threshold_auc_endpoint_16_32_64_128": motion.get("threshold_auc_endpoint_16_32_64_128"),
        "MissRate@64": allm.get("MissRate@64"),
        "MissRate@128": allm.get("MissRate@128"),
        "visibility_F1": allm.get("visibility_F1"),
        "relative_deformation_layout_error": allm.get("relative_deformation_layout_error"),
    }


def compare(m512: dict[str, Any] | None, m128: dict[str, Any] | None) -> dict[str, Any]:
    a = metric_slice(m512)
    b = metric_slice(m128)
    out = {"m512": a, "m128": b}
    if a and b:
        for key in ("minFDE_K", "motion_minFDE_K", "threshold_auc_endpoint_16_32_64_128", "MissRate@64", "MissRate@128", "visibility_F1", "relative_deformation_layout_error"):
            av = a.get(key)
            bv = b.get(key)
            if av is not None and bv is not None:
                out[f"m512_minus_m128_{key}"] = float(av) - float(bv)
        out["m512_beats_m128_minFDE_K"] = bool((a.get("minFDE_K") is not None and b.get("minFDE_K") is not None) and float(a["minFDE_K"]) < float(b["minFDE_K"]))
        out["same_protocol"] = bool(a.get("steps") == b.get("steps") and a.get("eval_interval") == b.get("eval_interval") and a.get("batch_size") == b.get("batch_size"))
    else:
        out["same_protocol"] = False
    return out


def main() -> int:
    counts: dict[str, Any] = {}
    blockers = []
    for h in HORIZONS:
        for m in M_VALUES:
            key = f"M{m}_H{h}"
            counts[key] = {}
            for split in ("train", "val", "test"):
                try:
                    entries = manifest_entries(split, h, m)
                    _ = OSTFExternalGTDataset(split, horizon=h, m_points=m, max_items=1)
                    counts[key][split] = {
                        "item_count": len(entries),
                        "motion_count": sum("motion" in e.get("v30_subset_tags", []) for e in entries),
                        "cache_path_exists_sample": bool(entries and (ROOT / entries[0]["cache_path"]).exists()),
                    }
                except Exception as exc:
                    counts[key][split] = {"item_count": 0, "motion_count": 0, "error": f"{type(exc).__name__}: {exc}"}
                    blockers.append(f"{key}/{split}: {type(exc).__name__}: {exc}")

    existing: dict[str, Any] = {}
    protocol_notes: dict[str, Any] = {}
    for h in HORIZONS:
        m512 = run_payload(f"v30_extgt_m512_h{h}_seed42")
        m128 = run_payload(f"v30_extgt_m128_h{h}_seed42")
        existing[f"H{h}"] = compare(m512, m128)
        a = existing[f"H{h}"].get("m512")
        protocol_notes[f"H{h}"] = {
            "existing_m512_report_found": m512 is not None,
            "existing_m512_same_protocol_as_m128_seed42": existing[f"H{h}"].get("same_protocol"),
            "rerun_seed42_recommended": bool(m512 is None or not existing[f"H{h}"].get("same_protocol")),
            "reason": "missing or not 4000-step/current batch/eval protocol" if m512 is None or not existing[f"H{h}"].get("same_protocol") else "existing report compatible",
            "existing_m512_steps": a.get("steps") if a else None,
            "existing_m512_batch_size": a.get("batch_size") if a else None,
        }

    all_m1024_ready = all(counts[f"M1024_H{h}"][s].get("item_count", 0) > 0 for h in HORIZONS for s in ("train", "val", "test"))
    payload = {
        "audit_name": "stwm_ostf_v30_density_scaling_readiness_audit",
        "generated_at_utc": utc_now(),
        "cache_item_counts_by_M_H_split": counts,
        "existing_m512_seed42_vs_m128_seed42": existing,
        "existing_m512_protocol_audit": protocol_notes,
        "m1024_memory_batch_feasibility_estimate": {
            "M1024_cache_ready": all_m1024_ready,
            "recommended_batch_size": {"H32": 1, "H64": 1, "H96": 1},
            "recommended_grad_accum_steps": {"H32": 8, "H64": 8, "H96": 8},
            "expected_risk": "moderate: point encoder is per-point MLP but hypotheses tensor scales linearly with M*H*modes",
        },
        "possible_m512_failure_modes_to_monitor": [
            "smaller effective batch",
            "insufficient steps",
            "GPU memory constraints",
            "point encoder mean-pooling bottleneck",
            "metric insensitivity to density",
        ],
        "density_scaling_ready": bool(not blockers and all_m1024_ready),
        "exact_blocker": "; ".join(blockers) if blockers else None,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V30 Density Scaling Readiness Audit",
        payload,
        [
            "density_scaling_ready",
            "cache_item_counts_by_M_H_split",
            "existing_m512_protocol_audit",
            "existing_m512_seed42_vs_m128_seed42",
            "m1024_memory_batch_feasibility_estimate",
            "exact_blocker",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
