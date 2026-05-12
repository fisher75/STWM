#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


OLD_BANK = ROOT / "outputs/cache/stwm_ostf_v34_semantic_measurement_bank/pointodyssey"
IDENTITY_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128/semantic_identity_targets/pointodyssey"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_9_TRACE_PRESERVING_SEMANTIC_MEASUREMENT_BANK_20260512.md"


def scalar(x: np.ndarray) -> str:
    return str(np.asarray(x).item())


def source_npz_for(split: str, uid: str) -> Path | None:
    idp = IDENTITY_ROOT / split / f"{uid}.npz"
    if not idp.exists():
        return None
    sid = np.load(idp, allow_pickle=True)
    rel = scalar(sid["source_npz"])
    p = ROOT / rel
    return p if p.exists() else None


def stats(vals: list[float]) -> dict[str, float | None]:
    return {"mean": None if not vals else float(np.mean(vals)), "p10": None if not vals else float(np.percentile(vals, 10)), "p90": None if not vals else float(np.percentile(vals, 90))}


def main() -> int:
    rows: dict[str, Any] = {}
    blockers: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        (OUT_ROOT / split).mkdir(parents=True, exist_ok=True)
        built = 0
        obs_cov, trace_cov, zero_ratios, meas_conf, agree_vals = [], [], [], [], []
        split_blockers: list[str] = []
        for oldp in sorted((OLD_BANK / split).glob("*.npz")):
            z = np.load(oldp, allow_pickle=True)
            uid = scalar(z["sample_uid"])
            src = source_npz_for(split, uid)
            if src is None:
                split_blockers.append(f"missing source trace for {uid}")
                continue
            tr = np.load(src, allow_pickle=True)
            obs_points = np.asarray(tr["obs_points"], dtype=np.float32)
            obs_vis = np.asarray(tr["obs_vis"]).astype(bool)
            obs_conf = np.asarray(tr["obs_conf"], dtype=np.float32)
            obs_sem = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
            obs_mask = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
            if obs_points.shape[:2] != obs_sem.shape[:2]:
                split_blockers.append(f"shape mismatch for {uid}")
                continue
            zero_ratio = float(np.mean(np.abs(obs_points) < 1e-8))
            if zero_ratio > 0.999:
                split_blockers.append(f"zero trace points for {uid}")
                continue
            conf = np.asarray(z["obs_measurement_confidence"], dtype=np.float32)
            agree = np.asarray(z["teacher_agreement_score"], dtype=np.float32)
            if obs_mask.any():
                obs_cov.append(float(obs_mask.mean()))
                meas_conf.extend(conf[obs_mask].astype(float).tolist())
                agree_vals.extend(agree[obs_mask].astype(float).tolist())
            trace_cov.append(float(obs_vis.mean()))
            zero_ratios.append(zero_ratio)
            np.savez_compressed(
                OUT_ROOT / split / f"{uid}.npz",
                sample_uid=uid,
                split=split,
                point_id=np.asarray(z["point_id"], dtype=np.int64),
                point_to_instance_id=np.asarray(z["point_to_instance_id"], dtype=np.int64),
                obs_points=obs_points,
                obs_vis=obs_vis,
                obs_conf=obs_conf,
                obs_semantic_measurements=obs_sem,
                obs_semantic_measurement_mask=obs_mask,
                obs_measurement_teacher_name=scalar(z["obs_measurement_teacher_name"]),
                obs_measurement_confidence=conf,
                instance_observed_semantic_measurement=np.asarray(z["instance_observed_semantic_measurement"], dtype=np.float32),
                teacher_agreement_score=agree,
                fut_teacher_embedding=np.asarray(z["fut_teacher_embedding"], dtype=np.float32),
                fut_teacher_available_mask=np.asarray(z["fut_teacher_available_mask"]).astype(bool),
                fut_teacher_confidence=np.asarray(z["fut_teacher_confidence"], dtype=np.float32),
                future_teacher_embeddings_supervision_only=True,
                future_teacher_embeddings_input_allowed=False,
                leakage_safe=True,
                trace_source_npz=str(src.relative_to(ROOT)),
                old_measurement_bank=str(oldp.relative_to(ROOT)),
            )
            built += 1
        rows[split] = {
            "sample_count": built,
            "measurement_coverage": float(np.mean(obs_cov)) if obs_cov else 0.0,
            "trace_coverage": float(np.mean(trace_cov)) if trace_cov else 0.0,
            "obs_points_zero_ratio": float(np.mean(zero_ratios)) if zero_ratios else 1.0,
            "measurement_confidence": stats(meas_conf),
            "teacher_agreement": stats(agree_vals),
        }
        if split_blockers:
            blockers[split] = split_blockers[:20]
    trace_pass = bool(rows and all(v["sample_count"] > 0 and v["obs_points_zero_ratio"] < 0.999 for v in rows.values()))
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.9 measurement bank 已用真实 V30 external-GT trace sidecar 重建 obs_points/obs_vis/obs_conf；teacher feature 仍只作为 observed measurement 和 future supervision。",
        "semantic_measurement_bank_built": trace_pass,
        "trace_preserving_measurement_bank_built": trace_pass,
        "output_root": str(OUT_ROOT.relative_to(ROOT)),
        "measurement_coverage_by_split": {k: v["measurement_coverage"] for k, v in rows.items()},
        "trace_coverage_by_split": {k: v["trace_coverage"] for k, v in rows.items()},
        "obs_points_zero_ratio_by_split": {k: v["obs_points_zero_ratio"] for k, v in rows.items()},
        "obs_vis_source": "V30 external-GT trace obs_vis via V33.8 semantic_identity source_npz",
        "obs_conf_source": "V30 external-GT trace obs_conf via V33.8 semantic_identity source_npz",
        "trace_state_contract_passed": trace_pass,
        "teacher_agreement_stats": {k: v["teacher_agreement"] for k, v in rows.items()},
        "measurement_confidence_stats": {k: v["measurement_confidence"] for k, v in rows.items()},
        "rows": rows,
        "leakage_safe": True,
        "future_teacher_embeddings_input_allowed": False,
        "exact_blockers": blockers,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "V34.9 trace-preserving semantic measurement bank 中文报告", payload, ["中文结论", "trace_preserving_measurement_bank_built", "output_root", "measurement_coverage_by_split", "trace_coverage_by_split", "obs_points_zero_ratio_by_split", "obs_vis_source", "obs_conf_source", "trace_state_contract_passed", "teacher_agreement_stats", "measurement_confidence_stats", "leakage_safe", "exact_blockers"])
    print(f"已写出 V34.9 trace-preserving measurement bank 报告: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
