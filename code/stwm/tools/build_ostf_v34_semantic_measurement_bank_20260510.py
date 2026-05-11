#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


FEATURE_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_14_teacher_features/pointodyssey/dinov2_base/point_local_crop"
IDENTITY_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128/semantic_identity_targets/pointodyssey"
OUT = ROOT / "outputs/cache/stwm_ostf_v34_semantic_measurement_bank/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_semantic_measurement_bank_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V34_SEMANTIC_MEASUREMENT_BANK_20260510.md"


def main() -> int:
    rows: dict[str, Any] = {}
    blockers: list[str] = []
    if not FEATURE_ROOT.exists():
        blockers.append(f"missing teacher feature root: {FEATURE_ROOT.relative_to(ROOT)}")
    for split in ("train", "val", "test"):
        (OUT / split).mkdir(parents=True, exist_ok=True)
        files = sorted((FEATURE_ROOT / split).glob("*.npz")) if FEATURE_ROOT.exists() else []
        obs_cov = []
        conf_vals = []
        agree_vals = []
        built = 0
        for fp in files:
            z = np.load(fp, allow_pickle=True)
            uid = str(np.asarray(z["sample_uid"]).item())
            idp = IDENTITY_ROOT / split / f"{uid}.npz"
            if not idp.exists():
                continue
            sid = np.load(idp, allow_pickle=True)
            obs = np.asarray(z["obs_teacher_embedding"], dtype=np.float32)
            obs_mask = np.asarray(z["obs_teacher_available_mask"]).astype(bool)
            conf = np.asarray(z["crop_confidence_obs"], dtype=np.float32) if "crop_confidence_obs" in z.files else obs_mask.astype(np.float32)
            # Agreement is temporal cosine consistency to the per-point observed mean.
            norm = obs / np.maximum(np.linalg.norm(obs, axis=-1, keepdims=True), 1e-6)
            mean = (norm * obs_mask[..., None]).sum(axis=1) / np.maximum(obs_mask.sum(axis=1, keepdims=True), 1)
            mean = mean / np.maximum(np.linalg.norm(mean, axis=-1, keepdims=True), 1e-6)
            agree = (norm * mean[:, None, :]).sum(axis=-1)
            agree = np.where(obs_mask, agree, 0.0).astype(np.float32)
            if obs_mask.any():
                obs_cov.append(float(obs_mask.mean()))
                conf_vals.extend(conf[obs_mask].astype(float).tolist())
                agree_vals.extend(agree[obs_mask].astype(float).tolist())
            np.savez_compressed(
                OUT / split / f"{uid}.npz",
                sample_uid=uid,
                split=split,
                point_id=np.asarray(z["point_id"], dtype=np.int64),
                point_to_instance_id=np.asarray(z["point_to_instance_id"], dtype=np.int64),
                obs_points=np.zeros((obs.shape[0], obs.shape[1], 2), dtype=np.float32),
                obs_vis=obs_mask.astype(bool),
                obs_semantic_measurements=obs.astype(np.float32),
                obs_semantic_measurement_mask=obs_mask.astype(bool),
                obs_measurement_teacher_name=str(np.asarray(z["teacher_name"]).item()),
                obs_measurement_confidence=conf.astype(np.float32),
                instance_observed_semantic_measurement=np.asarray(z["obs_instance_teacher_embedding"], dtype=np.float32),
                teacher_agreement_score=agree.astype(np.float32),
                fut_teacher_embedding=np.asarray(z["fut_teacher_embedding"], dtype=np.float32),
                fut_teacher_available_mask=np.asarray(z["fut_teacher_available_mask"]).astype(bool),
                fut_teacher_confidence=np.asarray(z["crop_confidence_fut"], dtype=np.float32) if "crop_confidence_fut" in z.files else np.asarray(z["fut_teacher_available_mask"]).astype(np.float32),
                future_teacher_embeddings_supervision_only=True,
                future_teacher_embeddings_input_allowed=False,
                leakage_safe=True,
                identity_sidecar=str(idp.relative_to(ROOT)),
                source_teacher_feature=str(fp.relative_to(ROOT)),
            )
            built += 1
        rows[split] = {
            "sample_count": built,
            "measurement_coverage": float(np.mean(obs_cov)) if obs_cov else 0.0,
            "measurement_confidence_mean": float(np.mean(conf_vals)) if conf_vals else None,
            "measurement_confidence_p10": float(np.percentile(conf_vals, 10)) if conf_vals else None,
            "teacher_agreement_mean": float(np.mean(agree_vals)) if agree_vals else None,
            "teacher_agreement_p10": float(np.percentile(agree_vals, 10)) if agree_vals else None,
        }
    payload = {
        "generated_at_utc": utc_now(),
        "semantic_measurement_bank_built": bool(not blockers and sum(r["sample_count"] for r in rows.values()) > 0),
        "output_root": str(OUT.relative_to(ROOT)),
        "measurement_coverage_by_split": {k: v["measurement_coverage"] for k, v in rows.items()},
        "teacher_agreement_stats": {k: {"mean": v["teacher_agreement_mean"], "p10": v["teacher_agreement_p10"]} for k, v in rows.items()},
        "measurement_confidence_stats": {k: {"mean": v["measurement_confidence_mean"], "p10": v["measurement_confidence_p10"]} for k, v in rows.items()},
        "rows": rows,
        "leakage_safe": True,
        "future_teacher_embeddings_input_allowed": False,
        "exact_blockers": blockers,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V34 Semantic Measurement Bank", payload, ["semantic_measurement_bank_built", "output_root", "measurement_coverage_by_split", "teacher_agreement_stats", "measurement_confidence_stats", "leakage_safe", "exact_blockers"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
