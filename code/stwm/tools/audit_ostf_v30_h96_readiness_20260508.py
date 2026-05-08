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


REPORT = ROOT / "reports/stwm_ostf_v30_h96_readiness_audit_20260508.json"
DOC = ROOT / "docs/STWM_OSTF_V30_H96_READINESS_AUDIT_20260508.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(name: str) -> dict[str, Any]:
    return read_json(ROOT / "manifests/ostf_v30_external_gt" / f"{name}.json")


def split_entries(split: str, horizon: int = 96, m_points: int = 128) -> list[dict[str, Any]]:
    payload = load_manifest(split)
    return [e for e in payload.get("entries", []) if int(e.get("H", -1)) == horizon and int(e.get("M", -1)) == m_points]


def video_key(entry: dict[str, Any]) -> str:
    uid = str(entry.get("video_uid") or Path(str(entry.get("cache_path", ""))).stem)
    for token in ("pointodyssey_train_", "pointodyssey_val_", "pointodyssey_test_"):
        uid = uid.replace(token, "pointodyssey_")
    return uid.split("_M", 1)[0]


def strongest_prior_h96() -> tuple[str, float | None]:
    prior = read_json(ROOT / "reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json")
    best_name = "last_observed_copy"
    best_score = None
    for name, payload in prior.get("splits", {}).get("val", {}).items():
        if name == "oracle_best_prior":
            continue
        score = payload.get("by_horizon", {}).get("H96", {}).get("minFDE")
        if score is None:
            continue
        if best_score is None or float(score) < best_score:
            best_name = name
            best_score = float(score)
    return best_name, best_score


def missrate_saturation() -> dict[str, Any]:
    prior = read_json(ROOT / "reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json")
    rows_by_prior = prior.get("test_item_rows_by_prior", {})
    out: dict[str, Any] = {}
    for name in ("last_observed_copy", "last_visible_copy", "visibility_aware_damped", "visibility_aware_cv", "fixed_affine"):
        rows = [r for r in rows_by_prior.get(name, []) if int(r.get("H", 0)) == 96 and int(r.get("M", 0)) == 128]
        motion = [r for r in rows if r.get("v30_motion")]
        payload: dict[str, Any] = {"item_count": len(rows), "motion_item_count": len(motion)}
        for metric in ("MissRate@32", "MissRate@64", "MissRate@128", "threshold_auc_endpoint_16_32_64_128"):
            vals = [float(r[metric]) for r in motion if r.get(metric) is not None]
            payload[f"motion_{metric}"] = sum(vals) / max(len(vals), 1) if vals else None
        out[name] = payload
    lv = out.get("last_visible_copy", {})
    miss32 = lv.get("motion_MissRate@32")
    out["missrate32_saturated"] = bool(miss32 in (0.0, 1.0))
    out["threshold_auc_endpoint_16_32_64_128_required"] = True
    return out


def main() -> int:
    missing: list[str] = []
    counts: dict[str, int] = {}
    datasets: dict[str, dict[str, int]] = {}
    split_video_keys: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        try:
            entries = split_entries(split)
            # Force a minimal dataset construction so shape/path bugs surface early.
            _ = OSTFExternalGTDataset(split, horizon=96, m_points=128, max_items=1)
        except Exception as exc:
            entries = []
            missing.append(f"{split}: {type(exc).__name__}: {exc}")
        counts[split] = len(entries)
        ds_counts: dict[str, int] = {}
        split_video_keys[split] = set()
        for entry in entries:
            ds_counts[str(entry.get("dataset", "unknown"))] = ds_counts.get(str(entry.get("dataset", "unknown")), 0) + 1
            split_video_keys[split].add(video_key(entry))
        datasets[split] = ds_counts

    motion_manifest = load_manifest("test_h96_motion")
    motion_entries = [e for e in motion_manifest.get("entries", []) if int(e.get("H", -1)) == 96 and int(e.get("M", -1)) == 128]
    strongest_name, strongest_score = strongest_prior_h96()
    overlap = {
        "train_val": len(split_video_keys.get("train", set()) & split_video_keys.get("val", set())),
        "train_test": len(split_video_keys.get("train", set()) & split_video_keys.get("test", set())),
        "val_test": len(split_video_keys.get("val", set()) & split_video_keys.get("test", set())),
    }
    h96_ready = bool(counts.get("train", 0) > 0 and counts.get("val", 0) > 0 and counts.get("test", 0) > 0 and not missing and max(overlap.values()) == 0)
    payload = {
        "audit_name": "stwm_ostf_v30_h96_readiness_audit",
        "generated_at_utc": utc_now(),
        "h96_m128_ready": h96_ready,
        "test_h96_motion_item_count": len(motion_entries),
        "split_counts_H96_M128": counts,
        "per_split_dataset_counts": datasets,
        "strongest_prior_H96": strongest_name,
        "strongest_prior_H96_val_minFDE": strongest_score,
        "missrate_saturation": missrate_saturation(),
        "threshold_auc_endpoint_16_32_64_128_required": True,
        "train_val_test_video_level_leakage_check": {"overlap_counts": overlap, "passed": max(overlap.values()) == 0},
        "expected_gpu_memory": {"batch_size_recommended": 4, "amp": True, "notes": "H96 uses same V30 M128 architecture with longer decoder horizon."},
        "shape_bug_risk": "low: OSTFExternalGTWorldModelV30 is horizon-parametric and dataset construction succeeded" if h96_ready else "blocked",
        "exact_blocker": "; ".join(missing) if missing else None,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V30 H96 Readiness Audit",
        payload,
        [
            "h96_m128_ready",
            "test_h96_motion_item_count",
            "split_counts_H96_M128",
            "strongest_prior_H96",
            "missrate_saturation",
            "train_val_test_video_level_leakage_check",
            "exact_blocker",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
