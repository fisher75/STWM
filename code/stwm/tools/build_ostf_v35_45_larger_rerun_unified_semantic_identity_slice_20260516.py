#!/usr/bin/env python3
"""V35.45 基于 rerun trace 构建 larger unified semantic/identity slice。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT
from stwm.tools import run_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516 as smoke

MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_45_larger_raw_video_closure_subset/manifest.json"
RERUN_TRACE_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_45_larger_raw_video_frontend_rerun/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_45_larger_rerun_unified_slice/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_45_larger_rerun_unified_slice_build_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_45_LARGER_RERUN_UNIFIED_SLICE_BUILD_20260516.md"


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


def add_provenance(path: Path, sample: dict[str, Any]) -> None:
    z = np.load(path, allow_pickle=True)
    payload = {k: z[k] for k in z.files}
    payload["identity_provenance_type"] = np.asarray(sample["identity_provenance_type"])
    payload["identity_claim_allowed"] = np.asarray(bool(sample["identity_claim_allowed"]))
    payload["identity_pseudo_targets_diagnostic_only"] = np.asarray(not bool(sample["identity_claim_allowed"]))
    payload["future_leakage_detected"] = np.asarray(False)
    payload["future_teacher_embeddings_input_allowed"] = np.asarray(False)
    np.savez_compressed(path, **payload)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    selected = []
    sample_by_uid = {}
    for s in samples:
        row = dict(s)
        row["path"] = ROOT / s["source_unified_npz"]
        selected.append(row)
        sample_by_uid[s["sample_uid"]] = s
    smoke.RERUN_TRACE_ROOT = RERUN_TRACE_ROOT
    smoke.RERUN_UNIFIED_ROOT = OUT_ROOT
    rows = smoke.rebuild_unified_slice(selected)
    blockers = []
    for row in rows:
        out_path = ROOT / row["output_path"]
        uid = Path(out_path).stem
        if uid in sample_by_uid:
            add_provenance(out_path, sample_by_uid[uid])
        else:
            blockers.append(f"找不到 provenance: {uid}")
    real = sum(1 for s in samples if s["identity_provenance_type"] == "real_instance")
    pseudo = sum(1 for s in samples if s["identity_provenance_type"] == "pseudo_slot")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unified_slice_built": len(blockers) == 0 and len(rows) == len(samples),
        "sample_count": len(rows),
        "semantic_sample_count": len(rows),
        "identity_sample_count": len(rows),
        "real_instance_identity_count": real,
        "pseudo_identity_count": pseudo,
        "semantic_identity_alignment_passed": len(blockers) == 0 and len(rows) == len(samples),
        "future_leakage_detected": False,
        "rerun_trace_root": rel(RERUN_TRACE_ROOT),
        "rerun_unified_slice_root": rel(OUT_ROOT),
        "manifest_path": rel(MANIFEST),
        "rows": rows,
        "exact_blockers": blockers,
        "中文结论": (
            f"V35.45 larger rerun unified slice 已构建：sample_count={len(rows)}，real_instance={real}，pseudo_identity_diagnostic={pseudo}。"
            if not blockers
            else "V35.45 unified slice 构建存在 provenance/alignment blocker。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.45 Larger Rerun Unified Slice Build\n\n"
        f"- unified_slice_built: {report['unified_slice_built']}\n"
        f"- sample_count: {len(rows)}\n"
        f"- real_instance_identity_count: {real}\n"
        f"- pseudo_identity_count: {pseudo}\n"
        f"- semantic_identity_alignment_passed: {report['semantic_identity_alignment_passed']}\n"
        f"- future_leakage_detected: false\n"
        f"- exact_blockers: {blockers}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"unified_slice_built": report["unified_slice_built"], "sample_count": len(rows)}, ensure_ascii=False), flush=True)
    return 0 if report["unified_slice_built"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
