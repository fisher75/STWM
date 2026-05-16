#!/usr/bin/env python3
"""V35.53: 从 V35.52 package manifest 做独立 dry-run 复验。"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

MANIFEST = ROOT / "reports/stwm_ostf_v35_52_reproducibility_package_manifest_20260516.json"
CARD = ROOT / "reports/stwm_ostf_v35_52_benchmark_card_20260516.json"
V35_52 = ROOT / "reports/stwm_ostf_v35_52_reproducibility_package_and_benchmark_card_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v35_53_reproducibility_dry_run_from_manifest_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_53_REPRODUCIBILITY_DRY_RUN_FROM_MANIFEST_20260516.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_files(path: Path, rel: str) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    if "cache" in rel:
        return sum(1 for p in path.rglob("*.npz") if p.is_file())
    if "figures" in rel:
        return sum(1 for p in path.rglob("*.png") if p.is_file())
    return sum(1 for p in path.rglob("*") if p.is_file())


def main() -> int:
    manifest = load(MANIFEST)
    card = load(CARD)
    v35_52 = load(V35_52)
    entries = list(manifest.get("artifact_entries", []))
    dry_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for entry in entries:
        rel = str(entry.get("path", ""))
        path = ROOT / rel
        exists_now = path.exists()
        count_now = count_files(path, rel)
        sha_now = None if path.is_dir() else sha256_file(path)
        row = {
            "path": rel,
            "group": entry.get("group"),
            "manifest_exists": entry.get("exists"),
            "exists_now": exists_now,
            "manifest_file_count": entry.get("file_count"),
            "file_count_now": count_now,
            "manifest_sha256": entry.get("sha256"),
            "sha256_now": sha_now,
            "passed": bool(
                bool(entry.get("exists")) == exists_now
                and int(entry.get("file_count") or 0) == int(count_now)
                and (entry.get("sha256") in (None, sha_now))
            ),
        }
        if not row["passed"]:
            mismatches.append(row)
        dry_rows.append(row)

    card_consistency = {
        "card_exists": CARD.exists(),
        "input_contract_present": bool(card.get("input_contract")),
        "output_contract_present": bool(card.get("output_contract")),
        "scope_present": bool(card.get("scope")),
        "metrics_present": bool(card.get("metrics")),
        "claim_boundary_present": bool(card.get("safety_and_claim_boundary")),
        "future_teacher_embedding_as_input_false": card.get("input_contract", {}).get("future_teacher_embedding_as_input") is False,
        "pseudo_identity_diagnostic_only": card.get("output_contract", {}).get("pseudo_identity_role") == "VSPW pseudo slot identity diagnostic-only",
        "full_cvpr_scale_claim_allowed_false": card.get("safety_and_claim_boundary", {}).get("full_cvpr_scale_claim_allowed") is False,
    }
    card_passed = all(bool(v) for v in card_consistency.values())
    manifest_counts_passed = bool(
        int(manifest.get("selected_clip_count", 0) or 0) >= 300
        and int(manifest.get("frontend_npz_count", 0) or 0) >= int(manifest.get("selected_clip_count", 0) or 0)
        and int(manifest.get("unified_npz_count", 0) or 0) >= int(manifest.get("selected_clip_count", 0) or 0)
        and int(manifest.get("figure_png_count", 0) or 0) >= 12
    )
    dry_run_passed = bool(not mismatches and card_passed and manifest_counts_passed and v35_52.get("reproducibility_package_ready", False))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.53",
        "source_completed_version": "V35.52",
        "reproducibility_dry_run_done": True,
        "dry_run_passed": dry_run_passed,
        "manifest_path": str(MANIFEST.relative_to(ROOT)),
        "benchmark_card_path": str(CARD.relative_to(ROOT)),
        "artifact_count": len(entries),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "manifest_counts_passed": manifest_counts_passed,
        "card_consistency": card_consistency,
        "card_passed": card_passed,
        "reproducibility_package_ready": bool(v35_52.get("reproducibility_package_ready", False)),
        "m128_h32_full_325_video_system_benchmark_claim_allowed": bool(
            dry_run_passed
            and v35_52.get("m128_h32_full_325_video_system_benchmark_claim_allowed", False)
        ),
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": "prepare_submission_ready_benchmark_bundle_or_external_sanity_review",
        "中文结论": (
            "V35.53 已从 V35.52 package manifest 独立复验 artifact 文件、目录计数、hash 和 benchmark card 字段；"
            "dry-run 通过，说明当前 full 325 M128/H32 raw-video closure 证据链具备可打包复验基础。"
            if dry_run_passed
            else "V35.53 dry-run 未通过，必须先修复缺失 artifact 或 benchmark card 字段。"
        ),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.53 Reproducibility Dry-Run From Manifest\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n\n"
        "## Dry-run 状态\n"
        f"- dry_run_passed: {dry_run_passed}\n"
        f"- artifact_count: {len(entries)}\n"
        f"- mismatch_count: {len(mismatches)}\n"
        f"- manifest_counts_passed: {manifest_counts_passed}\n"
        f"- card_passed: {card_passed}\n"
        f"- m128_h32_full_325_video_system_benchmark_claim_allowed: {report['m128_h32_full_325_video_system_benchmark_claim_allowed']}\n"
        "- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "v35_53_reproducibility_dry_run_done": True,
        "dry_run_passed": dry_run_passed,
        "mismatch_count": len(mismatches),
        "m128_h32_full_325_video_system_benchmark_claim_allowed": report["m128_h32_full_325_video_system_benchmark_claim_allowed"],
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": report["recommended_next_step"],
    }, ensure_ascii=False), flush=True)
    return 0 if dry_run_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
