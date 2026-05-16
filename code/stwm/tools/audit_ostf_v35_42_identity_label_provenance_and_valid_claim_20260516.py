#!/usr/bin/env python3
"""V35.42 审计 identity label provenance，并在真实 instance subset 上重算 claim gate。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT
from stwm.tools.run_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516 import load_identity_model, load_identity_sample
from stwm.tools.train_eval_ostf_v35_16_video_identity_pairwise_retrieval_head_20260515 import (
    aggregate,
    model_embedding,
    retrieval_metrics_for_sample,
)

V35_38_REPORT = ROOT / "reports/stwm_ostf_v35_38_eval_balanced_raw_video_frontend_rerun_subset_20260516.json"
V35_40_REPORT = ROOT / "reports/stwm_ostf_v35_40_identity_hard_case_failure_modes_20260516.json"
SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_38_eval_balanced_raw_video_frontend_rerun_unified_slice/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_42_identity_label_provenance_and_valid_claim_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_42_IDENTITY_LABEL_PROVENANCE_AND_VALID_CLAIM_20260516.md"
LOG = ROOT / "outputs/logs/stwm_ostf_v35_42_identity_label_provenance_and_valid_claim_20260516.log"
SEEDS = [42, 123, 456]


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


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def pass_identity(m: dict[str, float | None]) -> bool:
    return bool(
        (m.get("identity_retrieval_exclude_same_point_top1") or 0.0) >= 0.65
        and (m.get("identity_retrieval_same_frame_top1") or 0.0) >= 0.65
        and (m.get("identity_retrieval_instance_pooled_top1") or 0.0) >= 0.65
        and (m.get("identity_confuser_avoidance_top1") or 0.0) >= 0.65
    )


def sample_paths(smoke: dict[str, Any], root: Path) -> list[Path]:
    return [root / str(r["split"]) / f"{r['sample_uid']}.npz" for r in smoke.get("selected_samples", [])]


def identity_provenance(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    dataset = str(np.asarray(z["dataset"]).item())
    inst = np.asarray(z["point_to_instance_id"], dtype=np.int64)
    sem = np.asarray(z["source_semantic_id"], dtype=np.int64)
    last = np.asarray(z["obs_semantic_last_id"], dtype=np.int64)
    valid_inst = inst[inst >= 0]
    counts = [int((inst == k).sum()) for k in np.unique(valid_inst)] if valid_inst.size else []
    equal_slot_blocks = bool(counts and len(set(counts)) == 1 and counts[0] == 128)
    sem_unique = int(len(np.unique(sem[sem >= 0]))) if np.any(sem >= 0) else 0
    last_unique = int(len(np.unique(last[last >= 0]))) if np.any(last >= 0) else 0
    real_instance = dataset.upper() == "VIPSEG"
    pseudo_slot = dataset.upper() == "VSPW" or (equal_slot_blocks and sem_unique <= 1 and last_unique <= 1)
    return {
        "sample_uid": str(np.asarray(z["sample_uid"]).item()),
        "split": str(np.asarray(z["split"]).item()),
        "dataset": dataset,
        "instance_count": int(len(np.unique(valid_inst))) if valid_inst.size else 0,
        "semantic_unique_count": sem_unique,
        "obs_last_semantic_unique_count": last_unique,
        "equal_128_point_slot_blocks": equal_slot_blocks,
        "identity_label_provenance": "real_instance" if real_instance else "semantic_or_track_slot_pseudo_identity" if pseudo_slot else "unknown",
        "identity_pairwise_target_valid_for_claim": bool(real_instance),
    }


@torch.no_grad()
def eval_filtered_identity(paths: list[Path], seed: int, device: torch.device) -> dict[str, Any]:
    model = load_identity_model(seed, device)
    rows_by_split: dict[str, list[dict[str, float]]] = defaultdict(list)
    sample_rows = []
    for p in paths:
        s = load_identity_sample(p)
        emb = model_embedding(model, np.asarray(s["x"], dtype=np.float32), device)
        row = retrieval_metrics_for_sample(emb, s)
        rows_by_split[str(s["split"])].append(row)
        sample_rows.append({"sample_uid": str(s["sample_uid"]), "split": str(s["split"]), "metrics": aggregate([row])})
    return {"by_split": {k: aggregate(v) for k, v in rows_by_split.items()}, "by_sample": sample_rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v35-38-report", default=str(V35_38_REPORT))
    ap.add_argument("--v35-40-report", default=str(V35_40_REPORT))
    ap.add_argument("--slice-root", default=str(SLICE_ROOT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    smoke_path = Path(args.v35_38_report)
    hard_path = Path(args.v35_40_report)
    root = Path(args.slice_root)
    if not smoke_path.is_absolute():
        smoke_path = ROOT / smoke_path
    if not hard_path.is_absolute():
        hard_path = ROOT / hard_path
    if not root.is_absolute():
        root = ROOT / root
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu and args.device == "cuda" else "cpu")
    LOG.write_text("", encoding="utf-8")
    log("开始 V35.42 identity label provenance audit。")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    paths = sample_paths(smoke, root)
    provenance_rows = [identity_provenance(p) for p in paths]
    valid_paths = [p for p, r in zip(paths, provenance_rows) if r["identity_pairwise_target_valid_for_claim"]]
    invalid_rows = [r for r in provenance_rows if not r["identity_pairwise_target_valid_for_claim"]]
    filtered_eval = {str(seed): eval_filtered_identity(valid_paths, seed, device) for seed in SEEDS}
    filtered_pass = bool(
        valid_paths
        and all(pass_identity(filtered_eval[str(seed)]["by_split"].get("val", {})) for seed in SEEDS)
        and all(pass_identity(filtered_eval[str(seed)]["by_split"].get("test", {})) for seed in SEEDS)
    )
    semantic_pass = bool(smoke.get("semantic_smoke_passed_all_seeds", False))
    drift_ok = bool(smoke.get("cached_vs_rerun_drift", {}).get("drift_ok", False))
    m128_claim = bool(semantic_pass and drift_ok and filtered_pass)
    recommended = "render_case_mined_visualization_and_write_unified_raw_video_decision" if m128_claim else "fix_identity_targets_or_video_instance_supervision"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_42_identity_label_provenance_audit_done": True,
        "v35_38_report": rel(smoke_path),
        "v35_40_report": rel(hard_path),
        "slice_root": rel(root),
        "identity_label_provenance_rows": provenance_rows,
        "invalid_identity_claim_rows": invalid_rows,
        "vspw_identity_targets_marked_diagnostic_only": any(r["dataset"].upper() == "VSPW" for r in invalid_rows),
        "identity_valid_instance_sample_count": len(valid_paths),
        "identity_invalid_or_pseudo_sample_count": len(invalid_rows),
        "filtered_real_instance_identity_eval": filtered_eval,
        "filtered_real_instance_identity_passed_all_seeds": filtered_pass,
        "semantic_smoke_passed_all_seeds": semantic_pass,
        "raw_frontend_drift_ok": drift_ok,
        "m128_h32_video_system_benchmark_claim_allowed": m128_claim,
        "m128_h32_video_system_benchmark_claim_boundary": "identity claim 仅限真实 instance-labeled subset；VSPW identity 伪 slot 只做诊断，不作为 identity pass gate。",
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": recommended,
        "中文结论": (
            "V35.42 修正了 identity claim boundary：VSPW 在当前目标中更像 semantic/track-slot pseudo identity，不能作为真实 identity retrieval pass gate；"
            "在 VIPSeg 真实 instance subset 上，raw-video rerun 的 semantic 与 identity 闭环可以作为 M128/H32 bounded video system benchmark。"
            if m128_claim
            else "V35.42 发现 identity label provenance 仍不足以支撑 raw-video M128/H32 claim，需要继续修 identity targets 或补真实 instance supervision。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.42 Identity Label Provenance And Valid Claim\n\n"
        f"- v35_42_identity_label_provenance_audit_done: true\n"
        f"- identity_valid_instance_sample_count: {len(valid_paths)}\n"
        f"- identity_invalid_or_pseudo_sample_count: {len(invalid_rows)}\n"
        f"- vspw_identity_targets_marked_diagnostic_only: {report['vspw_identity_targets_marked_diagnostic_only']}\n"
        f"- filtered_real_instance_identity_passed_all_seeds: {filtered_pass}\n"
        f"- semantic_smoke_passed_all_seeds: {semantic_pass}\n"
        f"- raw_frontend_drift_ok: {drift_ok}\n"
        f"- m128_h32_video_system_benchmark_claim_allowed: {m128_claim}\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {recommended}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n\n"
        "## Claim boundary\n"
        "允许的只是 M128/H32 bounded video system benchmark：semantic 可跨 VSPW/VIPSeg，identity 只在真实 instance-labeled subset 上评估；full CVPR-scale claim 仍不允许。\n",
        encoding="utf-8",
    )
    log(f"V35.42 完成；m128_claim={m128_claim} recommended_next_step={recommended}")
    print(json.dumps({"v35_42_identity_label_provenance_audit_done": True, "m128_h32_video_system_benchmark_claim_allowed": m128_claim, "recommended_next_step": recommended}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
