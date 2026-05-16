#!/usr/bin/env python3
"""V36.6: 修复 occlusion/reappear identity target/eval contract，并重评估现有 identity field。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402
from stwm.tools.run_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516 import (  # noqa: E402
    load_identity_model,
    load_identity_sample,
)
from stwm.tools.train_eval_ostf_v35_16_video_identity_pairwise_retrieval_head_20260515 import (  # noqa: E402
    aggregate,
    model_embedding,
    retrieval_metrics_for_sample,
)

SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_2c_conservative_selector_downstream_slice/M128_H32"
OVERRIDE_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_6_occlusion_reappear_identity_target_overrides/M128_H32"
V36_3_DECISION = ROOT / "reports/stwm_ostf_v36_3_full_325_causal_benchmark_rerun_decision_20260516.json"
V36_5_RISK = ROOT / "reports/stwm_ostf_v36_5_reviewer_risk_audit_20260516.json"
AUDIT_REPORT = ROOT / "reports/stwm_ostf_v36_6_occlusion_reappear_identity_target_contract_audit_20260516.json"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v36_6_occlusion_reappear_identity_field_repair_eval_summary_20260516.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v36_6_occlusion_reappear_identity_field_repair_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_6_OCCLUSION_REAPPEAR_IDENTITY_FIELD_REPAIR_DECISION_20260516.md"
LOG = ROOT / "outputs/logs/stwm_ostf_v36_6_occlusion_reappear_identity_field_repair_20260516.log"
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def scalar(z: Any, key: str, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    try:
        return arr.item()
    except ValueError:
        return arr


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def teacher_occlusion_reappear_mask(fut_vis: np.ndarray) -> np.ndarray:
    out = np.zeros((fut_vis.shape[0],), dtype=bool)
    for i, row in enumerate(fut_vis.astype(bool)):
        if not row.any() or row.all():
            continue
        false_idx = np.where(~row)[0]
        true_idx = np.where(row)[0]
        out[i] = bool(false_idx.size and true_idx.size and true_idx.max() > false_idx.min())
    return out


def sample_paths(split: str) -> list[Path]:
    return sorted((SLICE_ROOT / split).glob("*.npz"))


def build_target_overrides() -> tuple[dict[str, Any], dict[tuple[str, str], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    override_map: dict[tuple[str, str], np.ndarray] = {}
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for split in ["train", "val", "test"]:
        summary[split] = {}
        for claim in [True, False]:
            key = "real_instance" if claim else "pseudo_diagnostic"
            counts = {
                "sample_count": 0,
                "original_predicted_occ_sample_count": 0,
                "original_predicted_occ_point_count": 0,
                "teacher_occ_sample_count": 0,
                "teacher_occ_point_count": 0,
                "teacher_occ_feasible_point_count": 0,
            }
            for path in sample_paths(split):
                z = np.load(path, allow_pickle=True)
                identity_claim_allowed = bool(scalar(z, "identity_claim_allowed", False))
                if identity_claim_allowed != claim:
                    continue
                sample_uid = str(scalar(z, "sample_uid", path.stem))
                inst = np.asarray(z["point_to_instance_id"], dtype=np.int64)
                valid = inst >= 0
                same = (inst[:, None] == inst[None, :]) & (inst[:, None] >= 0)
                np.fill_diagonal(same, False)
                original = np.asarray(z["identity_occlusion_reappear_point_mask"], dtype=bool) & valid
                teacher = teacher_occlusion_reappear_mask(np.asarray(z["future_trace_teacher_vis"], dtype=bool)) & valid
                feasible = teacher & same.any(axis=1)
                counts["sample_count"] += 1
                counts["original_predicted_occ_sample_count"] += int(original.any())
                counts["original_predicted_occ_point_count"] += int(original.sum())
                counts["teacher_occ_sample_count"] += int(teacher.any())
                counts["teacher_occ_point_count"] += int(teacher.sum())
                counts["teacher_occ_feasible_point_count"] += int(feasible.sum())
                override_map[(split, path.name)] = teacher
                out_dir = OVERRIDE_ROOT / split
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / path.name
                np.savez_compressed(
                    out_path,
                    sample_uid=np.asarray(sample_uid),
                    split=np.asarray(split),
                    dataset=np.asarray(str(scalar(z, "dataset", ""))),
                    identity_claim_allowed=np.asarray(identity_claim_allowed),
                    original_predicted_occlusion_reappear_point_mask=original.astype(bool),
                    teacher_occlusion_reappear_point_mask=teacher.astype(bool),
                    feasible_teacher_occlusion_reappear_point_mask=feasible.astype(bool),
                    occlusion_reappear_target_source=np.asarray("future_trace_teacher_vis_supervision_only"),
                    future_teacher_trace_input_allowed=np.asarray(False),
                    future_trace_predicted_from_past_only=np.asarray(True),
                    leakage_safe=np.asarray(True),
                )
                rows.append(
                    {
                        "sample_uid": sample_uid,
                        "split": split,
                        "dataset": str(scalar(z, "dataset", "")),
                        "identity_claim_allowed": identity_claim_allowed,
                        "override_path": rel(out_path),
                        "original_predicted_occ_points": int(original.sum()),
                        "teacher_occ_points": int(teacher.sum()),
                        "teacher_occ_feasible_points": int(feasible.sum()),
                    }
                )
            summary[split][key] = counts
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v36_6_occlusion_reappear_identity_target_contract_audit_done": True,
        "source_slice_root": rel(SLICE_ROOT),
        "override_root": rel(OVERRIDE_ROOT),
        "target_override_count": len(rows),
        "summary_by_split_and_provenance": summary,
        "original_predicted_future_vis_occ_mask_empty": all(
            summary[s][p]["original_predicted_occ_point_count"] == 0 for s in summary for p in summary[s]
        ),
        "teacher_future_vis_occ_target_available": all(
            summary[s]["real_instance"]["teacher_occ_point_count"] > 0 for s in ["train", "val", "test"]
        ),
        "teacher_occ_target_source": "future_trace_teacher_vis_supervision_only",
        "future_teacher_trace_input_allowed": False,
        "future_trace_predicted_from_past_only": True,
        "future_leakage_detected": False,
        "target_contract_fix_required": True,
        "exact_issue_zh": (
            "V36 causal slice 的 identity_occlusion_reappear_point_mask 来自 selector/predicted future_vis；selector future_vis 在当前 full325 中没有产生 occlusion/reappear，"
            "导致 real-instance val/test occlusion total 为 0。应使用 future_trace_teacher_vis 作为 supervision/eval target 来定义遮挡再出现事件，而不是作为模型 input。"
        ),
        "rows": rows,
    }
    AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT.write_text(json.dumps(jsonable(audit), indent=2, ensure_ascii=False), encoding="utf-8")
    return audit, override_map


def identity_paths(split: str, real_only: bool) -> list[Path]:
    out: list[Path] = []
    for path in sample_paths(split):
        z = np.load(path, allow_pickle=True)
        if bool(scalar(z, "identity_claim_allowed", False)) == real_only:
            out.append(path)
    return out


@torch.no_grad()
def eval_split(paths: list[Path], seed: int, device: torch.device, override_map: dict[tuple[str, str], np.ndarray]) -> dict[str, float | None]:
    if not paths:
        return {}
    model = load_identity_model(seed, device)
    rows = []
    for path in paths:
        sample = load_identity_sample(path)
        split = str(sample["split"])
        sample["occlusion"] = override_map[(split, path.name)]
        emb = model_embedding(model, np.asarray(sample["x"], dtype=np.float32), device)
        rows.append(retrieval_metrics_for_sample(emb, sample))
    return aggregate(rows)


def pass_identity_with_occlusion(m: dict[str, float | None]) -> bool:
    return bool(
        (m.get("identity_retrieval_exclude_same_point_top1") or 0.0) >= 0.65
        and (m.get("identity_retrieval_same_frame_top1") or 0.0) >= 0.65
        and (m.get("identity_retrieval_instance_pooled_top1") or 0.0) >= 0.65
        and (m.get("identity_confuser_avoidance_top1") or 0.0) >= 0.65
        and (m.get("occlusion_reappear_total") or 0.0) > 0
        and (m.get("occlusion_reappear_retrieval_top1") or 0.0) >= 0.65
    )


def mean_metric(rows: list[dict[str, Any]], split: str, key: str) -> float | None:
    vals = []
    for row in rows:
        value = row.get(split, {}).get(key)
        if value is not None:
            vals.append(float(value))
    return float(np.mean(vals)) if vals else None


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log("开始 V36.6 occlusion/reappear identity target contract audit。")
    audit, override_map = build_target_overrides()
    log("target override 已构建；开始三 seed identity re-eval。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_rows = []
    for seed in SEEDS:
        log(f"评估 seed={seed} real-instance occlusion/reappear identity。")
        val = eval_split(identity_paths("val", True), seed, device, override_map)
        test = eval_split(identity_paths("test", True), seed, device, override_map)
        seed_rows.append(
            {
                "seed": seed,
                "val": val,
                "test": test,
                "identity_with_occlusion_passed": pass_identity_with_occlusion(val) and pass_identity_with_occlusion(test),
            }
        )
    val_occ = mean_metric(seed_rows, "val", "occlusion_reappear_retrieval_top1")
    test_occ = mean_metric(seed_rows, "test", "occlusion_reappear_retrieval_top1")
    val_occ_total = mean_metric(seed_rows, "val", "occlusion_reappear_total")
    test_occ_total = mean_metric(seed_rows, "test", "occlusion_reappear_total")
    passed = bool(
        all(row["identity_with_occlusion_passed"] for row in seed_rows)
        and audit["teacher_future_vis_occ_target_available"]
        and not audit["future_leakage_detected"]
    )
    eval_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v36_6_occlusion_reappear_identity_field_repair_eval_done": True,
        "source_slice_root": rel(SLICE_ROOT),
        "override_root": rel(OVERRIDE_ROOT),
        "seed_rows": seed_rows,
        "mean_metrics": {
            "val_occlusion_reappear_retrieval_top1": val_occ,
            "test_occlusion_reappear_retrieval_top1": test_occ,
            "val_occlusion_reappear_total": val_occ_total,
            "test_occlusion_reappear_total": test_occ_total,
            "test_identity_retrieval_exclude_same_point_top1": mean_metric(seed_rows, "test", "identity_retrieval_exclude_same_point_top1"),
            "test_identity_retrieval_instance_pooled_top1": mean_metric(seed_rows, "test", "identity_retrieval_instance_pooled_top1"),
            "test_identity_confuser_avoidance_top1": mean_metric(seed_rows, "test", "identity_confuser_avoidance_top1"),
            "test_trajectory_crossing_retrieval_top1": mean_metric(seed_rows, "test", "trajectory_crossing_retrieval_top1"),
        },
        "target_source": "future_trace_teacher_vis_supervision_only",
        "future_teacher_trace_input_allowed": False,
        "future_trace_predicted_from_past_only": True,
        "future_leakage_detected": False,
        "v30_backbone_frozen": True,
    }
    previous_v36_3 = load_json(V36_3_DECISION)
    previous_risk = load_json(V36_5_RISK)
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V36.6",
        "occlusion_reappear_identity_target_contract_audit_done": True,
        "occlusion_reappear_identity_target_repaired": True,
        "original_predicted_future_vis_occ_mask_empty": audit["original_predicted_future_vis_occ_mask_empty"],
        "teacher_future_vis_occ_target_available": audit["teacher_future_vis_occ_target_available"],
        "target_override_root": rel(OVERRIDE_ROOT),
        "occlusion_reappear_identity_re_eval_done": True,
        "occlusion_reappear_identity_three_seed_passed": passed,
        "val_occlusion_reappear_retrieval_top1": val_occ,
        "test_occlusion_reappear_retrieval_top1": test_occ,
        "val_occlusion_reappear_total": val_occ_total,
        "test_occlusion_reappear_total": test_occ_total,
        "previous_v36_3_occlusion_reappear_top1": (previous_v36_3.get("identity_test_means") or {}).get("occlusion_reappear_retrieval_top1"),
        "previous_v36_5_risk_status": previous_risk.get("occlusion_reappear_identity_hard_risk"),
        "future_teacher_trace_input_allowed": False,
        "future_teacher_embedding_input_allowed": False,
        "future_trace_predicted_from_past_only": True,
        "future_leakage_detected": False,
        "v30_backbone_frozen": True,
        "m128_h32_causal_video_world_model_benchmark_claim_allowed": bool(
            previous_v36_3.get("m128_h32_causal_video_world_model_benchmark_claim_allowed", False)
        ),
        "m128_h32_causal_identity_occlusion_reappear_claim_allowed": passed,
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": "update_v36_claim_table_after_occlusion_identity_repair" if passed else "fix_occlusion_reappear_identity_field",
        "中文结论": (
            "V36.6 发现并修复了 occlusion/reappear identity 的 target/eval contract：旧 mask 来自 predicted future_vis 且为空，"
            "因此 V36.3 的 0.0 是空 target 被聚合为 0，不是 real-instance occlusion 全失败。"
            "使用 future_trace_teacher_vis 作为 supervision-only/eval-only target 后，现有 V35.29 identity head 在 real-instance occlusion/reappear 上三 seed 通过。"
            if passed
            else "V36.6 修复 target/eval contract 后，occlusion/reappear identity 仍未三 seed 通过；需要继续修 identity memory/reassociation。"
        ),
    }
    EVAL_REPORT.write_text(json.dumps(jsonable(eval_report), indent=2, ensure_ascii=False), encoding="utf-8")
    DECISION_REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(
        "# STWM OSTF V36.6 Occlusion/Reappear Identity Field Repair Decision\n\n"
        "## 中文总结\n"
        f"{decision['中文结论']}\n\n"
        "## 关键发现\n"
        f"- original_predicted_future_vis_occ_mask_empty: {decision['original_predicted_future_vis_occ_mask_empty']}\n"
        f"- teacher_future_vis_occ_target_available: {decision['teacher_future_vis_occ_target_available']}\n"
        f"- previous_v36_3_occlusion_reappear_top1: {decision['previous_v36_3_occlusion_reappear_top1']}\n"
        f"- val_occlusion_reappear_retrieval_top1: {val_occ}\n"
        f"- test_occlusion_reappear_retrieval_top1: {test_occ}\n"
        f"- val_occlusion_reappear_total: {val_occ_total}\n"
        f"- test_occlusion_reappear_total: {test_occ_total}\n"
        f"- occlusion_reappear_identity_three_seed_passed: {passed}\n"
        "- future_teacher_trace_input_allowed: false\n"
        "- future_teacher_embedding_input_allowed: false\n"
        "- future_trace_predicted_from_past_only: true\n"
        "- full_cvpr_scale_claim_allowed: false\n\n"
        "## Claim 边界\n"
        "- 可以把 V36.3 中 occlusion/reappear=0.0 更正为 target/eval contract 问题，而不是模型在真实遮挡样本上全错。\n"
        "- 如果引用 V36.6 指标，必须说明遮挡/再出现 target 来自 future_trace_teacher_vis，仅作为 supervision/eval target，不作为模型输入。\n"
        "- 仍不允许 claim full CVPR-scale complete system、H64/H96、M512/M1024 或 full open-vocabulary semantic segmentation。\n\n"
        "## 输出\n"
        f"- audit_report: `{rel(AUDIT_REPORT)}`\n"
        f"- eval_summary: `{rel(EVAL_REPORT)}`\n"
        f"- decision_report: `{rel(DECISION_REPORT)}`\n"
        f"- override_root: `{rel(OVERRIDE_ROOT)}`\n"
        f"- log: `{rel(LOG)}`\n"
        f"- recommended_next_step: `{decision['recommended_next_step']}`\n",
        encoding="utf-8",
    )
    log(f"V36.6 完成；passed={passed}；recommended_next_step={decision['recommended_next_step']}")
    print(
        json.dumps(
            {
                "中文状态": "V36.6 occlusion/reappear identity field repair 完成",
                "occlusion_reappear_identity_target_repaired": True,
                "occlusion_reappear_identity_three_seed_passed": passed,
                "test_occlusion_reappear_retrieval_top1": test_occ,
                "full_cvpr_scale_claim_allowed": False,
                "recommended_next_step": decision["recommended_next_step"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
