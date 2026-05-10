#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


SUMMARY = ROOT / "reports/stwm_ostf_v33_9_fresh_expanded_train_summary_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_9_FRESH_EXPANDED_TRAIN_SUMMARY_20260510.md"
RUN_DIR = ROOT / "reports/stwm_ostf_v33_9_fresh_expanded_runs"
CKPT_ROOT = ROOT / "outputs/checkpoints/stwm_ostf_v33_9_fresh_expanded_h32_m128"
COMPLETE = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128"
MASK = ROOT / "manifests/ostf_v33_8_split_matched_hard_identity_semantic/H32_M128_seed42.json"


def selected_k(default: int = 32) -> int:
    report = ROOT / "reports/stwm_ostf_v33_8_complete_h32_m128_target_coverage_20260510.json"
    if report.exists():
        return int(json.loads(report.read_text(encoding="utf-8")).get("selected_K", default))
    return default


def complete_train_count() -> int | None:
    report = ROOT / "reports/stwm_ostf_v33_8_complete_h32_m128_target_coverage_20260510.json"
    if report.exists():
        return json.loads(report.read_text(encoding="utf-8")).get("complete_train_sample_count")
    return None


def roots(k: int) -> dict[str, str]:
    return {
        "semantic_identity_sidecar_root": str(COMPLETE / "semantic_identity_targets/pointodyssey"),
        "global_identity_label_root": str(COMPLETE / "global_identity_labels/pointodyssey"),
        "visual_teacher_root": str(COMPLETE / "visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"),
        "semantic_prototype_target_root": str(COMPLETE / f"semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K{k}"),
        "prototype_vocab_path": str(ROOT / f"outputs/cache/stwm_ostf_v33_8_semantic_prototypes/pointodyssey/clip_vit_b32_local/K{k}/prototype_vocab.npz"),
    }


def candidate_names() -> list[str]:
    return [
        "v33_9_v33_6_global_contrastive_fresh_seed42",
        "v33_9_v33_7_no_fused_logits_fresh_seed42",
        "v33_9_v33_7_full_identity_belief_fresh_seed42",
    ]


def timestamped_name(name: str) -> str:
    ckpt = CKPT_ROOT / f"{name}_best.pt"
    if not ckpt.exists():
        return name
    return f"{name}_rerun_{time.strftime('%Y%m%d_%H%M%S')}"


def namespace_for(candidate: str, args: argparse.Namespace, k: int) -> argparse.Namespace:
    r = roots(k)
    name = timestamped_name(candidate)
    base = {
        "experiment_name": name,
        "v30_checkpoint": str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"),
        **r,
        "m_points": 128,
        "horizon": 32,
        "seed": 42,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_train_items": 0,
        "teacher_embedding_dim": 512,
        "lr": args.lr,
        "contrastive_max_tokens": args.contrastive_max_tokens,
        "use_observed_instance_context": False,
        "cpu": args.cpu,
    }
    if candidate.startswith("v33_9_v33_6"):
        base.update(
            {
                "hard_subset_manifest": str(MASK),
                "eval_interval": args.steps,
                "max_eval_items": 0,
                "identity_loss_weight": 0.20,
                "identity_retrieval_loss_weight": 0.20,
                "same_frame_loss_weight": 0.12,
                "semantic_confuser_loss_weight": 0.08,
                "consistency_loss_weight": 0.10,
                "use_local_instance_contrastive_control": False,
            }
        )
    else:
        base.update(
            {
                "hard_train_mask_manifest": str(MASK),
                "full_bce_weight": 0.15,
                "hard_bce_weight": 1.25,
                "embedding_bce_weight": 0.85,
                "fused_bce_weight": 1.35,
                "disable_hard_bce": False,
                "disable_embedding_similarity_logits": False,
                "disable_fused_logits": "no_fused_logits" in candidate,
                "write_main_summary": False,
            }
        )
    return argparse.Namespace(**base)


def patch_modules() -> tuple[Any, Any]:
    import stwm.tools.train_ostf_v33_6_identity_contrastive_repair_20260509 as t6
    import stwm.tools.train_ostf_v33_7_identity_belief_calibration_20260509 as t7

    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    t6.CKPT_DIR = CKPT_ROOT
    t6.RUN_DIR = RUN_DIR
    t6.SUMMARY = ROOT / "reports/stwm_ostf_v33_9_fresh_expanded_v33_6_last_train_summary_20260510.json"
    t6.DOC = ROOT / "docs/STWM_OSTF_V33_9_FRESH_EXPANDED_V33_6_LAST_TRAIN_SUMMARY_20260510.md"
    t7.CKPT_DIR = CKPT_ROOT
    t7.RUN_DIR = RUN_DIR
    t7.SUMMARY = ROOT / "reports/stwm_ostf_v33_9_fresh_expanded_v33_7_last_train_summary_20260510.json"
    t7.DOC = ROOT / "docs/STWM_OSTF_V33_9_FRESH_EXPANDED_V33_7_LAST_TRAIN_SUMMARY_20260510.md"
    return t6, t7


def run_candidate(candidate: str, args: argparse.Namespace) -> dict[str, Any]:
    k = selected_k()
    t6, t7 = patch_modules()
    ns = namespace_for(candidate, args, k)
    if candidate.startswith("v33_9_v33_6"):
        payload = t6.train_one(ns)
        kind = "v33_6"
    else:
        payload = t7.train_one(ns)
        kind = "v33_7"
    ckpt = CKPT_ROOT / f"{ns.experiment_name}_best.pt"
    out = {
        "generated_at_utc": utc_now(),
        "candidate": candidate,
        "experiment_name": ns.experiment_name,
        "kind": kind,
        "completed": bool(payload.get("completed") and ckpt.exists()),
        "fresh_training": True,
        "skipped_existing": False,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "target_root": str(COMPLETE.relative_to(ROOT)),
        "hard_mask_manifest": str(MASK.relative_to(ROOT)),
        "actual_train_sample_count": payload.get("train_sample_count") or payload.get("complete_train_sample_count"),
        "global_identity_labels_used": bool(payload.get("global_identity_labels_used_in_training", payload.get("global_identity_labels_used", True))),
        "future_teacher_leakage_detected": bool(payload.get("future_teacher_leakage_detected", False)),
        "v30_backbone_frozen": bool(payload.get("v30_backbone_frozen", False)),
        "raw_train_payload": payload,
    }
    dump_json(RUN_DIR / f"{ns.experiment_name}.v33_9.json", out)
    return out


def aggregate() -> dict[str, Any]:
    names = candidate_names()
    rows = []
    for path in sorted(RUN_DIR.glob("v33_9_*.v33_9.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("candidate") in names:
            rows.append(row)
    by_candidate: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_candidate[row["candidate"]] = row
    selected = [by_candidate[c] for c in names if c in by_candidate]
    payload = {
        "generated_at_utc": utc_now(),
        "candidate_count": len(names),
        "completed_candidate_count": sum(1 for r in selected if r.get("completed")),
        "fresh_training": True,
        "skipped_existing_candidate_count": 0,
        "target_root": str(COMPLETE.relative_to(ROOT)),
        "hard_mask_manifest": str(MASK.relative_to(ROOT)),
        "complete_train_sample_count": complete_train_count(),
        "actual_train_sample_count_by_candidate": {r["candidate"]: r.get("actual_train_sample_count") for r in selected},
        "checkpoint_root": str(CKPT_ROOT.relative_to(ROOT)),
        "checkpoint_paths": {r["candidate"]: r.get("checkpoint_path") for r in selected},
        "global_identity_labels_used": all(bool(r.get("global_identity_labels_used")) for r in selected) if selected else False,
        "future_teacher_leakage_detected": any(bool(r.get("future_teacher_leakage_detected")) for r in selected),
        "v30_backbone_frozen": all(bool(r.get("v30_backbone_frozen")) for r in selected) if selected else False,
        "candidates": selected,
        "exact_blockers": [] if len(selected) == len(names) else [f"missing completed run for {c}" for c in names if c not in by_candidate],
    }
    dump_json(SUMMARY, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.9 Fresh Expanded Train Summary",
        payload,
        ["candidate_count", "completed_candidate_count", "fresh_training", "skipped_existing_candidate_count", "target_root", "complete_train_sample_count", "actual_train_sample_count_by_candidate", "checkpoint_root", "global_identity_labels_used", "future_teacher_leakage_detected", "v30_backbone_frozen", "exact_blockers"],
    )
    print(SUMMARY.relative_to(ROOT))
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default="all")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--contrastive-max-tokens", type=int, default=2048)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.aggregate_only:
        aggregate()
        return 0
    names = candidate_names() if args.candidate == "all" else [x.strip() for x in args.candidate.split(",") if x.strip()]
    for name in names:
        if name not in candidate_names():
            raise SystemExit(f"unknown candidate: {name}")
        run_candidate(name, args)
    aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
