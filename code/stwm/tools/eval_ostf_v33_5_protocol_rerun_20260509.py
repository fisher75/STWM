#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_3_structured_semantic_identity_world_model import StructuredSemanticIdentityWorldModelV333
from stwm.tools.eval_ostf_v33_5_structured_semantic_identity_manifest_driven_20260509 import eval_available
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


SUMMARY = ROOT / "reports/stwm_ostf_v33_5_protocol_rerun_summary_20260509.json"
DECISION = ROOT / "reports/stwm_ostf_v33_5_protocol_rerun_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_5_PROTOCOL_RERUN_DECISION_20260509.md"


def stat(vals: list[float | None], higher: bool = True) -> dict[str, Any]:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return {"mean": None, "std": None, "worst": None}
    return {"mean": float(mean(xs)), "std": float(pstdev(xs)), "worst": float(min(xs) if higher else max(xs))}


def load_model(args: argparse.Namespace) -> tuple[StructuredSemanticIdentityWorldModelV333, argparse.Namespace, torch.device]:
    ck = torch.load(args.checkpoint, map_location="cpu")
    train_args = argparse.Namespace(**ck["args"])
    train_args.batch_size = args.batch_size
    train_args.num_workers = args.num_workers
    centers = torch.from_numpy(np.asarray(np.load(train_args.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StructuredSemanticIdentityWorldModelV333(train_args.v30_checkpoint, prototype_centers=centers, teacher_embedding_dim=train_args.teacher_embedding_dim, use_observed_instance_context=False).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, train_args, device


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v33_3_structured_semantic_identity/v33_3_structured_semantic_identity_m128_h32_seed42_smoke_best.pt"))
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    model, train_args, device = load_model(args)
    runs: dict[str, Any] = {}
    for seed in seeds:
        train_args.hard_subset_manifest = str(ROOT / f"manifests/ostf_v33_5_split_matched_hard_identity_semantic/H32_M128_seed{seed}.json")
        train_args.batch_size = args.batch_size
        train_args.num_workers = args.num_workers
        seed_payload = {}
        for split in ("val", "test"):
            metrics, _ = eval_available(split, train_args, model, device)
            seed_payload[split] = metrics
        seed_payload["split_shift_suspected"] = bool(
            seed_payload["val"].get("hard_identity_ROC_AUC_available") is not None
            and seed_payload["test"].get("hard_identity_ROC_AUC_available") is not None
            and abs(float(seed_payload["test"]["hard_identity_ROC_AUC_available"]) - float(seed_payload["val"]["hard_identity_ROC_AUC_available"])) > 0.15
        )
        seed_payload["val_test_agree"] = not seed_payload["split_shift_suspected"]
        runs[f"seed{seed}"] = seed_payload
    def vals(split: str, key: str) -> list[float | None]:
        return [runs[f"seed{s}"][split].get(key) for s in seeds]
    aggregate = {
        "hard_identity_ROC_AUC": {"val": stat(vals("val", "hard_identity_ROC_AUC_available")), "test": stat(vals("test", "hard_identity_ROC_AUC_available"))},
        "hard_identity_balanced_accuracy": {"val": stat(vals("val", "hard_identity_balanced_accuracy_available")), "test": stat(vals("test", "hard_identity_balanced_accuracy_available"))},
        "identity_retrieval_exclude_same_point_top1": {"val": stat(vals("val", "identity_retrieval_exclude_same_point_top1")), "test": stat(vals("test", "identity_retrieval_exclude_same_point_top1"))},
        "identity_retrieval_same_frame_top1": {"val": stat(vals("val", "identity_retrieval_same_frame_top1")), "test": stat(vals("test", "identity_retrieval_same_frame_top1"))},
        "identity_retrieval_instance_pooled_top1": {"val": stat(vals("val", "identity_retrieval_instance_pooled_top1")), "test": stat(vals("test", "identity_retrieval_instance_pooled_top1"))},
        "semantic_proto_top1": {"val": stat(vals("val", "semantic_proto_top1")), "test": stat(vals("test", "semantic_proto_top1"))},
        "semantic_proto_top5": {"val": stat(vals("val", "semantic_proto_top5")), "test": stat(vals("test", "semantic_proto_top5"))},
        "available_ratio": {"val": stat(vals("val", "available_ratio")), "test": stat(vals("test", "available_ratio"))},
    }
    manifest_full = all(runs[f"seed{s}"][sp].get("manifest_full_coverage_ok", False) for s in seeds for sp in ("val", "test"))
    available_ratio = min(float(runs[f"seed{s}"][sp].get("available_ratio", 0.0)) for s in seeds for sp in ("val", "test"))
    identity_balanced = all(runs[f"seed{s}"][sp].get("identity_hard_balanced", False) for s in seeds for sp in ("val", "test"))
    val_test_agree = all(runs[f"seed{s}"].get("val_test_agree", False) for s in seeds)
    identity_stable = bool(
        aggregate["hard_identity_ROC_AUC"]["val"]["mean"] is not None
        and aggregate["hard_identity_ROC_AUC"]["test"]["mean"] is not None
        and aggregate["hard_identity_ROC_AUC"]["val"]["mean"] >= 0.60
        and aggregate["hard_identity_ROC_AUC"]["test"]["mean"] >= 0.60
        and aggregate["hard_identity_balanced_accuracy"]["val"]["mean"] >= 0.55
        and aggregate["hard_identity_balanced_accuracy"]["test"]["mean"] >= 0.55
    )
    semantic_top5_stable = all(bool(runs[f"seed{s}"][sp].get("semantic_top5_copy_beaten_available", False)) for s in seeds for sp in ("val", "test"))
    semantic_top1_positive = all(bool(runs[f"seed{s}"][sp].get("semantic_top1_copy_beaten_available", False)) for s in seeds for sp in ("val", "test"))
    trajectory_degraded = any(bool(runs[f"seed{s}"][sp].get("trajectory_degraded", True)) for s in seeds for sp in ("val", "test"))
    if not manifest_full or available_ratio < 0.95:
        next_step = "fix_manifest_dataset_coverage"
    elif not identity_balanced:
        next_step = "fix_split_matched_hard_subset"
    elif not val_test_agree:
        next_step = "fix_split_matched_hard_subset"
    elif not identity_stable:
        next_step = "fix_identity_contrastive_loss"
    elif not semantic_top5_stable:
        next_step = "fix_semantic_prototype_loss"
    else:
        next_step = "run_v33_5_h32_full_data_smoke"
    summary = {
        "generated_at_utc": utc_now(),
        "seeds": seeds,
        "runs": runs,
        "aggregate": aggregate,
        "manifest_full_coverage_ok": manifest_full,
        "available_ratio": available_ratio,
        "identity_hard_balanced": identity_balanced,
        "split_shift_suspected": not val_test_agree,
        "whether_identity_signal_stable": identity_stable,
        "whether_semantic_ranking_signal_stable": semantic_top5_stable,
        "whether_semantic_top1_signal_positive": semantic_top1_positive,
        "whether_val_test_agree": val_test_agree,
        "trajectory_degraded": trajectory_degraded,
    }
    decision = {
        "generated_at_utc": utc_now(),
        "manifest_full_coverage_ok": manifest_full,
        "available_ratio": available_ratio,
        "identity_hard_balanced": identity_balanced,
        "split_shift_suspected": not val_test_agree,
        "hard_identity_ROC_AUC": aggregate["hard_identity_ROC_AUC"],
        "hard_identity_balanced_accuracy": aggregate["hard_identity_balanced_accuracy"],
        "identity_retrieval_exclude_same_point_top1": aggregate["identity_retrieval_exclude_same_point_top1"],
        "identity_retrieval_same_frame_top1": aggregate["identity_retrieval_same_frame_top1"],
        "identity_retrieval_instance_pooled_top1": aggregate["identity_retrieval_instance_pooled_top1"],
        "semantic_proto_top1": aggregate["semantic_proto_top1"],
        "semantic_proto_top5": aggregate["semantic_proto_top5"],
        "semantic_top1_copy_beaten": semantic_top1_positive,
        "semantic_top5_copy_beaten": semantic_top5_stable,
        "trajectory_degraded": trajectory_degraded,
        "identity_signal_stable": identity_stable,
        "semantic_ranking_signal_stable": semantic_top5_stable,
        "val_test_agree": val_test_agree,
        "recommended_next_step": next_step,
    }
    dump_json(SUMMARY, summary)
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V33.5 Protocol Rerun Decision", decision, ["manifest_full_coverage_ok", "available_ratio", "identity_hard_balanced", "split_shift_suspected", "hard_identity_ROC_AUC", "hard_identity_balanced_accuracy", "identity_retrieval_exclude_same_point_top1", "identity_retrieval_same_frame_top1", "identity_retrieval_instance_pooled_top1", "semantic_proto_top1", "semantic_proto_top5", "semantic_top1_copy_beaten", "semantic_top5_copy_beaten", "trajectory_degraded", "identity_signal_stable", "semantic_ranking_signal_stable", "val_test_agree", "recommended_next_step"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
