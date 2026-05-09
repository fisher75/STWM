#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_3_structured_semantic_identity_world_model import StructuredSemanticIdentityWorldModelV333
from stwm.tools.eval_ostf_v33_4_structured_semantic_identity_protocol_20260509 import evaluate
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


SUMMARY = ROOT / "reports/stwm_ostf_v33_4_protocol_rerun_summary_20260509.json"
DECISION = ROOT / "reports/stwm_ostf_v33_4_protocol_rerun_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_4_PROTOCOL_RERUN_DECISION_20260509.md"


def load(rel: str) -> dict:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v33_3_structured_semantic_identity/v33_3_structured_semantic_identity_m128_h32_seed42_smoke_best.pt"))
    p.add_argument("--hard-subset-manifest", default=str(ROOT / "manifests/ostf_v33_4_separated_hard_identity_semantic/H32_M128_seed42.json"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-items", type=int, default=128)
    args = p.parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    ck = torch.load(ckpt_path, map_location="cpu")
    train_args = argparse.Namespace(**ck["args"])
    train_args.hard_subset_manifest = str(args.hard_subset_manifest)
    train_args.batch_size = args.batch_size
    train_args.num_workers = args.num_workers
    train_args.max_items = args.max_items
    centers = torch.from_numpy(np.asarray(np.load(train_args.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StructuredSemanticIdentityWorldModelV333(train_args.v30_checkpoint, prototype_centers=centers, teacher_embedding_dim=train_args.teacher_embedding_dim, use_observed_instance_context=False).to(device)
    model.load_state_dict(ck["model"], strict=True)
    val, _ = evaluate("val", train_args, model, device)
    test, _ = evaluate("test", train_args, model, device)
    split_audit = load("reports/stwm_ostf_v33_4_split_shift_audit_20260509.json")
    forensics = load("reports/stwm_ostf_v33_4_result_forensics_20260509.json")
    val_gap = None
    if val.get("hard_identity_ROC_AUC") is not None and test.get("hard_identity_ROC_AUC") is not None:
        val_gap = abs(float(test["hard_identity_ROC_AUC"]) - float(val["hard_identity_ROC_AUC"]))
    split_shift = bool(split_audit.get("split_shift_suspected", False) or (val_gap is not None and val_gap > 0.15))
    strict_top1 = test.get("identity_retrieval_exclude_same_point_top1")
    strict_prior = test.get("identity_retrieval_exclude_same_point_prior_top1")
    strict_beats = strict_top1 is not None and strict_prior is not None and float(strict_top1) > float(strict_prior) + 1e-9
    identity_ok = bool((test.get("hard_identity_ROC_AUC") or 0.0) >= 0.60 and (test.get("hard_identity_balanced_accuracy") or 0.0) >= 0.55 and strict_beats)
    semantic_top1 = bool(test.get("semantic_top1_copy_beaten", False))
    semantic_top5 = bool(test.get("semantic_top5_copy_beaten", False))
    if not test.get("manifest_sample_match_ok", False):
        next_step = "fix_hard_identity_subset_again"
    elif not test.get("identity_hard_balanced", False):
        next_step = "fix_hard_identity_subset_again"
    elif split_shift:
        next_step = "fix_split_shift_or_data_protocol"
    elif not identity_ok:
        next_step = "fix_identity_contrastive_loss"
    elif not semantic_top1 and semantic_top5:
        next_step = "fix_semantic_prototype_loss"
    elif not semantic_top1 and not semantic_top5:
        next_step = "fix_semantic_prototype_loss"
    else:
        next_step = "run_v33_4_h64_h96_smoke"
    summary = {
        "generated_at_utc": utc_now(),
        "result_forensics_done": bool(forensics),
        "separated_hard_subset_built": bool(load("reports/stwm_ostf_v33_4_separated_hard_subset_20260509.json").get("separated_hard_subset_built", False)),
        "val_metrics": val,
        "test_metrics": test,
        "val_test_gap": val_gap,
        "split_shift_suspected": split_shift,
    }
    decision = {
        "generated_at_utc": utc_now(),
        "result_forensics_done": summary["result_forensics_done"],
        "separated_hard_subset_built": summary["separated_hard_subset_built"],
        "manifest_loaded": bool(test.get("manifest_loaded", False)),
        "manifest_sample_match_ok": bool(test.get("manifest_sample_match_ok", False)),
        "identity_hard_balanced": bool(test.get("identity_hard_balanced", False)),
        "semantic_hard_nonempty": bool(test.get("semantic_hard_mask_nonempty_ratio", 0.0) > 0.0),
        "split_shift_suspected": split_shift,
        "hard_identity_ROC_AUC": test.get("hard_identity_ROC_AUC"),
        "hard_identity_balanced_accuracy": test.get("hard_identity_balanced_accuracy"),
        "identity_strict_retrieval_top1": strict_top1,
        "identity_instance_pooled_retrieval_top1": test.get("identity_retrieval_instance_pooled_top1"),
        "semantic_proto_top1": test.get("semantic_proto_top1"),
        "semantic_proto_top5": test.get("semantic_proto_top5"),
        "semantic_top1_copy_beaten": semantic_top1,
        "semantic_top5_copy_beaten": semantic_top5,
        "semantic_ranking_signal_positive": semantic_top5,
        "trajectory_degraded": bool(test.get("trajectory_degraded", True)),
        "integrated_identity_field_claim_allowed": bool(identity_ok and not split_shift and test.get("identity_hard_balanced", False)),
        "integrated_semantic_field_claim_allowed": bool(semantic_top1 and not split_shift),
        "recommended_next_step": next_step,
    }
    dump_json(SUMMARY, summary)
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V33.4 Protocol Rerun Decision", decision, ["result_forensics_done", "separated_hard_subset_built", "manifest_loaded", "manifest_sample_match_ok", "identity_hard_balanced", "semantic_hard_nonempty", "split_shift_suspected", "hard_identity_ROC_AUC", "hard_identity_balanced_accuracy", "identity_strict_retrieval_top1", "identity_instance_pooled_retrieval_top1", "semantic_proto_top1", "semantic_proto_top5", "semantic_top1_copy_beaten", "semantic_top5_copy_beaten", "trajectory_degraded", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"])
    print(DECISION.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
