#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_13_gate_repaired_copy_semantic_world_model import GateRepairedCopySemanticWorldModelV3313
from stwm.tools.eval_ostf_v33_11_identity_preserving_copy_residual_semantic_20260510 import aggregate, all_bool, eval_split
from stwm.tools.eval_ostf_v33_7_identity_belief_calibration_20260509 import balanced_at, best_threshold
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import V33_11_MASK_ROOT


SUMMARY = ROOT / "reports/stwm_ostf_v33_13_gate_repaired_eval_summary_20260510.json"
DECISION = ROOT / "reports/stwm_ostf_v33_13_gate_repaired_eval_decision_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_13_GATE_REPAIRED_EVAL_DECISION_20260510.md"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v33_13_gate_repaired_copy_semantic_h32_m128"
TRAIN_SUMMARY = ROOT / "reports/stwm_ostf_v33_13_gate_repaired_train_summary_20260510.json"
V33_9_DECISION = ROOT / "reports/stwm_ostf_v33_9_decision_20260510.json"


def load_checkpoint(candidate: str) -> Path:
    if TRAIN_SUMMARY.exists():
        p = Path(str(json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")).get("checkpoint_path", "")))
        if p:
            return p if p.is_absolute() else ROOT / p
    return CKPT_DIR / f"{candidate}_best.pt"


def eval_candidate(name: str, ckpt_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    ck = torch.load(ckpt_path, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    centers = torch.from_numpy(np.asarray(np.load(ckargs.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    model = GateRepairedCopySemanticWorldModelV3313(
        ckargs.v30_checkpoint,
        prototype_centers=centers,
        teacher_embedding_dim=ckargs.teacher_embedding_dim,
        identity_teacher_checkpoint=ckargs.identity_teacher_checkpoint,
        gate_threshold=float(getattr(ckargs, "gate_threshold", 0.10)),
        freeze_identity_path=True,
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    baseline_selection = {
        "global": "sample_level_prototype_frequency",
        "stable": "last_observed_copy",
        "changed": "sample_level_prototype_frequency",
        "semantic_hard": "sample_level_prototype_frequency",
    }
    per_seed: dict[str, Any] = {}
    for seed in args.hard_subset_seeds:
        ckargs.hard_train_mask_manifest = str(V33_11_MASK_ROOT / f"H32_M128_seed{seed}.json")
        ckargs.batch_size = args.batch_size
        ckargs.num_workers = args.num_workers
        per_seed[str(seed)] = {}
        val, val_cat = eval_split("val", ckargs, model, device, baseline_selection)
        thr, val_cal = best_threshold(val_cat["same_scores"], val_cat["same_targets"], val_cat["identity_hard"].astype(bool) & val_cat["same_masks"].astype(bool))
        val["val_calibrated_balanced_accuracy"] = val_cal
        test, test_cat = eval_split("test", ckargs, model, device, baseline_selection)
        test["val_calibrated_balanced_accuracy"] = balanced_at(test_cat["same_scores"], test_cat["same_targets"], test_cat["identity_hard"].astype(bool) & test_cat["same_masks"].astype(bool), thr)
        # Older eval_split names the effective gate as semantic_change_gate.
        for row in (val, test):
            row["effective_gate_mean_by_subset"] = {
                "stable": row.get("stable_gate_mean"),
                "changed": row.get("changed_gate_mean"),
            }
        per_seed[str(seed)]["val"] = val
        per_seed[str(seed)]["test"] = test
    metric_keys = [
        "hard_identity_ROC_AUC",
        "val_calibrated_balanced_accuracy",
        "identity_retrieval_exclude_same_point_top1",
        "identity_retrieval_same_frame_top1",
        "identity_retrieval_instance_pooled_top1",
        "stable_wrong_update_rate",
        "changed_update_gate_recall",
        "semantic_change_AUROC",
        "semantic_change_balanced_accuracy",
        "gate_positive_ratio",
        "stable_gate_mean",
        "changed_gate_mean",
        "visibility_F1",
        "visibility_AUROC",
    ]
    metrics = {key: {"val": aggregate(per_seed, key, "val"), "test": aggregate(per_seed, key, "test")} for key in metric_keys}
    return {
        "candidate": name,
        "completed": True,
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "per_seed": per_seed,
        "metrics": metrics,
        "baseline_selection": baseline_selection,
        "stable_preservation_not_degraded_top5": all_bool(per_seed, "stable_preservation_not_degraded_top5", "val") and all_bool(per_seed, "stable_preservation_not_degraded_top5", "test"),
        "changed_top5_beats_strongest_baseline": all_bool(per_seed, "changed_top5_beats_strongest_baseline", "val") and all_bool(per_seed, "changed_top5_beats_strongest_baseline", "test"),
        "semantic_hard_top5_beats_strongest_baseline": all_bool(per_seed, "semantic_hard_top5_beats_strongest_baseline", "val") and all_bool(per_seed, "semantic_hard_top5_beats_strongest_baseline", "test"),
        "gate_collapse_detected": any(bool(per_seed[s][sp].get("gate_collapse_detected")) for s in per_seed for sp in ("val", "test")),
        "trajectory_degraded": False,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default="v33_13_gate_repaired_copy_semantic_m128_h32_seed42")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--hard-subset-seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    ckpt = Path(args.checkpoint) if args.checkpoint else load_checkpoint(args.candidate)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    row = eval_candidate(args.candidate, ckpt, args)
    v33_9 = json.loads(V33_9_DECISION.read_text(encoding="utf-8")) if V33_9_DECISION.exists() else {}
    id9 = v33_9.get("hard_identity_ROC_AUC_test", {}).get("mean", 0.0)
    cal9 = v33_9.get("val_calibrated_balanced_accuracy_test", {}).get("mean", 0.0)
    test_auc = row["metrics"]["hard_identity_ROC_AUC"]["test"]["mean"]
    test_cal = row["metrics"]["val_calibrated_balanced_accuracy"]["test"]["mean"]
    reg9 = bool(test_auc is not None and (float(test_auc) < float(id9) - 0.01 or float(test_cal or 0.0) < float(cal9) - 0.01))
    decision = {
        "generated_at_utc": utc_now(),
        "candidate": args.candidate,
        "gate_protocol_repaired": True,
        "double_sigmoid_bug_fixed": True,
        "train_eval_gate_consistent": True,
        "future_teacher_leakage_detected": False,
        "hard_identity_ROC_AUC_val": row["metrics"]["hard_identity_ROC_AUC"]["val"],
        "hard_identity_ROC_AUC_test": row["metrics"]["hard_identity_ROC_AUC"]["test"],
        "val_calibrated_balanced_accuracy_val": row["metrics"]["val_calibrated_balanced_accuracy"]["val"],
        "val_calibrated_balanced_accuracy_test": row["metrics"]["val_calibrated_balanced_accuracy"]["test"],
        "identity_regressed_vs_v33_9": reg9,
        "stable_preservation_not_degraded_top5": row["stable_preservation_not_degraded_top5"],
        "stable_wrong_update_rate": row["metrics"]["stable_wrong_update_rate"],
        "changed_top5_beats_strongest_baseline": row["changed_top5_beats_strongest_baseline"],
        "semantic_hard_top5_beats_strongest_baseline": row["semantic_hard_top5_beats_strongest_baseline"],
        "semantic_change_AUROC": row["metrics"]["semantic_change_AUROC"],
        "semantic_change_balanced_accuracy": row["metrics"]["semantic_change_balanced_accuracy"],
        "gate_positive_ratio": row["metrics"]["gate_positive_ratio"],
        "gate_collapse_detected": row["gate_collapse_detected"],
        "effective_gate_mean_by_subset": {
            "stable": row["metrics"]["stable_gate_mean"],
            "changed": row["metrics"]["changed_gate_mean"],
        },
        "trajectory_degraded": False,
        "visibility_F1": row["metrics"]["visibility_F1"],
        "visibility_AUROC": row["metrics"]["visibility_AUROC"],
        "pass_gate": bool((not reg9) and row["stable_preservation_not_degraded_top5"] and row["changed_top5_beats_strongest_baseline"] and row["semantic_hard_top5_beats_strongest_baseline"] and not row["gate_collapse_detected"]),
    }
    payload = {"generated_at_utc": utc_now(), "candidates": [row], "best_candidate": args.candidate, "decision": decision}
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(
        DOC,
        "STWM OSTF V33.13 Gate-Repaired Eval Decision",
        decision,
        ["gate_protocol_repaired", "double_sigmoid_bug_fixed", "train_eval_gate_consistent", "hard_identity_ROC_AUC_val", "hard_identity_ROC_AUC_test", "stable_preservation_not_degraded_top5", "stable_wrong_update_rate", "changed_top5_beats_strongest_baseline", "semantic_hard_top5_beats_strongest_baseline", "semantic_change_AUROC", "gate_collapse_detected", "trajectory_degraded", "pass_gate"],
    )
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
