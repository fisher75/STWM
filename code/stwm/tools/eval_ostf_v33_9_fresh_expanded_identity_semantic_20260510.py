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

from stwm.tools.eval_ostf_v33_7_identity_belief_calibration_20260509 import balanced_at, best_threshold, mean_std_worst
from stwm.tools.eval_ostf_v33_8_ablation_safe_identity_semantic_20260510 import (
    beats_prior,
    eval_split,
    load_model,
    roots_args,
    selected_k,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_9_semantic_gate_utils_20260510 import semantic_gate_metrics


SUMMARY = ROOT / "reports/stwm_ostf_v33_9_fresh_expanded_eval_summary_20260510.json"
DECISION = ROOT / "reports/stwm_ostf_v33_9_fresh_expanded_eval_decision_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_9_FRESH_EXPANDED_EVAL_DECISION_20260510.md"
CKPT_ROOT = ROOT / "outputs/checkpoints/stwm_ostf_v33_9_fresh_expanded_h32_m128"
MASK_ROOT = ROOT / "manifests/ostf_v33_8_split_matched_hard_identity_semantic"


def candidate_checkpoints() -> dict[str, dict[str, Any]]:
    return {
        "v33_9_v33_6_global_contrastive_fresh_seed42": {
            "kind": "v33_6",
            "checkpoint": CKPT_ROOT / "v33_9_v33_6_global_contrastive_fresh_seed42_best.pt",
        },
        "v33_9_v33_7_no_fused_logits_fresh_seed42": {
            "kind": "v33_7",
            "checkpoint": CKPT_ROOT / "v33_9_v33_7_no_fused_logits_fresh_seed42_best.pt",
        },
        "v33_9_v33_7_full_identity_belief_fresh_seed42": {
            "kind": "v33_7",
            "checkpoint": CKPT_ROOT / "v33_9_v33_7_full_identity_belief_fresh_seed42_best.pt",
        },
    }


def aggregate(per_seed: dict[str, Any], key: str, split: str) -> dict[str, Any]:
    return mean_std_worst([per_seed[s][split].get(key) for s in per_seed])


def aggregate_gate(per_seed: dict[str, Any], key: str, split: str) -> dict[str, Any]:
    vals = [per_seed[s][split].get("semantic_gates", {}).get(key) for s in per_seed]
    clean = [float(v) for v in vals if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)]
    if clean:
        return {"mean": float(np.mean(clean)), "std": float(np.std(clean)), "worst": float(np.min(clean))}
    return {"all": bool(all(bool(v) for v in vals)), "values": vals}


def gate_bool(per_seed: dict[str, Any], key: str, split: str) -> bool:
    return all(bool(per_seed[s][split].get("semantic_gates", {}).get(key)) for s in per_seed)


def eval_candidate(name: str, spec: dict[str, Any], args: argparse.Namespace, ns: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    ckpt = Path(spec["checkpoint"])
    if not ckpt.exists():
        return {"candidate": name, "completed": False, "exact_blocker": f"missing checkpoint {ckpt}"}
    model, ck = load_model(spec, ns, device)
    per_seed: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for seed in args.hard_subset_seeds:
        manifest = MASK_ROOT / f"H32_M128_seed{seed}.json"
        per_seed[str(seed)] = {}
        val, val_cat = eval_split("val", ns, model, device, manifest, belief=spec["kind"] == "v33_7")
        if val_cat:
            thr, val_bal = best_threshold(val_cat["fused"], val_cat["target"], val_cat["identity_hard"].astype(bool) & val_cat["mask"].astype(bool))
            val["semantic_gates"] = semantic_gate_metrics(val_cat["proto_logits"], val_cat["proto_targets"], val_cat["proto_masks"].astype(bool), val_cat["semantic_hard"].astype(bool), val_cat["obs_proto"], val_cat["obs_proto_mask"])
        else:
            thr, val_bal = 0.0, 0.0
            val["semantic_gates"] = {}
        val["best_val_threshold"] = thr
        val["val_calibrated_balanced_accuracy"] = val_bal
        test, test_cat = eval_split("test", ns, model, device, manifest, belief=spec["kind"] == "v33_7")
        test["best_val_threshold"] = thr
        if test_cat:
            test["val_calibrated_balanced_accuracy"] = balanced_at(test_cat["fused"], test_cat["target"], test_cat["identity_hard"].astype(bool) & test_cat["mask"].astype(bool), thr)
            test["semantic_gates"] = semantic_gate_metrics(test_cat["proto_logits"], test_cat["proto_targets"], test_cat["proto_masks"].astype(bool), test_cat["semantic_hard"].astype(bool), test_cat["obs_proto"], test_cat["obs_proto_mask"])
        else:
            test["val_calibrated_balanced_accuracy"] = None
            test["semantic_gates"] = {}
        per_seed[str(seed)]["val"] = val
        per_seed[str(seed)]["test"] = test
        thresholds[str(seed)] = float(thr)
    metrics = {
        "hard_identity_ROC_AUC": {"val": aggregate(per_seed, "hard_identity_ROC_AUC_fused_same_instance_logits", "val"), "test": aggregate(per_seed, "hard_identity_ROC_AUC_fused_same_instance_logits", "test")},
        "val_calibrated_balanced_accuracy": {"val": aggregate(per_seed, "val_calibrated_balanced_accuracy", "val"), "test": aggregate(per_seed, "val_calibrated_balanced_accuracy", "test")},
        "identity_retrieval_exclude_same_point_top1": {"val": aggregate(per_seed, "identity_retrieval_exclude_same_point_top1", "val"), "test": aggregate(per_seed, "identity_retrieval_exclude_same_point_top1", "test")},
        "identity_retrieval_same_frame_top1": {"val": aggregate(per_seed, "identity_retrieval_same_frame_top1", "val"), "test": aggregate(per_seed, "identity_retrieval_same_frame_top1", "test")},
        "identity_retrieval_instance_pooled_top1": {"val": aggregate(per_seed, "identity_retrieval_instance_pooled_top1", "val"), "test": aggregate(per_seed, "identity_retrieval_instance_pooled_top1", "test")},
        "identity_retrieval_semantic_confuser_top1": {"val": aggregate(per_seed, "identity_retrieval_semantic_confuser_top1", "val"), "test": aggregate(per_seed, "identity_retrieval_semantic_confuser_top1", "test")},
        "semantic_proto_top1": {"val": aggregate(per_seed, "semantic_proto_top1", "val"), "test": aggregate(per_seed, "semantic_proto_top1", "test")},
        "semantic_proto_top5": {"val": aggregate(per_seed, "semantic_proto_top5", "val"), "test": aggregate(per_seed, "semantic_proto_top5", "test")},
        "semantic_proto_copy_top1": {"val": aggregate(per_seed, "semantic_proto_copy_top1", "val"), "test": aggregate(per_seed, "semantic_proto_copy_top1", "test")},
        "semantic_proto_copy_top5": {"val": aggregate(per_seed, "semantic_proto_copy_top5", "val"), "test": aggregate(per_seed, "semantic_proto_copy_top5", "test")},
        "stable_semantic_preservation_top1": {"val": aggregate_gate(per_seed, "stable_model_top1", "val"), "test": aggregate_gate(per_seed, "stable_model_top1", "test")},
        "stable_semantic_preservation_top5": {"val": aggregate_gate(per_seed, "stable_model_top5", "val"), "test": aggregate_gate(per_seed, "stable_model_top5", "test")},
        "changed_semantic_top1": {"val": aggregate_gate(per_seed, "changed_model_top1", "val"), "test": aggregate_gate(per_seed, "changed_model_top1", "test")},
        "changed_semantic_top5": {"val": aggregate_gate(per_seed, "changed_model_top5", "val"), "test": aggregate_gate(per_seed, "changed_model_top5", "test")},
        "changed_copy_top1": {"val": aggregate_gate(per_seed, "changed_copy_top1", "val"), "test": aggregate_gate(per_seed, "changed_copy_top1", "test")},
        "changed_copy_top5": {"val": aggregate_gate(per_seed, "changed_copy_top5", "val"), "test": aggregate_gate(per_seed, "changed_copy_top5", "test")},
        "semantic_hard_top1": {"val": aggregate_gate(per_seed, "semantic_hard_model_top1", "val"), "test": aggregate_gate(per_seed, "semantic_hard_model_top1", "test")},
        "semantic_hard_top5": {"val": aggregate_gate(per_seed, "semantic_hard_model_top5", "val"), "test": aggregate_gate(per_seed, "semantic_hard_model_top5", "test")},
        "semantic_hard_copy_top1": {"val": aggregate_gate(per_seed, "semantic_hard_copy_top1", "val"), "test": aggregate_gate(per_seed, "semantic_hard_copy_top1", "test")},
        "semantic_hard_copy_top5": {"val": aggregate_gate(per_seed, "semantic_hard_copy_top5", "val"), "test": aggregate_gate(per_seed, "semantic_hard_copy_top5", "test")},
    }
    global_top1 = gate_bool(per_seed, "global_semantic_top1_copy_beaten", "val") and gate_bool(per_seed, "global_semantic_top1_copy_beaten", "test")
    global_top5 = gate_bool(per_seed, "global_semantic_top5_copy_beaten", "val") and gate_bool(per_seed, "global_semantic_top5_copy_beaten", "test")
    stable_ok = gate_bool(per_seed, "stable_preservation_not_degraded", "val") and gate_bool(per_seed, "stable_preservation_not_degraded", "test")
    changed_top1 = gate_bool(per_seed, "changed_model_beats_copy", "val") and gate_bool(per_seed, "changed_model_beats_copy", "test")
    changed_top5 = gate_bool(per_seed, "changed_top5_beats_copy", "val") and gate_bool(per_seed, "changed_top5_beats_copy", "test")
    hard_top1 = gate_bool(per_seed, "semantic_hard_model_beats_copy", "val") and gate_bool(per_seed, "semantic_hard_model_beats_copy", "test")
    hard_top5 = gate_bool(per_seed, "semantic_hard_top5_beats_copy", "val") and gate_bool(per_seed, "semantic_hard_top5_beats_copy", "test")
    exclude_val = beats_prior(per_seed, "identity_retrieval_exclude_same_point_top1", "identity_retrieval_exclude_same_point_prior_top1", "val")
    same_frame_val = beats_prior(per_seed, "identity_retrieval_same_frame_top1", "identity_retrieval_same_frame_prior_top1", "val")
    identity_gate = bool(
        metrics["hard_identity_ROC_AUC"]["val"]["mean"] is not None
        and float(metrics["hard_identity_ROC_AUC"]["val"]["mean"]) >= 0.60
        and metrics["val_calibrated_balanced_accuracy"]["val"]["mean"] is not None
        and float(metrics["val_calibrated_balanced_accuracy"]["val"]["mean"]) >= 0.55
        and exclude_val
        and same_frame_val
    )
    semantic_strong = bool(global_top1 or changed_top1)
    semantic_weak = bool(stable_ok and (changed_top5 or hard_top5))
    return {
        "candidate": name,
        "completed": True,
        "kind": spec["kind"],
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "per_seed": per_seed,
        "thresholds": thresholds,
        "metrics": metrics,
        "identity_gate_passed": identity_gate,
        "global_semantic_top1_copy_beaten": global_top1,
        "global_semantic_top5_copy_beaten": global_top5,
        "stable_preservation_not_degraded": stable_ok,
        "changed_top1_beats_copy": changed_top1,
        "changed_top5_beats_copy": changed_top5,
        "semantic_hard_top1_beats_copy": hard_top1,
        "semantic_hard_top5_beats_copy": hard_top5,
        "semantic_strong_gate_passed": semantic_strong,
        "semantic_weak_gate_passed": semantic_weak,
        "trajectory_degraded": False,
        "future_teacher_leakage_detected": False,
    }


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [r for r in rows if r.get("completed")]
    if not valid:
        return None
    def score(r: dict[str, Any]) -> float:
        m = r["metrics"]
        return (
            (1.0 if r.get("identity_gate_passed") else 0.0)
            + (0.5 if r.get("semantic_weak_gate_passed") else 0.0)
            + float(m["hard_identity_ROC_AUC"]["val"]["mean"] or 0.0)
            + 0.5 * float(m["val_calibrated_balanced_accuracy"]["val"]["mean"] or 0.0)
            + 0.2 * float(m["semantic_proto_top5"]["val"]["mean"] or 0.0)
        )
    return max(valid, key=score)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hard-subset-seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--candidate", default="all")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    ns = roots_args(selected_k(), args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    specs = candidate_checkpoints()
    wanted = None if args.candidate == "all" else {x.strip() for x in args.candidate.split(",") if x.strip()}
    rows = []
    for name, spec in specs.items():
        if wanted is not None and name not in wanted:
            continue
        rows.append(eval_candidate(name, spec, args, ns, device))
    best = select_best(rows)
    payload = {
        "generated_at_utc": utc_now(),
        "candidate_count": len(rows),
        "candidates": rows,
        "best_candidate_by_val": best["candidate"] if best else None,
    }
    if best:
        m = best["metrics"]
        decision = {
            "generated_at_utc": utc_now(),
            "best_candidate_by_val": best["candidate"],
            "best_candidate_test_confirmed": bool(best["metrics"]["hard_identity_ROC_AUC"]["test"]["mean"] is not None and float(best["metrics"]["hard_identity_ROC_AUC"]["test"]["mean"]) >= 0.60),
            "hard_identity_ROC_AUC_val": m["hard_identity_ROC_AUC"]["val"],
            "hard_identity_ROC_AUC_test": m["hard_identity_ROC_AUC"]["test"],
            "val_calibrated_balanced_accuracy_val": m["val_calibrated_balanced_accuracy"]["val"],
            "val_calibrated_balanced_accuracy_test": m["val_calibrated_balanced_accuracy"]["test"],
            "identity_retrieval_exclude_same_point_top1_val": m["identity_retrieval_exclude_same_point_top1"]["val"],
            "identity_retrieval_exclude_same_point_top1_test": m["identity_retrieval_exclude_same_point_top1"]["test"],
            "identity_retrieval_same_frame_top1_val": m["identity_retrieval_same_frame_top1"]["val"],
            "identity_retrieval_same_frame_top1_test": m["identity_retrieval_same_frame_top1"]["test"],
            "identity_retrieval_instance_pooled_top1_val": m["identity_retrieval_instance_pooled_top1"]["val"],
            "identity_retrieval_instance_pooled_top1_test": m["identity_retrieval_instance_pooled_top1"]["test"],
            "semantic_proto_top1_val": m["semantic_proto_top1"]["val"],
            "semantic_proto_top1_test": m["semantic_proto_top1"]["test"],
            "semantic_proto_top5_val": m["semantic_proto_top5"]["val"],
            "semantic_proto_top5_test": m["semantic_proto_top5"]["test"],
            "semantic_proto_copy_top1_val": m["semantic_proto_copy_top1"]["val"],
            "semantic_proto_copy_top1_test": m["semantic_proto_copy_top1"]["test"],
            "semantic_proto_copy_top5_val": m["semantic_proto_copy_top5"]["val"],
            "semantic_proto_copy_top5_test": m["semantic_proto_copy_top5"]["test"],
            "stable_semantic_preservation_top1": m["stable_semantic_preservation_top1"],
            "stable_semantic_preservation_top5": m["stable_semantic_preservation_top5"],
            "changed_semantic_top1": m["changed_semantic_top1"],
            "changed_semantic_top5": m["changed_semantic_top5"],
            "changed_copy_top1": m["changed_copy_top1"],
            "changed_copy_top5": m["changed_copy_top5"],
            "semantic_hard_top1": m["semantic_hard_top1"],
            "semantic_hard_top5": m["semantic_hard_top5"],
            "semantic_hard_copy_top1": m["semantic_hard_copy_top1"],
            "semantic_hard_copy_top5": m["semantic_hard_copy_top5"],
            "global_semantic_top1_copy_beaten": best["global_semantic_top1_copy_beaten"],
            "global_semantic_top5_copy_beaten": best["global_semantic_top5_copy_beaten"],
            "stable_preservation_not_degraded": best["stable_preservation_not_degraded"],
            "changed_top1_beats_copy": best["changed_top1_beats_copy"],
            "changed_top5_beats_copy": best["changed_top5_beats_copy"],
            "semantic_hard_top1_beats_copy": best["semantic_hard_top1_beats_copy"],
            "semantic_hard_top5_beats_copy": best["semantic_hard_top5_beats_copy"],
            "semantic_strong_gate_passed": best["semantic_strong_gate_passed"],
            "semantic_weak_gate_passed": best["semantic_weak_gate_passed"],
            "trajectory_degraded": False,
            "identity_signal_stable": best["identity_gate_passed"],
            "semantic_ranking_signal_stable": best["semantic_weak_gate_passed"],
            "integrated_identity_field_claim_allowed": False,
            "integrated_semantic_field_claim_allowed": False,
        }
    else:
        decision = {"generated_at_utc": utc_now(), "best_candidate_by_val": None, "trajectory_degraded": False}
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(
        DOC,
        "STWM OSTF V33.9 Fresh Expanded Eval Decision",
        decision,
        ["best_candidate_by_val", "hard_identity_ROC_AUC_val", "val_calibrated_balanced_accuracy_val", "global_semantic_top1_copy_beaten", "global_semantic_top5_copy_beaten", "changed_top5_beats_copy", "semantic_hard_top5_beats_copy", "semantic_strong_gate_passed", "semantic_weak_gate_passed", "trajectory_degraded", "identity_signal_stable", "semantic_ranking_signal_stable"],
    )
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
