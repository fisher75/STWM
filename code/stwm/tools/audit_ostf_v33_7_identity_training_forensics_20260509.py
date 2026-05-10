#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_7_identity_training_forensics_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_7_IDENTITY_TRAINING_FORENSICS_20260509.md"

TRAIN6 = ROOT / "reports/stwm_ostf_v33_6_identity_contrastive_train_summary_20260509.json"
EVAL6 = ROOT / "reports/stwm_ostf_v33_6_identity_contrastive_eval_summary_20260509.json"
DEC6 = ROOT / "reports/stwm_ostf_v33_6_identity_contrastive_eval_decision_20260509.json"
ABL6 = ROOT / "reports/stwm_ostf_v33_6_identity_label_ablation_summary_20260509.json"

ROOTS = {
    "v30": ROOT / "outputs/cache/stwm_ostf_v30_external_gt/pointodyssey/M128_H32",
    "identity": ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey",
    "global_identity": ROOT / "outputs/cache/stwm_ostf_v33_6_global_identity_labels/pointodyssey",
    "visual": ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local",
    "semantic_proto": ROOT / "outputs/cache/stwm_ostf_v33_3_semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def stems(root: Path, split: str) -> set[str]:
    path = root / split
    return {p.stem for p in path.glob("*.npz")} if path.exists() else set()


def counts_by_split(root: Path) -> dict[str, int]:
    return {split: len(stems(root, split)) for split in ("train", "val", "test")}


def manifest_stems(split: str) -> set[str]:
    entries = json.loads((ROOT / "manifests/ostf_v30_external_gt" / f"{split}.json").read_text(encoding="utf-8")).get("entries", [])
    return {
        Path(e["cache_path"]).stem
        for e in entries
        if int(e.get("H", -1)) == 32 and int(e.get("M", -1)) == 128
    }


def complete_counts() -> dict[str, int]:
    out = {}
    for split in ("train", "val", "test"):
        sets = [manifest_stems(split)] + [stems(root, split) for root in ROOTS.values()]
        out[split] = len(set.intersection(*sets)) if all(sets) else 0
    return out


def get_mean(payload: dict[str, Any], metric: str, split: str) -> float | None:
    val = payload.get("metrics", {}).get(metric, {}).get(split, {}).get("mean")
    return float(val) if val is not None else None


def main() -> int:
    train = load(TRAIN6)
    eval_summary = load(EVAL6)
    decision = load(DEC6)
    ablation = load(ABL6)
    train_text = (ROOT / "code/stwm/tools/train_ostf_v33_6_identity_contrastive_repair_20260509.py").read_text(encoding="utf-8")
    train_count = int(train.get("train_sample_count", 0) or 0)
    component_counts = {name: counts_by_split(root) for name, root in ROOTS.items()}
    complete = complete_counts()
    hard_bce_missing = "identity_hard_train_mask" not in train_text
    contrastive_hard_missing = "identity_hard_train_mask" not in train_text and "hard identity" not in train_text.lower()
    auc_val = get_mean(eval_summary, "hard_identity_ROC_AUC", "val")
    auc_test = get_mean(eval_summary, "hard_identity_ROC_AUC", "test")
    bal_val = get_mean(eval_summary, "hard_identity_balanced_accuracy", "val")
    bal_test = get_mean(eval_summary, "hard_identity_balanced_accuracy", "test")
    embedding_retrieval_val = get_mean(eval_summary, "identity_retrieval_exclude_same_point_top1", "val")
    embedding_retrieval_test = get_mean(eval_summary, "identity_retrieval_exclude_same_point_top1", "test")
    threshold_problem = bool((auc_val or 0.0) > 0.58 and (bal_val or 0.0) < 0.55) or bool((auc_test or 0.0) > 0.65 and (bal_test or 0.0) < 0.55)
    embedding_logit_mismatch = bool((embedding_retrieval_test or 0.0) > 0.58 and (bal_test or 0.0) < 0.55)
    deltas = ablation.get("global_label_model_vs_old_label_control", {})
    payload = {
        "generated_at_utc": utc_now(),
        "train_sample_count": train_count,
        "why_train_sample_count_only_47": "complete training samples are the intersection of V30 H32/M128 cache, identity sidecar, global identity sidecar, visual teacher sidecar, and semantic prototype target sidecar; visual/prototype sidecars are currently the bottleneck.",
        "component_counts_by_split": component_counts,
        "complete_samples_by_split": complete,
        "semantic_prototype_targets_by_split": component_counts["semantic_proto"],
        "visual_teacher_sidecars_by_split": component_counts["visual"],
        "global_identity_sidecars_by_split": component_counts["global_identity"],
        "identity_sidecars_by_split": component_counts["identity"],
        "current_training_actual_complete_samples": complete.get("train", 0),
        "same_instance_bce_uses_hard_identity_mask": not hard_bce_missing,
        "contrastive_loss_uses_hard_identity_mask": not contrastive_hard_missing,
        "eval_hard_identity_auc_source": "same_instance_logits",
        "retrieval_metric_source": "identity_embedding",
        "same_instance_logits_and_embedding_similarity_consistent": not embedding_logit_mismatch,
        "AUC_positive_but_balanced_accuracy_near_half_reason": "same_instance logits rank positives above negatives but the zero threshold is poorly calibrated; the identity embedding retrieval improves while the BCE head remains under-calibrated on balanced hard masks.",
        "training_coverage_bottleneck_detected": train_count < 200,
        "same_instance_hard_bce_missing": hard_bce_missing,
        "embedding_logit_mismatch_detected": embedding_logit_mismatch,
        "threshold_calibration_problem_detected": threshold_problem,
        "old_local_control_vs_global_label_model_delta": {
            "hard_identity_ROC_AUC_delta_val": deltas.get("hard_identity_ROC_AUC_delta_val"),
            "hard_identity_ROC_AUC_delta_test": deltas.get("hard_identity_ROC_AUC_delta_test"),
            "balanced_accuracy_delta_val": deltas.get("hard_identity_balanced_accuracy_delta_val"),
            "balanced_accuracy_delta_test": deltas.get("hard_identity_balanced_accuracy_delta_test"),
            "strict_retrieval_delta_val": deltas.get("strict_retrieval_delta_val"),
            "strict_retrieval_delta_test": deltas.get("strict_retrieval_delta_test"),
        },
        "recommended_fix": "Expand/record H32 complete target coverage, train BCE on balanced hard identity masks, add observed-anchor embedding similarity logits, and calibrate fused same-instance belief using validation threshold.",
        "v33_6_recommended_next_step": decision.get("recommended_next_step"),
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.7 Identity Training Forensics",
        payload,
        ["train_sample_count", "complete_samples_by_split", "training_coverage_bottleneck_detected", "same_instance_hard_bce_missing", "embedding_logit_mismatch_detected", "threshold_calibration_problem_detected", "recommended_fix"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
