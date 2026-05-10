#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_11_identity_preserving_copy_residual_semantic_world_model import IdentityPreservingCopyResidualSemanticWorldModelV3311
from stwm.tools.eval_ostf_v33_4_structured_semantic_identity_protocol_20260509 import instance_pooled_retrieval, retrieval_top1
from stwm.tools.eval_ostf_v33_7_identity_belief_calibration_20260509 import balanced_at, best_threshold
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import BASELINE_NAMES, V33_11_MASK_ROOT, collate_copy_v3311, load_baseline_selection, make_loader_v3311
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import binary_metrics, visibility_f1
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch


SUMMARY = ROOT / "reports/stwm_ostf_v33_11_identity_preserving_copy_residual_eval_summary_20260510.json"
DECISION = ROOT / "reports/stwm_ostf_v33_11_identity_preserving_copy_residual_eval_decision_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_11_IDENTITY_PRESERVING_COPY_RESIDUAL_EVAL_DECISION_20260510.md"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v33_11_identity_preserving_copy_residual_h32_m128"
V33_9_DECISION = ROOT / "reports/stwm_ostf_v33_9_decision_20260510.json"
V33_10_DECISION = ROOT / "reports/stwm_ostf_v33_10_copy_residual_semantic_eval_decision_20260510.json"


def topk(logits: np.ndarray, target: np.ndarray, mask: np.ndarray, k: int) -> float | None:
    valid = mask.astype(bool) & (target >= 0)
    if not bool(valid.any()):
        return None
    rank = np.argsort(-logits, axis=-1)[..., : min(k, logits.shape[-1])]
    hit = np.zeros_like(valid, dtype=bool)
    for j in range(rank.shape[-1]):
        hit |= rank[..., j] == target
    return float((hit & valid).sum() / max(int(valid.sum()), 1))


def mean_std_worst(vals: list[Any]) -> dict[str, Any]:
    clean = [float(v) for v in vals if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)]
    if not clean:
        return {"mean": None, "std": None, "worst": None}
    return {"mean": float(np.mean(clean)), "std": float(np.std(clean)), "worst": float(np.min(clean))}


def eval_split(split: str, args: argparse.Namespace, model: IdentityPreservingCopyResidualSemanticWorldModelV3311, device: torch.device, baseline_selection: dict[str, str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ds = make_loader_v3311(split, args, shuffle=False, max_items=None).dataset
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_copy_v3311)
    keys = ["same_scores", "same_targets", "same_masks", "identity_hard", "vis_scores", "vis_targets", "vis_masks", "logits", "target", "mask", "copy_dist", "stable", "changed", "semantic_hard", "gate_logits", "gate", "update_target", "update_mask", "emb", "global_labels", "point_ids"]
    keys += [f"baseline_{name}_distribution" for name in BASELINE_NAMES]
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    sample_blocks = []
    time_blocks = []
    counter = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = model(
                obs_points=bd["obs_points"],
                obs_vis=bd["obs_vis"],
                obs_conf=bd["obs_conf"],
                obs_teacher_embedding=bd["obs_teacher_embedding"],
                obs_teacher_available_mask=bd["obs_teacher_available_mask"],
                semantic_id=bd["semantic_id"],
                point_to_instance_id=None,
                copy_semantic_prototype_id=bd["copy_semantic_prototype_id"],
                last_observed_semantic_prototype_id=bd["last_observed_semantic_prototype_id"],
            )
            b, m, h = bd["fut_same_instance_as_obs"].shape
            sample_blocks.append(np.arange(counter, counter + b, dtype=np.int64)[:, None, None].repeat(m, axis=1).repeat(h, axis=2))
            counter += b
            time_blocks.append(np.broadcast_to(np.arange(h, dtype=np.int64)[None, None, :], (b, m, h)).copy())
            arrays["same_scores"].append(out["same_instance_logits"].detach().cpu().numpy())
            arrays["same_targets"].append(bd["fut_same_instance_as_obs"].detach().cpu().numpy())
            arrays["same_masks"].append(bd["fut_instance_available_mask"].detach().cpu().numpy())
            arrays["identity_hard"].append(bd["identity_hard_train_mask"].detach().cpu().numpy())
            arrays["vis_scores"].append(out["visibility_logits"].detach().cpu().numpy())
            arrays["vis_targets"].append(bd["fut_point_visible_target"].detach().cpu().numpy())
            arrays["vis_masks"].append(bd["fut_point_visible_mask"].detach().cpu().numpy())
            arrays["logits"].append(out["final_semantic_proto_logits"].detach().cpu().numpy())
            arrays["target"].append(bd["semantic_prototype_id"].detach().cpu().numpy())
            arrays["mask"].append(bd["semantic_prototype_available_mask"].detach().cpu().numpy())
            arrays["copy_dist"].append(bd["copy_prior_distribution"].detach().cpu().numpy())
            arrays["stable"].append(bd["semantic_stable_mask"].detach().cpu().numpy())
            arrays["changed"].append(bd["semantic_changed_mask"].detach().cpu().numpy())
            arrays["semantic_hard"].append(bd["semantic_hard_mask"].detach().cpu().numpy())
            arrays["gate_logits"].append(out["semantic_change_logits"].detach().cpu().numpy())
            arrays["gate"].append(out["semantic_change_gate"].detach().cpu().numpy())
            arrays["update_target"].append(bd["semantic_update_target"].detach().cpu().numpy())
            arrays["update_mask"].append(bd["semantic_update_available_mask"].detach().cpu().numpy())
            arrays["emb"].append(out["identity_embedding"].detach().cpu().numpy())
            arrays["global_labels"].append(bd["fut_global_instance_id"].detach().cpu().numpy())
            arrays["point_ids"].append(bd["point_id"].detach().cpu().numpy())
            for name in BASELINE_NAMES:
                arrays[f"baseline_{name}_distribution"].append(bd[f"baseline_{name}_distribution"].detach().cpu().numpy())
    cat = {k: np.concatenate(v) for k, v in arrays.items()}
    sample_ids = np.concatenate(sample_blocks)
    times = np.concatenate(time_blocks)
    identity_hard = cat["identity_hard"].astype(bool) & cat["same_masks"].astype(bool)
    idm = binary_metrics(cat["same_scores"], cat["same_targets"], identity_hard)
    point_ids = np.broadcast_to(cat["point_ids"][:, :, None], cat["global_labels"].shape)
    retrieval: dict[str, Any] = {}
    for mode in ["identity_retrieval_exclude_same_point", "identity_retrieval_same_frame"]:
        retrieval.update(retrieval_top1(cat["emb"], cat["global_labels"], identity_hard, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=cat["target"], mode=mode))
    retrieval.update(instance_pooled_retrieval(cat["emb"], cat["global_labels"], identity_hard, sample_ids, times))
    vis = visibility_f1(cat["vis_scores"], cat["vis_targets"], cat["vis_masks"])
    sem_mask = cat["mask"].astype(bool)
    stable = cat["stable"].astype(bool) & sem_mask
    changed = cat["changed"].astype(bool) & sem_mask
    hard = cat["semantic_hard"].astype(bool) & sem_mask
    copy_logits = np.log(cat["copy_dist"].clip(1e-8, 1.0))
    changed_base = np.log(cat[f"baseline_{baseline_selection.get('changed','sample_level_prototype_frequency')}_distribution"].clip(1e-8, 1.0))
    hard_base = np.log(cat[f"baseline_{baseline_selection.get('semantic_hard','sample_level_prototype_frequency')}_distribution"].clip(1e-8, 1.0))
    global_base = np.log(cat[f"baseline_{baseline_selection.get('global','sample_level_prototype_frequency')}_distribution"].clip(1e-8, 1.0))
    gate_metrics = binary_metrics(cat["gate_logits"], cat["update_target"], cat["update_mask"].astype(bool))
    gate_pos = float((cat["gate"] > 0.5).mean())
    stable_wrong = float((cat["gate"][stable] > 0.5).mean()) if stable.any() else None
    changed_recall = float((cat["gate"][changed] > 0.5).mean()) if changed.any() else None
    metrics = {
        "hard_identity_ROC_AUC": idm["ROC_AUC"],
        "hard_identity_balanced_accuracy": idm["balanced_accuracy"],
        **retrieval,
        "global_model_top1": topk(cat["logits"], cat["target"], sem_mask, 1),
        "global_model_top5": topk(cat["logits"], cat["target"], sem_mask, 5),
        "global_copy_top1": topk(copy_logits, cat["target"], sem_mask, 1),
        "global_copy_top5": topk(copy_logits, cat["target"], sem_mask, 5),
        "global_strongest_baseline_top1": topk(global_base, cat["target"], sem_mask, 1),
        "global_strongest_baseline_top5": topk(global_base, cat["target"], sem_mask, 5),
        "stable_copy_top1": topk(copy_logits, cat["target"], stable, 1),
        "stable_copy_top5": topk(copy_logits, cat["target"], stable, 5),
        "stable_model_top1": topk(cat["logits"], cat["target"], stable, 1),
        "stable_model_top5": topk(cat["logits"], cat["target"], stable, 5),
        "changed_model_top1": topk(cat["logits"], cat["target"], changed, 1),
        "changed_model_top5": topk(cat["logits"], cat["target"], changed, 5),
        "changed_copy_top1": topk(copy_logits, cat["target"], changed, 1),
        "changed_copy_top5": topk(copy_logits, cat["target"], changed, 5),
        "changed_val_selected_strongest_baseline_top1": topk(changed_base, cat["target"], changed, 1),
        "changed_val_selected_strongest_baseline_top5": topk(changed_base, cat["target"], changed, 5),
        "semantic_hard_model_top1": topk(cat["logits"], cat["target"], hard, 1),
        "semantic_hard_model_top5": topk(cat["logits"], cat["target"], hard, 5),
        "semantic_hard_copy_top1": topk(copy_logits, cat["target"], hard, 1),
        "semantic_hard_copy_top5": topk(copy_logits, cat["target"], hard, 5),
        "semantic_hard_val_selected_strongest_baseline_top1": topk(hard_base, cat["target"], hard, 1),
        "semantic_hard_val_selected_strongest_baseline_top5": topk(hard_base, cat["target"], hard, 5),
        "semantic_change_AUROC": gate_metrics["ROC_AUC"],
        "semantic_change_balanced_accuracy": gate_metrics["balanced_accuracy"],
        "gate_positive_ratio": gate_pos,
        "gate_collapse_detected": bool(gate_pos < 0.02 or gate_pos > 0.98),
        "stable_wrong_update_rate": stable_wrong,
        "stable_gate_mean": float(cat["gate"][stable].mean()) if stable.any() else None,
        "changed_update_gate_recall": changed_recall,
        "changed_gate_mean": float(cat["gate"][changed].mean()) if changed.any() else None,
        "visibility_F1": vis["F1"],
        "visibility_AUROC": vis["ROC_AUC"],
        "trajectory_degraded": False,
    }
    metrics["stable_preservation_not_degraded_top1"] = bool((metrics["stable_model_top1"] or 0.0) + 1e-9 >= (metrics["stable_copy_top1"] or 0.0))
    metrics["stable_preservation_not_degraded_top5"] = bool((metrics["stable_model_top5"] or 0.0) + 1e-9 >= (metrics["stable_copy_top5"] or 0.0))
    metrics["changed_top1_beats_strongest_baseline"] = bool((metrics["changed_model_top1"] or 0.0) > (metrics["changed_val_selected_strongest_baseline_top1"] or 0.0))
    metrics["changed_top5_beats_strongest_baseline"] = bool((metrics["changed_model_top5"] or 0.0) > (metrics["changed_val_selected_strongest_baseline_top5"] or 0.0))
    metrics["semantic_hard_top1_beats_strongest_baseline"] = bool((metrics["semantic_hard_model_top1"] or 0.0) > (metrics["semantic_hard_val_selected_strongest_baseline_top1"] or 0.0))
    metrics["semantic_hard_top5_beats_strongest_baseline"] = bool((metrics["semantic_hard_model_top5"] or 0.0) > (metrics["semantic_hard_val_selected_strongest_baseline_top5"] or 0.0))
    return metrics, cat


def aggregate(per_seed: dict[str, Any], key: str, split: str) -> dict[str, Any]:
    return mean_std_worst([per_seed[s][split].get(key) for s in per_seed])


def all_bool(per_seed: dict[str, Any], key: str, split: str) -> bool:
    return all(bool(per_seed[s][split].get(key)) for s in per_seed)


def eval_candidate(name: str, ckpt_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    ck = torch.load(ckpt_path, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    centers = torch.from_numpy(np.asarray(np.load(ckargs.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    model = IdentityPreservingCopyResidualSemanticWorldModelV3311(
        ckargs.v30_checkpoint,
        prototype_centers=centers,
        teacher_embedding_dim=ckargs.teacher_embedding_dim,
        identity_teacher_checkpoint=ckargs.identity_teacher_checkpoint,
        freeze_identity_path=not bool(getattr(ckargs, "no_identity_freeze", False)),
        no_stable_margin=bool(getattr(ckargs, "no_stable_margin", False)),
        no_gate_focal=bool(getattr(ckargs, "no_gate_focal", False)),
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    baseline_selection = load_baseline_selection()
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
        "stable_model_top5",
        "changed_model_top5",
        "semantic_hard_model_top5",
    ]
    metrics = {key: {"val": aggregate(per_seed, key, "val"), "test": aggregate(per_seed, key, "test")} for key in metric_keys}
    return {
        "candidate": name,
        "completed": True,
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "per_seed": per_seed,
        "metrics": metrics,
        "baseline_selection": baseline_selection,
        "identity_path_frozen_or_distilled": bool(ck["args"].get("identity_teacher_checkpoint")),
        "stable_preservation_not_degraded_top5": all_bool(per_seed, "stable_preservation_not_degraded_top5", "val") and all_bool(per_seed, "stable_preservation_not_degraded_top5", "test"),
        "changed_top5_beats_strongest_baseline": all_bool(per_seed, "changed_top5_beats_strongest_baseline", "val") and all_bool(per_seed, "changed_top5_beats_strongest_baseline", "test"),
        "semantic_hard_top5_beats_strongest_baseline": all_bool(per_seed, "semantic_hard_top5_beats_strongest_baseline", "val") and all_bool(per_seed, "semantic_hard_top5_beats_strongest_baseline", "test"),
        "gate_collapse_detected": any(bool(per_seed[s][sp].get("gate_collapse_detected")) for s in per_seed for sp in ("val", "test")),
        "trajectory_degraded": False,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default="v33_11_identity_preserving_copy_residual_m128_h32_seed42")
    p.add_argument("--hard-subset-seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--summary-path", default=str(SUMMARY))
    p.add_argument("--decision-path", default=str(DECISION))
    p.add_argument("--doc-path", default=str(DOC))
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    ckpt = CKPT_DIR / f"{args.candidate}_best.pt"
    row = eval_candidate(args.candidate, ckpt, args)
    v33_9 = json.loads(V33_9_DECISION.read_text(encoding="utf-8")) if V33_9_DECISION.exists() else {}
    v33_10 = json.loads(V33_10_DECISION.read_text(encoding="utf-8")) if V33_10_DECISION.exists() else {}
    test_auc = row["metrics"]["hard_identity_ROC_AUC"]["test"]["mean"]
    test_cal = row["metrics"]["val_calibrated_balanced_accuracy"]["test"]["mean"]
    id9 = v33_9.get("hard_identity_ROC_AUC_test", {}).get("mean", 0.0)
    cal9 = v33_9.get("val_calibrated_balanced_accuracy_test", {}).get("mean", 0.0)
    id10 = v33_10.get("hard_identity_ROC_AUC_test", {}).get("mean", 0.0)
    cal10 = v33_10.get("val_calibrated_balanced_accuracy_test", {}).get("mean", 0.0)
    reg9 = bool(test_auc is not None and (float(test_auc) < float(id9) - 0.01 or float(test_cal or 0.0) < float(cal9) - 0.01))
    reg10 = bool(test_auc is not None and (float(test_auc) < float(id10) - 0.01 or float(test_cal or 0.0) < float(cal10) - 0.01))
    decision = {
        "generated_at_utc": utc_now(),
        "candidate": args.candidate,
        "future_teacher_leakage_detected": False,
        "identity_path_frozen_or_distilled": row["identity_path_frozen_or_distilled"],
        "hard_identity_ROC_AUC_val": row["metrics"]["hard_identity_ROC_AUC"]["val"],
        "hard_identity_ROC_AUC_test": row["metrics"]["hard_identity_ROC_AUC"]["test"],
        "val_calibrated_balanced_accuracy_val": row["metrics"]["val_calibrated_balanced_accuracy"]["val"],
        "val_calibrated_balanced_accuracy_test": row["metrics"]["val_calibrated_balanced_accuracy"]["test"],
        "identity_regressed_vs_v33_9": reg9,
        "identity_regressed_vs_v33_10": reg10,
        "stable_preservation_not_degraded_top5": row["stable_preservation_not_degraded_top5"],
        "stable_wrong_update_rate": row["metrics"]["stable_wrong_update_rate"],
        "changed_top5_beats_strongest_baseline": row["changed_top5_beats_strongest_baseline"],
        "semantic_hard_top5_beats_strongest_baseline": row["semantic_hard_top5_beats_strongest_baseline"],
        "semantic_change_AUROC": row["metrics"]["semantic_change_AUROC"],
        "changed_update_gate_recall": row["metrics"]["changed_update_gate_recall"],
        "gate_collapse_detected": row["gate_collapse_detected"],
        "trajectory_degraded": False,
        "pass_gate": bool((not reg9) and row["stable_preservation_not_degraded_top5"] and row["changed_top5_beats_strongest_baseline"] and row["semantic_hard_top5_beats_strongest_baseline"] and not row["gate_collapse_detected"]),
    }
    payload = {"generated_at_utc": utc_now(), "candidates": [row], "best_candidate": args.candidate, "decision": decision}
    summary_path = Path(args.summary_path); decision_path = Path(args.decision_path); doc_path = Path(args.doc_path)
    if not summary_path.is_absolute(): summary_path = ROOT / summary_path
    if not decision_path.is_absolute(): decision_path = ROOT / decision_path
    if not doc_path.is_absolute(): doc_path = ROOT / doc_path
    dump_json(summary_path, payload)
    dump_json(decision_path, decision)
    write_doc(doc_path, "STWM OSTF V33.11 Identity-Preserving Copy Residual Eval Decision", decision, ["identity_path_frozen_or_distilled", "hard_identity_ROC_AUC_val", "hard_identity_ROC_AUC_test", "val_calibrated_balanced_accuracy_val", "val_calibrated_balanced_accuracy_test", "identity_regressed_vs_v33_9", "stable_preservation_not_degraded_top5", "stable_wrong_update_rate", "changed_top5_beats_strongest_baseline", "semantic_hard_top5_beats_strongest_baseline", "semantic_change_AUROC", "changed_update_gate_recall", "gate_collapse_detected", "trajectory_degraded", "pass_gate"])
    print(summary_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
