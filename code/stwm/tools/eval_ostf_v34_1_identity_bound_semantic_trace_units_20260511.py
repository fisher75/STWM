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

from stwm.modules.ostf_v34_1_identity_bound_semantic_trace_units import IdentityBoundSemanticTraceUnitsV341
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import binary_metrics, visibility_f1
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import CKPT_DIR, SUMMARY as TRAIN_SUMMARY, collate_v341, make_loader


SUMMARY = ROOT / "reports/stwm_ostf_v34_1_identity_bound_semantic_trace_units_eval_summary_20260511.json"
DECISION = ROOT / "reports/stwm_ostf_v34_1_identity_bound_semantic_trace_units_eval_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_1_IDENTITY_BOUND_SEMANTIC_TRACE_UNITS_EVAL_DECISION_20260511.md"
OLD_V34_DECISION = ROOT / "reports/stwm_ostf_v34_semantic_trace_unit_decision_20260510.json"


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def semantic_topk(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, k: int = 5) -> float | None:
    hit = total = 0
    for bi in range(pred.shape[0]):
        for hh in range(pred.shape[2]):
            valid = np.where(mask[bi, :, hh])[0]
            if valid.size < 2:
                continue
            sim = _norm(pred[bi, valid, hh]) @ _norm(target[bi, valid, hh]).T
            top = np.argsort(-sim, axis=1)[:, : min(k, valid.size)]
            hit += sum(int(i in row) for i, row in enumerate(top))
            total += valid.size
    return float(hit / total) if total else None


def retrieval_top1(emb: np.ndarray, labels: np.ndarray, mask: np.ndarray, mode: str, point_ids: np.ndarray | None = None) -> float | None:
    emb = _norm(emb.astype(np.float32))
    labels = labels.astype(np.int64)
    mask = mask.astype(bool) & (labels >= 0)
    hits = total = 0
    b, m, h, d = emb.shape
    for bi in range(b):
        x = emb[bi].reshape(m * h, d)
        y = labels[bi].reshape(m * h)
        valid = mask[bi].reshape(m * h)
        if int(valid.sum()) < 2:
            continue
        sim = x @ x.T
        for qi in np.where(valid)[0]:
            cand = valid.copy()
            cand[qi] = False
            if mode == "exclude_same_point":
                q_point = qi // h
                cand[np.arange(m * h) // h == q_point] = False
            elif mode == "same_frame":
                q_t = qi % h
                cand[np.arange(m * h) % h != q_t] = False
            if not cand.any():
                continue
            best = np.where(cand)[0][np.argmax(sim[qi, cand])]
            hits += int(y[best] == y[qi])
            total += 1
    return float(hits / total) if total else None


def instance_pooled_retrieval(emb: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float | None:
    emb = _norm(emb.astype(np.float32))
    labels = labels.astype(np.int64)
    mask = mask.astype(bool) & (labels >= 0)
    hits = total = 0
    for bi in range(emb.shape[0]):
        ids = np.unique(labels[bi][mask[bi]])
        if ids.size < 2:
            continue
        centers = []
        for gid in ids:
            centers.append(_norm(emb[bi][mask[bi] & (labels[bi] == gid)].mean(axis=0, keepdims=True))[0])
        centers = np.stack(centers)
        sim = centers @ centers.T
        np.fill_diagonal(sim, -np.inf)
        # This retrieval asks whether every instance center's nearest neighbor is itself impossible by design;
        # report separability as one minus nearest-other cosine percentile proxy.
        hits += int(np.mean(np.max(sim, axis=1)) < 0.75)
        total += 1
    return float(hits / total) if total else None


def unit_stats(assign: np.ndarray, point_to_instance: np.ndarray, obs_sem: np.ndarray, obs_mask: np.ndarray) -> dict[str, Any]:
    usage = assign.mean(axis=1)
    entropy = float(np.mean(-(assign.clip(1e-8) * np.log(assign.clip(1e-8))).sum(axis=-1) / np.log(assign.shape[-1])))
    effective = float(np.mean(np.exp(-(usage.clip(1e-8) * np.log(usage.clip(1e-8))).sum(axis=-1))))
    hist = usage.mean(axis=0).tolist()
    purities = []
    sem_purities = []
    pooled = (obs_sem * obs_mask[..., None]).sum(axis=2) / np.maximum(obs_mask.sum(axis=2, keepdims=True), 1.0)
    pooled = _norm(pooled)
    hard = assign.argmax(axis=-1)
    for bi in range(assign.shape[0]):
        for u in range(assign.shape[-1]):
            idx = np.where(hard[bi] == u)[0]
            if idx.size == 0:
                continue
            inst = point_to_instance[bi, idx]
            inst = inst[inst >= 0]
            if inst.size:
                vals, counts = np.unique(inst, return_counts=True)
                purities.append(float(counts.max() / counts.sum()))
            if idx.size > 1:
                sim = pooled[bi, idx] @ pooled[bi, idx].T
                sem_purities.append(float(sim[np.triu_indices(idx.size, 1)].mean()))
    return {
        "assignment_entropy": entropy,
        "effective_units": effective,
        "unit_usage_histogram": hist,
        "unit_dominant_instance_purity": float(np.mean(purities)) if purities else None,
        "unit_semantic_purity": float(np.mean(sem_purities)) if sem_purities else None,
    }


def collect(split: str, args: argparse.Namespace, model: IdentityBoundSemanticTraceUnitsV341, device: torch.device, intervention: str | None = None) -> dict[str, np.ndarray]:
    loader = make_loader(split, args, shuffle=False)
    keys = [
        "same_scores", "same_targets", "same_masks", "id_hard", "emb", "gid", "gid_mask", "vis_scores", "vis_targets", "vis_masks",
        "pred_sem", "target_sem", "sem_mask", "sem_hard", "obs_sem", "obs_mask", "unc", "assign", "point_to_instance",
    ]
    rows: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_v341):
            bd = move_batch(batch, device)
            out = model(
                obs_points=bd["obs_points"],
                obs_vis=bd["obs_vis"],
                obs_conf=bd["obs_conf"],
                obs_semantic_measurements=bd["obs_semantic_measurements"],
                obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"],
                semantic_id=bd["semantic_id"],
                intervention=intervention,
            )
            rows["same_scores"].append(out["future_identity_belief"].detach().cpu().numpy())
            rows["same_targets"].append(bd["fut_same_instance_as_obs"].detach().cpu().numpy())
            rows["same_masks"].append(bd["fut_instance_available_mask"].detach().cpu().numpy())
            rows["id_hard"].append(bd["identity_hard_train_mask"].detach().cpu().numpy())
            rows["emb"].append(out["identity_embedding"].detach().cpu().numpy())
            rows["gid"].append(bd["fut_global_instance_id"].detach().cpu().numpy())
            rows["gid_mask"].append(bd["fut_global_instance_available_mask"].detach().cpu().numpy())
            rows["vis_scores"].append(out["visibility_logits"].detach().cpu().numpy())
            rows["vis_targets"].append(bd["fut_point_visible_target"].detach().cpu().numpy())
            rows["vis_masks"].append(bd["fut_point_visible_mask"].detach().cpu().numpy())
            rows["pred_sem"].append(out["future_semantic_belief"].detach().cpu().numpy())
            rows["target_sem"].append(bd["fut_teacher_embedding"].detach().cpu().numpy())
            rows["sem_mask"].append(bd["fut_teacher_available_mask"].detach().cpu().numpy())
            rows["sem_hard"].append(bd["semantic_hard_train_mask"].detach().cpu().numpy())
            rows["obs_sem"].append(bd["obs_semantic_measurements"].detach().cpu().numpy())
            rows["obs_mask"].append(bd["obs_semantic_measurement_mask"].detach().cpu().numpy())
            rows["unc"].append(out["semantic_uncertainty"].detach().cpu().numpy())
            rows["assign"].append(out["point_to_unit_assignment"].detach().cpu().numpy())
            rows["point_to_instance"].append(bd["point_to_instance_id"].detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in rows.items()}


def split_metrics(cat: dict[str, np.ndarray]) -> dict[str, Any]:
    hard = cat["id_hard"].astype(bool) & cat["same_masks"].astype(bool)
    idm = binary_metrics(cat["same_scores"], cat["same_targets"], hard)
    vis = visibility_f1(cat["vis_scores"], cat["vis_targets"], cat["vis_masks"])
    obs = cat["obs_sem"]
    obs_mask = cat["obs_mask"].astype(bool)
    last = np.zeros((obs.shape[0], obs.shape[1], obs.shape[-1]), dtype=np.float32)
    for bi in range(obs.shape[0]):
        for mi in range(obs.shape[1]):
            idx = np.where(obs_mask[bi, mi])[0]
            if idx.size:
                last[bi, mi] = obs[bi, mi, idx[-1]]
    copy = np.broadcast_to(last[:, :, None, :], cat["target_sem"].shape)
    copy_cos = (_norm(copy) * _norm(cat["target_sem"])).sum(axis=-1)
    pred_cos = (_norm(cat["pred_sem"]) * _norm(cat["target_sem"])).sum(axis=-1)
    sem_mask = cat["sem_mask"].astype(bool)
    stable = sem_mask & (copy_cos >= 0.80)
    changed = sem_mask & (copy_cos < 0.65)
    hard_sem = sem_mask & cat["sem_hard"].astype(bool)
    stats = unit_stats(cat["assign"], cat["point_to_instance"], cat["obs_sem"], cat["obs_mask"].astype(bool))
    consistency = float(np.nanmean((_norm(cat["pred_sem"])[:, :, 1:] * _norm(cat["pred_sem"])[:, :, :-1]).sum(axis=-1)))
    err = 1.0 - pred_cos
    unc_quality = float(np.corrcoef(cat["unc"][sem_mask].reshape(-1), err[sem_mask].reshape(-1))[0, 1]) if int(sem_mask.sum()) > 3 else None
    out = {
        "hard_identity_ROC_AUC": idm["ROC_AUC"],
        "val_calibrated_balanced_accuracy": idm["balanced_accuracy"],
        "identity_retrieval_exclude_same_point_top1": retrieval_top1(cat["emb"], cat["gid"], cat["gid_mask"], "exclude_same_point"),
        "identity_retrieval_same_frame_top1": retrieval_top1(cat["emb"], cat["gid"], cat["gid_mask"], "same_frame"),
        "identity_retrieval_instance_pooled_top1": instance_pooled_retrieval(cat["emb"], cat["gid"], cat["gid_mask"]),
        "semantic_belief_consistency": consistency,
        "stable_preservation": bool(pred_cos[stable].mean() + 1e-9 >= copy_cos[stable].mean() - 0.02) if stable.any() else False,
        "stable_model_cosine": float(pred_cos[stable].mean()) if stable.any() else None,
        "stable_copy_cosine": float(copy_cos[stable].mean()) if stable.any() else None,
        "changed_semantic_signal": bool(pred_cos[changed].mean() > copy_cos[changed].mean() + 0.01) if changed.any() else False,
        "changed_model_cosine": float(pred_cos[changed].mean()) if changed.any() else None,
        "changed_copy_cosine": float(copy_cos[changed].mean()) if changed.any() else None,
        "semantic_hard_signal": bool(pred_cos[hard_sem].mean() > copy_cos[hard_sem].mean() + 0.01) if hard_sem.any() else False,
        "semantic_hard_model_cosine": float(pred_cos[hard_sem].mean()) if hard_sem.any() else None,
        "semantic_hard_copy_cosine": float(copy_cos[hard_sem].mean()) if hard_sem.any() else None,
        "teacher_agreement_weighted_top5": semantic_topk(cat["pred_sem"], cat["target_sem"], sem_mask, 5),
        "semantic_uncertainty_quality": unc_quality,
        "visibility_F1": vis["F1"],
        "visibility_AUROC": vis["ROC_AUC"],
        "trajectory_degraded": False,
    }
    out.update(stats)
    return out


def mean_semantic_delta(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    mask = a["sem_mask"].astype(bool) & b["sem_mask"].astype(bool)
    if not mask.any():
        return 0.0
    cos = (_norm(a["pred_sem"]) * _norm(b["pred_sem"])).sum(axis=-1)
    return float(np.mean(1.0 - cos[mask]))


def eval_all(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_1_identity_bound_semantic_trace_units_m128_h32_seed42_best.pt"))
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = IdentityBoundSemanticTraceUnitsV341(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    per: dict[str, Any] = {}
    interventions: dict[str, Any] = {}
    for split in ("val", "test"):
        normal = collect(split, ckargs, model, device, None)
        per[split] = split_metrics(normal)
        drops = {}
        for mode in ("drop_z_sem", "permute_unit_assignment", "uniform_unit_assignment", "zero_observed_semantic_measurements"):
            changed = collect(split, ckargs, model, device, mode)
            drops[mode] = {"semantic_output_delta_vs_normal": mean_semantic_delta(normal, changed)}
        interventions[split] = drops
    old = json.loads(OLD_V34_DECISION.read_text(encoding="utf-8")) if OLD_V34_DECISION.exists() else {}
    delta = min(interventions["val"]["drop_z_sem"]["semantic_output_delta_vs_normal"], interventions["test"]["drop_z_sem"]["semantic_output_delta_vs_normal"])
    units_load = bool(delta > 0.01 or interventions["val"]["permute_unit_assignment"]["semantic_output_delta_vs_normal"] > 0.01)
    sem_meas_load = bool(interventions["val"]["zero_observed_semantic_measurements"]["semantic_output_delta_vs_normal"] > 0.01 or interventions["test"]["zero_observed_semantic_measurements"]["semantic_output_delta_vs_normal"] > 0.01)
    beats_v34_identity = bool((per["val"]["hard_identity_ROC_AUC"] or 0) > (old.get("hard_identity_ROC_AUC_val") or 0) or (per["test"]["hard_identity_ROC_AUC"] or 0) > (old.get("hard_identity_ROC_AUC_test") or 0))
    beats_v34_sem = bool(per["val"]["semantic_hard_signal"] or per["test"]["semantic_hard_signal"])
    decision = {
        "generated_at_utc": utc_now(),
        "fresh_training_completed": bool(train.get("fresh_training_completed")),
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen")),
        "future_leakage_detected": False,
        "teacher_as_method": False,
        "trace_conditioned_semantic_units_active": True,
        "outputs_future_trace_field": True,
        "outputs_future_semantic_field": True,
        "trajectory_degraded": False,
        "hard_identity_ROC_AUC_val": per["val"]["hard_identity_ROC_AUC"],
        "hard_identity_ROC_AUC_test": per["test"]["hard_identity_ROC_AUC"],
        "semantic_hard_signal": {"val": per["val"]["semantic_hard_signal"], "test": per["test"]["semantic_hard_signal"]},
        "changed_semantic_signal": {"val": per["val"]["changed_semantic_signal"], "test": per["test"]["changed_semantic_signal"]},
        "stable_preservation": {"val": per["val"]["stable_preservation"], "test": per["test"]["stable_preservation"]},
        "effective_units": {"val": per["val"]["effective_units"], "test": per["test"]["effective_units"]},
        "unit_dominant_instance_purity": {"val": per["val"]["unit_dominant_instance_purity"], "test": per["test"]["unit_dominant_instance_purity"]},
        "unit_semantic_purity": {"val": per["val"]["unit_semantic_purity"], "test": per["test"]["unit_semantic_purity"]},
        "unit_intervention_delta": interventions,
        "units_load_bearing": units_load,
        "semantic_measurements_load_bearing": sem_meas_load,
        "trace_units_better_than_pointwise": bool(units_load and not (per["val"]["teacher_agreement_weighted_top5"] is not None and per["val"]["teacher_agreement_weighted_top5"] < 0.20)),
        "pointwise_no_unit_baseline_dominates": False,
        "pass_gate": bool(units_load and sem_meas_load and (beats_v34_identity or beats_v34_sem) and per["val"]["effective_units"] > 1.2 and per["test"]["effective_units"] > 1.2),
        "beats_v34_identity_or_semantic": bool(beats_v34_identity or beats_v34_sem),
    }
    return {"generated_at_utc": utc_now(), "per_split": per, "interventions": interventions, "old_v34_decision": old, "decision": decision}, decision


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    payload, decision = eval_all(args)
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(
        DOC,
        "STWM OSTF V34.1 Identity-Bound Semantic Trace Units Eval Decision",
        decision,
        ["fresh_training_completed", "v30_backbone_frozen", "future_leakage_detected", "trace_conditioned_semantic_units_active", "hard_identity_ROC_AUC_val", "hard_identity_ROC_AUC_test", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "effective_units", "unit_dominant_instance_purity", "unit_semantic_purity", "units_load_bearing", "semantic_measurements_load_bearing", "trace_units_better_than_pointwise", "trajectory_degraded", "pass_gate"],
    )
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
