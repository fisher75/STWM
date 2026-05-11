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

from stwm.modules.ostf_v34_semantic_trace_units import SemanticTraceUnitsWorldModelV34
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import binary_metrics
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_semantic_trace_units_20260510 import CKPT_DIR, SUMMARY as TRAIN_SUMMARY, collate_v34, make_loader


REPORT = ROOT / "reports/stwm_ostf_v34_1_unit_intervention_probe_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_1_UNIT_INTERVENTION_PROBE_20260511.md"


INTERVENTIONS = [
    "normal",
    "zero_observed_semantic_measurements",
    "shuffle_observed_semantic_measurements_across_points",
    "shuffle_observed_semantic_measurements_across_samples",
    "uniform_unit_assignment",
    "permute_unit_assignment",
    "drop_z_sem",
    "drop_unit_semantics",
    "randomize_unit_semantics",
    "pointwise_semantic_no_units",
    "copy_observed_semantic",
    "frozen_teacher_measurement_nearest",
]


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def forward_with_intervention(model: SemanticTraceUnitsWorldModelV34, bd: dict[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor]:
    if mode == "normal":
        return model(
            obs_points=bd["obs_points"],
            obs_vis=bd["obs_vis"],
            obs_conf=bd["obs_conf"],
            obs_semantic_measurements=bd["obs_semantic_measurements"],
            obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"],
            semantic_id=bd["semantic_id"],
        )
    with torch.no_grad():
        features, _ = model._v30_features(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], semantic_id=bd["semantic_id"])
        v30_out = model.v30(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], semantic_id=bd["semantic_id"])
    point_token = features["point_token"].detach()
    step_hidden = features["step_hidden"].detach()
    point_pred = v30_out["point_pred"].detach()
    rel_pred = ((point_pred - features["last_visible"][:, :, None, :]) / features["spread"].squeeze(2)[:, :, None, :]).clamp(-16.0, 16.0)
    sem = bd["obs_semantic_measurements"]
    mask = bd["obs_semantic_measurement_mask"]
    b, m = sem.shape[:2]
    if mode == "zero_observed_semantic_measurements":
        sem = torch.zeros_like(sem)
        mask = torch.zeros_like(mask)
    elif mode == "shuffle_observed_semantic_measurements_across_points":
        idx = torch.randperm(m, device=sem.device)
        sem = sem[:, idx]
        mask = mask[:, idx]
    elif mode == "shuffle_observed_semantic_measurements_across_samples" and b > 1:
        idx = torch.randperm(b, device=sem.device)
        sem = sem[idx]
        mask = mask[idx]
    assign, unit_sem = model.tokenizer(point_token, sem, mask)
    if mode == "uniform_unit_assignment":
        assign = torch.full_like(assign, 1.0 / assign.shape[-1])
    elif mode == "permute_unit_assignment":
        assign = assign[..., torch.randperm(assign.shape[-1], device=assign.device)]
    state = model.factorized_state(unit_sem)
    z_sem = state["z_sem"]
    if mode in {"drop_z_sem", "drop_unit_semantics"}:
        z_sem = torch.zeros_like(z_sem)
    elif mode == "randomize_unit_semantics":
        z_sem = torch.randn_like(z_sem)
    unit_point_sem = torch.einsum("bmu,bud->bmd", assign, z_sem)
    if mode in {"pointwise_semantic_no_units", "copy_observed_semantic", "frozen_teacher_measurement_nearest"}:
        obs = torch.nan_to_num(bd["obs_semantic_measurements"].float())
        obs_mask = bd["obs_semantic_measurement_mask"].float()
        pooled = (obs * obs_mask[..., None]).sum(dim=2) / obs_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        pred = torch.nn.functional.normalize(pooled[:, :, None, :].expand(-1, -1, point_pred.shape[2], -1), dim=-1)
        return {
            "future_semantic_belief": pred,
            "future_identity_belief": torch.zeros_like(v30_out["visibility_logits"]),
            "identity_embedding": torch.zeros((*point_pred.shape[:3], 64), device=point_pred.device),
            "semantic_uncertainty": torch.ones_like(v30_out["visibility_logits"]),
            "point_to_unit_assignment": assign,
            "visibility_logits": v30_out["visibility_logits"].detach(),
        }
    vis_prob = torch.sigmoid(v30_out["visibility_logits"].detach())
    hidden = model.rollout(model.handshake(point_token, step_hidden, unit_point_sem, rel_pred, vis_prob))
    out = model.readout(hidden)
    out.update({"point_to_unit_assignment": assign, "visibility_logits": v30_out["visibility_logits"].detach()})
    return out


def collect(split: str, args: argparse.Namespace, model: SemanticTraceUnitsWorldModelV34, device: torch.device, mode: str) -> dict[str, np.ndarray]:
    loader = make_loader(split, args, shuffle=False)
    keys = ["same_scores", "same_targets", "same_masks", "id_hard", "pred_sem", "target_sem", "sem_mask", "sem_hard", "obs_sem", "obs_mask", "unc", "assign", "point_to_instance"]
    rows: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_v34):
            bd = move_batch(batch, device)
            out = forward_with_intervention(model, bd, mode)
            rows["same_scores"].append(out["future_identity_belief"].detach().cpu().numpy())
            rows["same_targets"].append(bd["fut_same_instance_as_obs"].detach().cpu().numpy())
            rows["same_masks"].append(bd["fut_instance_available_mask"].detach().cpu().numpy())
            rows["id_hard"].append(bd["identity_hard_train_mask"].detach().cpu().numpy())
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


def unit_stats(cat: dict[str, np.ndarray]) -> dict[str, Any]:
    assign = cat["assign"]
    usage = assign.mean(axis=1).clip(1e-8)
    entropy = float(np.mean(-(assign.clip(1e-8) * np.log(assign.clip(1e-8))).sum(axis=-1) / np.log(assign.shape[-1])))
    effective = float(np.mean(np.exp(-(usage * np.log(usage)).sum(axis=-1))))
    hard = assign.argmax(axis=-1)
    pur = []
    for bi in range(assign.shape[0]):
        for u in range(assign.shape[-1]):
            idx = np.where(hard[bi] == u)[0]
            inst = cat["point_to_instance"][bi, idx]
            inst = inst[inst >= 0]
            if inst.size:
                _, cnt = np.unique(inst, return_counts=True)
                pur.append(float(cnt.max() / cnt.sum()))
    return {"assignment_entropy": entropy, "unit_usage_count": usage.mean(axis=0).tolist(), "effective_units": effective, "unit_dominant_instance_purity": float(np.mean(pur)) if pur else None, "unit_semantic_purity": None}


def metrics(cat: dict[str, np.ndarray], normal: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    hard = cat["id_hard"].astype(bool) & cat["same_masks"].astype(bool)
    idm = binary_metrics(cat["same_scores"], cat["same_targets"], hard)
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
    consistency = float(np.nanmean((_norm(cat["pred_sem"])[:, :, 1:] * _norm(cat["pred_sem"])[:, :, :-1]).sum(axis=-1)))
    output_delta = semantic_delta = identity_delta = 0.0
    if normal is not None:
        nmask = sem_mask & normal["sem_mask"].astype(bool)
        semantic_delta = float(np.mean(1.0 - (_norm(cat["pred_sem"]) * _norm(normal["pred_sem"])).sum(axis=-1)[nmask])) if nmask.any() else 0.0
        identity_delta = float(np.mean(np.abs(cat["same_scores"] - normal["same_scores"])))
        output_delta = semantic_delta + identity_delta
    out = {
        "hard_identity_ROC_AUC": idm["ROC_AUC"],
        "val_calibrated_balanced_accuracy": idm["balanced_accuracy"],
        "strict_retrieval": None,
        "stable_preservation_cosine": float(pred_cos[stable].mean()) if stable.any() else None,
        "stable_copy_cosine": float(copy_cos[stable].mean()) if stable.any() else None,
        "changed_semantic_signal": bool(pred_cos[changed].mean() > copy_cos[changed].mean() + 0.01) if changed.any() else False,
        "semantic_hard_signal": bool(pred_cos[hard_sem].mean() > copy_cos[hard_sem].mean() + 0.01) if hard_sem.any() else False,
        "teacher_agreement_weighted_top5": None,
        "semantic_belief_consistency": consistency,
        "semantic_uncertainty_quality": None,
        "output_delta_vs_normal": output_delta,
        "semantic_output_delta_vs_normal": semantic_delta,
        "identity_output_delta_vs_normal": identity_delta,
    }
    out.update(unit_stats(cat))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_semantic_trace_units_m128_h32_seed42_best.pt"))
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SemanticTraceUnitsWorldModelV34(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units).to(device)
    model.load_state_dict(ck["model"], strict=True)
    per: dict[str, Any] = {}
    for split in ("val", "test"):
        normal = collect(split, ckargs, model, device, "normal")
        per[split] = {}
        for mode in INTERVENTIONS:
            cat = normal if mode == "normal" else collect(split, ckargs, model, device, mode)
            per[split][mode] = metrics(cat, None if mode == "normal" else normal)
    dz = min(per["val"]["drop_z_sem"]["semantic_output_delta_vs_normal"], per["test"]["drop_z_sem"]["semantic_output_delta_vs_normal"])
    ua = min(per["val"]["uniform_unit_assignment"]["semantic_output_delta_vs_normal"], per["test"]["uniform_unit_assignment"]["semantic_output_delta_vs_normal"])
    sm = min(per["val"]["shuffle_observed_semantic_measurements_across_points"]["semantic_output_delta_vs_normal"], per["test"]["shuffle_observed_semantic_measurements_across_points"]["semantic_output_delta_vs_normal"])
    pointwise_not_weaker = bool((per["val"]["pointwise_semantic_no_units"]["stable_preservation_cosine"] or 0) >= (per["val"]["normal"]["stable_preservation_cosine"] or 0))
    copy_stronger = bool((per["val"]["copy_observed_semantic"]["stable_preservation_cosine"] or 0) >= (per["val"]["normal"]["stable_preservation_cosine"] or 0))
    decision = {
        "generated_at_utc": utc_now(),
        "units_not_load_bearing": bool(dz < 0.01 and ua < 0.01),
        "semantic_measurements_not_load_bearing": bool(sm < 0.01),
        "trace_units_not_better_than_pointwise": pointwise_not_weaker,
        "semantic_belief_not_world_model": copy_stronger,
        "units_load_bearing": bool(dz >= 0.01 or ua >= 0.01),
        "semantic_measurements_load_bearing": bool(sm >= 0.01),
        "trace_units_better_than_pointwise": bool(not pointwise_not_weaker),
        "drop_z_sem_delta_min": dz,
        "uniform_assignment_delta_min": ua,
        "shuffle_semantic_measurements_delta_min": sm,
    }
    payload = {"generated_at_utc": utc_now(), "interventions": per, "decision": decision}
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V34.1 Unit Intervention Probe", decision, ["units_not_load_bearing", "semantic_measurements_not_load_bearing", "trace_units_not_better_than_pointwise", "semantic_belief_not_world_model", "units_load_bearing", "semantic_measurements_load_bearing", "trace_units_better_than_pointwise", "drop_z_sem_delta_min", "uniform_assignment_delta_min", "shuffle_semantic_measurements_delta_min"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
