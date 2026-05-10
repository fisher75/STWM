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

from stwm.modules.ostf_v33_12_copy_conservative_semantic_world_model import CopyConservativeSemanticWorldModelV3312
from stwm.modules.ostf_v33_13_gate_repaired_copy_semantic_world_model import GateRepairedCopySemanticWorldModelV3313
from stwm.tools.eval_ostf_v33_11_identity_preserving_copy_residual_semantic_20260510 import topk
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import V33_11_MASK_ROOT, collate_copy_v3311, make_loader_v3311
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch


REPORT = ROOT / "reports/stwm_ostf_v33_13_gate_calibration_sweep_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_13_GATE_CALIBRATION_SWEEP_20260510.md"
V3312_TRAIN = ROOT / "reports/stwm_ostf_v33_12_copy_conservative_semantic_train_summary_20260510.json"
V3313_TRAIN = ROOT / "reports/stwm_ostf_v33_13_gate_repaired_train_summary_20260510.json"


def load_train(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def softmax(x: np.ndarray) -> np.ndarray:
    y = x - x.max(axis=-1, keepdims=True)
    y = np.exp(y)
    return y / y.sum(axis=-1, keepdims=True).clip(1e-8)


def logmix(copy_logits: np.ndarray, resid_logits: np.ndarray, gate: np.ndarray) -> np.ndarray:
    cp = softmax(copy_logits)
    rp = softmax(resid_logits)
    return np.log(((1.0 - gate[..., None]) * cp + gate[..., None] * rp).clip(1e-8, 1.0))


def collect(model: torch.nn.Module, ckargs: argparse.Namespace, split: str, device: torch.device) -> dict[str, np.ndarray]:
    ckargs.hard_train_mask_manifest = str(V33_11_MASK_ROOT / "H32_M128_seed42.json")
    loader = make_loader_v3311(split, ckargs, shuffle=False, max_items=None)
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ["copy", "resid", "final", "target", "mask", "stable", "changed", "hard", "raw", "prob"]}
    with torch.no_grad():
        for batch in DataLoader(loader.dataset, batch_size=ckargs.batch_size, shuffle=False, num_workers=ckargs.num_workers, collate_fn=collate_copy_v3311):
            bd = move_batch(batch, device)
            out = model(
                obs_points=bd["obs_points"],
                obs_vis=bd["obs_vis"],
                obs_conf=bd["obs_conf"],
                obs_teacher_embedding=bd["obs_teacher_embedding"],
                obs_teacher_available_mask=bd["obs_teacher_available_mask"],
                semantic_id=bd["semantic_id"],
                copy_semantic_prototype_id=bd["copy_semantic_prototype_id"],
                last_observed_semantic_prototype_id=bd["last_observed_semantic_prototype_id"],
            )
            raw = out.get("semantic_change_logits_raw", out["semantic_change_logits"])
            prob = out.get("semantic_change_prob", torch.sigmoid(raw))
            arrays["copy"].append(out["copy_prior_semantic_logits"].detach().cpu().numpy())
            arrays["resid"].append(out["semantic_residual_logits"].detach().cpu().numpy())
            arrays["final"].append(out["final_semantic_proto_logits"].detach().cpu().numpy())
            arrays["target"].append(bd["semantic_prototype_id"].detach().cpu().numpy())
            arrays["mask"].append(bd["semantic_prototype_available_mask"].detach().cpu().numpy())
            arrays["stable"].append(bd["semantic_stable_mask"].detach().cpu().numpy())
            arrays["changed"].append(bd["semantic_changed_mask"].detach().cpu().numpy())
            arrays["hard"].append(bd["semantic_hard_mask"].detach().cpu().numpy())
            arrays["raw"].append(raw.detach().cpu().numpy())
            arrays["prob"].append(prob.detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in arrays.items()}


def eval_mode(logits: np.ndarray, cat: dict[str, np.ndarray]) -> dict[str, Any]:
    target = cat["target"]
    mask = cat["mask"].astype(bool)
    stable = cat["stable"].astype(bool) & mask
    changed = cat["changed"].astype(bool) & mask
    hard = cat["hard"].astype(bool) & mask
    copy = cat["copy"]
    sample = np.zeros_like(copy)
    sample[..., 0] = 0.0
    out = {
        "global_top1": topk(logits, target, mask, 1),
        "global_top5": topk(logits, target, mask, 5),
        "stable_top1": topk(logits, target, stable, 1),
        "stable_top5": topk(logits, target, stable, 5),
        "stable_copy_top5": topk(copy, target, stable, 5),
        "changed_top1": topk(logits, target, changed, 1),
        "changed_top5": topk(logits, target, changed, 5),
        "semantic_hard_top1": topk(logits, target, hard, 1),
        "semantic_hard_top5": topk(logits, target, hard, 5),
        "stable_wrong_update_rate": float((softmax(logits).argmax(axis=-1)[stable] != target[stable]).mean()) if stable.any() else None,
        "gate_positive_ratio": None,
    }
    return out


def run_model(name: str, ckpt_path: Path, repaired: bool, args: argparse.Namespace) -> dict[str, Any]:
    ck = torch.load(ckpt_path, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    centers = torch.from_numpy(np.asarray(np.load(ckargs.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    cls = GateRepairedCopySemanticWorldModelV3313 if repaired else CopyConservativeSemanticWorldModelV3312
    kwargs = dict(
        v30_checkpoint_path=ckargs.v30_checkpoint,
        prototype_centers=centers,
        teacher_embedding_dim=ckargs.teacher_embedding_dim,
        identity_teacher_checkpoint=ckargs.identity_teacher_checkpoint,
        gate_threshold=float(getattr(ckargs, "gate_threshold", 0.10)),
        freeze_identity_path=True,
    )
    if not repaired:
        kwargs["residual_update_budget"] = float(getattr(ckargs, "residual_update_budget", 0.35))
    model = cls(**kwargs).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    out: dict[str, Any] = {"checkpoint_path": str(ckpt_path.relative_to(ROOT)), "splits": {}}
    thresholds = [0.00, 0.01, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75]
    val_scores: dict[float, float] = {}
    for split in ("val", "test"):
        cat = collect(model, ckargs, split, device)
        modes: dict[str, dict[str, Any]] = {}
        modes["copy_only"] = eval_mode(cat["copy"], cat)
        modes["residual_only"] = eval_mode(cat["resid"], cat)
        modes["learned_gate_current_threshold"] = eval_mode(cat["final"], cat)
        modes["oracle_gate"] = eval_mode(np.where((cat["changed"].astype(bool) | cat["hard"].astype(bool))[..., None], cat["resid"], cat["copy"]), cat)
        for thr in thresholds:
            gate = ((cat["prob"] - thr) / max(1.0 - thr, 1e-6)).clip(0.0, 1.0)
            m = eval_mode(logmix(cat["copy"], cat["resid"], gate), cat)
            m["gate_positive_ratio"] = float((gate > 0.5).mean())
            modes[f"threshold_{thr:g}"] = m
            if split == "val":
                val_scores[thr] = (m.get("stable_top5") or 0.0) + (m.get("changed_top5") or 0.0) + (m.get("semantic_hard_top5") or 0.0)
        out["splits"][split] = modes
    best_thr = max(val_scores, key=val_scores.get) if val_scores else 0.10
    out["best_val_threshold"] = best_thr
    out["gate_threshold_bottleneck"] = bool(out["splits"]["val"][f"threshold_{best_thr:g}"].get("stable_top5", 0) > out["splits"]["val"]["learned_gate_current_threshold"].get("stable_top5", 0))
    out["residual_classifier_bottleneck"] = bool((out["splits"]["val"]["residual_only"].get("changed_top5") or 0) <= (out["splits"]["val"]["copy_only"].get("changed_top5") or 0))
    out["stable_preservation_can_be_fixed_by_threshold"] = bool(out["splits"]["val"][f"threshold_{best_thr:g}"].get("stable_top5") >= out["splits"]["val"]["copy_only"].get("stable_top5"))
    out["changed_hard_can_be_fixed_by_threshold"] = bool((out["splits"]["val"][f"threshold_{best_thr:g}"].get("changed_top5") or 0) > (out["splits"]["val"]["copy_only"].get("changed_top5") or 0))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    runs = {}
    t12 = load_train(V3312_TRAIN)
    t13 = load_train(V3313_TRAIN)
    if t12.get("checkpoint_path"):
        runs["v33_12"] = run_model("v33_12", ROOT / t12["checkpoint_path"], False, args)
    if t13.get("checkpoint_path"):
        runs["v33_13"] = run_model("v33_13", ROOT / t13["checkpoint_path"], True, args)
    payload = {
        "generated_at_utc": utc_now(),
        "models": runs,
        "gate_threshold_bottleneck": any(bool(v.get("gate_threshold_bottleneck")) for v in runs.values()),
        "residual_classifier_bottleneck": any(bool(v.get("residual_classifier_bottleneck")) for v in runs.values()),
        "stable_preservation_can_be_fixed_by_threshold": any(bool(v.get("stable_preservation_can_be_fixed_by_threshold")) for v in runs.values()),
        "changed_hard_can_be_fixed_by_threshold": any(bool(v.get("changed_hard_can_be_fixed_by_threshold")) for v in runs.values()),
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.13 Gate Calibration Sweep", payload, ["gate_threshold_bottleneck", "residual_classifier_bottleneck", "stable_preservation_can_be_fixed_by_threshold", "changed_hard_can_be_fixed_by_threshold"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
