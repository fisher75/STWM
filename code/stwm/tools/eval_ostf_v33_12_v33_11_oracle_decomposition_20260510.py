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
from stwm.tools.eval_ostf_v33_11_identity_preserving_copy_residual_semantic_20260510 import topk
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import V33_11_MASK_ROOT, collate_copy_v3311, load_baseline_selection, make_loader_v3311
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch


REPORT = ROOT / "reports/stwm_ostf_v33_12_v33_11_oracle_decomposition_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_12_V33_11_ORACLE_DECOMPOSITION_20260510.md"


def eval_logits(logits: np.ndarray, target: np.ndarray, masks: dict[str, np.ndarray], bases: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for subset, mask in masks.items():
        out[f"{subset}_top1"] = topk(logits, target, mask, 1)
        out[f"{subset}_top5"] = topk(logits, target, mask, 5)
        base = bases.get(subset, bases["global"])
        out[f"{subset}_strongest_baseline_top1"] = topk(base, target, mask, 1)
        out[f"{subset}_strongest_baseline_top5"] = topk(base, target, mask, 5)
    out["stable_preservation_not_degraded"] = bool((out.get("stable_top5") or 0.0) + 1e-9 >= (out.get("stable_strongest_baseline_top5") or 0.0))
    out["changed_top5_beats_strongest_baseline"] = bool((out.get("changed_top5") or 0.0) > (out.get("changed_strongest_baseline_top5") or 0.0))
    out["semantic_hard_top5_beats_strongest_baseline"] = bool((out.get("semantic_hard_top5") or 0.0) > (out.get("semantic_hard_strongest_baseline_top5") or 0.0))
    return out


def mean_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({k for r in rows for k in r if k not in {"seed", "split"}})
    out: dict[str, Any] = {}
    for key in keys:
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float, bool, np.floating))]
        if not vals:
            out[key] = None
        elif isinstance(vals[0], bool):
            out[key] = all(bool(x) for x in vals)
        else:
            out[key] = float(np.mean(vals))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    train_summary = ROOT / "reports/stwm_ostf_v33_11_identity_preserving_copy_residual_train_summary_20260510.json"
    ckpt_default = ROOT / "outputs/checkpoints/stwm_ostf_v33_11_identity_preserving_copy_residual_h32_m128/v33_11_identity_preserving_copy_residual_m128_h32_seed42_best.pt"
    if train_summary.exists():
        ckpt_default = ROOT / json.loads(train_summary.read_text(encoding="utf-8")).get("checkpoint_path", str(ckpt_default.relative_to(ROOT)))
    p.add_argument("--checkpoint", default=str(ckpt_default))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    centers = torch.from_numpy(np.asarray(np.load(ckargs.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
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
    selection = load_baseline_selection()
    modes = ["copy_only", "residual_only", "learned_gate", "oracle_gate", "val_threshold_gate"] + [f"fixed_gate_{g}" for g in ("0.01", "0.03", "0.05", "0.1", "0.2", "0.5")]
    rows: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
    val_gate_scores = []
    val_gate_targets = []
    cached_splits: list[tuple[int, str, dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    for seed in (42, 123, 456):
        ckargs.hard_train_mask_manifest = str(V33_11_MASK_ROOT / f"H32_M128_seed{seed}.json")
        for split in ("val", "test"):
            ds = make_loader_v3311(split, ckargs, shuffle=False, max_items=None).dataset
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_copy_v3311)
            chunks = {k: [] for k in ["copy", "resid", "learned", "gate", "target", "mask", "stable", "changed", "hard", "global_base", "changed_base", "hard_base"]}
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
                        copy_semantic_prototype_id=bd["copy_semantic_prototype_id"],
                        last_observed_semantic_prototype_id=bd["last_observed_semantic_prototype_id"],
                    )
                    chunks["copy"].append(out["copy_prior_semantic_logits"].detach().cpu().numpy())
                    chunks["resid"].append(out["semantic_residual_logits"].detach().cpu().numpy())
                    chunks["learned"].append(out["final_semantic_proto_logits"].detach().cpu().numpy())
                    chunks["gate"].append(out["semantic_change_gate"].detach().cpu().numpy())
                    chunks["target"].append(bd["semantic_prototype_id"].detach().cpu().numpy())
                    chunks["mask"].append(bd["semantic_prototype_available_mask"].detach().cpu().numpy())
                    chunks["stable"].append(bd["semantic_stable_mask"].detach().cpu().numpy())
                    chunks["changed"].append(bd["semantic_changed_mask"].detach().cpu().numpy())
                    chunks["hard"].append(bd["semantic_hard_mask"].detach().cpu().numpy())
                    chunks["global_base"].append(torch.log(bd[f"baseline_{selection['global']}_distribution"].clamp_min(1e-8)).detach().cpu().numpy())
                    chunks["changed_base"].append(torch.log(bd[f"baseline_{selection['changed']}_distribution"].clamp_min(1e-8)).detach().cpu().numpy())
                    chunks["hard_base"].append(torch.log(bd[f"baseline_{selection['semantic_hard']}_distribution"].clamp_min(1e-8)).detach().cpu().numpy())
            cat = {k: np.concatenate(v) for k, v in chunks.items()}
            cached_splits.append((seed, split, cat, selection))
            if split == "val":
                valid = (cat["stable"] | cat["changed"]) & cat["mask"]
                val_gate_scores.append(cat["gate"][valid])
                val_gate_targets.append(cat["changed"][valid].astype(np.int32))
    scores = np.concatenate(val_gate_scores)
    targets = np.concatenate(val_gate_targets)
    best_thr = 0.5
    best_acc = -1.0
    for thr in np.linspace(0.01, 0.99, 99):
        pred = scores >= thr
        pos = targets == 1
        neg = targets == 0
        acc = 0.5 * ((pred[pos].mean() if pos.any() else 0.0) + ((~pred[neg]).mean() if neg.any() else 0.0))
        if acc > best_acc:
            best_acc = float(acc)
            best_thr = float(thr)
    best_fixed_gate = None
    best_fixed_val = -1.0
    for seed, split, cat, _ in cached_splits:
        masks = {"global": cat["mask"].astype(bool), "stable": cat["stable"].astype(bool) & cat["mask"].astype(bool), "changed": cat["changed"].astype(bool) & cat["mask"].astype(bool), "semantic_hard": cat["hard"].astype(bool) & cat["mask"].astype(bool)}
        bases = {"global": cat["global_base"], "stable": cat["copy"], "changed": cat["changed_base"], "semantic_hard": cat["hard_base"]}
        copy_probs = np.exp(cat["copy"] - cat["copy"].max(axis=-1, keepdims=True)); copy_probs /= copy_probs.sum(axis=-1, keepdims=True)
        resid_probs = np.exp(cat["resid"] - cat["resid"].max(axis=-1, keepdims=True)); resid_probs /= resid_probs.sum(axis=-1, keepdims=True)
        oracle = np.where((masks["changed"] | masks["semantic_hard"])[..., None], cat["resid"], cat["copy"])
        threshold = np.where((cat["gate"] >= best_thr)[..., None], cat["resid"], cat["copy"])
        mode_logits = {
            "copy_only": cat["copy"],
            "residual_only": cat["resid"],
            "learned_gate": cat["learned"],
            "oracle_gate": oracle,
            "val_threshold_gate": threshold,
        }
        for g in (0.01, 0.03, 0.05, 0.10, 0.20, 0.50):
            mode_logits[f"fixed_gate_{g}"] = np.log(((1.0 - g) * copy_probs + g * resid_probs).clip(1e-8, 1.0))
        for mode, logits in mode_logits.items():
            r = eval_logits(logits, cat["target"], masks, bases)
            r.update({"seed": seed, "split": split})
            rows[mode].append(r)
    agg = {mode: {"val": mean_rows([r for r in vals if r["split"] == "val"]), "test": mean_rows([r for r in vals if r["split"] == "test"])} for mode, vals in rows.items()}
    for g in ("0.01", "0.03", "0.05", "0.1", "0.2", "0.5"):
        val = agg[f"fixed_gate_{g}"]["val"].get("global_top5") or 0.0
        if val > best_fixed_val:
            best_fixed_val = float(val)
            best_fixed_gate = float(g)
    oracle_passes = bool(agg["oracle_gate"]["val"].get("stable_preservation_not_degraded") and agg["oracle_gate"]["test"].get("stable_preservation_not_degraded") and agg["oracle_gate"]["val"].get("changed_top5_beats_strongest_baseline") and agg["oracle_gate"]["test"].get("changed_top5_beats_strongest_baseline") and agg["oracle_gate"]["val"].get("semantic_hard_top5_beats_strongest_baseline") and agg["oracle_gate"]["test"].get("semantic_hard_top5_beats_strongest_baseline"))
    residual_sufficient = bool(agg["residual_only"]["val"].get("changed_top5_beats_strongest_baseline") and agg["residual_only"]["test"].get("changed_top5_beats_strongest_baseline"))
    payload = {
        "generated_at_utc": utc_now(),
        "checkpoint_path": str(Path(args.checkpoint).relative_to(ROOT) if Path(args.checkpoint).is_absolute() else args.checkpoint),
        "uses_v33_11_checkpoint": "v33_11_identity_preserving" in str(args.checkpoint),
        "baseline_selection": selection,
        "aggregate": agg,
        "oracle_gate_passes": oracle_passes,
        "residual_classifier_sufficient": residual_sufficient,
        "gate_is_primary_bottleneck": bool(oracle_passes and not agg["learned_gate"]["val"].get("stable_preservation_not_degraded")),
        "prototype_target_space_bottleneck": not residual_sufficient,
        "copy_preservation_possible_with_threshold": bool(agg["val_threshold_gate"]["val"].get("stable_preservation_not_degraded")),
        "best_fixed_gate": best_fixed_gate,
        "best_val_gate_threshold": best_thr,
        "rows": rows,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.12 V33.11 Oracle Decomposition", payload, ["uses_v33_11_checkpoint", "oracle_gate_passes", "residual_classifier_sufficient", "gate_is_primary_bottleneck", "prototype_target_space_bottleneck", "copy_preservation_possible_with_threshold", "best_fixed_gate", "best_val_gate_threshold", "baseline_selection"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
