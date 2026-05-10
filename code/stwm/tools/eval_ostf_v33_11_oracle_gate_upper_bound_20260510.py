#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_10_copy_residual_semantic_world_model import CopyResidualSemanticWorldModelV3310
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import V33_11_MASK_ROOT, collate_copy_v3311, load_baseline_selection, make_loader_v3311
from stwm.tools.eval_ostf_v33_11_identity_preserving_copy_residual_semantic_20260510 import topk
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch


REPORT = ROOT / "reports/stwm_ostf_v33_11_oracle_gate_upper_bound_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_11_ORACLE_GATE_UPPER_BOUND_20260510.md"


def eval_logits(logits: np.ndarray, target: np.ndarray, masks: dict[str, np.ndarray], baseline_logits: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for subset, mask in masks.items():
        out[f"{subset}_top1"] = topk(logits, target, mask, 1)
        out[f"{subset}_top5"] = topk(logits, target, mask, 5)
        base = baseline_logits.get(subset, baseline_logits["global"])
        out[f"{subset}_beats_baseline_top5"] = bool((out[f"{subset}_top5"] or 0.0) > (topk(base, target, mask, 5) or 0.0))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v33_10_copy_residual_semantic_h32_m128/v33_10_copy_residual_semantic_m128_h32_seed42_best.pt"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.copy_residual_semantic_target_root = str(ROOT / "outputs/cache/stwm_ostf_v33_11_copy_residual_semantic_targets/pointodyssey/clip_vit_b32_local/K32")
    ckargs.semantic_baseline_bank_root = str(ROOT / "outputs/cache/stwm_ostf_v33_11_semantic_baseline_bank/pointodyssey/clip_vit_b32_local/K32")
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    centers = torch.from_numpy(np.asarray(np.load(ckargs.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = CopyResidualSemanticWorldModelV3310(ckargs.v30_checkpoint, prototype_centers=centers, teacher_embedding_dim=ckargs.teacher_embedding_dim).to(device)
    model.load_state_dict(ck["model"], strict=True)
    selection = load_baseline_selection()
    rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in ["copy_only", "residual_only", "learned_gate", "oracle_gate", "fixed_gate_0.05", "fixed_gate_0.1", "fixed_gate_0.2", "fixed_gate_0.5"]}
    for seed in (42, 123, 456):
        ckargs.hard_train_mask_manifest = str(V33_11_MASK_ROOT / f"H32_M128_seed{seed}.json")
        for split in ("val", "test"):
            ds = make_loader_v3311(split, ckargs, shuffle=False, max_items=None).dataset
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_copy_v3311)
            chunks: dict[str, list[np.ndarray]] = {k: [] for k in ["copy", "resid", "learned", "target", "mask", "stable", "changed", "hard", "global_base", "changed_base", "hard_base"]}
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
                    chunks["target"].append(bd["semantic_prototype_id"].detach().cpu().numpy())
                    chunks["mask"].append(bd["semantic_prototype_available_mask"].detach().cpu().numpy())
                    chunks["stable"].append(bd["semantic_stable_mask"].detach().cpu().numpy())
                    chunks["changed"].append(bd["semantic_changed_mask"].detach().cpu().numpy())
                    chunks["hard"].append(bd["semantic_hard_mask"].detach().cpu().numpy())
                    chunks["global_base"].append(torch.log(bd[f"baseline_{selection['global']}_distribution"].clamp_min(1e-8)).detach().cpu().numpy())
                    chunks["changed_base"].append(torch.log(bd[f"baseline_{selection['changed']}_distribution"].clamp_min(1e-8)).detach().cpu().numpy())
                    chunks["hard_base"].append(torch.log(bd[f"baseline_{selection['semantic_hard']}_distribution"].clamp_min(1e-8)).detach().cpu().numpy())
            cat = {k: np.concatenate(v) for k, v in chunks.items()}
            masks = {"global": cat["mask"].astype(bool), "stable": cat["stable"].astype(bool), "changed": cat["changed"].astype(bool), "semantic_hard": cat["hard"].astype(bool)}
            bases = {"global": cat["global_base"], "changed": cat["changed_base"], "semantic_hard": cat["hard_base"], "stable": cat["copy"]}
            copy_probs = np.exp(cat["copy"] - cat["copy"].max(axis=-1, keepdims=True)); copy_probs /= copy_probs.sum(axis=-1, keepdims=True)
            resid_probs = np.exp(cat["resid"] - cat["resid"].max(axis=-1, keepdims=True)); resid_probs /= resid_probs.sum(axis=-1, keepdims=True)
            oracle = np.where((masks["changed"] | masks["semantic_hard"])[..., None], cat["resid"], cat["copy"])
            mode_logits = {
                "copy_only": cat["copy"],
                "residual_only": cat["resid"],
                "learned_gate": cat["learned"],
                "oracle_gate": oracle,
            }
            for g in (0.05, 0.1, 0.2, 0.5):
                mode_logits[f"fixed_gate_{g}"] = np.log(((1.0 - g) * copy_probs + g * resid_probs).clip(1e-8, 1.0))
            for mode, logits in mode_logits.items():
                r = eval_logits(logits, cat["target"], masks, bases)
                r.update({"seed": seed, "split": split})
                rows[mode].append(r)
    agg: dict[str, Any] = {}
    for mode, vals in rows.items():
        agg[mode] = {}
        for key in ["global_top5", "stable_top5", "changed_top5", "semantic_hard_top5", "changed_beats_baseline_top5", "semantic_hard_beats_baseline_top5"]:
            clean = [v.get(key) for v in vals if isinstance(v.get(key), (int, float, bool, np.floating))]
            agg[mode][key] = float(np.mean(clean)) if clean and not isinstance(clean[0], bool) else (all(bool(x) for x in clean) if clean else None)
    oracle_passes = bool(agg.get("oracle_gate", {}).get("stable_top5", 0.0) >= agg.get("copy_only", {}).get("stable_top5", 1.0) and agg.get("oracle_gate", {}).get("changed_beats_baseline_top5"))
    residual_sufficient = bool(agg.get("residual_only", {}).get("changed_beats_baseline_top5") and agg.get("residual_only", {}).get("semantic_hard_beats_baseline_top5"))
    learned_stable = float(agg.get("learned_gate", {}).get("stable_top5") or 0.0)
    copy_stable = float(agg.get("copy_only", {}).get("stable_top5") or 0.0)
    payload = {
        "generated_at_utc": utc_now(),
        "oracle_gate_upper_bound_done": True,
        "baseline_selection": selection,
        "aggregate": agg,
        "oracle_gate_passes": oracle_passes,
        "residual_classifier_sufficient": residual_sufficient,
        "gate_is_primary_bottleneck": bool(oracle_passes and learned_stable + 1e-9 < copy_stable),
        "prototype_target_space_bottleneck": not residual_sufficient,
        "rows": rows,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.11 Oracle Gate Upper Bound", payload, ["oracle_gate_upper_bound_done", "oracle_gate_passes", "residual_classifier_sufficient", "gate_is_primary_bottleneck", "prototype_target_space_bottleneck", "baseline_selection"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
