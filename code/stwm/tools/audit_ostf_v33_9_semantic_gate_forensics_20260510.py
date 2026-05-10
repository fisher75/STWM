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

from stwm.modules.ostf_v33_3_structured_semantic_identity_world_model import StructuredSemanticIdentityWorldModelV333
from stwm.modules.ostf_v33_7_identity_belief_world_model import IdentityBeliefWorldModelV337
from stwm.tools.eval_ostf_v33_7_identity_belief_calibration_20260509 import BeliefEvalDataset, collate_belief_eval
from stwm.tools.eval_ostf_v33_8_ablation_safe_identity_semantic_20260510 import candidate_checkpoints, roots_args, selected_k
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_9_semantic_gate_utils_20260510 import semantic_gate_metrics


REPORT = ROOT / "reports/stwm_ostf_v33_9_semantic_gate_forensics_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_9_SEMANTIC_GATE_FORENSICS_20260510.md"
V33_8_EVAL = ROOT / "reports/stwm_ostf_v33_8_ablation_safe_eval_summary_20260510.json"
MASK_ROOT = ROOT / "manifests/ostf_v33_8_split_matched_hard_identity_semantic"


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def load_model(spec: dict[str, Any], ns: argparse.Namespace, device: torch.device):
    ck = torch.load(spec["checkpoint"], map_location="cpu")
    ck_args = argparse.Namespace(**ck.get("args", {}))
    centers = torch.from_numpy(np.asarray(np.load(ns.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    v30 = getattr(ck_args, "v30_checkpoint", str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    teacher_dim = int(getattr(ck_args, "teacher_embedding_dim", 512))
    if spec["kind"] == "v33_7":
        model = IdentityBeliefWorldModelV337(
            v30,
            prototype_centers=centers,
            teacher_embedding_dim=teacher_dim,
            use_observed_instance_context=False,
            disable_embedding_similarity_logits=bool(getattr(ck_args, "disable_embedding_similarity_logits", False)),
            disable_fused_logits=bool(getattr(ck_args, "disable_fused_logits", False)),
        )
    else:
        model = StructuredSemanticIdentityWorldModelV333(v30, prototype_centers=centers, teacher_embedding_dim=teacher_dim, use_observed_instance_context=False)
    model.load_state_dict(ck["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def eval_semantic(split: str, ns: argparse.Namespace, model: torch.nn.Module, device: torch.device, manifest: Path) -> dict[str, Any]:
    ds = BeliefEvalDataset(split, ns, manifest)
    loader = DataLoader(ds, batch_size=ns.batch_size, shuffle=False, num_workers=ns.num_workers, collate_fn=collate_belief_eval)
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ["proto_logits", "proto_targets", "proto_masks", "obs_proto", "obs_proto_mask", "semantic_hard"]}
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
            )
            arrays["proto_logits"].append(out["future_semantic_proto_logits"].detach().cpu().numpy())
            arrays["proto_targets"].append(bd["semantic_prototype_id"].detach().cpu().numpy())
            arrays["proto_masks"].append(bd["semantic_prototype_available_mask"].detach().cpu().numpy())
            arrays["obs_proto"].append(bd["obs_semantic_prototype_id"].detach().cpu().numpy())
            arrays["obs_proto_mask"].append(bd["obs_semantic_prototype_available_mask"].detach().cpu().numpy())
            arrays["semantic_hard"].append(bd["semantic_hard_eval_mask"].detach().cpu().numpy())
    cat = {k: np.concatenate(v) for k, v in arrays.items()}
    return semantic_gate_metrics(cat["proto_logits"], cat["proto_targets"], cat["proto_masks"].astype(bool), cat["semantic_hard"].astype(bool), cat["obs_proto"], cat["obs_proto_mask"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--hard-subset-seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    v33_8 = json.loads(V33_8_EVAL.read_text(encoding="utf-8")) if V33_8_EVAL.exists() else {}
    best_name = v33_8.get("best_candidate_by_val") or (v33_8.get("candidates", [{}])[0].get("candidate"))
    specs = candidate_checkpoints()
    spec = specs.get(best_name)
    k = selected_k()
    ns = roots_args(k, args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    per_split = {}
    if spec and Path(spec["checkpoint"]).exists():
        model = load_model(spec, ns, device)
        manifest = MASK_ROOT / f"H32_M128_seed{args.hard_subset_seed}.json"
        for split in ("val", "test"):
            per_split[split] = eval_semantic(split, ns, model, device, manifest)
    payload = {
        "generated_at_utc": utc_now(),
        "best_candidate_by_val": best_name,
        "selected_K": k,
        "per_split": per_split,
        "rules": {
            "strong_semantic_field_claim_requires": "global top1 copy beaten OR changed top1 copy beaten",
            "weak_semantic_ranking_claim_requires": "stable preservation not degraded AND changed top5 or semantic-hard top5 beats copy",
            "stable_states_do_not_need_to_beat_copy": True,
        },
    }
    for split, row in per_split.items():
        payload[f"{split}_global_semantic_top1_copy_beaten"] = row.get("global_semantic_top1_copy_beaten")
        payload[f"{split}_global_semantic_top5_copy_beaten"] = row.get("global_semantic_top5_copy_beaten")
        payload[f"{split}_changed_top1_beats_copy"] = row.get("changed_model_beats_copy")
        payload[f"{split}_changed_top5_beats_copy"] = row.get("changed_top5_beats_copy")
        payload[f"{split}_semantic_hard_model_beats_copy"] = row.get("semantic_hard_model_beats_copy")
        payload[f"{split}_semantic_hard_top5_beats_copy"] = row.get("semantic_hard_top5_beats_copy")
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.9 Semantic Gate Forensics",
        payload,
        ["best_candidate_by_val", "test_global_semantic_top1_copy_beaten", "test_global_semantic_top5_copy_beaten", "test_changed_top1_beats_copy", "test_changed_top5_beats_copy", "test_semantic_hard_top5_beats_copy"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
