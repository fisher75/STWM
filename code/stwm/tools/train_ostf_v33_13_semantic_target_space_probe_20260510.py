#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_13_gate_repaired_copy_semantic_world_model import GateRepairedCopySemanticWorldModelV3313
from stwm.tools.eval_ostf_v33_11_identity_preserving_copy_residual_semantic_20260510 import topk
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import V33_11_MASK_ROOT, collate_copy_v3311, make_loader_v3311
from stwm.tools.train_ostf_v33_13_gate_repaired_copy_semantic_20260510 import TARGET_ROOT, VOCAB
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch


SUMMARY = ROOT / "reports/stwm_ostf_v33_13_semantic_target_space_probe_summary_20260510.json"
DECISION = ROOT / "reports/stwm_ostf_v33_13_semantic_target_space_probe_decision_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_13_SEMANTIC_TARGET_SPACE_PROBE_DECISION_20260510.md"
OUT = ROOT / "outputs/cache/stwm_ostf_v33_13_semantic_target_space_probe"
TRAIN_SUMMARY = ROOT / "reports/stwm_ostf_v33_13_gate_repaired_train_summary_20260510.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(args: argparse.Namespace, device: torch.device) -> GateRepairedCopySemanticWorldModelV3313:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    ckpt = ROOT / train.get("checkpoint_path", "")
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu")
        ckargs = argparse.Namespace(**ck["args"])
    else:
        ck = None
        ckargs = args
    centers = torch.from_numpy(np.asarray(np.load(VOCAB)["prototype_centers"], dtype=np.float32))
    model = GateRepairedCopySemanticWorldModelV3313(
        ckargs.v30_checkpoint,
        prototype_centers=centers,
        teacher_embedding_dim=ckargs.teacher_embedding_dim,
        identity_teacher_checkpoint=ckargs.identity_teacher_checkpoint,
        gate_threshold=float(getattr(ckargs, "gate_threshold", 0.10)),
        freeze_identity_path=True,
    ).to(device)
    if ck is not None:
        model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model


def collect(split: str, args: argparse.Namespace, model: GateRepairedCopySemanticWorldModelV3313, device: torch.device) -> dict[str, np.ndarray]:
    args.hard_train_mask_manifest = str(V33_11_MASK_ROOT / "H32_M128_seed42.json")
    loader = make_loader_v3311(split, args, shuffle=False, max_items=None)
    rows: dict[str, list[np.ndarray]] = {k: [] for k in ["teacher", "v30", "v30_teacher", "obsfreq", "samplefreq", "copy", "target", "mask", "stable", "changed", "hard"]}
    with torch.no_grad():
        for batch in DataLoader(loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_copy_v3311):
            bd = move_batch(batch, device)
            features, _ = model._v30_features(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], semantic_id=bd["semantic_id"])
            v30_out = model.v30(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], semantic_id=bd["semantic_id"])
            point_token = features["point_token"].detach()
            step_hidden = features["step_hidden"].detach()
            b, m, h = bd["semantic_prototype_id"].shape
            visual = model._visual_context(bd["obs_teacher_embedding"], bd["obs_teacher_available_mask"]).detach()
            v30_feat = torch.cat(
                [
                    point_token[:, :, None, :].expand(-1, -1, h, -1),
                    step_hidden[:, None, :, :].expand(-1, m, -1, -1),
                    v30_out["visibility_logits"].detach().sigmoid()[:, :, :, None],
                ],
                dim=-1,
            )
            teacher_feat = visual[:, :, None, :].expand(-1, -1, h, -1)
            rows["teacher"].append(teacher_feat.detach().cpu().numpy())
            rows["v30"].append(v30_feat.detach().cpu().numpy())
            rows["v30_teacher"].append(torch.cat([v30_feat, teacher_feat], dim=-1).detach().cpu().numpy())
            rows["obsfreq"].append(bd["observed_frequency_prior_distribution"].detach().cpu().numpy())
            rows["samplefreq"].append(bd["sample_level_frequency_prior_distribution"].detach().cpu().numpy())
            rows["copy"].append(bd["copy_prior_distribution"].detach().cpu().numpy())
            rows["target"].append(bd["semantic_prototype_id"].detach().cpu().numpy())
            rows["mask"].append(bd["semantic_prototype_available_mask"].detach().cpu().numpy())
            rows["stable"].append(bd["semantic_stable_mask"].detach().cpu().numpy())
            rows["changed"].append(bd["semantic_changed_mask"].detach().cpu().numpy())
            rows["hard"].append(bd["semantic_hard_mask"].detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in rows.items()}


def flat(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return x.reshape(-1, x.shape[-1])[mask.reshape(-1)]


def train_probe(name: str, x: np.ndarray, target: np.ndarray, mask: np.ndarray, k: int, *, mlp: bool, steps: int, device: torch.device) -> nn.Module:
    valid_x = flat(x, mask)
    valid_y = target.reshape(-1)[mask.reshape(-1)]
    n = min(valid_x.shape[0], 180000)
    idx = np.random.choice(valid_x.shape[0], size=n, replace=False) if valid_x.shape[0] > n else np.arange(valid_x.shape[0])
    xt = torch.from_numpy(valid_x[idx].astype(np.float32)).to(device)
    yt = torch.from_numpy(valid_y[idx].astype(np.int64)).to(device)
    if mlp:
        model = nn.Sequential(nn.LayerNorm(xt.shape[-1]), nn.Linear(xt.shape[-1], 512), nn.GELU(), nn.Linear(512, k)).to(device)
    else:
        model = nn.Sequential(nn.LayerNorm(xt.shape[-1]), nn.Linear(xt.shape[-1], k)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    bs = 8192
    for _ in range(steps):
        ii = torch.randint(0, xt.shape[0], (min(bs, xt.shape[0]),), device=device)
        loss = torch.nn.functional.cross_entropy(model(xt[ii]), yt[ii])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    y = []
    flat_x = x.reshape(-1, x.shape[-1]).astype(np.float32)
    with torch.no_grad():
        for i in range(0, flat_x.shape[0], 32768):
            y.append(model(torch.from_numpy(flat_x[i : i + 32768]).to(device)).detach().cpu().numpy())
    return np.concatenate(y).reshape(*x.shape[:-1], -1)


def eval_logits(logits: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, Any]:
    mask = data["mask"].astype(bool)
    stable = data["stable"].astype(bool) & mask
    changed = data["changed"].astype(bool) & mask
    hard = data["hard"].astype(bool) & mask
    target = data["target"]
    sample = np.log(data["samplefreq"].clip(1e-8, 1.0))
    copy = np.log(data["copy"].clip(1e-8, 1.0))
    return {
        "global_top1": topk(logits, target, mask, 1),
        "global_top5": topk(logits, target, mask, 5),
        "stable_top1": topk(logits, target, stable, 1),
        "stable_top5": topk(logits, target, stable, 5),
        "stable_copy_top5": topk(copy, target, stable, 5),
        "stable_preservation_not_degraded": bool((topk(logits, target, stable, 5) or 0.0) >= (topk(copy, target, stable, 5) or 0.0)),
        "changed_top1": topk(logits, target, changed, 1),
        "changed_top5": topk(logits, target, changed, 5),
        "changed_samplefreq_top5": topk(sample, target, changed, 5),
        "changed_top5_beats_strongest_baseline": bool((topk(logits, target, changed, 5) or 0.0) > (topk(sample, target, changed, 5) or 0.0)),
        "semantic_hard_top1": topk(logits, target, hard, 1),
        "semantic_hard_top5": topk(logits, target, hard, 5),
        "semantic_hard_samplefreq_top5": topk(sample, target, hard, 5),
        "semantic_hard_top5_beats_strongest_baseline": bool((topk(logits, target, hard, 5) or 0.0) > (topk(sample, target, hard, 5) or 0.0)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    args.v30_checkpoint = str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt")
    args.identity_teacher_checkpoint = str(ROOT / "outputs/checkpoints/stwm_ostf_v33_9_fresh_expanded_h32_m128/v33_9_v33_6_global_contrastive_fresh_seed42_best.pt")
    args.teacher_embedding_dim = 512
    args.m_points = 128
    args.horizon = 32
    args.semantic_ideanity_sidecar_root = ""
    args.semantic_identity_sidecar_root = str(ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128/semantic_identity_targets/pointodyssey")
    args.global_identity_label_root = str(ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128/global_identity_labels/pointodyssey")
    args.visual_teacher_root = str(ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128/visual_teacher_prototypes/pointodyssey/clip_vit_b32_local")
    args.semantic_prototype_target_root = str(TARGET_ROOT)
    args.copy_residual_semantic_target_root = str(ROOT / "outputs/cache/stwm_ostf_v33_12_copy_conservative_semantic_targets/pointodyssey/clip_vit_b32_local/K256")
    args.semantic_baseline_bank_root = str(ROOT / "outputs/cache/stwm_ostf_v33_12_semantic_baseline_bank/pointodyssey/clip_vit_b32_local/K256")
    args.prototype_vocab_path = str(VOCAB)
    args.hard_train_mask_manifest = str(V33_11_MASK_ROOT / "H32_M128_seed42.json")
    args.use_observed_instance_context = False
    args.enable_global_identity_labels = True
    args.require_global_identity_labels = True
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args, device)
    data = {split: collect(split, args, model, device) for split in ("train", "val", "test")}
    k = int(np.load(VOCAB)["prototype_centers"].shape[0])
    probes: dict[str, Any] = {}
    baseline_logits = {
        "observed_frequency_probe": lambda d: np.log(d["obsfreq"].clip(1e-8, 1.0)),
        "sample_frequency_probe": lambda d: np.log(d["samplefreq"].clip(1e-8, 1.0)),
        "copy_probe": lambda d: np.log(d["copy"].clip(1e-8, 1.0)),
    }
    for name, fn in baseline_logits.items():
        probes[name] = {"val": eval_logits(fn(data["val"]), data["val"]), "test": eval_logits(fn(data["test"]), data["test"]), "trained": False}
    train_specs = [
        ("observed_teacher_history_mlp_probe", "teacher", True),
        ("V30_hidden_linear_probe", "v30", False),
        ("V30_hidden_mlp_probe", "v30", True),
        ("V30_hidden_plus_teacher_mlp_probe", "v30_teacher", True),
    ]
    for name, key, mlp in train_specs:
        m = train_probe(name, data["train"][key], data["train"]["target"], data["train"]["mask"].astype(bool), k, mlp=mlp, steps=args.steps, device=device)
        probes[name] = {
            "val": eval_logits(predict(m, data["val"][key], device), data["val"]),
            "test": eval_logits(predict(m, data["test"][key], device), data["test"]),
            "trained": True,
        }
    best = max(probes, key=lambda n: (probes[n]["val"].get("changed_top5") or 0.0) + (probes[n]["val"].get("semantic_hard_top5") or 0.0))
    best_row = probes[best]
    learnability = bool(best_row["val"].get("changed_top5_beats_strongest_baseline") and best_row["val"].get("semantic_hard_top5_beats_strongest_baseline"))
    payload = {
        "generated_at_utc": utc_now(),
        "target_space_probe_done": True,
        "teacher": "clip_vit_b32_local",
        "K": k,
        "probes": probes,
        "best_probe_by_val": best,
        "target_space_learnability_passed": learnability,
        "target_space_learnability_failed": not learnability,
        "v30_hidden_features_loaded": True,
    }
    decision = {
        "generated_at_utc": utc_now(),
        "target_space_probe_done": True,
        "best_probe_by_val": best,
        "target_space_learnability_passed": learnability,
        "target_space_learnability_failed": not learnability,
        "architecture_or_loss_bottleneck": bool(learnability),
        "recommended_next_step": "fix_gate_repaired_model_loss" if learnability else "build_real_stronger_teacher_targets",
    }
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V33.13 Semantic Target-Space Probe Decision", decision, ["target_space_probe_done", "best_probe_by_val", "target_space_learnability_passed", "target_space_learnability_failed", "architecture_or_loss_bottleneck", "recommended_next_step"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
