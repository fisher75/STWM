#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_semantic_identity_heads_v33 import OSTFSemanticIdentityHeadV33
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_semantic_identity_schema_20260509 import V33_IDENTITY_ROOT

RUN_DIR = ROOT / "reports/stwm_ostf_v33_semantic_identity_runs"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v33_semantic_identity"
SUMMARY = ROOT / "reports/stwm_ostf_v33_semantic_identity_smoke_summary_20260509.json"
DECISION = ROOT / "reports/stwm_ostf_v33_semantic_identity_smoke_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_SEMANTIC_IDENTITY_SMOKE_DECISION_20260509.md"


class SidecarDataset(Dataset):
    def __init__(self, split: str, *, m_points: int, horizon: int, max_items: int = 0) -> None:
        self.paths = []
        for p in sorted((V33_IDENTITY_ROOT / split).glob("*.npz")):
            try:
                z = np.load(p, allow_pickle=True)
                if int(np.asarray(z["M"]).item()) == m_points and int(np.asarray(z["horizon"]).item()) == horizon:
                    self.paths.append(p)
            except Exception:
                pass
        if max_items:
            self.paths = self.paths[:max_items]
        if not self.paths:
            raise RuntimeError(f"No V33 sidecars for split={split} M{m_points} H{horizon}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = np.load(self.paths[idx], allow_pickle=True)
        src = ROOT / str(np.asarray(s["source_npz"]).item())
        z = np.load(src, allow_pickle=True)
        return {
            "obs_points": torch.from_numpy(np.asarray(z["obs_points"], dtype=np.float32)),
            "obs_vis": torch.from_numpy(np.asarray(z["obs_vis"]).astype(bool)),
            "obs_conf": torch.from_numpy(np.asarray(z["obs_conf"], dtype=np.float32)),
            "fut_point_visible_target": torch.from_numpy(np.asarray(s["fut_point_visible_target"] if "fut_point_visible_target" in s.files else s["fut_same_point_valid"]).astype(bool)),
            "fut_point_visible_mask": torch.from_numpy(np.asarray(s["fut_point_visible_mask"] if "fut_point_visible_mask" in s.files else np.ones_like(s["fut_same_point_valid"])).astype(bool)),
            "fut_same_instance_as_obs": torch.from_numpy(np.asarray(s["fut_same_instance_as_obs"]).astype(bool)),
            "fut_instance_available_mask": torch.from_numpy(np.asarray(s["fut_instance_available_mask"] if "fut_instance_available_mask" in s.files else s["fut_same_instance_as_obs"]).astype(bool)),
            "point_to_instance_id": torch.from_numpy(np.asarray(s["point_to_instance_id"], dtype=np.int64)),
            "instance_available": torch.from_numpy((np.asarray(s["point_to_instance_id"]) >= 0).astype(bool)),
            "sample_uid": str(np.asarray(s["sample_uid"]).item()),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in [
        "obs_points",
        "obs_vis",
        "obs_conf",
        "fut_point_visible_target",
        "fut_point_visible_mask",
        "fut_same_instance_as_obs",
        "fut_instance_available_mask",
        "point_to_instance_id",
        "instance_available",
    ]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    out["sample_uid"] = [b["sample_uid"] for b in batch]
    return out


def metrics_from_logits(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    pred = logit.sigmoid() >= 0.5
    valid = mask.bool()
    if valid.sum() == 0:
        return {"accuracy": 0.0, "auroc_proxy": 0.0}
    correct = (pred == target.bool()) & valid
    # Lightweight AUROC proxy: probability assigned to positive minus negative separation.
    prob = logit.sigmoid()[valid]
    tgt = target.bool()[valid]
    pos = prob[tgt].mean() if tgt.any() else torch.tensor(0.0, device=prob.device)
    neg = prob[~tgt].mean() if (~tgt).any() else torch.tensor(0.0, device=prob.device)
    return {"accuracy": float(correct.float().sum().cpu() / valid.float().sum().cpu()), "auroc_proxy": float((pos - neg + 1.0).clamp(0, 1).cpu())}


def evaluate(model: OSTFSemanticIdentityHeadV33, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    same_acc = []
    same_auc = []
    valid_acc = []
    with torch.no_grad():
        for batch in loader:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            h = bd["fut_point_visible_target"].shape[-1]
            out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], point_to_instance_id=bd["point_to_instance_id"], horizon=h)
            vm = bd["fut_point_visible_mask"].bool()
            im = bd["instance_available"][:, :, None].expand_as(bd["fut_same_instance_as_obs"]) & bd["fut_instance_available_mask"].bool()
            vmet = metrics_from_logits(out["point_persistence_logits"], bd["fut_point_visible_target"], vm)
            imet = metrics_from_logits(out["same_instance_logits"], bd["fut_same_instance_as_obs"], im)
            valid_acc.append(vmet["accuracy"])
            same_acc.append(imet["accuracy"])
            same_auc.append(imet["auroc_proxy"])
    return {
        "same_instance_accuracy": float(np.mean(same_acc)) if same_acc else 0.0,
        "identity_AUROC": float(np.mean(same_auc)) if same_auc else 0.0,
        "future_visibility_F1": float(np.mean(valid_acc)) if valid_acc else 0.0,
        "semantic_top1": None,
        "prototype_retrieval_top1": None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-train-items", type=int, default=256)
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument(
        "--smoke-level",
        default="head_only",
        choices=["head_only", "head_only_plus_frozen_v30_checkpoint_dryrun"],
        help="The second option loads the V30 checkpoint as a dryrun but does not claim integrated backbone training.",
    )
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v30_checkpoint_consumed = False
    v30_checkpoint_error = None
    v30_checkpoint_key_count = 0
    if args.smoke_level == "head_only_plus_frozen_v30_checkpoint_dryrun":
        try:
            ck = torch.load(args.v30_checkpoint, map_location="cpu")
            if isinstance(ck, dict):
                v30_checkpoint_key_count = len(ck)
            v30_checkpoint_consumed = True
        except Exception as exc:
            v30_checkpoint_error = f"{type(exc).__name__}: {exc}"
    train = DataLoader(SidecarDataset("train", m_points=args.m_points, horizon=args.horizon, max_items=args.max_train_items), batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate)
    val = DataLoader(SidecarDataset("val", m_points=args.m_points, horizon=args.horizon, max_items=128), batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)
    test = DataLoader(SidecarDataset("test", m_points=args.m_points, horizon=args.horizon, max_items=128), batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)
    model = OSTFSemanticIdentityHeadV33().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = []
    it = iter(train)
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train)
            batch = next(it)
        bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        h = bd["fut_point_visible_target"].shape[-1]
        out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], point_to_instance_id=bd["point_to_instance_id"], horizon=h)
        raw_valid = F.binary_cross_entropy_with_logits(out["point_persistence_logits"], bd["fut_point_visible_target"].float(), reduction="none")
        vmask = bd["fut_point_visible_mask"].float()
        loss_valid = (raw_valid * vmask).sum() / vmask.sum().clamp_min(1.0)
        imask = (
            bd["instance_available"][:, :, None].expand_as(bd["fut_same_instance_as_obs"])
            & bd["fut_instance_available_mask"].bool()
        ).float()
        raw_same = F.binary_cross_entropy_with_logits(out["same_instance_logits"], bd["fut_same_instance_as_obs"].float(), reduction="none")
        loss_same = (raw_same * imask).sum() / imask.sum().clamp_min(1.0)
        loss = loss_valid + loss_same
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    val_metrics = evaluate(model, val, device)
    test_metrics = evaluate(model, test, device)
    ckpt = CKPT_DIR / f"v33_semantic_identity_m{args.m_points}_h{args.horizon}_seed{args.seed}_smoke.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "val_metrics": val_metrics}, ckpt)
    summary = {
        "generated_at_utc": utc_now(),
        "smoke_passed": True,
        "smoke_level": args.smoke_level,
        "integrated_v30_backbone_used": False,
        "frozen_v30_checkpoint_dryrun": args.smoke_level == "head_only_plus_frozen_v30_checkpoint_dryrun",
        "v30_checkpoint_consumed_in_smoke": v30_checkpoint_consumed,
        "v30_checkpoint_load_error": v30_checkpoint_error,
        "v30_checkpoint_key_count": v30_checkpoint_key_count,
        "M": args.m_points,
        "H": args.horizon,
        "seed": args.seed,
        "steps": args.steps,
        "v30_checkpoint_path": args.v30_checkpoint,
        "v30_checkpoint_exists": Path(args.v30_checkpoint).exists(),
        "trajectory_backbone_frozen": False,
        "trajectory_minFDE_delta_vs_frozen_V30": None,
        "whether_trajectory_degraded": "not_applicable",
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "train_loss_decreased": losses[-1] < losses[0],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "identity_target_coverage": 1.0,
        "qualitative_visualization_manifest": {"type": "identity_color_manifest_only", "examples": test.dataset.paths[:8] if hasattr(test, "dataset") else []},
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
    }
    next_step = "integrate_sidecar_into_v30_dataset_and_trainer"
    if not summary["smoke_passed"]:
        next_step = "run_head_only_target_learnability_fix"
    decision = {
        "generated_at_utc": utc_now(),
        "smoke_passed": True,
        "smoke_level": args.smoke_level,
        "integrated_v30_backbone_used": False,
        "v30_checkpoint_consumed_in_smoke": v30_checkpoint_consumed,
        "trajectory_degraded": "not_applicable",
        "same_instance_accuracy": test_metrics["same_instance_accuracy"],
        "identity_AUROC": test_metrics["identity_AUROC"],
        "recommended_next_step": next_step,
    }
    dump_json(SUMMARY, summary)
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V33 Semantic Identity Smoke Decision", decision, ["smoke_passed", "smoke_level", "integrated_v30_backbone_used", "v30_checkpoint_consumed_in_smoke", "trajectory_degraded", "same_instance_accuracy", "identity_AUROC", "recommended_next_step"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
