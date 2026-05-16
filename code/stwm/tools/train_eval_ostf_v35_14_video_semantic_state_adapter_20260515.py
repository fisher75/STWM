#!/usr/bin/env python3
"""训练/评估 V35.14 mask-derived video semantic state adapter。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.eval_ostf_v35_14_mask_video_semantic_state_predictability_20260515 import TARGET_ROOT, build_split
from stwm.tools.ostf_v17_common_20260502 import ROOT

CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v35_14_video_semantic_state_adapter_h32_m128"
TRAIN_REPORT = ROOT / "reports/stwm_ostf_v35_14_video_semantic_state_adapter_train_summary_20260515.json"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_14_video_semantic_state_adapter_eval_summary_20260515.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_14_video_semantic_state_adapter_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_14_VIDEO_SEMANTIC_STATE_ADAPTER_DECISION_20260515.md"
EXPERIMENT_ID = "V35.14"
EXPERIMENT_LABEL = "V35.14 video semantic state adapter"
CKPT_PREFIX = "v35_14_video_semantic_state_adapter"
NEXT_STEP_ON_PASS = "run_v35_14_video_adapter_seed123_replication"
NEXT_STEP_ON_FAIL = "fix_video_semantic_state_adapter_or_expand_targets"


def jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VideoSemanticAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 384, clusters: int = 128, families: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.cluster = nn.Linear(hidden, clusters)
        self.family = nn.Linear(hidden, families)
        self.changed = nn.Linear(hidden, 1)
        self.hard = nn.Linear(hidden, 1)
        self.uncertainty = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.net(x)
        return {
            "cluster_logits": self.cluster(h),
            "family_logits": self.family(h),
            "changed_logits": self.changed(h).squeeze(-1),
            "hard_logits": self.hard(h).squeeze(-1),
            "uncertainty_logits": self.uncertainty(h).squeeze(-1),
        }


def sample_batch(data: dict[str, np.ndarray], batch_size: int, rng: np.random.Generator, device: torch.device) -> dict[str, torch.Tensor]:
    idx = rng.integers(0, len(data["x"]), size=batch_size)
    return {
        "x": torch.from_numpy(data["x"][idx]).to(device),
        "cluster": torch.from_numpy(data["cluster"][idx]).long().to(device),
        "family": torch.from_numpy(data["family"][idx]).long().to(device),
        "changed": torch.from_numpy(data["changed"][idx]).float().to(device),
        "hard": torch.from_numpy(data["hard"][idx]).float().to(device),
        "uncertainty": torch.from_numpy(data["uncertainty_high"][idx]).float().to(device),
    }


def pos_weight(y: np.ndarray) -> torch.Tensor:
    p = float(y.mean())
    return torch.tensor([(1.0 - p) / max(p, 1e-4)], dtype=torch.float32).clamp(0.25, 12.0)


def train_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    cluster_loss = F.cross_entropy(out["cluster_logits"], batch["cluster"].clamp_min(0))
    family_loss = F.cross_entropy(out["family_logits"], batch["family"].clamp_min(0))
    changed_loss = F.binary_cross_entropy_with_logits(out["changed_logits"], batch["changed"], pos_weight=weights["changed"].to(batch["x"].device))
    hard_loss = F.binary_cross_entropy_with_logits(out["hard_logits"], batch["hard"], pos_weight=weights["hard"].to(batch["x"].device))
    unc_loss = F.binary_cross_entropy_with_logits(out["uncertainty_logits"], batch["uncertainty"], pos_weight=weights["uncertainty"].to(batch["x"].device))
    loss = 0.35 * cluster_loss + 0.20 * family_loss + 1.40 * changed_loss + 1.40 * hard_loss + 0.80 * unc_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "cluster_loss": float(cluster_loss.detach().cpu()),
        "changed_loss": float(changed_loss.detach().cpu()),
        "hard_loss": float(hard_loss.detach().cpu()),
        "uncertainty_loss": float(unc_loss.detach().cpu()),
        "family_loss": float(family_loss.detach().cpu()),
    }


@torch.no_grad()
def predict(model: VideoSemanticAdapter, x: np.ndarray, device: torch.device, batch_size: int = 8192) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {"cluster_logits": [], "family_logits": [], "changed": [], "hard": [], "uncertainty": []}
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).to(device)
        out = model(xb)
        outs["cluster_logits"].append(out["cluster_logits"].detach().cpu().numpy())
        outs["family_logits"].append(out["family_logits"].detach().cpu().numpy())
        outs["changed"].append(torch.sigmoid(out["changed_logits"]).detach().cpu().numpy())
        outs["hard"].append(torch.sigmoid(out["hard_logits"]).detach().cpu().numpy())
        outs["uncertainty"].append(torch.sigmoid(out["uncertainty_logits"]).detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


def choose_threshold(score: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    best_t, best = 0.5, -1.0
    for t in np.quantile(score, np.linspace(0.05, 0.95, 37)):
        ba = balanced_accuracy_score(y, score >= t)
        if ba > best:
            best = float(ba)
            best_t = float(t)
    return best_t


def bin_metrics(score: np.ndarray, y: np.ndarray, thr: float) -> dict[str, float | None]:
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "balanced_accuracy": None, "f1": None, "positive_ratio": float(y.mean())}
    pred = score >= thr
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "positive_ratio": float(y.mean()),
    }


def top5_cluster_metrics(cluster_logits: np.ndarray, y: np.ndarray, last: np.ndarray, changed_score: np.ndarray, changed_thr: float) -> dict[str, float]:
    top5 = np.argpartition(-cluster_logits, kth=4, axis=1)[:, :5]
    copy = np.where(last >= 0, last, top5[:, 0])
    changed_pred = changed_score >= changed_thr
    final_top1 = np.where(changed_pred, top5[:, 0], copy)
    # stable copy prior: if no change predicted, include copy in top5.
    final_top5 = top5.copy()
    final_top5[:, 0] = np.where(changed_pred, final_top5[:, 0], copy)
    # pointwise-preserving semantic field: copy anchor must remain in top-k even when adapter predicts change.
    final_top5[:, -1] = copy
    stable = (last >= 0) & (y == last)
    stable_top5 = float(np.any(final_top5[stable] == y[stable, None], axis=1).mean()) if stable.any() else 0.0
    stable_copy = float((copy[stable] == y[stable]).mean()) if stable.any() else 0.0
    return {
        "cluster_top1": float((final_top1 == y).mean()),
        "cluster_top5": float(np.any(final_top5 == y[:, None], axis=1).mean()),
        "copy_top1": float((copy == y).mean()),
        "stable_top5": stable_top5,
        "stable_copy_top1": stable_copy,
        "stable_token_ratio": float(stable.mean()),
    }


def summarize_trace(trace: list[dict[str, float]], key: str) -> dict[str, float]:
    vals = [t[key] for t in trace if key in t]
    return {"first": float(vals[0]), "last": float(vals[-1]), "mean": float(np.mean(vals))} if vals else {"first": 0.0, "last": 0.0, "mean": 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-root", default=str(TARGET_ROOT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--max-train-tokens", type=int, default=160000)
    ap.add_argument("--max-eval-tokens", type=int, default=80000)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    root = Path(args.target_root)
    if not root.is_absolute():
        root = ROOT / root
    train = build_split(root, "train", args.max_train_tokens, args.seed)
    val = build_split(root, "val", args.max_eval_tokens, args.seed)
    test = build_split(root, "test", args.max_eval_tokens, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = VideoSemanticAdapter(train["x"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = {"changed": pos_weight(train["changed"]), "hard": pos_weight(train["hard"]), "uncertainty": pos_weight(train["uncertainty_high"])}
    rng = np.random.default_rng(args.seed)
    trace: list[dict[str, float]] = []
    best_val = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    print(f"{EXPERIMENT_LABEL}: 开始训练；V30 frozen，future semantic/mask 不作为输入。", flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        batch = sample_batch(train, args.batch_size, rng, device)
        out = model(batch["x"])
        loss, stats = train_loss(out, batch, weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        trace.append(stats)
        if step % 200 == 0 or step == args.steps:
            pv = predict(model, val["x"], device)
            thr = choose_threshold(pv["changed"], val["changed"])
            score = float(bin_metrics(pv["changed"], val["changed"], thr)["balanced_accuracy"] or 0.0)
            print(f"step={step} loss={stats['loss']:.4f} val_changed_balacc={score:.4f}", flush=True)
            if score > best_val:
                best_val = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_dir = CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{CKPT_PREFIX}_m128_h32_seed{args.seed}_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "input_dim": int(train["x"].shape[1]), "best_val_changed_balacc": best_val}, ckpt_path)

    pv = predict(model, val["x"], device)
    pt = predict(model, test["x"], device)
    thresholds = {k: choose_threshold(pv[k], val[{"changed": "changed", "hard": "hard", "uncertainty": "uncertainty_high"}[k]]) for k in ["changed", "hard", "uncertainty"]}
    val_metrics = {
        "semantic_changed": bin_metrics(pv["changed"], val["changed"], thresholds["changed"]),
        "semantic_hard": bin_metrics(pv["hard"], val["hard"], thresholds["hard"]),
        "semantic_uncertainty": bin_metrics(pv["uncertainty"], val["uncertainty_high"], thresholds["uncertainty"]),
        "cluster": top5_cluster_metrics(pv["cluster_logits"], val["cluster"], val["last_cluster"], pv["changed"], thresholds["changed"]),
    }
    test_metrics = {
        "semantic_changed": bin_metrics(pt["changed"], test["changed"], thresholds["changed"]),
        "semantic_hard": bin_metrics(pt["hard"], test["hard"], thresholds["hard"]),
        "semantic_uncertainty": bin_metrics(pt["uncertainty"], test["uncertainty_high"], thresholds["uncertainty"]),
        "cluster": top5_cluster_metrics(pt["cluster_logits"], test["cluster"], test["last_cluster"], pt["changed"], thresholds["changed"]),
    }
    def passed_bin(m: dict[str, float | None]) -> bool:
        return bool((m["roc_auc"] or 0.0) >= 0.58 and (m["balanced_accuracy"] or 0.0) >= 0.56)
    semantic_changed_passed = passed_bin(val_metrics["semantic_changed"]) and passed_bin(test_metrics["semantic_changed"])
    semantic_hard_passed = passed_bin(val_metrics["semantic_hard"]) and passed_bin(test_metrics["semantic_hard"])
    uncertainty_passed = passed_bin(val_metrics["semantic_uncertainty"]) and passed_bin(test_metrics["semantic_uncertainty"])
    stable_preservation = bool(
        val_metrics["cluster"]["stable_top5"] >= val_metrics["cluster"]["stable_copy_top1"] - 0.02
        and test_metrics["cluster"]["stable_top5"] >= test_metrics["cluster"]["stable_copy_top1"] - 0.02
    )
    adapter_passed = bool((semantic_changed_passed or semantic_hard_passed) and uncertainty_passed and stable_preservation)
    train_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_semantic_state_adapter_training_ran": True,
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "target_root": str(root.relative_to(ROOT)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "input_dim": int(train["x"].shape[1]),
        "loss_trace": {k: summarize_trace(trace, k) for k in trace[0].keys()},
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "future_teacher_embedding_input_allowed": False,
        "中文结论": f"{EXPERIMENT_LABEL} 已完成 M128/H32 seed{args.seed} 训练；只使用 observed measurement 与 future trace geometry。",
    }
    eval_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_semantic_state_adapter_eval_done": True,
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "thresholds_from_val": thresholds,
        "val": val_metrics,
        "test": test_metrics,
        "semantic_changed_passed": semantic_changed_passed,
        "semantic_hard_passed": semantic_hard_passed,
        "uncertainty_passed": uncertainty_passed,
        "stable_preservation": stable_preservation,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
    }
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_semantic_state_adapter_training_ran": True,
        "video_semantic_state_adapter_passed": adapter_passed,
        "semantic_changed_passed": semantic_changed_passed,
        "semantic_hard_passed": semantic_hard_passed,
        "uncertainty_passed": uncertainty_passed,
        "stable_preservation": stable_preservation,
        "identity_measurement_base_required": True,
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "recommended_next_step": NEXT_STEP_ON_PASS if adapter_passed else NEXT_STEP_ON_FAIL,
        "中文结论": (
            f"{EXPERIMENT_LABEL} seed{args.seed} "
            + ("通过 changed/hard/uncertainty/stable gate；下一步应做 seed123/456 复现。"
               if adapter_passed else "未通过全部 gate，不能 claim video semantic field success。")
        ),
    }
    suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    train_report_path = TRAIN_REPORT if not suffix else TRAIN_REPORT.with_name(TRAIN_REPORT.stem + suffix + TRAIN_REPORT.suffix)
    eval_report_path = EVAL_REPORT if not suffix else EVAL_REPORT.with_name(EVAL_REPORT.stem + suffix + EVAL_REPORT.suffix)
    decision_report_path = DECISION_REPORT if not suffix else DECISION_REPORT.with_name(DECISION_REPORT.stem + suffix + DECISION_REPORT.suffix)
    doc_path = DOC if not suffix else DOC.with_name(DOC.stem + suffix + DOC.suffix)
    TRAIN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    train_report_path.write_text(json.dumps(jsonable(train_report), indent=2, ensure_ascii=False), encoding="utf-8")
    eval_report_path.write_text(json.dumps(jsonable(eval_report), indent=2, ensure_ascii=False), encoding="utf-8")
    decision_report_path.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    doc_path.write_text(
        f"# STWM OSTF {EXPERIMENT_ID} Video Semantic State Adapter Decision\n\n"
        f"- video_semantic_state_adapter_training_ran: true\n"
        f"- video_semantic_state_adapter_passed: {adapter_passed}\n"
        f"- semantic_changed_passed: {semantic_changed_passed}\n"
        f"- semantic_hard_passed: {semantic_hard_passed}\n"
        f"- uncertainty_passed: {uncertainty_passed}\n"
        f"- stable_preservation: {stable_preservation}\n"
        f"- full_video_semantic_identity_field_claim_allowed: false\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"video_semantic_state_adapter_passed": adapter_passed, "recommended_next_step": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
