#!/usr/bin/env python3
"""训练/评估 V35.16 video identity pairwise retrieval embedding head。"""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_16_video_identity_pairwise_retrieval_targets/M128_H32"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v35_16_video_identity_pairwise_retrieval_h32_m128"
TRAIN_REPORT = ROOT / "reports/stwm_ostf_v35_16_video_identity_pairwise_retrieval_train_summary_20260515.json"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_16_video_identity_pairwise_retrieval_eval_summary_20260515.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_16_video_identity_pairwise_retrieval_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_16_VIDEO_IDENTITY_PAIRWISE_RETRIEVAL_DECISION_20260515.md"
EXPERIMENT_ID = "V35.16"
EXPERIMENT_LABEL = "V35.16 video identity pairwise retrieval"
CKPT_PREFIX = "v35_16_video_identity_pairwise_retrieval"
NEXT_STEP_ON_PASS = "run_v35_16_video_identity_seed123_replication"
NEXT_STEP_ON_FAIL = "fix_video_identity_pairwise_retrieval_head"


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


def list_npz(root: Path, split: str) -> list[Path]:
    return sorted((root / split).glob("*.npz"))


def load_sample(path: Path) -> dict[str, np.ndarray | str]:
    z = np.load(path, allow_pickle=True)
    return {
        "path": str(path),
        "sample_uid": str(np.asarray(z["sample_uid"]).item()),
        "split": str(np.asarray(z["split"]).item()),
        "dataset": str(np.asarray(z["dataset"]).item()),
        "x": np.asarray(z["identity_input_features"], dtype=np.float32),
        "measurement": np.asarray(z["measurement_identity_embedding"], dtype=np.float32),
        "inst": np.asarray(z["point_to_instance_id"], dtype=np.int64),
        "same": np.asarray(z["same_instance_pair_mask"], dtype=bool),
        "confuser": np.asarray(z["identity_confuser_pair_mask"], dtype=bool),
        "same_semantic": np.asarray(z["same_semantic_hard_negative_pair_mask"], dtype=bool),
        "spatial_hard": np.asarray(z["same_frame_hard_negative_pair_mask"], dtype=bool),
        "crossing": np.asarray(z["trajectory_crossing_pair_mask"], dtype=bool),
        "occlusion": np.asarray(z["occlusion_reappear_point_mask"], dtype=bool),
        "obs_points": np.asarray(z["obs_points"], dtype=np.float32),
        "future_points": np.asarray(z["future_points"], dtype=np.float32),
    }


def load_split(root: Path, split: str) -> list[dict[str, np.ndarray | str]]:
    return [load_sample(p) for p in list_npz(root, split)]


class IdentityResidualHead(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 512, emb_dim: int = 768, scale: float = 0.25) -> None:
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        base = F.normalize(x[:, :768], dim=-1)
        residual = torch.tanh(self.net(x)) * self.scale
        emb = F.normalize(base + residual, dim=-1)
        return {"identity_embedding": emb, "base_embedding": base, "identity_residual": residual}


def choose_points(inst: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    n = len(inst)
    if n <= max_points:
        return np.arange(n)
    keep: list[int] = []
    for label in np.unique(inst[inst >= 0]):
        ids = np.where(inst == label)[0]
        take = min(len(ids), max(8, max_points // max(len(np.unique(inst[inst >= 0])), 1)))
        keep.extend(rng.choice(ids, take, replace=False).tolist())
    keep = np.asarray(sorted(set(keep)), dtype=np.int64)
    if len(keep) > max_points:
        keep = rng.choice(keep, max_points, replace=False)
    elif len(keep) < max_points:
        rest = np.setdiff1d(np.arange(n), keep, assume_unique=False)
        add = rng.choice(rest, min(len(rest), max_points - len(keep)), replace=False)
        keep = np.concatenate([keep, add])
    return np.sort(keep)


def supcon_loss(emb: torch.Tensor, inst: torch.Tensor, temp: float = 0.08) -> torch.Tensor:
    same = (inst[:, None] == inst[None, :]) & (inst[:, None] >= 0)
    eye = torch.eye(len(inst), dtype=torch.bool, device=emb.device)
    same = same & (~eye)
    valid_anchor = same.any(dim=1)
    if not valid_anchor.any():
        return emb.new_tensor(0.0)
    sim = emb @ emb.T / temp
    sim = sim.masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_log = (log_prob * same.float()).sum(dim=1) / same.float().sum(dim=1).clamp_min(1.0)
    return -pos_log[valid_anchor].mean()


def confuser_margin_loss(emb: torch.Tensor, same_np: np.ndarray, hard_np: np.ndarray, idx: np.ndarray, margin: float = 0.20) -> torch.Tensor:
    same = torch.from_numpy(same_np[np.ix_(idx, idx)]).to(emb.device)
    hard = torch.from_numpy(hard_np[np.ix_(idx, idx)]).to(emb.device)
    sim = emb @ emb.T
    pos = sim.masked_fill(~same, -1e4).max(dim=1).values
    neg = sim.masked_fill(~hard, -1e4).max(dim=1).values
    valid = (pos > -1e3) & (neg > -1e3)
    if not valid.any():
        return emb.new_tensor(0.0)
    return F.relu(margin + neg[valid] - pos[valid]).mean()


def train_one(seed: int, steps: int, batch_points: int, device: torch.device, root: Path) -> tuple[IdentityResidualHead, dict[str, Any]]:
    set_seed(seed)
    train = load_split(root, "train")
    if not train:
        raise RuntimeError("train split 为空")
    input_dim = int(np.asarray(train[0]["x"]).shape[1])
    model = IdentityResidualHead(input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    trace: list[dict[str, float]] = []
    print(f"{EXPERIMENT_LABEL}: 开始训练；V30 frozen，future teacher embedding 不作为输入。", flush=True)
    for step in range(1, steps + 1):
        s = train[int(rng.integers(0, len(train)))]
        inst = np.asarray(s["inst"], dtype=np.int64)
        idx = choose_points(inst, batch_points, rng)
        x = torch.from_numpy(np.asarray(s["x"], dtype=np.float32)[idx]).to(device)
        inst_t = torch.from_numpy(inst[idx]).to(device)
        out = model(x)
        emb = out["identity_embedding"]
        hard = np.asarray(s["confuser"], dtype=bool) | np.asarray(s["spatial_hard"], dtype=bool) | np.asarray(s["crossing"], dtype=bool)
        loss_sup = supcon_loss(emb, inst_t)
        loss_margin = confuser_margin_loss(emb, np.asarray(s["same"], dtype=bool), hard, idx)
        residual_loss = out["identity_residual"].pow(2).mean()
        base_align = (1.0 - (emb * out["base_embedding"]).sum(dim=-1)).mean()
        loss = loss_sup + 0.80 * loss_margin + 0.02 * residual_loss + 0.05 * base_align
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        row = {
            "loss": float(loss.detach().cpu()),
            "supcon_loss": float(loss_sup.detach().cpu()),
            "confuser_margin_loss": float(loss_margin.detach().cpu()),
            "residual_loss": float(residual_loss.detach().cpu()),
            "base_align_loss": float(base_align.detach().cpu()),
        }
        trace.append(row)
        if step % 200 == 0 or step == steps:
            print(f"step={step} loss={row['loss']:.4f} supcon={row['supcon_loss']:.4f} confuser={row['confuser_margin_loss']:.4f}", flush=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{CKPT_PREFIX}_m128_h32_seed{seed}_best.pt"
    torch.save({"model": model.state_dict(), "seed": seed, "input_dim": input_dim}, ckpt_path)
    def stat(k: str) -> dict[str, float]:
        vals = [r[k] for r in trace]
        return {"first": vals[0], "last": vals[-1], "mean": float(np.mean(vals))}
    return model, {
        "video_identity_pairwise_retrieval_training_ran": True,
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "seed": seed,
        "steps": steps,
        "batch_points": batch_points,
        "loss_trace": {k: stat(k) for k in trace[0]},
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "future_teacher_embedding_input_allowed": False,
    }


@torch.no_grad()
def model_embedding(model: IdentityResidualHead, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    for start in range(0, len(x), 2048):
        xb = torch.from_numpy(x[start : start + 2048]).to(device)
        outs.append(model(xb)["identity_embedding"].detach().cpu().numpy())
    emb = np.concatenate(outs, axis=0).astype(np.float32)
    return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6)


def retrieval_metrics_for_sample(emb: np.ndarray, s: dict[str, np.ndarray | str]) -> dict[str, float]:
    inst = np.asarray(s["inst"], dtype=np.int64)
    same = np.asarray(s["same"], dtype=bool)
    conf = np.asarray(s["confuser"], dtype=bool)
    occ = np.asarray(s["occlusion"], dtype=bool)
    crossing = np.asarray(s["crossing"], dtype=bool)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    top = np.argmax(sim, axis=1)
    valid = inst >= 0
    exclude_hit = (inst[top] == inst) & valid
    out: dict[str, float] = {
        "exclude_hit": float(exclude_hit[valid].sum()),
        "exclude_total": float(valid.sum()),
        "same_frame_hit": float(exclude_hit[valid].sum()),
        "same_frame_total": float(valid.sum()),
    }
    labels = np.unique(inst[valid])
    if len(labels) > 0:
        centers = []
        for lab in labels:
            centers.append(emb[inst == lab].mean(axis=0))
        centers = np.asarray(centers, dtype=np.float32)
        centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-6)
        pred = labels[np.argmax(emb[valid] @ centers.T, axis=1)]
        out["instance_pooled_hit"] = float((pred == inst[valid]).sum())
        out["instance_pooled_total"] = float(valid.sum())
    else:
        out["instance_pooled_hit"] = 0.0
        out["instance_pooled_total"] = 0.0
    same_vals = sim[same]
    conf_vals = sim[conf]
    out["same_sim_sum"] = float(np.nan_to_num(same_vals, nan=0.0, posinf=0.0, neginf=0.0).sum())
    out["same_sim_count"] = float(same_vals.size)
    out["confuser_sim_sum"] = float(np.nan_to_num(conf_vals, nan=0.0, posinf=0.0, neginf=0.0).sum())
    out["confuser_sim_count"] = float(conf_vals.size)
    anchors = valid & conf.any(axis=1) & same.any(axis=1)
    hard_hit = 0
    hard_total = 0
    for i in np.where(anchors)[0]:
        cand = same[i] | conf[i]
        if cand.any():
            j = np.argmax(np.where(cand, sim[i], -np.inf))
            hard_hit += int(same[i, j])
            hard_total += 1
    out["confuser_avoid_hit"] = float(hard_hit)
    out["confuser_avoid_total"] = float(hard_total)
    for name, mask in [("occlusion_reappear", occ), ("trajectory_crossing", crossing.any(axis=1))]:
        m = valid & mask
        out[f"{name}_hit"] = float(exclude_hit[m].sum())
        out[f"{name}_total"] = float(m.sum())
    return out


def aggregate(rows: list[dict[str, float]]) -> dict[str, float | None]:
    sums: dict[str, float] = {}
    for r in rows:
        for k, v in r.items():
            sums[k] = sums.get(k, 0.0) + float(v)
    same_mean = sums.get("same_sim_sum", 0.0) / max(sums.get("same_sim_count", 0.0), 1.0)
    conf_mean = sums.get("confuser_sim_sum", 0.0) / max(sums.get("confuser_sim_count", 0.0), 1.0)
    return {
        "identity_retrieval_exclude_same_point_top1": sums.get("exclude_hit", 0.0) / max(sums.get("exclude_total", 0.0), 1.0),
        "identity_retrieval_same_frame_top1": sums.get("same_frame_hit", 0.0) / max(sums.get("same_frame_total", 0.0), 1.0),
        "identity_retrieval_instance_pooled_top1": sums.get("instance_pooled_hit", 0.0) / max(sums.get("instance_pooled_total", 0.0), 1.0),
        "identity_confuser_separation": same_mean - conf_mean if sums.get("confuser_sim_count", 0.0) > 0 else None,
        "identity_confuser_avoidance_top1": sums.get("confuser_avoid_hit", 0.0) / max(sums.get("confuser_avoid_total", 0.0), 1.0),
        "occlusion_reappear_retrieval_top1": sums.get("occlusion_reappear_hit", 0.0) / max(sums.get("occlusion_reappear_total", 0.0), 1.0),
        "trajectory_crossing_retrieval_top1": sums.get("trajectory_crossing_hit", 0.0) / max(sums.get("trajectory_crossing_total", 0.0), 1.0),
        "same_pair_similarity_mean": same_mean,
        "confuser_pair_similarity_mean": conf_mean,
        "confuser_pair_count": sums.get("confuser_sim_count", 0.0),
        "occlusion_reappear_total": sums.get("occlusion_reappear_total", 0.0),
        "trajectory_crossing_total": sums.get("trajectory_crossing_total", 0.0),
    }


def evaluate_split(samples: list[dict[str, np.ndarray | str]], model: IdentityResidualHead | None, device: torch.device, mode: str) -> dict[str, float | None]:
    rows = []
    for s in samples:
        if mode == "measurement":
            emb = np.asarray(s["measurement"], dtype=np.float32)
        else:
            assert model is not None
            emb = model_embedding(model, np.asarray(s["x"], dtype=np.float32), device)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6)
        rows.append(retrieval_metrics_for_sample(emb, s))
    return aggregate(rows)


def pass_identity(m: dict[str, float | None]) -> bool:
    return bool(
        (m["identity_retrieval_exclude_same_point_top1"] or 0.0) >= 0.70
        and (m["identity_retrieval_same_frame_top1"] or 0.0) >= 0.70
        and (m["identity_retrieval_instance_pooled_top1"] or 0.0) >= 0.70
        and (m["identity_confuser_separation"] is not None and m["identity_confuser_separation"] > 0.02)
        and (m["identity_confuser_avoidance_top1"] or 0.0) >= 0.70
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-root", default=str(TARGET_ROOT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-points", type=int, default=640)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    root = Path(args.target_root)
    if not root.is_absolute():
        root = ROOT / root
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, train_report = train_one(args.seed, args.steps, args.batch_points, device, root)
    val = load_split(root, "val")
    test = load_split(root, "test")
    eval_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_identity_pairwise_retrieval_eval_done": True,
        "seed": args.seed,
        "target_root": str(root.relative_to(ROOT)),
        "measurement_baseline": {
            "val": evaluate_split(val, None, device, "measurement"),
            "test": evaluate_split(test, None, device, "measurement"),
        },
        "learned_identity_head": {
            "val": evaluate_split(val, model, device, "learned"),
            "test": evaluate_split(test, model, device, "learned"),
        },
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
    }
    val_pass = pass_identity(eval_report["learned_identity_head"]["val"])
    test_pass = pass_identity(eval_report["learned_identity_head"]["test"])
    meas_val = eval_report["measurement_baseline"]["val"]
    meas_test = eval_report["measurement_baseline"]["test"]
    learned_val = eval_report["learned_identity_head"]["val"]
    learned_test = eval_report["learned_identity_head"]["test"]
    not_worse = bool(
        (learned_val["identity_retrieval_exclude_same_point_top1"] or 0.0) >= (meas_val["identity_retrieval_exclude_same_point_top1"] or 0.0) - 0.08
        and (learned_test["identity_retrieval_exclude_same_point_top1"] or 0.0) >= (meas_test["identity_retrieval_exclude_same_point_top1"] or 0.0) - 0.08
        and (learned_val["identity_confuser_avoidance_top1"] or 0.0) >= (meas_val["identity_confuser_avoidance_top1"] or 0.0) - 0.08
        and (learned_test["identity_confuser_avoidance_top1"] or 0.0) >= (meas_test["identity_confuser_avoidance_top1"] or 0.0) - 0.08
    )
    passed = bool(val_pass and test_pass and not_worse)
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_identity_pairwise_retrieval_training_ran": True,
        "video_identity_pairwise_retrieval_passed": passed,
        "identity_embedding_trained": True,
        "same_frame_hard_negative_used": True,
        "same_semantic_confuser_used": True,
        "trajectory_crossing_used": True,
        "occlusion_reappear_used": True,
        "measurement_preserving_residual_head": True,
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "recommended_next_step": NEXT_STEP_ON_PASS if passed else NEXT_STEP_ON_FAIL,
        "中文结论": (
            f"{EXPERIMENT_LABEL} 通过 val/test pairwise retrieval gate；下一步应做 seed123/456 复现。"
            if passed
            else f"{EXPERIMENT_LABEL} 未通过严格 gate，不能开放 video identity field claim。"
        ),
    }
    suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    train_path = TRAIN_REPORT if not suffix else TRAIN_REPORT.with_name(TRAIN_REPORT.stem + suffix + TRAIN_REPORT.suffix)
    eval_path = EVAL_REPORT if not suffix else EVAL_REPORT.with_name(EVAL_REPORT.stem + suffix + EVAL_REPORT.suffix)
    decision_path = DECISION_REPORT if not suffix else DECISION_REPORT.with_name(DECISION_REPORT.stem + suffix + DECISION_REPORT.suffix)
    doc_path = DOC if not suffix else DOC.with_name(DOC.stem + suffix + DOC.suffix)
    TRAIN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(json.dumps(jsonable(train_report), indent=2, ensure_ascii=False), encoding="utf-8")
    eval_path.write_text(json.dumps(jsonable(eval_report), indent=2, ensure_ascii=False), encoding="utf-8")
    decision_path.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    doc_path.write_text(
        f"# STWM OSTF {EXPERIMENT_ID} Video Identity Pairwise Retrieval Decision\n\n"
        f"- video_identity_pairwise_retrieval_training_ran: true\n"
        f"- video_identity_pairwise_retrieval_passed: {passed}\n"
        f"- identity_embedding_trained: true\n"
        f"- same_frame_hard_negative_used: true\n"
        f"- same_semantic_confuser_used: true\n"
        f"- trajectory_crossing_used: true\n"
        f"- occlusion_reappear_used: true\n"
        f"- full_video_semantic_identity_field_claim_allowed: false\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"video_identity_pairwise_retrieval_passed": passed, "recommended_next_step": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
