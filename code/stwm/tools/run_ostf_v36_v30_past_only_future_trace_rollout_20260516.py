#!/usr/bin/env python3
"""V36: 用 frozen V30 从 past-only observed trace 预测未来 trace。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.modules.ostf_v33_integrated_semantic_identity_world_model import build_v30_from_checkpoint  # noqa: E402
from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402

INPUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_past_only_observed_trace_input/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_v30_past_only_future_trace_rollout/M128_H32"
CKPT = ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"
REPORT = ROOT / "reports/stwm_ostf_v36_v30_past_only_future_trace_rollout_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_V30_PAST_ONLY_FUTURE_TRACE_ROLLOUT_20260516.md"
LOG = ROOT / "outputs/logs/stwm_ostf_v36_v30_past_only_future_trace_rollout_20260516.log"


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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def list_npz(root: Path) -> list[Path]:
    return sorted(root.glob("*/*.npz"))


def safe_mean(vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


def fde(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float | None:
    vals = []
    for i in range(pred.shape[0]):
        idx = np.where(valid[i])[0]
        if idx.size:
            t = int(idx[-1])
            vals.append(float(np.linalg.norm(pred[i, t] - target[i, t])))
    return safe_mean(vals)


def ade(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float | None:
    mask = valid.astype(bool)
    if not mask.any():
        return None
    return float(np.linalg.norm(pred - target, axis=-1)[mask].mean())


def visibility_f1(pred: np.ndarray, target: np.ndarray) -> float | None:
    y = target.reshape(-1).astype(bool)
    if y.size == 0:
        return None
    p = (pred.reshape(-1) >= 0.5).astype(bool)
    if len(np.unique(y)) < 2:
        return float((p == y).mean())
    return float(f1_score(y, p))


def constant_velocity(obs: np.ndarray) -> np.ndarray:
    h = 32
    vel = obs[:, -1] - obs[:, -2]
    t = np.arange(1, h + 1, dtype=np.float32)[None, :, None]
    return obs[:, -1:, :] + vel[:, None, :] * t


def damped_velocity(obs: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    h = 32
    vel = obs[:, -1] - obs[:, -2]
    t = np.arange(1, h + 1, dtype=np.float32)[None, :, None]
    return obs[:, -1:, :] + float(gamma) * vel[:, None, :] * t


def last_observed_copy(obs: np.ndarray) -> np.ndarray:
    return np.repeat(obs[:, -1:, :], 32, axis=1)


def last_visible_copy(obs: np.ndarray, vis: np.ndarray) -> np.ndarray:
    out = np.zeros((obs.shape[0], 2), dtype=np.float32)
    for i in range(obs.shape[0]):
        idx = np.where(vis[i].astype(bool))[0]
        out[i] = obs[i, idx[-1]] if idx.size else obs[i, -1]
    return np.repeat(out[:, None, :], 32, axis=1)


def metric_pack(pred: np.ndarray, vis_pred: np.ndarray, target: np.ndarray, target_vis: np.ndarray) -> dict[str, float | None]:
    return {
        "ADE": ade(pred, target, target_vis),
        "FDE": fde(pred, target, target_vis),
        "visibility_F1": visibility_f1(vis_pred, target_vis),
    }


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    paths = list_npz(INPUT_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, args = build_v30_from_checkpoint(CKPT, map_location=device)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    with torch.no_grad():
        for idx, p in enumerate(paths, start=1):
            try:
                z = np.load(p, allow_pickle=True)
                split = str(np.asarray(z["split"]).item())
                obs = np.asarray(z["obs_points"], dtype=np.float32)
                obs_vis = np.asarray(z["obs_vis"], dtype=bool)
                obs_conf = np.asarray(z["obs_conf"], dtype=np.float32)
                tgt = np.asarray(z["future_trace_teacher_points"], dtype=np.float32)
                tgt_vis = np.asarray(z["future_trace_teacher_vis"], dtype=bool)
                sem_id = np.asarray(z["semantic_id"], dtype=np.int64)
                semantic_batch = torch.tensor([int(np.bincount(np.clip(sem_id, 0, 8191)).argmax())], device=device, dtype=torch.long) if sem_id.size else None
                out = model(
                    obs_points=torch.from_numpy(obs[None]).to(device),
                    obs_vis=torch.from_numpy(obs_vis[None]).to(device),
                    obs_conf=torch.from_numpy(obs_conf[None]).to(device),
                    semantic_id=semantic_batch,
                )
                pred = out["point_pred"][0].detach().cpu().numpy().astype(np.float32)
                pred_vis = torch.sigmoid(out["visibility_logits"][0]).detach().cpu().numpy().astype(np.float32)
                priors = {
                    "last_visible_copy": last_visible_copy(obs, obs_vis),
                    "last_observed_copy": last_observed_copy(obs),
                    "constant_velocity": constant_velocity(obs),
                    "damped_velocity": damped_velocity(obs),
                }
                prior_metrics = {name: metric_pack(pr, np.repeat(obs_vis[:, -1:, None], 32, axis=1).squeeze(-1), tgt, tgt_vis) for name, pr in priors.items()}
                v30_metrics = metric_pack(pred, pred_vis, tgt, tgt_vis)
                best_prior_name = min(prior_metrics, key=lambda k: float(prior_metrics[k]["ADE"] if prior_metrics[k]["ADE"] is not None else 1e18))
                best_prior_ade = prior_metrics[best_prior_name]["ADE"]
                v30_beats = bool(v30_metrics["ADE"] is not None and best_prior_ade is not None and float(v30_metrics["ADE"]) <= float(best_prior_ade))
                out_dir = OUT_ROOT / split
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / p.name
                np.savez_compressed(
                    out_path,
                    sample_uid=z["sample_uid"],
                    dataset=z["dataset"],
                    split=z["split"],
                    point_id=z["point_id"],
                    obs_points=obs,
                    obs_vis=obs_vis,
                    obs_conf=obs_conf,
                    predicted_future_points=pred,
                    predicted_future_vis=pred_vis,
                    predicted_future_conf=pred_vis,
                    future_trace_teacher_points=tgt,
                    future_trace_teacher_vis=tgt_vis,
                    future_trace_teacher_conf=np.asarray(z["future_trace_teacher_conf"], dtype=np.float32),
                    prior_last_visible_copy=priors["last_visible_copy"],
                    prior_last_observed_copy=priors["last_observed_copy"],
                    prior_constant_velocity=priors["constant_velocity"],
                    prior_damped_velocity=priors["damped_velocity"],
                    future_trace_predicted_from_past_only=np.asarray(True),
                    v30_backbone_frozen=np.asarray(True),
                    future_trace_teacher_input_allowed=np.asarray(False),
                    leakage_safe=np.asarray(True),
                    source_past_only_input_npz=np.asarray(rel(p)),
                )
                row = {
                    "sample_uid": str(np.asarray(z["sample_uid"]).item()),
                    "split": split,
                    "output_path": rel(out_path),
                    "v30_metrics": v30_metrics,
                    "prior_metrics": prior_metrics,
                    "strongest_prior": best_prior_name,
                    "v30_beats_strongest_prior": v30_beats,
                }
                rows.append(row)
                metric_rows.append(row)
                split_counts[split] += 1
                if idx % 25 == 0:
                    with LOG.open("a", encoding="utf-8") as f:
                        f.write(f"已完成 {idx}/{len(paths)} 个 causal V30 rollout\n")
            except Exception as e:  # noqa: BLE001
                failures.append({"path": rel(p), "reason": repr(e)})

    v30_ades = [float(r["v30_metrics"]["ADE"]) for r in metric_rows if r["v30_metrics"]["ADE"] is not None]
    v30_fdes = [float(r["v30_metrics"]["FDE"]) for r in metric_rows if r["v30_metrics"]["FDE"] is not None]
    v30_f1 = [float(r["v30_metrics"]["visibility_F1"]) for r in metric_rows if r["v30_metrics"]["visibility_F1"] is not None]
    prior_ades: dict[str, list[float]] = {}
    for r in metric_rows:
        for name, m in r["prior_metrics"].items():
            if m["ADE"] is not None:
                prior_ades.setdefault(name, []).append(float(m["ADE"]))
    prior_mean = {k: float(np.mean(v)) for k, v in prior_ades.items()}
    strongest_prior = min(prior_mean, key=prior_mean.get) if prior_mean else None
    v30_ade_mean = float(np.mean(v30_ades)) if v30_ades else None
    strongest_prior_ade = prior_mean.get(strongest_prior) if strongest_prior else None
    v30_beats_strongest_prior = bool(v30_ade_mean is not None and strongest_prior_ade is not None and v30_ade_mean <= strongest_prior_ade)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v30_past_only_future_trace_rollout_done": bool(rows and not failures),
        "input_root": rel(INPUT_ROOT),
        "output_root": rel(OUT_ROOT),
        "v30_checkpoint": rel(CKPT),
        "v30_checkpoint_loaded": CKPT.exists(),
        "v30_backbone_frozen": not any(p.requires_grad for p in model.parameters()),
        "future_trace_predicted_from_past_only": bool(rows and not failures),
        "sample_count": len(rows),
        "split_counts": dict(split_counts),
        "ADE_mean": v30_ade_mean,
        "FDE_mean": float(np.mean(v30_fdes)) if v30_fdes else None,
        "visibility_F1_mean": float(np.mean(v30_f1)) if v30_f1 else None,
        "minFDE": float(np.min(v30_fdes)) if v30_fdes else None,
        "prior_ADE_mean": prior_mean,
        "strongest_prior": strongest_prior,
        "strongest_prior_ADE_mean": strongest_prior_ade,
        "v30_beats_strongest_prior": v30_beats_strongest_prior,
        "trajectory_degraded": bool(not v30_ades or not np.isfinite(v30_ade_mean)),
        "rows": rows,
        "exact_blockers": failures,
        "中文总结": (
            "V36 已用 frozen V30 从 past-only observed trace 预测 future trace，并与 full-clip teacher trace 仅做 target 对比。"
            if rows
            else "V36 V30 rollout 没有可用输出，需要检查 past-only 输入或 checkpoint。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V36 V30 Past-Only Future Trace Rollout\n\n"
        f"- v30_checkpoint_loaded: {report['v30_checkpoint_loaded']}\n"
        f"- v30_backbone_frozen: {report['v30_backbone_frozen']}\n"
        f"- future_trace_predicted_from_past_only: {report['future_trace_predicted_from_past_only']}\n"
        f"- sample_count: {len(rows)}\n"
        f"- ADE_mean: {report['ADE_mean']}\n"
        f"- FDE_mean: {report['FDE_mean']}\n"
        f"- visibility_F1_mean: {report['visibility_F1_mean']}\n"
        f"- strongest_prior: {strongest_prior}\n"
        f"- v30_beats_strongest_prior: {v30_beats_strongest_prior}\n"
        f"- trajectory_degraded: {report['trajectory_degraded']}\n\n"
        "## 中文总结\n"
        + report["中文总结"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"V30因果rollout完成": bool(rows and not failures), "样本数": len(rows), "V30赢strongest_prior": v30_beats_strongest_prior}, ensure_ascii=False), flush=True)
    return 0 if rows and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
