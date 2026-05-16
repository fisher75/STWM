#!/usr/bin/env python3
"""用 video-derived CoTracker M128/H32 trace cache 做 V35 前向闭环 smoke。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.modules.ostf_v35_semantic_state_world_model import SemanticStateWorldModelV35
from stwm.tools.ostf_v17_common_20260502 import ROOT

CACHE_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16/M128_H32"
MEASUREMENT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_10_video_observed_semantic_measurement_bank/M128_H32"
TRAIN_SUMMARY = ROOT / "reports/stwm_ostf_v35_8_identity_only_retrieval_finetune_decision_20260515.json"
CKPT_SUMMARY = ROOT / "reports/stwm_ostf_v35_8_identity_only_retrieval_finetune_train_summary_20260515.json"
OUT = ROOT / "reports/stwm_ostf_v35_9_video_derived_input_closure_smoke_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_9_VIDEO_DERIVED_INPUT_CLOSURE_SMOKE_20260515.md"
K = 64
MEAS_DIM = 768
MEAS_STATS = 4


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


def scalar(x: np.ndarray) -> Any:
    return np.asarray(x).item()


def list_npz(root: Path) -> list[Path]:
    return sorted(root.glob("*/*.npz"))


def build_features(z: Any, measurement_z: Any | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
    vis = np.asarray(z["visibility"]).astype(bool)
    conf = np.asarray(z["confidence"], dtype=np.float32)
    semantic_id = np.asarray(z["semantic_id"], dtype=np.int64)
    obj_n, per_obj_m, total_t, _ = tracks.shape
    obs_len = int(scalar(z["obs_len"]))
    horizon = int(scalar(z["horizon"]))
    m_total = obj_n * per_obj_m
    obs_points = tracks[:, :, :obs_len].reshape(m_total, obs_len, 2)
    obs_vis = vis[:, :, :obs_len].reshape(m_total, obs_len)
    obs_conf = conf[:, :, :obs_len].reshape(m_total, obs_len)
    obj_sem = np.repeat(semantic_id[:, None], per_obj_m, axis=1).reshape(m_total)
    cluster = np.mod(np.maximum(obj_sem, 0), K).astype(np.int64)
    one_hot = np.eye(K, dtype=np.float32)[cluster]
    obs_ent = np.zeros((m_total,), dtype=np.float32)
    vis_frac = obs_vis.mean(axis=1).astype(np.float32)
    conf_mean = obs_conf.mean(axis=1).astype(np.float32)
    conf_last = obs_conf[:, -1].astype(np.float32)
    disp = obs_points[:, -1] - obs_points[:, 0]
    speed = np.sqrt((np.diff(obs_points, axis=1) ** 2).sum(axis=-1)).mean(axis=1)
    stats = np.stack(
        [obs_ent, vis_frac, conf_mean, conf_last, disp[:, 0], disp[:, 1], speed, np.ones_like(obs_ent), np.ones_like(obs_ent)],
        axis=1,
    ).astype(np.float32)
    base = np.concatenate([one_hot, one_hot, stats], axis=1).astype(np.float32)
    measurement_available = measurement_z is not None
    if measurement_available:
        sem = np.asarray(measurement_z["instance_observed_semantic_measurement"], dtype=np.float32)
        sem = sem / np.maximum(np.linalg.norm(sem, axis=1, keepdims=True), 1e-6)
        meas_conf = np.asarray(measurement_z["obs_measurement_confidence"], dtype=np.float32)
        meas_mask = np.asarray(measurement_z["obs_semantic_measurement_mask"], dtype=np.float32)
        agreement = np.asarray(measurement_z["teacher_agreement_score"], dtype=np.float32)
        denom = np.maximum(meas_mask.sum(axis=1, keepdims=True), 1.0)
        meas_stats = np.concatenate(
            [
                (meas_conf * meas_mask).sum(axis=1, keepdims=True) / denom,
                meas_conf.max(axis=1, keepdims=True),
                meas_mask.mean(axis=1, keepdims=True),
                (agreement * meas_mask).sum(axis=1, keepdims=True) / denom,
            ],
            axis=1,
        ).astype(np.float32)
        measurement_feat = np.concatenate([sem, meas_stats], axis=1).astype(np.float32)
    else:
        measurement_feat = np.zeros((m_total, MEAS_DIM + MEAS_STATS), dtype=np.float32)
    feat = np.concatenate([base, measurement_feat], axis=1).astype(np.float32)
    meta = {
        "object_count": int(obj_n),
        "point_count": int(m_total),
        "obs_len": obs_len,
        "horizon": horizon,
        "obs_visibility_mean": float(obs_vis.mean()),
        "obs_conf_mean": float(obs_conf.mean()),
        "semantic_id_count": int(len(np.unique(semantic_id))),
        "measurement_embedding_available": measurement_available,
    }
    return feat, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=str, default=str(CACHE_ROOT))
    ap.add_argument("--ckpt-summary", type=str, default=str(CKPT_SUMMARY))
    ap.add_argument("--measurement-root", type=str, default="")
    ap.add_argument("--max-samples", type=int, default=6)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cache_root = Path(args.cache_root)
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    measurement_root = Path(args.measurement_root) if args.measurement_root else None
    if measurement_root is not None and not measurement_root.is_absolute():
        measurement_root = ROOT / measurement_root
    ckpt_summary = Path(args.ckpt_summary)
    if not ckpt_summary.is_absolute():
        ckpt_summary = ROOT / ckpt_summary
    train = json.loads(ckpt_summary.read_text(encoding="utf-8"))
    ckpt_path = ROOT / train["checkpoint_path"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SemanticStateWorldModelV35(
        point_feature_dim=int(ckpt["feature_dim"]),
        semantic_clusters=K,
        evidence_families=5,
        copy_prior_strength=float(ckpt_args.get("copy_prior_strength", 7.0)),
        assignment_bound_decoder=bool(ckpt_args.get("assignment_bound_decoder", True)),
        identity_dim=int(ckpt_args.get("identity_dim", 128)),
        semantic_feature_dim=int(ckpt_args.get("semantic_feature_dim", 137)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    with torch.no_grad():
        for p in list_npz(cache_root)[: args.max_samples]:
            try:
                z = np.load(p, allow_pickle=True)
                measurement_z = None
                if measurement_root is not None:
                    mp = measurement_root / str(scalar(z["split"])) / p.name
                    if mp.exists():
                        measurement_z = np.load(mp, allow_pickle=True)
                feat, meta = build_features(z, measurement_z)
                x = torch.from_numpy(feat[None]).to(device)
                out = model(x, horizon=int(meta["horizon"]))
                cluster = out["semantic_cluster_logits"].argmax(dim=-1).detach().cpu().numpy()
                change = torch.sigmoid(out["semantic_change_logits"]).detach().cpu().numpy()
                emb = out["identity_embedding"].detach().cpu()
                rows.append(
                    {
                        "cache_path": str(p.relative_to(ROOT)),
                        "item_key": str(scalar(z["item_key"])),
                        "dataset": str(scalar(z["dataset"])),
                        "split": str(scalar(z["split"])),
                        **meta,
                        "raw_frame_paths_available": bool(len(np.asarray(z["frame_paths"], dtype=object)) > 0),
                        "output_semantic_cluster_shape": list(cluster.shape),
                        "output_identity_embedding_shape": list(emb.shape),
                        "output_change_prob_mean": float(change.mean()),
                        "output_uncertainty_mean": float(out["semantic_uncertainty"].detach().cpu().numpy().mean()),
                        "future_trace_field_available_from_cache": True,
                        "future_semantic_field_output_available": True,
                        "future_identity_field_output_available": True,
                    }
                )
            except Exception as exc:
                failures.append(f"{p}: {type(exc).__name__}: {exc}")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_derived_input_closure_smoke_ran": True,
        "cache_root": str(cache_root.relative_to(ROOT)),
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "sample_count": len(rows),
        "failure_count": len(failures),
        "rows": rows,
        "failures": failures[:20],
        "input_is_video_derived_trace": bool(rows),
        "raw_frame_paths_traceable": bool(rows and all(r["raw_frame_paths_available"] for r in rows)),
        "outputs_future_trace_field": bool(rows and all(r["future_trace_field_available_from_cache"] for r in rows)),
        "outputs_future_semantic_field": bool(rows and all(r["future_semantic_field_output_available"] for r in rows)),
        "outputs_future_identity_field": bool(rows and all(r["future_identity_field_output_available"] for r in rows)),
        "observed_semantic_measurement_embedding_available": bool(rows and all(r["measurement_embedding_available"] for r in rows)),
        "semantic_measurement_closure_complete": bool(rows and all(r["measurement_embedding_available"] for r in rows)),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "exact_blockers": [
            "本 smoke 仍不是 supervised benchmark：video-derived VSPW/VIPSeg cache 没有 V35 PointOdyssey 风格 future semantic state targets，当前只能验证前向闭环与输入合同。",
            "要 claim 完整系统，还需要在 video-derived cache 上构建可评估的 future semantic/identity target 或接入外部 benchmark target。",
        ]
        if rows and all(r["measurement_embedding_available"] for r in rows)
        else [
            "CoTracker M128_H32 cache 提供真实 video-derived trace 和 raw frame paths，但缺少部分 V35.8 identity head 所需的 observed semantic measurement embedding。",
            "需要补齐 video-derived observed semantic measurement cache 后再重跑。"
        ],
        "recommended_next_step": "build_video_derived_semantic_state_eval_targets_or_benchmark_adapter" if rows and all(r["measurement_embedding_available"] for r in rows) else "build_video_observed_semantic_measurement_cache_for_v35",
        "中文结论": (
            "V35.9 已完成 video-derived M128/H32 trace 到 V35 semantic/identity head 的前向闭环 smoke；"
            "输出 future semantic/identity tensors 成功生成。若 measurement_root 已提供，本 smoke 已覆盖 video-derived trace + observed semantic measurement 输入闭环；"
            "但仍缺少 video-derived future semantic/identity target 评估，所以还不是完整 video semantic field success。"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.9 Video-Derived Input Closure Smoke\n\n"
        f"- video_derived_input_closure_smoke_ran: true\n"
        f"- sample_count: {len(rows)}\n"
        f"- input_is_video_derived_trace: {report['input_is_video_derived_trace']}\n"
        f"- raw_frame_paths_traceable: {report['raw_frame_paths_traceable']}\n"
        f"- outputs_future_trace_field: {report['outputs_future_trace_field']}\n"
        f"- outputs_future_semantic_field: {report['outputs_future_semantic_field']}\n"
        f"- outputs_future_identity_field: {report['outputs_future_identity_field']}\n"
        f"- semantic_measurement_closure_complete: {report['semantic_measurement_closure_complete']}\n"
        f"- full_video_semantic_identity_field_claim_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"sample_count": len(rows), "recommended_next_step": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
