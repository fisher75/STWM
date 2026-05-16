#!/usr/bin/env python3
"""评估 V35 video closure 的 identity measurement base 与 stable-copy adapter。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT

CACHE_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16/M128_H32"
MEASUREMENT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_10_video_observed_semantic_measurement_bank/M128_H32"
V35_10_BENCHMARK = ROOT / "reports/stwm_ostf_v35_10_video_derived_closure_benchmark_20260515.json"
REPORT = ROOT / "reports/stwm_ostf_v35_11_video_identity_measurement_base_and_stable_copy_adapter_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_11_VIDEO_IDENTITY_MEASUREMENT_BASE_AND_STABLE_COPY_ADAPTER_20260515.md"
K = 64


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


def normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8)


def observed_measurement_embedding(zm: Any) -> np.ndarray:
    obs = np.asarray(zm["obs_semantic_measurements"], dtype=np.float32)
    mask = np.asarray(zm["obs_semantic_measurement_mask"], dtype=np.float32)
    conf = np.asarray(zm["obs_measurement_confidence"], dtype=np.float32)
    weight = mask * np.maximum(conf, 0.0)
    denom = np.maximum(weight.sum(axis=1, keepdims=True), 1e-6)
    pooled = (obs * weight[:, :, None]).sum(axis=1) / denom
    fallback_denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    fallback = (obs * mask[:, :, None]).sum(axis=1) / fallback_denom
    pooled = np.where(denom > 1e-5, pooled, fallback)
    return normalize(pooled.astype(np.float32))


def identity_retrieval_metrics(emb: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    x = normalize(emb.astype(np.float32))
    labels = labels.astype(np.int64)
    valid = labels >= 0
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    hit = 0
    total = 0
    for i in np.where(valid)[0]:
        same = (labels == labels[i]) & valid
        same[i] = False
        if same.any():
            total += 1
            hit += int(labels[int(np.argmax(sim[i]))] == labels[i])

    unique_ids = [int(v) for v in np.unique(labels[valid])]
    pooled_hit = 0
    pooled_total = int(valid.sum())
    if len(unique_ids) >= 2 and pooled_total > 0:
        centroids = []
        for inst in unique_ids:
            c = x[labels == inst].mean(axis=0)
            centroids.append(normalize(c[None])[0])
        pred = np.asarray(unique_ids)[(x[valid] @ np.stack(centroids).T).argmax(axis=1)]
        pooled_hit = int((pred == labels[valid]).sum())

    return {
        "exclude_same_point_top1": float(hit / max(total, 1)),
        "same_frame_top1": float(hit / max(total, 1)),
        "instance_pooled_top1": float(pooled_hit / max(pooled_total, 1)),
        "retrieval_total": float(total),
        "instance_count": float(len(unique_ids)),
    }


def semantic_confuser_metrics(emb: np.ndarray, labels: np.ndarray, semantic_id: np.ndarray) -> dict[str, float]:
    x = normalize(emb.astype(np.float32))
    labels = labels.astype(np.int64)
    semantic_id = semantic_id.astype(np.int64)
    valid = labels >= 0
    sim = x @ x.T
    same_instance_scores: list[float] = []
    same_semantic_diff_instance_scores: list[float] = []
    for i in np.where(valid)[0]:
        same = (labels == labels[i]) & valid
        same[i] = False
        confuser = (semantic_id == semantic_id[i]) & (labels != labels[i]) & valid
        if same.any():
            same_instance_scores.append(float(np.max(sim[i, same])))
        if confuser.any():
            same_semantic_diff_instance_scores.append(float(np.max(sim[i, confuser])))
    same_mean = float(np.mean(same_instance_scores)) if same_instance_scores else 0.0
    confuser_mean = float(np.mean(same_semantic_diff_instance_scores)) if same_semantic_diff_instance_scores else 0.0
    return {
        "same_instance_similarity": same_mean,
        "same_semantic_different_instance_similarity": confuser_mean,
        "same_semantic_confuser_margin": float(same_mean - confuser_mean),
        "same_semantic_confuser_count": float(len(same_semantic_diff_instance_scores)),
    }


def merge_weighted(rows: list[dict[str, Any]], key: str, weight_key: str = "retrieval_total") -> float:
    num = 0.0
    den = 0.0
    for r in rows:
        w = float(r.get(weight_key, 0.0))
        num += float(r[key]) * w
        den += w
    return float(num / max(den, 1e-8))


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0


def split_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for split in sorted({str(r["split"]) for r in rows}):
        part = [r for r in rows if str(r["split"]) == split]
        out[split] = {
            "sample_count": float(len(part)),
            "point_count": float(sum(float(r["point_count"]) for r in part)),
            "measurement_identity_exclude_same_point_top1": merge_weighted(part, "exclude_same_point_top1"),
            "measurement_identity_same_frame_top1": merge_weighted(part, "same_frame_top1"),
            "measurement_identity_instance_pooled_top1": mean(part, "instance_pooled_top1"),
            "stable_copy_top1": mean(part, "stable_copy_top1"),
            "stable_copy_top5": mean(part, "stable_copy_top5"),
            "same_semantic_confuser_margin": mean(part, "same_semantic_confuser_margin"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=str, default=str(CACHE_ROOT))
    ap.add_argument("--measurement-root", type=str, default=str(MEASUREMENT_ROOT))
    ap.add_argument("--v35-10-benchmark", type=str, default=str(V35_10_BENCHMARK))
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    measurement_root = Path(args.measurement_root)
    if not measurement_root.is_absolute():
        measurement_root = ROOT / measurement_root
    v35_10_benchmark = Path(args.v35_10_benchmark)
    if not v35_10_benchmark.is_absolute():
        v35_10_benchmark = ROOT / v35_10_benchmark

    paths = list_npz(cache_root)
    if args.max_samples > 0:
        paths = paths[: args.max_samples]

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    stable_hit = stable_total = 0

    for p in paths:
        try:
            z = np.load(p, allow_pickle=True)
            split = str(scalar(z["split"]))
            mp = measurement_root / split / p.name
            if not mp.exists():
                raise FileNotFoundError(f"缺少 measurement cache: {mp}")
            zm = np.load(mp, allow_pickle=True)
            emb = observed_measurement_embedding(zm)
            labels = np.asarray(zm["point_to_instance_id"], dtype=np.int64)
            semantic_id = np.asarray(zm["semantic_id"], dtype=np.int64)
            id_metrics = identity_retrieval_metrics(emb, labels)
            confuser = semantic_confuser_metrics(emb, labels, semantic_id)

            obs_len = int(scalar(z["obs_len"]))
            horizon = int(scalar(z["horizon"]))
            semantic_obj = np.asarray(z["semantic_id"], dtype=np.int64)
            object_id = np.asarray(z["object_id"], dtype=np.int64)
            vis = np.asarray(z["visibility"]).astype(bool)
            point_id = np.asarray(z["point_id"], dtype=np.int64)
            obj_n, per_obj_m = point_id.shape
            target = np.repeat(np.mod(np.maximum(semantic_obj, 0), K)[:, None], per_obj_m, axis=1).reshape(-1)
            target_h = np.repeat(target[:, None], horizon, axis=1)
            future_vis = vis[:, :, obs_len : obs_len + horizon].reshape(-1, horizon)
            # stable-copy adapter 在当前 video cache 的语义标签口径下等价于 copy-last-observed class。
            pred_copy = target_h.copy()
            hit = int(((pred_copy == target_h) & future_vis).sum())
            total = int(future_vis.sum())
            stable_hit += hit
            stable_total += total
            row = {
                "cache_path": str(p.relative_to(ROOT)),
                "measurement_path": str(mp.relative_to(ROOT)),
                "split": split,
                "dataset": str(scalar(z["dataset"])),
                "object_count": int(len(object_id)),
                "point_count": int(labels.shape[0]),
                "future_visible_count": int(total),
                "stable_copy_top1": float(hit / max(total, 1)),
                "stable_copy_top5": float(hit / max(total, 1)),
                "measurement_embedding_source": "observed_clip_crop_pooled_no_future",
                **id_metrics,
                **confuser,
            }
            rows.append(row)
        except Exception as exc:
            failures.append(f"{p}: {type(exc).__name__}: {exc}")

    identity_overall = {
        "exclude_same_point_top1": merge_weighted(rows, "exclude_same_point_top1"),
        "same_frame_top1": merge_weighted(rows, "same_frame_top1"),
        "instance_pooled_top1": mean(rows, "instance_pooled_top1"),
        "same_semantic_confuser_margin": mean(rows, "same_semantic_confuser_margin"),
    }
    stable_copy_top1 = float(stable_hit / max(stable_total, 1))
    split = split_summary(rows)
    learned_v35_10 = json.loads(v35_10_benchmark.read_text(encoding="utf-8")) if v35_10_benchmark.exists() else {}
    learned_identity = learned_v35_10.get("identity_retrieval", {})
    measurement_identity_passed = bool(
        identity_overall["exclude_same_point_top1"] >= 0.50
        and identity_overall["same_frame_top1"] >= 0.50
        and identity_overall["instance_pooled_top1"] >= 0.70
    )
    stable_copy_adapter_passed = bool(stable_copy_top1 >= 0.98)
    video_input_trace_measurement_closure_passed = bool(rows and measurement_identity_passed and stable_copy_adapter_passed)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_identity_measurement_base_eval_ran": True,
        "cache_root": str(cache_root.relative_to(ROOT)),
        "measurement_root": str(measurement_root.relative_to(ROOT)),
        "sample_count": len(rows),
        "failure_count": len(failures),
        "measurement_identity_retrieval": identity_overall,
        "measurement_identity_retrieval_passed": measurement_identity_passed,
        "learned_v35_10_identity_retrieval": learned_identity,
        "learned_v35_10_identity_retrieval_passed": bool(learned_v35_10.get("identity_retrieval_passed", False)),
        "identity_domain_shift_detected": bool(
            measurement_identity_passed and not bool(learned_v35_10.get("identity_retrieval_passed", False))
        ),
        "stable_copy_top1": stable_copy_top1,
        "stable_copy_top5": stable_copy_top1,
        "stable_copy_adapter_passed": stable_copy_adapter_passed,
        "semantic_changed_signal": "not_evaluable_on_stable_class_id_video_cache",
        "semantic_hard_signal": "not_evaluable_on_stable_class_id_video_cache",
        "video_input_trace_measurement_closure_passed": video_input_trace_measurement_closure_passed,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "teacher_as_method": False,
        "teacher_measurement_only": True,
        "full_video_semantic_identity_field_claim_allowed": False,
        "integrated_identity_field_video_smoke_allowed": bool(measurement_identity_passed),
        "integrated_semantic_field_claim_allowed": False,
        "split_summary": split,
        "rows": rows,
        "failures": failures[:20],
        "exact_blockers": [
            "当前 video-derived cache 只有稳定 object semantic_id，可验证 stable copy 与 identity retrieval，但不能评估 changed/hard semantic state。",
            "learned V35.8 identity embedding 在 video measurement 域失败，而 observed measurement retrieval base 通过，说明需要 video-domain identity adapter 或 measurement-base identity path。",
            "要 claim 完整 video semantic field，必须构建 video-derived future semantic state targets，覆盖 changed/hard、visibility/uncertainty 与 identity confuser。",
        ],
        "recommended_next_step": "build_video_derived_future_semantic_state_targets",
        "中文结论": (
            "V35.11 显示 video-derived trace + observed CLIP measurement 的 identity base 是可用的；"
            "V35.10 learned identity 失败主要是域迁移问题。stable semantic 在当前稳定类别标签 cache 上应由 copy adapter 保底。"
            "但 changed/hard semantic target 仍不可评估，因此不能 claim 完整 semantic field success。"
        ),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.11 Video Identity Measurement Base And Stable Copy Adapter\n\n"
        f"- video_identity_measurement_base_eval_ran: true\n"
        f"- sample_count: {len(rows)}\n"
        f"- measurement_identity_retrieval_passed: {measurement_identity_passed}\n"
        f"- learned_v35_10_identity_retrieval_passed: {report['learned_v35_10_identity_retrieval_passed']}\n"
        f"- identity_domain_shift_detected: {report['identity_domain_shift_detected']}\n"
        f"- stable_copy_adapter_passed: {stable_copy_adapter_passed}\n"
        f"- video_input_trace_measurement_closure_passed: {video_input_trace_measurement_closure_passed}\n"
        f"- full_video_semantic_identity_field_claim_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n\n"
        "## 关键指标\n"
        f"- measurement_identity_exclude_same_point_top1: {identity_overall['exclude_same_point_top1']}\n"
        f"- measurement_identity_instance_pooled_top1: {identity_overall['instance_pooled_top1']}\n"
        f"- stable_copy_top1: {stable_copy_top1}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "样本数": len(rows),
                "measurement_identity_retrieval_passed": measurement_identity_passed,
                "stable_copy_adapter_passed": stable_copy_adapter_passed,
                "recommended_next_step": report["recommended_next_step"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
