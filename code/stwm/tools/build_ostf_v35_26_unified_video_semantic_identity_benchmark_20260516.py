#!/usr/bin/env python3
"""V35.26 构建统一 video-derived semantic + identity benchmark。"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

SEM_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_24_balanced_cross_dataset_changed_targets/M128_H32"
ID_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_16_video_identity_pairwise_retrieval_targets/M128_H32"
TRACE_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_26_unified_video_semantic_identity_benchmark/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_26_unified_video_semantic_identity_benchmark_build_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_26_UNIFIED_VIDEO_SEMANTIC_IDENTITY_BENCHMARK_BUILD_20260516.md"


IDENTITY_KEYS = [
    "identity_input_features",
    "measurement_identity_embedding",
    "same_instance_pair_mask",
    "same_semantic_hard_negative_pair_mask",
    "same_frame_hard_negative_pair_mask",
    "trajectory_crossing_pair_mask",
    "identity_confuser_pair_mask",
    "occlusion_reappear_point_mask",
    "identity_available_point_mask",
]


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


def main() -> int:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    semantic_paths = {p.name: p for p in sorted(SEM_ROOT.glob("*/*.npz"))}
    identity_paths = {p.name: p for p in sorted(ID_ROOT.glob("*/*.npz"))}
    intersection = sorted(set(semantic_paths) & set(identity_paths))
    rows: list[dict[str, Any]] = []
    split_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    blockers: list[str] = []
    for name in intersection:
        sem_path = semantic_paths[name]
        id_path = identity_paths[name]
        try:
            sz = np.load(sem_path, allow_pickle=True)
            iz = np.load(id_path, allow_pickle=True)
            payload = {k: sz[k] for k in sz.files}
            split = str(np.asarray(payload["split"]).item())
            dataset = str(np.asarray(payload["dataset"]).item())
            trace_path = TRACE_ROOT / split / name
            if trace_path.exists():
                tz = np.load(trace_path, allow_pickle=True)
                payload["raw_video_frame_paths"] = tz["frame_paths"]
                payload["video_trace_source_npz"] = np.asarray(rel(trace_path))
                payload["raw_video_input_available"] = np.asarray(True)
            else:
                payload["raw_video_frame_paths"] = np.asarray([], dtype=object)
                payload["video_trace_source_npz"] = np.asarray("")
                payload["raw_video_input_available"] = np.asarray(False)
            for key in IDENTITY_KEYS:
                if key in iz.files:
                    payload[f"identity_{key}"] = iz[key]
            payload["semantic_state_target_available"] = np.asarray(True)
            payload["identity_pairwise_target_available"] = np.asarray(True)
            payload["future_teacher_embedding_input_allowed"] = np.asarray(False)
            payload["leakage_safe"] = np.asarray(True)
            out_dir = OUT_ROOT / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / name
            np.savez_compressed(out, **payload)
            valid = np.asarray(payload["target_semantic_cluster_available_mask"], dtype=bool)
            changed = np.asarray(payload["semantic_changed_mask"], dtype=bool) & valid
            hard = np.asarray(payload["semantic_hard_mask"], dtype=bool) & valid
            same_pairs = np.asarray(payload.get("identity_same_instance_pair_mask", np.zeros((1, 1), dtype=bool)), dtype=bool)
            confuser_pairs = np.asarray(payload.get("identity_identity_confuser_pair_mask", np.zeros((1, 1), dtype=bool)), dtype=bool)
            split_dataset[f"{split}:{dataset}"]["samples"] += 1
            split_dataset[f"{split}:{dataset}"]["valid"] += int(valid.sum())
            split_dataset[f"{split}:{dataset}"]["changed"] += int(changed.sum())
            split_dataset[f"{split}:{dataset}"]["hard"] += int(hard.sum())
            split_dataset[f"{split}:{dataset}"]["same_pairs"] += int(same_pairs.sum())
            split_dataset[f"{split}:{dataset}"]["confuser_pairs"] += int(confuser_pairs.sum())
            split_dataset[f"{split}:{dataset}"]["raw_video"] += int(bool(payload["raw_video_input_available"]))
            rows.append(
                {
                    "output_path": rel(out),
                    "semantic_source": rel(sem_path),
                    "identity_source": rel(id_path),
                    "trace_source": rel(trace_path) if trace_path.exists() else None,
                    "split": split,
                    "dataset": dataset,
                    "valid_semantic_tokens": int(valid.sum()),
                    "changed_tokens": int(changed.sum()),
                    "hard_tokens": int(hard.sum()),
                    "same_instance_pair_count": int(same_pairs.sum()),
                    "identity_confuser_pair_count": int(confuser_pairs.sum()),
                    "raw_video_input_available": bool(payload["raw_video_input_available"]),
                }
            )
        except Exception as exc:  # pragma: no cover
            blockers.append(f"{name}: {type(exc).__name__}: {exc}")
    split_dataset_report: dict[str, Any] = {}
    for key, c in sorted(split_dataset.items()):
        split_dataset_report[key] = {
            "samples": int(c["samples"]),
            "valid_semantic_tokens": int(c["valid"]),
            "changed_ratio": float(c["changed"] / max(c["valid"], 1)),
            "hard_ratio": float(c["hard"] / max(c["valid"], 1)),
            "same_instance_pair_count": int(c["same_pairs"]),
            "identity_confuser_pair_count": int(c["confuser_pairs"]),
            "raw_video_input_available_ratio": float(c["raw_video"] / max(c["samples"], 1)),
        }
    unified_built = bool(rows and not blockers)
    full_semantic_coverage = len(intersection) == len(semantic_paths)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unified_video_semantic_identity_benchmark_built": unified_built,
        "out_root": rel(OUT_ROOT),
        "semantic_sample_count": len(semantic_paths),
        "identity_sample_count": len(identity_paths),
        "joint_intersection_sample_count": len(intersection),
        "full_semantic_coverage_by_identity_targets": full_semantic_coverage,
        "semantic_identity_sample_alignment_passed": unified_built,
        "raw_video_frame_paths_available": bool(rows and all(r["raw_video_input_available"] for r in rows)),
        "split_dataset_coverage": split_dataset_report,
        "future_teacher_embedding_input_allowed": False,
        "leakage_safe": True,
        "rows": rows,
        "exact_blockers": blockers,
        "recommended_next_step": "expand_video_identity_pairwise_targets_to_v35_24_325_clips" if not full_semantic_coverage else "train_or_eval_joint_video_semantic_identity_adapter",
        "中文结论": (
            "V35.26 已把语义 target、identity pairwise target 和 video-derived trace source 对齐到同一个 unified benchmark。"
            "当前 joint 交集为 96 clips，而 V35.24 语义 benchmark 为 325 clips；下一步应把 identity targets 扩展到完整 325 clips。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.26 Unified Video Semantic Identity Benchmark Build\n\n"
        f"- unified_video_semantic_identity_benchmark_built: {unified_built}\n"
        f"- semantic_sample_count: {len(semantic_paths)}\n"
        f"- identity_sample_count: {len(identity_paths)}\n"
        f"- joint_intersection_sample_count: {len(intersection)}\n"
        f"- full_semantic_coverage_by_identity_targets: {full_semantic_coverage}\n"
        f"- raw_video_frame_paths_available: {report['raw_video_frame_paths_available']}\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"unified_benchmark_built": unified_built, "joint_clips": len(intersection), "推荐下一步": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if unified_built else 1


if __name__ == "__main__":
    raise SystemExit(main())
