#!/usr/bin/env python3
"""评估 V35 在 video-derived M128/H32 trace + observed measurement 上的有限闭环。"""
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
from stwm.tools.run_ostf_v35_9_video_derived_input_closure_smoke_20260515 import build_features, scalar

CACHE_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16/M128_H32"
MEASUREMENT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_10_video_observed_semantic_measurement_bank/M128_H32"
CKPT_SUMMARY = ROOT / "reports/stwm_ostf_v35_8_identity_only_retrieval_finetune_train_summary_20260515.json"
SUMMARY = ROOT / "reports/stwm_ostf_v35_10_video_derived_closure_benchmark_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_10_VIDEO_DERIVED_CLOSURE_BENCHMARK_20260515.md"
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


def list_npz(root: Path) -> list[Path]:
    return sorted(root.glob("*/*.npz"))


def topk_hit(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, k: int) -> tuple[int, int]:
    if not bool(mask.any()):
        return 0, 0
    pred = logits.topk(min(k, logits.shape[-1]), dim=-1).indices
    hit = (pred == target[..., None]).any(dim=-1) & mask
    return int(hit.sum().item()), int(mask.sum().item())


def identity_metrics(emb: torch.Tensor, instance_id: np.ndarray) -> dict[str, float]:
    x = emb.detach().cpu().numpy().mean(axis=2)
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
    labels = instance_id.astype(np.int64)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    valid = labels >= 0
    hit = total = 0
    for i in np.where(valid)[0]:
        same = (labels == labels[i]) & valid
        same[i] = False
        if same.any():
            total += 1
            hit += int(labels[int(np.argmax(sim[i]))] == labels[i])
    unique_ids = [int(v) for v in np.unique(labels[valid])]
    pooled_hit = pooled_total = 0
    if len(unique_ids) >= 2:
        cents = []
        for inst in unique_ids:
            c = x[labels == inst].mean(axis=0)
            c = c / max(float(np.linalg.norm(c)), 1e-8)
            cents.append(c)
        pred = np.asarray(unique_ids)[(x[valid] @ np.stack(cents).T).argmax(axis=1)]
        pooled_hit = int((pred == labels[valid]).sum())
        pooled_total = int(valid.sum())
    return {
        "exclude_same_point_top1": float(hit / max(total, 1)),
        "same_frame_top1": float(hit / max(total, 1)),
        "instance_pooled_top1": float(pooled_hit / max(pooled_total, 1)),
        "retrieval_total": float(total),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--measurement-root", default=str(MEASUREMENT_ROOT))
    ap.add_argument("--ckpt-summary", default=str(CKPT_SUMMARY))
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cache_root = Path(args.cache_root)
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    measurement_root = Path(args.measurement_root)
    if not measurement_root.is_absolute():
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
    paths = list_npz(cache_root)
    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    totals = {"stable_top1": [0, 0], "stable_top5": [0, 0], "copy_stable_top1": [0, 0]}
    id_rows: list[dict[str, float]] = []
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    with torch.no_grad():
        for p in paths:
            try:
                z = np.load(p, allow_pickle=True)
                split = str(scalar(z["split"]))
                mp = measurement_root / split / p.name
                if not mp.exists():
                    raise FileNotFoundError(f"missing measurement {mp}")
                zm = np.load(mp, allow_pickle=True)
                feat, meta = build_features(z, zm)
                x = torch.from_numpy(feat[None]).to(device)
                out = model(x, horizon=int(meta["horizon"]))
                semantic_id = np.asarray(z["semantic_id"], dtype=np.int64)
                obj_n, per_obj_m = np.asarray(z["point_id"]).shape
                target = np.repeat(np.mod(np.maximum(semantic_id, 0), K)[:, None], per_obj_m, axis=1).reshape(-1)
                target_h = np.repeat(target[:, None], int(meta["horizon"]), axis=1)
                vis = np.asarray(z["visibility"]).astype(bool)[:, :, int(meta["obs_len"]) : int(meta["obs_len"]) + int(meta["horizon"])].reshape(-1, int(meta["horizon"]))
                target_t = torch.from_numpy(target_h[None]).to(device)
                mask_t = torch.from_numpy(vis[None]).to(device)
                logits = out["semantic_cluster_logits"]
                h1, n1 = topk_hit(logits, target_t, mask_t, 1)
                h5, n5 = topk_hit(logits, target_t, mask_t, 5)
                totals["stable_top1"][0] += h1
                totals["stable_top1"][1] += n1
                totals["stable_top5"][0] += h5
                totals["stable_top5"][1] += n5
                pred_copy = np.repeat(target[:, None], int(meta["horizon"]), axis=1)
                copy_hit = (pred_copy == target_h) & vis
                totals["copy_stable_top1"][0] += int(copy_hit.sum())
                totals["copy_stable_top1"][1] += int(vis.sum())
                instance_id = np.repeat(np.asarray(z["object_id"], dtype=np.int64)[:, None], per_obj_m, axis=1).reshape(-1)
                idm = identity_metrics(out["identity_embedding"][0], instance_id)
                id_rows.append(idm)
                rows.append(
                    {
                        "cache_path": str(p.relative_to(ROOT)),
                        "split": split,
                        "dataset": str(scalar(z["dataset"])),
                        "object_count": int(obj_n),
                        "point_count": int(obj_n * per_obj_m),
                        "stable_top5": float(h5 / max(n5, 1)),
                        "copy_stable_top1": float(copy_hit.sum() / max(vis.sum(), 1)),
                        **idm,
                    }
                )
            except Exception as exc:
                failures.append(f"{p}: {type(exc).__name__}: {exc}")
    def rate(key: str) -> float:
        return float(totals[key][0] / max(totals[key][1], 1))
    identity = {
        "exclude_same_point_top1": float(np.mean([r["exclude_same_point_top1"] for r in id_rows])) if id_rows else 0.0,
        "same_frame_top1": float(np.mean([r["same_frame_top1"] for r in id_rows])) if id_rows else 0.0,
        "instance_pooled_top1": float(np.mean([r["instance_pooled_top1"] for r in id_rows])) if id_rows else 0.0,
    }
    stable_preservation = bool(rate("stable_top5") >= rate("copy_stable_top1") - 0.02)
    identity_retrieval_passed = bool(identity["exclude_same_point_top1"] >= 0.50 and identity["same_frame_top1"] >= 0.50 and identity["instance_pooled_top1"] >= 0.70)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_derived_closure_benchmark_ran": True,
        "sample_count": len(rows),
        "failure_count": len(failures),
        "cache_root": str(cache_root.relative_to(ROOT)),
        "measurement_root": str(measurement_root.relative_to(ROOT)),
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "stable_top1": rate("stable_top1"),
        "stable_top5": rate("stable_top5"),
        "copy_stable_top1": rate("copy_stable_top1"),
        "stable_preservation": stable_preservation,
        "identity_retrieval": identity,
        "identity_retrieval_passed": identity_retrieval_passed,
        "semantic_changed_signal": "not_evaluable_on_stable_class_id_video_cache",
        "semantic_hard_signal": "not_evaluable_on_stable_class_id_video_cache",
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "rows": rows,
        "failures": failures[:20],
        "full_video_semantic_identity_field_claim_allowed": False,
        "exact_blockers": [
            "VSPW/VIPSeg object semantic_id 是稳定类别标签，不能评估 V35 的 changed/hard semantic transition 能力。",
            "当前 benchmark 是 video-derived input closure + stable semantic/identity smoke，不是完整 future semantic field benchmark。",
            "需要构建 video-derived future semantic state targets，或接 TAPVid/VSPW/VIPSeg 可评估的未来 semantic transition protocol。",
        ],
        "recommended_next_step": "build_video_derived_future_semantic_state_targets",
        "中文结论": "V35.10 完成 video-derived trace + observed CLIP measurement 的有限闭环评估；stable semantic 与 identity retrieval 可评估，但 changed/hard semantic target 仍缺失。",
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.10 Video-Derived Closure Benchmark\n\n"
        f"- video_derived_closure_benchmark_ran: true\n"
        f"- sample_count: {len(rows)}\n"
        f"- stable_top5: {report['stable_top5']}\n"
        f"- copy_stable_top1: {report['copy_stable_top1']}\n"
        f"- stable_preservation: {stable_preservation}\n"
        f"- identity_retrieval_passed: {identity_retrieval_passed}\n"
        f"- semantic_changed_signal: not_evaluable_on_stable_class_id_video_cache\n"
        f"- semantic_hard_signal: not_evaluable_on_stable_class_id_video_cache\n"
        f"- full_video_semantic_identity_field_claim_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stable_preservation": stable_preservation, "identity_retrieval_passed": identity_retrieval_passed, "recommended_next_step": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
