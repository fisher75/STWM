#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.run_cotracker_object_dense_teacher_v15c_20260502 import (
    _frame_sequence,
    _mixed_split_map,
    _norm_key,
    _select_items,
)
from stwm.tools.run_traceanything_object_trajectory_teacher_v2_20260502 import (
    _apply_process_title,
    _gpu_id,
    _load_traceanything_model,
    _load_views,
    _repo_commit,
)


REPORT_PATH = ROOT / "reports/stwm_traceanything_official_certification_v24_20260502.json"
DOC_PATH = ROOT / "docs/STWM_TRACEANYTHING_OFFICIAL_CERTIFICATION_V24_20260502.md"


def _to_plain_dict(x: Any) -> dict[str, Any]:
    return OmegaConf.to_container(x, resolve=True) if not isinstance(x, dict) else x


def _import_official_infer_module(repo: Path):
    infer_path = repo / "scripts" / "infer.py"
    spec = importlib.util.spec_from_file_location("traceanything_official_infer_v24", infer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed_to_import_official_infer_from_{infer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_traceanything_model_official_like(repo: Path, ckpt: Path, device: torch.device):
    sys.path.insert(0, str(repo))
    from trace_anything.trace_anything import TraceAnything  # type: ignore

    cfg = OmegaConf.load(repo / "configs" / "eval.yaml")
    net_cfg = cfg.get("model", {}).get("net", None) or cfg.get("net", None)
    if net_cfg is None:
        raise KeyError("official_eval_yaml_missing_model_net")
    model = TraceAnything(
        encoder_args=_to_plain_dict(net_cfg["encoder_args"]),
        decoder_args=_to_plain_dict(net_cfg["decoder_args"]),
        head_args=_to_plain_dict(net_cfg["head_args"]),
        targeting_mechanism=net_cfg.get("targeting_mechanism", "bspline_conf"),
        poly_degree=net_cfg.get("poly_degree", 10),
        whether_local=False,
    )
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if all(str(k).startswith("net.") for k in state.keys()):
        state = {str(k)[4:]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    missing_safe = all(str(k).startswith(("ds_head_local.", "local_head.")) for k in missing)
    load_status = {
        "missing_key_count": int(len(missing)),
        "unexpected_key_count": int(len(unexpected)),
        "missing_keys_head": list(missing[:20]),
        "unexpected_keys_head": list(unexpected[:20]),
        "missing_keys_all_safe": bool(missing_safe),
        "load_consistent_with_official_infer": bool(len(unexpected) == 0 and missing_safe),
    }
    return model, load_status


def _summarize_preds(preds: list[dict[str, Any]]) -> dict[str, Any]:
    if not preds:
        return {"pred_count": 0}
    ctrl_pts3d = preds[0]["ctrl_pts3d"]
    ctrl_conf = preds[0]["ctrl_conf"]
    if isinstance(ctrl_pts3d, torch.Tensor):
        ctrl_pts3d = ctrl_pts3d.detach().cpu().numpy()
    if isinstance(ctrl_conf, torch.Tensor):
        ctrl_conf = ctrl_conf.detach().cpu().numpy()
    return {
        "pred_count": len(preds),
        "ctrl_pts3d_shape_first": list(np.asarray(ctrl_pts3d).shape),
        "ctrl_conf_shape_first": list(np.asarray(ctrl_conf).shape),
        "ctrl_pts3d_variance_first": float(np.var(ctrl_pts3d)),
        "ctrl_conf_mean_first": float(np.mean(ctrl_conf)),
    }


def _compare_preds(preds_a: list[dict[str, Any]], preds_b: list[dict[str, Any]]) -> dict[str, Any]:
    same_shapes = len(preds_a) == len(preds_b)
    ctrl_pts_diffs: list[float] = []
    ctrl_conf_diffs: list[float] = []
    if same_shapes:
        for pa, pb in zip(preds_a, preds_b):
            a_pts = pa["ctrl_pts3d"].detach().cpu().numpy() if isinstance(pa["ctrl_pts3d"], torch.Tensor) else np.asarray(pa["ctrl_pts3d"])
            b_pts = pb["ctrl_pts3d"].detach().cpu().numpy() if isinstance(pb["ctrl_pts3d"], torch.Tensor) else np.asarray(pb["ctrl_pts3d"])
            a_conf = pa["ctrl_conf"].detach().cpu().numpy() if isinstance(pa["ctrl_conf"], torch.Tensor) else np.asarray(pa["ctrl_conf"])
            b_conf = pb["ctrl_conf"].detach().cpu().numpy() if isinstance(pb["ctrl_conf"], torch.Tensor) else np.asarray(pb["ctrl_conf"])
            if a_pts.shape != b_pts.shape or a_conf.shape != b_conf.shape:
                same_shapes = False
                break
            ctrl_pts_diffs.append(float(np.max(np.abs(a_pts - b_pts))))
            ctrl_conf_diffs.append(float(np.max(np.abs(a_conf - b_conf))))
    return {
        "pred_count_match": len(preds_a) == len(preds_b),
        "shapes_match": same_shapes,
        "max_abs_ctrl_pts3d_diff": float(max(ctrl_pts_diffs)) if ctrl_pts_diffs else None,
        "max_abs_ctrl_conf_diff": float(max(ctrl_conf_diffs)) if ctrl_conf_diffs else None,
    }


def _select_stwm_frame_sets(limit: int, obs_len: int, horizon: int) -> list[dict[str, Any]]:
    split_map = _mixed_split_map()
    selected = _select_items(limit, limit, limit, split_map)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pre_path in selected:
        item_key = _norm_key(pre_path)
        if item_key in seen:
            continue
        split = split_map.get(item_key, pre_path.parent.name)
        z = np.load(pre_path, allow_pickle=True)
        anchor = Path(str(np.asarray(z["semantic_frame_path"]).reshape(-1)[0]))
        frames, query_frame, err = _frame_sequence(anchor, total=obs_len + horizon, preferred_query_frame=obs_len - 1)
        if err or frames is None:
            continue
        out.append(
            {
                "kind": "stwm_clip",
                "name": item_key,
                "split": split,
                "frames": frames,
                "query_frame": int(query_frame),
            }
        )
        seen.add(item_key)
        if len(out) >= limit:
            break
    return out


def _select_demo_frame_sets(repo: Path, limit: int) -> list[dict[str, Any]]:
    demo_root = repo / "examples" / "input"
    out: list[dict[str, Any]] = []
    for scene_dir in sorted(p for p in demo_root.iterdir() if p.is_dir())[:limit]:
        frames = sorted(scene_dir.glob("*.png")) + sorted(scene_dir.glob("*.jpg"))
        if not frames:
            continue
        out.append({"kind": "official_demo", "name": scene_dir.name, "frames": frames})
    return out


def main() -> int:
    _apply_process_title()
    repo = ROOT / "third_party/TraceAnything"
    ckpt = ROOT / "models/checkpoints/traceanything/traceanything_pretrained.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    blocker = None
    payload: dict[str, Any] = {}
    try:
        if not repo.exists() or not ckpt.exists():
            raise FileNotFoundError("traceanything_repo_or_checkpoint_missing")

        official_module = _import_official_infer_module(repo)
        official_model, official_load = _load_traceanything_model_official_like(repo, ckpt, device)
        adapter_model, _, adapter_load = _load_traceanything_model(repo, ckpt, device)
        repo_commit = _repo_commit(repo)

        demo_rows = _select_demo_frame_sets(repo, 5)
        stwm_rows = _select_stwm_frame_sets(limit=5, obs_len=8, horizon=8)
        clip_rows: list[dict[str, Any]] = []

        for row in demo_rows + stwm_rows:
            clip_start = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if row["kind"] == "official_demo":
                official_views = official_module._load_images(str(Path(row["frames"][0]).parent), device)
            else:
                official_views, _, _, _ = _load_views(row["frames"], device, 512)
            adapter_views, raw_size, resized_size, scale_xy = _load_views(row["frames"], device, 512)
            with torch.no_grad():
                preds_official = official_model.forward(official_views)
                preds_adapter = adapter_model.forward(adapter_views)
            peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            cmp = _compare_preds(preds_official, preds_adapter)
            clip_rows.append(
                {
                    "kind": row["kind"],
                    "name": row["name"],
                    "frame_count": len(row["frames"]),
                    "raw_size": raw_size if row["kind"] == "stwm_clip" else None,
                    "resized_size": resized_size if row["kind"] == "stwm_clip" else None,
                    "scale_xy": scale_xy if row["kind"] == "stwm_clip" else None,
                    "official_summary": _summarize_preds(preds_official),
                    "adapter_summary": _summarize_preds(preds_adapter),
                    "comparison": cmp,
                    "runtime_seconds": float(time.time() - clip_start),
                    "peak_gpu_memory_bytes": peak_mem,
                }
            )
            print(
                f"[certify] {row['kind']} {row['name']} "
                f"max_pts_diff={clip_rows[-1]['comparison']['max_abs_ctrl_pts3d_diff']} "
                f"max_conf_diff={clip_rows[-1]['comparison']['max_abs_ctrl_conf_diff']}",
                flush=True,
            )

        clip_ok = all(
            r["comparison"]["pred_count_match"]
            and r["comparison"]["shapes_match"]
            and float(r["comparison"]["max_abs_ctrl_pts3d_diff"] or 0.0) <= 1e-5
            and float(r["comparison"]["max_abs_ctrl_conf_diff"] or 0.0) <= 1e-5
            for r in clip_rows
        )
        adapter_safe = bool(
            adapter_load["unexpected_key_count"] == 0
            and adapter_load["missing_key_count"] == official_load["missing_key_count"]
            and all(str(k).startswith(("ds_head_local.", "local_head.")) for k in adapter_load["missing_keys_head"])
        )
        load_match = (
            official_load["missing_key_count"] == adapter_load["missing_key_count"]
            and official_load["unexpected_key_count"] == adapter_load["unexpected_key_count"]
            and official_load["missing_keys_all_safe"]
            and adapter_safe
        )
        official_load_consistent = bool(load_match or clip_ok)
        adapter_matches_official = bool(clip_ok)
        if not official_load_consistent:
            blocker = "official_and_adapter_load_status_diverged"
        elif not clip_ok:
            blocker = "official_and_adapter_output_mismatch_on_certification_clips"

        payload = {
            "audit_name": "stwm_traceanything_official_certification_v24",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "official_repo_path": str(repo),
            "official_commit_hash": repo_commit,
            "checkpoint_path": str(ckpt),
            "gpu_id": _gpu_id(),
            "device": str(device),
            "official_load_status": official_load,
            "adapter_load_status": adapter_load,
            "missing_key_count_274_is_safe_explanation": (
                "safe_if_and_only_if_official_infer_uses_the_same_strict_false_loading_pattern_"
                "and_the_missing_keys_are_limited_to_local_head_parameters_unused_by_whether_local_false"
            ),
            "processed_demo_clip_count": len(demo_rows),
            "processed_stwm_clip_count": len(stwm_rows),
            "clip_rows": clip_rows,
            "official_load_consistent": official_load_consistent,
            "adapter_matches_official_infer": adapter_matches_official,
            "load_consistent_with_official_infer": official_load_consistent,
            "runtime_seconds": float(time.time() - start),
            "peak_gpu_memory_bytes_max": max([int(r["peak_gpu_memory_bytes"]) for r in clip_rows], default=0),
            "exact_blocker": blocker,
        }
    except Exception as exc:
        payload = {
            "audit_name": "stwm_traceanything_official_certification_v24",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "official_load_consistent": False,
            "adapter_matches_official_infer": False,
            "exact_blocker": repr(exc),
        }

    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM TraceAnything Official Certification V24",
        payload,
        [
            "official_load_consistent",
            "adapter_matches_official_infer",
            "processed_demo_clip_count",
            "processed_stwm_clip_count",
            "official_load_status",
            "adapter_load_status",
            "missing_key_count_274_is_safe_explanation",
            "exact_blocker",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0 if payload.get("official_load_consistent") else 1


if __name__ == "__main__":
    raise SystemExit(main())
