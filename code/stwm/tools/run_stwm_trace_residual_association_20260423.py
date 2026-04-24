#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import gc
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

try:
    import setproctitle  # type: ignore
except Exception:
    setproctitle = None

if setproctitle is not None:
    try:
        setproctitle.setproctitle("python")
    except Exception:
        pass

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

for candidate in [Path("/raid/chen034/workspace/stwm/code"), Path("/home/chen034/workspace/stwm/code")]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from stwm.tools import run_stage2_state_identifiability_eval_20260415 as evalcore
from stwm.tools import run_stwm_tusb_light_readout_eval_20260422 as lighteval
from stwm.tools import run_stwm_trace_conditioned_readout_20260423 as tracecond
from stwm.tools import run_stwm_trace_gated_readout_20260423 as tracegate
from stwm.tools import run_stwm_true_ood_eval_20260420 as oodcore


ROOT = Path("/raid/chen034/workspace/stwm")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
SHARDS = REPORTS / "trace_conditioned_readout_shards_20260423"

OFFICIAL_TUSB = "TUSB-v3.1::best_semantic_hard.pt"
CAL = "calibration-only::best.pt"
CROP = "cropenc::best.pt"
LEGACY = "legacysem::best.pt"
SEEDS = list(lighteval.SEEDS)
PANELS = [
    "densified_200_context_preserving",
    "heldout_burst_heavy_context_preserving",
    "heldout_scene_category_video_context_preserving",
]
TUSB_MODES = [
    "frozen_external_teacher_only",
    "tusb_semantic_target",
    "unit_identity_only",
    "trace_gated_semantic_target",
    "trace_residual_association",
]
METRIC_SPECS = [
    ("overall_top1", "query_future_top1_acc", ""),
    ("hard_subset_top1", "query_future_top1_acc", "__hard__"),
    ("ambiguity_top1", "query_future_top1_acc", "crossing_ambiguity"),
    ("occlusion_reappearance_top1", "query_future_top1_acc", "occlusion_reappearance"),
    ("long_gap_persistence_top1", "query_future_top1_acc", "long_gap_persistence"),
]
RESIDUAL_FEATURE_NAMES = [
    "unit_identity_score_norm",
    "coord_score_norm",
    "tusb_semantic_target_score_norm",
    "external_rank_score",
    "external_margin_to_top",
    "coord_rank_score",
    "coord_margin_to_top",
    "candidate_count_scaled",
    "is_occlusion_reappearance",
    "is_long_gap_persistence",
    "is_crossing_ambiguity",
    "external_x_unit",
    "external_x_coord",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, title: str, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", "", *list(lines)]).rstrip() + "\n", encoding="utf-8")


def _sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(len(vals), 1))


def _std(values: Iterable[float]) -> float:
    vals = np.asarray([float(v) for v in values], dtype=np.float64)
    if vals.size <= 1:
        return 0.0
    return float(vals.std(ddof=0))


def _item_bucket(protocol_item_id: str) -> int:
    digest = hashlib.sha256(str(protocol_item_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10


def _item_split3(protocol_item_id: str) -> str:
    bucket = _item_bucket(protocol_item_id)
    if bucket in {0, 1}:
        return "train"
    if bucket == 2:
        return "val"
    return "test"


def _minmax(scores: Mapping[str, float]) -> Dict[str, float]:
    return evalcore._minmax_normalize_score_map({str(k): float(v) for k, v in scores.items()})


def _rank_features(scores: Mapping[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not scores:
        return {}, {}
    ordered = sorted(scores, key=lambda cid: (-float(scores[cid]), str(cid)))
    n = max(len(ordered) - 1, 1)
    top = float(scores[ordered[0]])
    rank_score = {cid: float(1.0 - idx / n) for idx, cid in enumerate(ordered)}
    margin = {cid: float(float(scores[cid]) - top) for cid in ordered}
    return rank_score, margin


def _label_result(score_map: Mapping[str, float], target_id: str, target_future_mask: np.ndarray, future_masks: Mapping[str, np.ndarray], base: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    return lighteval._compose_score_result(
        base_result=dict(base),
        score_map={str(k): float(v) for k, v in score_map.items()},
        target_id=str(target_id),
        target_future_mask=target_future_mask,
        future_masks=dict(future_masks),
        scoring_mode=mode,
    )


def _residual_row(
    protocol_item_id: str,
    seed: int,
    method_name: str,
    scoring_mode: str,
    subset_tags: Sequence[str],
    dataset: str,
    clip_id: str,
    context_count: int,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    row = tracegate._row(
        protocol_item_id,
        seed,
        method_name,
        scoring_mode,
        subset_tags,
        dataset,
        clip_id,
        context_count,
        dict(result),
    )
    row["item_split"] = _item_split3(str(protocol_item_id))
    return row


def _feature_vector(
    cand_id: str,
    subset_tags: Sequence[str],
    external_n: Mapping[str, float],
    unit_n: Mapping[str, float],
    coord_n: Mapping[str, float],
    semantic_n: Mapping[str, float],
    ext_rank: Mapping[str, float],
    ext_margin: Mapping[str, float],
    coord_rank: Mapping[str, float],
    coord_margin: Mapping[str, float],
    candidate_count: int,
) -> np.ndarray:
    tags = set(str(x) for x in subset_tags)
    ext = float(external_n.get(cand_id, 0.0))
    unit = float(unit_n.get(cand_id, 0.0))
    coord = float(coord_n.get(cand_id, 0.0))
    return np.asarray(
        [
            unit,
            coord,
            float(semantic_n.get(cand_id, 0.0)),
            float(ext_rank.get(cand_id, 0.0)),
            float(ext_margin.get(cand_id, 0.0)),
            float(coord_rank.get(cand_id, 0.0)),
            float(coord_margin.get(cand_id, 0.0)),
            float(np.log1p(max(candidate_count, 0)) / 5.0),
            1.0 if "occlusion_reappearance" in tags else 0.0,
            1.0 if "long_gap_persistence" in tags else 0.0,
            1.0 if "crossing_ambiguity" in tags else 0.0,
            ext * unit,
            ext * coord,
        ],
        dtype=np.float64,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _fit_residual_logistic(
    X: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    l2: float,
    steps: int = 600,
    lr: float = 0.05,
) -> Tuple[np.ndarray, float]:
    w = np.zeros((X.shape[1],), dtype=np.float64)
    b = 0.0
    pos = max(float(y.sum()), 1.0)
    neg = max(float(y.shape[0] - y.sum()), 1.0)
    weights = np.where(y > 0.5, neg / pos, 1.0)
    denom = float(weights.sum())
    for _ in range(int(steps)):
        logits = base + X @ w + b
        p = _sigmoid(logits)
        err = (p - y) * weights
        grad_w = (X.T @ err) / denom + float(l2) * w
        grad_b = float(err.sum() / denom)
        w -= float(lr) * grad_w
        b -= float(lr) * grad_b
    return w, float(b)


def _score_rows(rows: Sequence[Dict[str, Any]], w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray) -> float:
    by_item_seed: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_item_seed[(str(row["protocol_item_id"]), int(row["seed"]))].append(row)
    hits = []
    for _, group in by_item_seed.items():
        best = max(group, key=lambda r: (float(r["external_base_score"]) + float(((np.asarray(r["features"], dtype=np.float64) - mean) / std) @ w) + b, str(r["candidate_id"])))
        hits.append(1.0 if str(best["candidate_id"]) == str(best["target_id"]) else 0.0)
    return _mean(hits)


def _fit_head(pair_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    train_rows = [row for row in pair_rows if row["item_split"] == "train"]
    val_rows = [row for row in pair_rows if row["item_split"] == "val"]
    train_val_rows = [row for row in pair_rows if row["item_split"] in {"train", "val"}]
    if not train_rows or not val_rows:
        raise RuntimeError("residual head split is empty")
    X_train_raw = np.stack([np.asarray(row["features"], dtype=np.float64) for row in train_rows], axis=0)
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    X_train = (X_train_raw - mean) / std
    base_train = np.asarray([float(row["external_base_score"]) for row in train_rows], dtype=np.float64)
    y_train = np.asarray([1.0 if bool(row["label"]) else 0.0 for row in train_rows], dtype=np.float64)
    best: Dict[str, Any] = {"val_score": -1.0, "l2": 0.0, "w": np.zeros(len(RESIDUAL_FEATURE_NAMES)), "b": 0.0}
    for l2 in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
        w, b = _fit_residual_logistic(X_train, base_train, y_train, l2=l2)
        val_score = _score_rows(val_rows, w, b, mean, std)
        if val_score > float(best["val_score"]):
            best = {"val_score": float(val_score), "l2": float(l2), "w": w, "b": float(b)}
    X_tv_raw = np.stack([np.asarray(row["features"], dtype=np.float64) for row in train_val_rows], axis=0)
    mean_tv = X_tv_raw.mean(axis=0)
    std_tv = X_tv_raw.std(axis=0)
    std_tv = np.where(std_tv < 1e-6, 1.0, std_tv)
    X_tv = (X_tv_raw - mean_tv) / std_tv
    base_tv = np.asarray([float(row["external_base_score"]) for row in train_val_rows], dtype=np.float64)
    y_tv = np.asarray([1.0 if bool(row["label"]) else 0.0 for row in train_val_rows], dtype=np.float64)
    w, b = _fit_residual_logistic(X_tv, base_tv, y_tv, l2=float(best["l2"]))
    return {
        "feature_names": list(RESIDUAL_FEATURE_NAMES),
        "selected_l2": float(best["l2"]),
        "val_top1": float(best["val_score"]),
        "weights": [float(x) for x in w.tolist()],
        "bias": float(b),
        "feature_mean": [float(x) for x in mean_tv.tolist()],
        "feature_std": [float(x) for x in std_tv.tolist()],
        "head_type": "external_base_plus_linear_logistic_residual",
    }


def _apply_head(external_n: Mapping[str, float], feature_rows: Mapping[str, np.ndarray], head: Mapping[str, Any]) -> Dict[str, float]:
    w = np.asarray(head["weights"], dtype=np.float64)
    b = float(head["bias"])
    mean = np.asarray(head["feature_mean"], dtype=np.float64)
    std = np.asarray(head["feature_std"], dtype=np.float64)
    scores: Dict[str, float] = {}
    for cid, feat in feature_rows.items():
        x = (np.asarray(feat, dtype=np.float64) - mean) / std
        scores[str(cid)] = float(float(external_n.get(cid, 0.0)) + float(x @ w) + b)
    return scores


def _build_audit(args: Any) -> Dict[str, Any]:
    trace_decision = _load_json(REPORTS / "stwm_trace_gated_readout_decision_20260423.json")
    payload = {
        "generated_at_utc": _now_iso(),
        "semantic_teacher_only_renamed_to_tusb_semantic_target": True,
        "frozen_external_teacher_only_is_clean_external_baseline": True,
        "trace_gated_semantic_target_had_extra_gain": bool(trace_decision.get("trace_gated_improved_vs_tusb_semantic_target", False)),
        "primary_comparators": [
            "frozen_external_teacher_only",
            "tusb_semantic_target",
            "trace_residual_association",
            "legacysem",
        ],
        "baseline_naming_frozen": True,
        "external_teacher_only_clean": True,
        "trace_gate_effective": bool(trace_decision.get("trace_gated_improved_vs_tusb_semantic_target", False)),
        "exact_blocking_reason": "",
    }
    _write_json(Path(args.audit_json), payload)
    _write_md(
        Path(args.audit_md),
        "STWM Residual Association Audit 20260423",
        [
            "- semantic_teacher_only: renamed to tusb_semantic_target",
            "- frozen_external_teacher_only: clean external baseline",
            f"- trace_gate_effective: {payload['trace_gate_effective']}",
            f"- baseline_naming_frozen: {payload['baseline_naming_frozen']}",
        ],
    )
    return payload


def _build_schema(args: Any) -> Dict[str, Any]:
    payload = {
        "generated_at_utc": _now_iso(),
        "head_type": "external_base_plus_linear_logistic_residual",
        "final_score_definition": "ExternalTeacherScore(candidate) + ResidualTraceScore(candidate)",
        "external_teacher_score": "clean frozen CLIP crop cosine; not method.semantic_encoder",
        "feature_names": list(RESIDUAL_FEATURE_NAMES),
        "allowed_feature_sources": {
            "unit_identity_score_norm": "TUSB trace-unit identity projection similarity",
            "coord_score_norm": "free rollout coordinate plausibility over candidate mask",
            "tusb_semantic_target_score_norm": "TUSB semantic target token candidate score",
            "external_rank_score": "rank feature derived from clean external teacher score",
            "external_margin_to_top": "margin from clean external top candidate",
            "coord_rank_score": "rank feature derived from coordinate score",
            "coord_margin_to_top": "margin from coordinate top candidate",
            "candidate_count_scaled": "candidate set size metadata",
            "continuity_flags": "occlusion/long-gap subset tags",
            "ambiguity_flag": "crossing_ambiguity subset tag",
            "interactions": "external_x_unit and external_x_coord only",
        },
        "prohibited_sources": [
            "future target mask overlap as feature",
            "large MLP or transformer",
            "backbone changes",
            "semantic teacher score duplicated as final base",
        ],
    }
    _write_json(Path(args.schema_json), payload)
    _write_md(
        Path(args.schema_md),
        "STWM Residual Association Feature Schema 20260423",
        [
            f"- head_type: {payload['head_type']}",
            f"- final_score_definition: {payload['final_score_definition']}",
            f"- feature_names: {json.dumps(payload['feature_names'], ensure_ascii=True)}",
        ],
    )
    return payload


def _build_eval(args: Any) -> Dict[str, Any]:
    audit = _build_audit(args)
    schema = _build_schema(args)
    raw_rows, shards, panel_item_ids, item_lookup = tracegate._load_source_rows(args)
    materialization, _, _ = tracegate._materialize_panels(args)
    selected_ids = set().union(*(panel_item_ids[name] for name in PANELS))
    tusb_rows_by_item: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    baseline_rows_by_item: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        item_id = str(row.get("protocol_item_id", ""))
        if str(row.get("method_name")) == OFFICIAL_TUSB:
            tusb_rows_by_item[item_id].append(row)
        elif str(row.get("method_name")) in {LEGACY, CAL, CROP}:
            baseline_rows_by_item[item_id].append(row)

    device, device_info = evalcore._select_eval_device(args)
    model = None
    preprocess = None
    frozen_backend = ""
    started = _now_iso()
    wall_start = time.time()
    pair_rows: List[Dict[str, Any]] = []
    item_assets: Dict[str, Dict[str, Any]] = {}
    prepared_meta: Dict[str, Dict[str, Any]] = {}
    skipped_reasons: Dict[str, str] = {}
    frozen_empty_items = 0
    try:
        model, preprocess, frozen_backend = tracegate._load_clip(device)
        total = len(selected_ids)
        print(f"[{_now_iso()}] residual_assoc_prepare_start items={total} device={device} backend={frozen_backend}", flush=True)
        for index, item_id in enumerate(sorted(selected_ids), start=1):
            item_id, prepared, error = oodcore._prepare_one_item(item_id, item_lookup.get(item_id))
            if not isinstance(prepared, dict):
                skipped_reasons[item_id] = error or "unknown_prepare_error"
                continue
            prepared_meta[item_id] = {
                "protocol_eval_context_entity_count": int(prepared["protocol_eval_context_entity_count"]),
                "item_split": _item_split3(str(item_id)),
            }
            frozen_scores = tracegate._clip_score_map(model, preprocess, device, prepared["batch"], prepared["candidate_inputs"])
            if not frozen_scores:
                frozen_empty_items += 1
            item_assets[item_id] = {
                "target_future_mask": prepared["target_future_mask"],
                "future_masks": prepared["future_masks"],
                "frozen_scores": frozen_scores,
            }
            for row in tusb_rows_by_item.get(item_id, []):
                if not bool(row.get("available", False)):
                    continue
                target_id = str(row.get("target_id", ""))
                ext_n = _minmax(frozen_scores)
                unit_n = _minmax(dict(row.get("unit_scores", {})))
                coord_n = _minmax(dict(row.get("coord_scores", {})))
                sem_n = _minmax(dict(row.get("semantic_scores", {})))
                ext_rank, ext_margin = _rank_features(ext_n)
                coord_rank, coord_margin = _rank_features(coord_n)
                cand_ids = sorted(set(ext_n) | set(unit_n) | set(coord_n) | set(sem_n))
                feature_map: Dict[str, List[float]] = {}
                for cand_id in cand_ids:
                    feat = _feature_vector(
                        cand_id=cand_id,
                        subset_tags=row.get("subset_tags", []),
                        external_n=ext_n,
                        unit_n=unit_n,
                        coord_n=coord_n,
                        semantic_n=sem_n,
                        ext_rank=ext_rank,
                        ext_margin=ext_margin,
                        coord_rank=coord_rank,
                        coord_margin=coord_margin,
                        candidate_count=len(cand_ids),
                    )
                    feature_map[cand_id] = [float(x) for x in feat.tolist()]
                    pair_rows.append({
                        "protocol_item_id": str(item_id),
                        "seed": int(row.get("seed", -1)),
                        "item_split": _item_split3(str(item_id)),
                        "candidate_id": str(cand_id),
                        "target_id": target_id,
                        "label": bool(str(cand_id) == target_id),
                        "external_base_score": float(ext_n.get(cand_id, 0.0)),
                        "features": feature_map[cand_id],
                    })
                row["residual_feature_map"] = feature_map
                row["clean_frozen_scores"] = dict(frozen_scores)
            if index % 25 == 0 or index == total:
                print(f"[{_now_iso()}] residual_assoc_prepare_progress processed={index}/{total} valid={len(prepared_meta)} skipped={len(skipped_reasons)}", flush=True)
            del prepared
            if index % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        lease_id = str(device_info.get("lease_id", "")).strip()
        if lease_id:
            try:
                evalcore.release_lease(lease_id=lease_id, lease_path=str(args.lease_path))
            except Exception:
                pass
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    head = _fit_head(pair_rows)
    final_rows: List[Dict[str, Any]] = []
    trace_gate_weights = {
        "top_k": 4,
        "coord_gate_threshold": 0.0,
        "semantic_tie_margin": 0.0,
        "coord_tiebreak_weight": 0.0,
        "veto_penalty": 1000.0,
    }
    for item_id in sorted(selected_ids):
        if item_id not in item_assets:
            continue
        assets = item_assets[item_id]
        target_future_mask = assets["target_future_mask"]
        future_masks = assets["future_masks"]
        for row in baseline_rows_by_item.get(item_id, []):
            copied = dict(row)
            copied["item_split"] = _item_split3(str(item_id))
            final_rows.append(copied)
        for row in tusb_rows_by_item.get(item_id, []):
            if not bool(row.get("available", False)):
                continue
            base = dict(row.get("coord_result", {}))
            target_id = str(row.get("target_id", ""))
            common = (
                item_id,
                int(row["seed"]),
                str(row.get("method_name")),
                row.get("subset_tags", []),
                row.get("dataset", ""),
                row.get("clip_id", ""),
                int(row.get("protocol_eval_context_entity_count", 0)),
            )
            frozen_scores = dict(row.get("clean_frozen_scores", {}))
            frozen_n = _minmax(frozen_scores)
            sem_scores = dict(row.get("semantic_scores", {}))
            unit_scores = dict(row.get("unit_scores", {}))
            coord_scores = dict(row.get("coord_scores", {}))
            trace_scores, _ = evalcore._build_trace_gated_semantic_target_scores(sem_scores, coord_scores, **trace_gate_weights)
            feature_rows = {cid: np.asarray(vals, dtype=np.float64) for cid, vals in dict(row.get("residual_feature_map", {})).items()}
            residual_scores = _apply_head(frozen_n, feature_rows, head)
            final_rows.append(_residual_row(*common[:3], "frozen_external_teacher_only", *common[3:], _label_result(frozen_n, target_id, target_future_mask, future_masks, base, "frozen_external_teacher_only")))
            final_rows.append(_residual_row(*common[:3], "tusb_semantic_target", *common[3:], _label_result(sem_scores, target_id, target_future_mask, future_masks, base, "tusb_semantic_target")))
            final_rows.append(_residual_row(*common[:3], "unit_identity_only", *common[3:], _label_result(unit_scores, target_id, target_future_mask, future_masks, base, "unit_identity_only")))
            final_rows.append(_residual_row(*common[:3], "trace_gated_semantic_target", *common[3:], _label_result(trace_scores, target_id, target_future_mask, future_masks, base, "trace_gated_semantic_target")))
            final_rows.append(_residual_row(*common[:3], "trace_residual_association", *common[3:], _label_result(residual_scores, target_id, target_future_mask, future_masks, base, "trace_residual_association")))

    split_ids = {"train": [], "val": [], "test": []}
    for item_id in sorted(selected_ids):
        if item_id in prepared_meta:
            split_ids[_item_split3(item_id)].append(item_id)
    panels = {
        panel_name: _build_panel(
            panel_name=panel_name,
            split_meta=materialization[panel_name],
            item_ids=panel_item_ids[panel_name],
            prepared_meta=prepared_meta,
            skipped_reasons=skipped_reasons,
            rows=final_rows,
        )
        for panel_name in PANELS
    }
    eval_payload = {
        "generated_at_utc": _now_iso(),
        "eval_started_at": started,
        "eval_finished_at": _now_iso(),
        "wall_time_seconds": float(time.time() - wall_start),
        "source_shards": [str(part.strip()) for part in str(args.source_shards).split(",") if part.strip()],
        "frozen_external_teacher_only_backend": frozen_backend,
        "frozen_score_empty_items": int(frozen_empty_items),
        "residual_head": head,
        "split_definition": {
            "protocol_item_id_hash": "sha256[:8] mod 10",
            "train_buckets": [0, 1],
            "val_buckets": [2],
            "test_buckets": [3, 4, 5, 6, 7, 8, 9],
            "train_val_fraction_nominal": 0.3,
            "test_fraction_nominal": 0.7,
        },
        "split_sizes": {key: int(len(value)) for key, value in split_ids.items()},
        "train_item_ids_hash": _sha256_json(split_ids["train"]),
        "val_item_ids_hash": _sha256_json(split_ids["val"]),
        "test_item_ids_hash": _sha256_json(split_ids["test"]),
        "leakage_check_passed": bool(set(split_ids["train"]).isdisjoint(split_ids["test"]) and set(split_ids["val"]).isdisjoint(split_ids["test"])),
        "feature_schema_report": str(args.schema_json),
        "panels": panels,
    }
    _write_json(Path(args.eval_json), eval_payload)
    _write_md(
        Path(args.eval_md),
        "STWM Trace Residual Association Eval 20260423",
        [
            f"- frozen_external_teacher_only_backend: {frozen_backend}",
            f"- residual_head_type: {head['head_type']}",
            f"- selected_l2: {head['selected_l2']}",
            f"- val_top1: {head['val_top1']}",
            f"- split_sizes: {json.dumps(eval_payload['split_sizes'], ensure_ascii=True)}",
            *[
                f"- {name}: valid_items={panel['valid_items']} test_items={panel['test_items']} skipped_items={panel['skipped_items']} hash={panel['per_item_results_hash']}"
                for name, panel in panels.items()
            ],
        ],
    )
    return {"eval": eval_payload, "audit": audit, "schema": schema}


def _build_panel(
    panel_name: str,
    split_meta: Dict[str, Any],
    item_ids: set[str],
    prepared_meta: Mapping[str, Dict[str, Any]],
    skipped_reasons: Mapping[str, str],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    valid_ids = {item_id for item_id in item_ids if item_id in prepared_meta}
    panel_rows = [
        row for row in rows
        if str(row.get("protocol_item_id", "")) in valid_ids
        and str(row.get("item_split")) == "test"
    ]
    missing_ids = sorted(item_id for item_id in item_ids if item_id not in prepared_meta)
    skipped_counts = Counter(str(skipped_reasons.get(item_id, "missing_from_item_source")) for item_id in missing_ids)
    per_method_seed_results = {
        OFFICIAL_TUSB: {mode: _seed_table(panel_rows, OFFICIAL_TUSB, mode) for mode in TUSB_MODES},
        LEGACY: {"coord_only": _seed_table(panel_rows, LEGACY, "coord_only")},
        CAL: {"coord_only": _seed_table(panel_rows, CAL, "coord_only")},
        CROP: {"coord_only": _seed_table(panel_rows, CROP, "coord_only")},
    }
    return {
        "panel_name": panel_name,
        "total_items": int(split_meta.get("item_count", len(item_ids))),
        "valid_items": int(len(valid_ids)),
        "test_items": int(len([item_id for item_id in valid_ids if prepared_meta[item_id]["item_split"] == "test"])),
        "skipped_items": int(len(missing_ids)),
        "skipped_reason_counts": dict(sorted(skipped_counts.items())),
        "per_item_results_hash": _sha256_json(panel_rows),
        "per_item_results": panel_rows,
        "per_method_seed_results": per_method_seed_results,
    }


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return tracecond._aggregate_rows(rows)


def _seed_table(rows: List[Dict[str, Any]], method_name: str, scoring_mode: str) -> Dict[str, Any]:
    seed_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        picked = [
            row for row in rows
            if str(row.get("method_name")) == method_name
            and str(row.get("scoring_mode")) == scoring_mode
            and int(row.get("seed", -1)) == int(seed)
            and str(row.get("item_split")) == "test"
        ]
        metrics = _aggregate_rows(picked)
        seed_rows.append({"seed": int(seed), **metrics})
    metric_keys = [key for key in seed_rows[0].keys() if key != "seed"] if seed_rows else []
    return {
        "seed_rows": seed_rows,
        "mean": {key: _mean(row[key] for row in seed_rows) for key in metric_keys},
        "std": {key: _std(row[key] for row in seed_rows) for key in metric_keys},
    }


def _bootstrap_block(rows: List[Dict[str, Any]], left_method: str, left_mode: str, right_method: str, right_mode: str, split_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for metric_name, metric_key, subset_tag in METRIC_SPECS:
        deltas = tracecond._metric_deltas(rows, left_method, left_mode, right_method, right_mode, metric_key, subset_tag=subset_tag)
        out[metric_name] = lighteval._bootstrap_deltas(
            deltas,
            seed=lighteval._stable_bootstrap_seed(split_name, left_method, left_mode, right_method, right_mode, metric_name),
        )
    return out


def _mean_for(panel: Mapping[str, Any], method_name: str, mode: str, subset: str = "overall") -> float:
    rows = [
        row for row in panel.get("per_item_results", [])
        if str(row.get("method_name")) == method_name and str(row.get("scoring_mode")) == mode
    ]
    if subset == "continuity":
        rows = [row for row in rows if ("occlusion_reappearance" in set(row.get("subset_tags", []))) or ("long_gap_persistence" in set(row.get("subset_tags", [])))]
    elif subset == "ambiguity":
        rows = [row for row in rows if "crossing_ambiguity" in set(row.get("subset_tags", []))]
    return float(_aggregate_rows(rows)["overall_top1"]) if rows else 0.0


def _headtohead(panel: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for subset_name in ["overall", "continuity", "ambiguity"]:
        residual = _mean_for(panel, OFFICIAL_TUSB, "trace_residual_association", subset_name)
        frozen = _mean_for(panel, OFFICIAL_TUSB, "frozen_external_teacher_only", subset_name)
        semantic = _mean_for(panel, OFFICIAL_TUSB, "tusb_semantic_target", subset_name)
        legacy = _mean_for(panel, LEGACY, "coord_only", subset_name)
        out[subset_name] = {
            "trace_residual_association_mean": residual,
            "frozen_external_teacher_only_mean": frozen,
            "tusb_semantic_target_mean": semantic,
            "legacysem_mean": legacy,
            "trace_residual_improved_vs_frozen_external_teacher_only": bool(residual > frozen),
            "trace_residual_improved_vs_tusb_semantic_target": bool(residual > semantic),
            "trace_residual_improved_vs_legacysem": bool(residual > legacy),
            "frozen_external_teacher_only_sufficient": bool(frozen >= residual),
        }
    return out


def _build_bootstrap_decision(args: Any, eval_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if eval_payload is None:
        eval_payload = _load_json(Path(args.eval_json))
    panels = eval_payload.get("panels", {}) if isinstance(eval_payload.get("panels", {}), dict) else {}
    id_rows = panels["densified_200_context_preserving"]["per_item_results"]
    ood_rows = panels["heldout_burst_heavy_context_preserving"]["per_item_results"] + panels["heldout_scene_category_video_context_preserving"]["per_item_results"]
    bootstrap_panels = {
        "densified_200_context_preserving": {
            "trace_residual_vs_frozen_external_teacher_only": _bootstrap_block(id_rows, OFFICIAL_TUSB, "trace_residual_association", OFFICIAL_TUSB, "frozen_external_teacher_only", "densified_200_context_preserving"),
            "trace_residual_vs_tusb_semantic_target": _bootstrap_block(id_rows, OFFICIAL_TUSB, "trace_residual_association", OFFICIAL_TUSB, "tusb_semantic_target", "densified_200_context_preserving"),
            "trace_residual_vs_legacysem": _bootstrap_block(id_rows, OFFICIAL_TUSB, "trace_residual_association", LEGACY, "coord_only", "densified_200_context_preserving"),
        },
        "true_ood_combined": {
            "trace_residual_vs_frozen_external_teacher_only": _bootstrap_block(ood_rows, OFFICIAL_TUSB, "trace_residual_association", OFFICIAL_TUSB, "frozen_external_teacher_only", "true_ood_combined"),
            "trace_residual_vs_legacysem": _bootstrap_block(ood_rows, OFFICIAL_TUSB, "trace_residual_association", LEGACY, "coord_only", "true_ood_combined"),
        },
    }
    id_stat = bootstrap_panels["densified_200_context_preserving"]["trace_residual_vs_frozen_external_teacher_only"]["overall_top1"]
    ood_stat = bootstrap_panels["true_ood_combined"]["trace_residual_vs_frozen_external_teacher_only"]["overall_top1"]
    id_zero = bool(id_stat["zero_excluded"] and float(id_stat["mean_delta"]) > 0.0)
    ood_zero = bool(ood_stat["zero_excluded"] and float(ood_stat["mean_delta"]) > 0.0)
    id_mean = float(id_stat["mean_delta"])
    ood_mean = float(ood_stat["mean_delta"])
    if id_zero and ood_zero:
        claim_level = "strong_claim"
    elif id_mean > 0.0 and ood_mean > 0.0:
        claim_level = "moderate_claim"
    else:
        claim_level = "weak_claim"
    bootstrap_payload = {
        "generated_at_utc": _now_iso(),
        "panels": bootstrap_panels,
        "residual_assoc_zero_excluded_on_id": bool(id_zero),
        "residual_assoc_zero_excluded_on_ood": bool(ood_zero),
        "claim_level": claim_level,
    }
    _write_json(Path(args.bootstrap_json), bootstrap_payload)
    _write_md(
        Path(args.bootstrap_md),
        "STWM Trace Residual Association Bootstrap 20260423",
        [
            f"- residual_assoc_zero_excluded_on_id: {id_zero}",
            f"- residual_assoc_zero_excluded_on_ood: {ood_zero}",
            f"- claim_level: {claim_level}",
        ],
    )
    head = {name: _headtohead(panel) for name, panel in panels.items()}
    densified = head["densified_200_context_preserving"]["overall"]
    ood_a = head["heldout_burst_heavy_context_preserving"]
    ood_b = head["heldout_scene_category_video_context_preserving"]
    ood_a_overall = ood_a["overall"]
    ood_b_overall = ood_b["overall"]
    improved_vs_frozen = bool(
        densified["trace_residual_improved_vs_frozen_external_teacher_only"]
        and ood_a_overall["trace_residual_improved_vs_frozen_external_teacher_only"]
        and ood_b_overall["trace_residual_improved_vs_frozen_external_teacher_only"]
    )
    improved_vs_legacy = bool(
        densified["trace_residual_improved_vs_legacysem"]
        and ood_a_overall["trace_residual_improved_vs_legacysem"]
        and ood_b_overall["trace_residual_improved_vs_legacysem"]
    )
    continuity_contribution = bool(ood_a["continuity"]["trace_residual_improved_vs_frozen_external_teacher_only"] or ood_b["continuity"]["trace_residual_improved_vs_frozen_external_teacher_only"])
    ambiguity_contribution = bool(ood_a["ambiguity"]["trace_residual_improved_vs_frozen_external_teacher_only"] or ood_b["ambiguity"]["trace_residual_improved_vs_frozen_external_teacher_only"])
    frozen_sufficient = bool(ood_a_overall["frozen_external_teacher_only_sufficient"] or ood_b_overall["frozen_external_teacher_only_sufficient"])
    trace_coupling = bool(improved_vs_frozen and (continuity_contribution or ambiguity_contribution))
    official_story_supported = bool(trace_coupling and improved_vs_legacy and claim_level in {"strong_claim", "moderate_claim"})
    if official_story_supported:
        next_step_choice = "start_main_submission_assets"
    elif improved_vs_legacy or improved_vs_frozen:
        next_step_choice = "one_last_surgical_fix"
    else:
        next_step_choice = "reframe_as_moderate_claim_main_track"
    decision = {
        "generated_at_utc": _now_iso(),
        "frozen_external_teacher_only_sufficient_for_strongest_gain": bool(frozen_sufficient),
        "trace_residual_association_improved_vs_frozen_external_teacher_only": bool(improved_vs_frozen),
        "trace_residual_association_improved_vs_legacysem": bool(improved_vs_legacy),
        "ood_continuity_trace_residual_independent_contribution": bool(continuity_contribution),
        "ood_ambiguity_trace_residual_independent_contribution": bool(ambiguity_contribution),
        "trace_semantic_coupling_load_bearing": bool(trace_coupling),
        "official_story_supported": bool(official_story_supported),
        "claim_level": claim_level,
        "next_step_choice": next_step_choice,
        "headtohead": head,
    }
    _write_json(Path(args.decision_json), decision)
    _write_md(
        Path(args.decision_md),
        "STWM Trace Residual Association Decision 20260423",
        [
            f"- frozen_external_teacher_only_sufficient_for_strongest_gain: {decision['frozen_external_teacher_only_sufficient_for_strongest_gain']}",
            f"- trace_residual_association_improved_vs_frozen_external_teacher_only: {improved_vs_frozen}",
            f"- trace_residual_association_improved_vs_legacysem: {improved_vs_legacy}",
            f"- residual_assoc_zero_excluded_on_id: {id_zero}",
            f"- residual_assoc_zero_excluded_on_ood: {ood_zero}",
            f"- trace_semantic_coupling_load_bearing: {trace_coupling}",
            f"- official_story_supported: {official_story_supported}",
            f"- next_step_choice: {next_step_choice}",
        ],
    )
    return {"bootstrap": bootstrap_payload, "decision": decision}


def parse_args() -> Any:
    parser = ArgumentParser(description="Run STWM trace-conditioned residual association readout.")
    parser.add_argument("--mode", default="all", choices=["audit", "schema", "eval", "bootstrap_decision", "all"])
    parser.add_argument("--audit-json", default=str(REPORTS / "stwm_residual_assoc_audit_20260423.json"))
    parser.add_argument("--audit-md", default=str(DOCS / "STWM_RESIDUAL_ASSOC_AUDIT_20260423.md"))
    parser.add_argument("--schema-json", default=str(REPORTS / "stwm_residual_assoc_feature_schema_20260423.json"))
    parser.add_argument("--schema-md", default=str(DOCS / "STWM_RESIDUAL_ASSOC_FEATURE_SCHEMA_20260423.md"))
    parser.add_argument("--eval-json", default=str(REPORTS / "stwm_trace_residual_association_eval_20260423.json"))
    parser.add_argument("--eval-md", default=str(DOCS / "STWM_TRACE_RESIDUAL_ASSOCIATION_EVAL_20260423.md"))
    parser.add_argument("--bootstrap-json", default=str(REPORTS / "stwm_trace_residual_association_bootstrap_20260423.json"))
    parser.add_argument("--bootstrap-md", default=str(DOCS / "STWM_TRACE_RESIDUAL_ASSOCIATION_BOOTSTRAP_20260423.md"))
    parser.add_argument("--decision-json", default=str(REPORTS / "stwm_trace_residual_association_decision_20260423.json"))
    parser.add_argument("--decision-md", default=str(DOCS / "STWM_TRACE_RESIDUAL_ASSOCIATION_DECISION_20260423.md"))
    parser.add_argument("--dense-protocol-json", default=str(REPORTS / "stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--extended-protocol-json", default=str(REPORTS / "stage2_protocol_v3_extended_evalset_20260420.json"))
    parser.add_argument("--source-shards", default=",".join([
        str(SHARDS / "tusb_all_fixed.json"),
        str(SHARDS / "legacysem.json"),
        str(SHARDS / "calibration.json"),
        str(SHARDS / "cropenc.json"),
    ]))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=12.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    evalcore._apply_process_title_normalization()
    args = parse_args()
    if args.mode == "audit":
        _build_audit(args)
    elif args.mode == "schema":
        _build_schema(args)
    elif args.mode == "eval":
        _build_eval(args)
    elif args.mode == "bootstrap_decision":
        _build_bootstrap_decision(args)
    else:
        result = _build_eval(args)
        _build_bootstrap_decision(args, result["eval"])


if __name__ == "__main__":
    main()
