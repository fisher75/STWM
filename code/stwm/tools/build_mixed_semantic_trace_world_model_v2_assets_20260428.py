#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


REPORT_DIR = Path("reports")
DOC_DIR = Path("docs")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, title: str, payload: dict[str, Any], *, notes: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    for note in notes or []:
        lines.append(f"- {note}")
    if notes:
        lines.append("")
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path.cwd() / path


def _load_npz(path_value: str | Path) -> dict[str, np.ndarray]:
    return dict(np.load(_resolve(path_value), allow_pickle=True))


def _observed_npz(report_path: str | Path, c: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = _load_json(report_path)
    return report, _load_npz(report["target_cache_paths_by_prototype_count"][str(c)])


def _future_npz(report_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = _load_json(report_path)
    return report, _load_npz(report["target_cache_path"])


def _dataset_of(key: str) -> str:
    return str(key).split("::", 1)[0].upper()


def _combine_observed_targets() -> dict[str, Any]:
    out_dir = Path("outputs/cache/stwm_mixed_observed_semantic_prototype_targets_v2_20260428")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    for c in (32, 64):
        full_report, full = _observed_npz(REPORT_DIR / "stwm_fullscale_observed_semantic_prototype_targets_v1_20260428.json", c)
        vip_report, vip = _observed_npz(REPORT_DIR / "stwm_vipseg_observed_semantic_prototype_targets_v2_20260428.json", c)
        item_keys = np.asarray(full["item_keys"], dtype=object)
        splits = np.asarray(full["splits"], dtype=object)
        datasets = np.asarray([_dataset_of(str(k)) for k in item_keys], dtype=object)
        target = np.asarray(full["observed_semantic_proto_target"], dtype=np.int64).copy()
        dist = np.asarray(full["observed_semantic_proto_distribution"], dtype=np.float32).copy()
        mask = np.asarray(full["observed_semantic_proto_mask"], dtype=bool).copy()
        vip_index = {str(k): i for i, k in enumerate(np.asarray(vip["item_keys"]).astype(str).tolist())}
        replaced = 0
        for i, key in enumerate(np.asarray(item_keys).astype(str).tolist()):
            if not key.startswith("VIPSEG::"):
                continue
            vi = vip_index.get(key)
            if vi is None:
                continue
            target[i] = np.asarray(vip["observed_semantic_proto_target"][vi], dtype=np.int64)
            dist[i] = np.asarray(vip["observed_semantic_proto_distribution"][vi], dtype=np.float32)
            mask[i] = np.asarray(vip["observed_semantic_proto_mask"][vi], dtype=bool)
            replaced += 1
        cache_path = out_dir / f"observed_proto_targets_c{c}.npz"
        np.savez_compressed(
            cache_path,
            item_keys=item_keys,
            splits=splits,
            datasets=datasets,
            observed_semantic_proto_target=target,
            observed_semantic_proto_distribution=dist,
            observed_semantic_proto_mask=mask,
            prototypes=np.asarray(full["prototypes"], dtype=np.float32),
            prototype_count=np.asarray(c, dtype=np.int64),
            no_future_leakage=np.asarray(True),
        )
        paths[str(c)] = str(cache_path)
        results.append(
            {
                "prototype_count": c,
                "target_cache_path": str(cache_path),
                "vipseg_rows_replaced": int(replaced),
                "observed_proto_valid_ratio": float(mask.mean()),
                "observed_item_count": int(mask.any(axis=1).sum()),
            }
        )
    report = {
        "audit_name": "stwm_mixed_observed_semantic_prototype_targets_v2",
        "item_count": int(len(item_keys)),
        "prototype_count": 64,
        "target_cache_path": paths["64"],
        "target_cache_paths_by_prototype_count": paths,
        "results_by_prototype_count": results,
        "source_vspw_observed_report": "reports/stwm_fullscale_observed_semantic_prototype_targets_v1_20260428.json",
        "source_vipseg_observed_report": "reports/stwm_vipseg_observed_semantic_prototype_targets_v2_20260428.json",
        "raw_vipseg_rebuild_used": True,
        "no_future_leakage": True,
    }
    _write_json(REPORT_DIR / "stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json", report)
    return report


def _stats(keys: list[str], future: dict[str, np.ndarray], observed: dict[str, np.ndarray]) -> dict[str, Any]:
    f_index = {str(k): i for i, k in enumerate(np.asarray(future["item_keys"]).astype(str).tolist())}
    o_index = {str(k): i for i, k in enumerate(np.asarray(observed["item_keys"]).astype(str).tolist())}
    changed = stable = obs_slots = fut_slots = overlap_slots = 0
    dataset_counts: dict[str, int] = {}
    for key in keys:
        dataset_counts[_dataset_of(key)] = dataset_counts.get(_dataset_of(key), 0) + 1
        fi = f_index.get(key)
        oi = o_index.get(key)
        if fi is None or oi is None:
            continue
        target = np.asarray(future["future_semantic_proto_target"][fi], dtype=np.int64)
        fmask = np.asarray(future["target_mask"][fi], dtype=bool) & (target >= 0)
        obs_target = np.asarray(observed["observed_semantic_proto_target"][oi], dtype=np.int64)
        omask = np.asarray(observed["observed_semantic_proto_mask"][oi], dtype=bool) & (obs_target >= 0)
        valid = fmask & omask[None, :]
        ch = valid & (target != obs_target[None, :])
        changed += int(ch.sum())
        stable += int((valid & ~ch).sum())
        obs_slots += int(omask.sum())
        fut_slots += int(fmask.any(axis=0).sum())
        overlap_slots += int(valid.any(axis=0).sum())
    return {
        "item_count": int(len(keys)),
        "dataset_counts": dataset_counts,
        "changed_count": int(changed),
        "stable_count": int(stable),
        "changed_ratio": float(changed / max(changed + stable, 1)),
        "observed_slot_count": int(obs_slots),
        "future_valid_slot_count": int(fut_slots),
        "overlap_slot_count": int(overlap_slots),
    }


def _build_splits() -> dict[str, Any]:
    _, future64 = _future_npz(REPORT_DIR / "stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json")
    mixed_report, observed64 = _observed_npz(REPORT_DIR / "stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json", 64)
    fkeys = np.asarray(future64["item_keys"]).astype(str).tolist()
    o_index = {str(k): i for i, k in enumerate(np.asarray(observed64["item_keys"]).astype(str).tolist())}
    eligible_by_dataset: dict[str, list[str]] = {"VSPW": [], "VIPSEG": []}
    for i, key in enumerate(fkeys):
        oi = o_index.get(key)
        if oi is None:
            continue
        fmask = np.asarray(future64["target_mask"][i], dtype=bool)
        omask = np.asarray(observed64["observed_semantic_proto_mask"][oi], dtype=bool)
        if bool(fmask.any()) and bool(omask.any()) and bool((fmask.any(axis=0) & omask).any()):
            ds = _dataset_of(key)
            if ds in eligible_by_dataset:
                eligible_by_dataset[ds].append(key)
    rng = np.random.default_rng(20260428)
    splits = {"train": [], "val": [], "test": []}
    for ds, keys in eligible_by_dataset.items():
        keys = list(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        n_train = min(n_train, n)
        n_val = min(n_val, max(n - n_train, 0))
        splits["train"].extend(keys[:n_train])
        splits["val"].extend(keys[n_train : n_train + n_val])
        splits["test"].extend(keys[n_train + n_val :])
    for name in splits:
        splits[name] = sorted(splits[name])
    payload = {
        "audit_name": "stwm_mixed_semantic_trace_world_model_v2_splits",
        "split_seed": 20260428,
        "item_level_split": True,
        "video_level_split_if_available": "item key is dataset::clip_id; no duplicate key crosses splits",
        "no_leakage": True,
        "mixed_protocol_available": True,
        "cross_dataset_protocol_available": True,
        "eligible_item_count": int(sum(len(x) for x in eligible_by_dataset.values())),
        "eligible_by_dataset": {k: int(len(v)) for k, v in eligible_by_dataset.items()},
        "train_item_count": int(len(splits["train"])),
        "val_item_count": int(len(splits["val"])),
        "test_item_count": int(len(splits["test"])),
        "splits": splits,
        "stats_c64": {name: _stats(keys, future64, observed64) for name, keys in splits.items()},
    }
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json", payload)
    _write_doc(
        DOC_DIR / "STWM_MIXED_SEMANTIC_TRACE_WORLD_MODEL_V2_SPLITS_20260428.md",
        "STWM Mixed Semantic Trace World Model V2 Splits",
        payload,
        notes=["Splits are constructed after raw VIPSeg observed-memory repair; model training/eval is intentionally not claimed here."],
    )
    return payload


def _write_skipped_training_eval_reports() -> None:
    skipped_reason = (
        "VIPSeg raw observed memory coverage is repaired and mixed splits are available; "
        "mixed fullscale training/evaluation/significance were intentionally not run in this "
        "data-pipeline repair command."
    )
    common = {
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "stage1_frozen": True,
        "trace_dynamic_path_frozen": True,
        "training_started": False,
        "eval_started": False,
        "significance_available": False,
        "skipped_reason": skipped_reason,
    }
    train = {
        "audit_name": "stwm_mixed_semantic_trace_world_model_v2_train_summary",
        **common,
    }
    eval_report = {
        "audit_name": "stwm_mixed_semantic_trace_world_model_v2_eval",
        **common,
        "free_rollout_path": True,
        "teacher_forced_path_used": False,
        "mixed_training_checkpoint_available": False,
    }
    sig = {
        "audit_name": "stwm_mixed_semantic_trace_world_model_v2_significance",
        **common,
        "bootstrap_run": False,
        "mixed_eval_available": False,
    }
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v2_train_summary_20260428.json", train)
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v2_eval_20260428.json", eval_report)
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v2_significance_20260428.json", sig)
    _write_doc(
        DOC_DIR / "STWM_MIXED_SEMANTIC_TRACE_WORLD_MODEL_V2_EVAL_20260428.md",
        "STWM Mixed Semantic Trace World Model V2 Eval",
        eval_report,
        notes=[
            "No mixed eval result is claimed here.",
            "The next executable step is mixed fullscale training/evaluation under the fixed raw VIPSeg observed-memory protocol.",
        ],
    )


def _write_guardrail() -> None:
    guardrail = {
        "guardrail_version": "v36",
        "current_status": (
            "raw VIPSeg observed memory rebuild completed; mixed protocol is available; "
            "mixed training/eval is not yet run."
        ),
        "allowed": [
            "raw VIPSeg observed memory rebuild",
            "mixed/cross-dataset free-rollout protocol after coverage is sufficient",
            "semantic trace world model output",
            "observed semantic memory as world-state input",
            "Stage1 frozen and trace dynamic path frozen",
        ],
        "forbidden": [
            "partial predecode cache accepted as paper-grade VIPSeg support",
            "candidate scorer",
            "SAM2/CoTracker plugin",
            "future candidate leakage",
            "hiding VSPW-only limitation",
            "changing method before VIPSeg data pipeline is fixed",
        ],
    }
    _write_json(REPORT_DIR / "stwm_world_model_no_drift_guardrail_v36_20260428.json", guardrail)
    _write_doc(
        DOC_DIR / "STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V36.md",
        "STWM World Model No-Drift Guardrail V36",
        guardrail,
    )


def main() -> None:
    mixed_observed = _combine_observed_targets()
    splits = _build_splits()
    _write_skipped_training_eval_reports()
    _write_guardrail()
    decision = _load_json(REPORT_DIR / "stwm_vipseg_raw_observed_memory_v2_decision_20260428.json")
    decision.update(
        {
            "mixed_training_started": False,
            "residual_beats_copy_mixed": "not_evaluated",
            "residual_beats_copy_vipseg": "not_evaluated",
            "changed_gain_CI_excludes_zero_mixed": False,
            "paper_world_model_claimable": "unclear",
            "paper_world_model_claim_scope": "VIPSeg raw observed memory is repaired and mixed protocol is available, but mixed training/eval has not been run; VSPW-only result remains the completed claim.",
            "semantic_field_branch_status": "vspw_only_with_limitation",
            "recommended_next_step_choice": "proceed_to_paper_assets_with_vspw_only_limitation",
            "mixed_observed_report": "reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json",
            "mixed_split_report": "reports/stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json",
        }
    )
    _write_json(REPORT_DIR / "stwm_vipseg_raw_observed_memory_v2_decision_20260428.json", decision)
    _write_doc(
        DOC_DIR / "STWM_VIPSEG_RAW_OBSERVED_MEMORY_V2_DECISION_20260428.md",
        "STWM VIPSeg Raw Observed Memory V2 Decision",
        decision,
        notes=[
            "VIPSeg raw observed memory is fixed.",
            "Mixed protocol is available, but no mixed fullscale training/evaluation has been completed in this command.",
        ],
    )
    pool = _load_json(REPORT_DIR / "stwm_mixed_semantic_trace_target_pool_v2_20260428.json")
    pool.update(
        {
            "mixed_observed_report": "reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json",
            "mixed_split_report": "reports/stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json",
            "train_item_count": int(splits["train_item_count"]),
            "val_item_count": int(splits["val_item_count"]),
            "test_item_count": int(splits["test_item_count"]),
        }
    )
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_target_pool_v2_20260428.json", pool)


if __name__ == "__main__":
    main()
