#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_lastobs_v28_common_20260502 import build_v28_rows, choose_visibility_aware_gamma_on_val
from stwm.tools.ostf_traceanything_metrics_v26_20260502 import paired_bootstrap_from_rows_v26
from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc
from stwm.tools.ostf_v29_benchmark_utils_20260508 import (
    COMBOS,
    PRIOR_NAMES,
    ROOT,
    V29_MANIFEST_DIR,
    aggregate_extended_rows,
    evaluate_prior_suite,
    logical_uid_from_row,
    sample_logical_uid,
    utc_now,
)


REPORT_PATH = ROOT / "reports/stwm_ostf_v29_antiprior_existing_eval_20260508.json"
BOOT_PATH = ROOT / "reports/stwm_ostf_v29_antiprior_existing_bootstrap_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V29_ANTIPRIOR_EXISTING_EVAL_20260508.md"
RUN_DIR = ROOT / "reports/stwm_ostf_v28_runs"


def _load_manifest(name: str) -> dict[str, Any]:
    path = V29_MANIFEST_DIR / f"{name}.json"
    if not path.exists():
        return {"entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_maps() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in [
        "test",
        "test_h32_mixed",
        "test_h64_motion",
        "test_occlusion_reappearance",
        "test_semantic_confuser",
        "test_nonlinear_large_disp",
    ]:
        payload = _load_manifest(name)
        by_combo: dict[str, set[str]] = {}
        tags_by_combo_uid: dict[tuple[str, str], list[str]] = {}
        for entry in payload.get("entries", []):
            combo = str(entry["combo"])
            logical = str(entry["logical_uid"])
            by_combo.setdefault(combo, set()).add(logical)
            # The same logical object can be evaluated at M512 by swapping the density in the combo.
            if combo.startswith("M128"):
                by_combo.setdefault(combo.replace("M128", "M512"), set()).add(logical)
                tags_by_combo_uid[(combo.replace("M128", "M512"), logical)] = list(entry.get("v29_subset_tags", []))
            tags_by_combo_uid[(combo, logical)] = list(entry.get("v29_subset_tags", []))
        out[name] = {"by_combo": by_combo, "tags_by_combo_uid": tags_by_combo_uid, "item_count": len(payload.get("entries", []))}
    return out


def _load_runs() -> dict[str, dict[str, Any]]:
    out = {}
    if not RUN_DIR.exists():
        return out
    for path in sorted(RUN_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("item_scores") and payload.get("test_metrics"):
            out[str(payload.get("experiment_name") or path.stem)] = payload
    return out


def _val_score(run: dict[str, Any]) -> float:
    subset = run.get("val_subset_metrics", {}).get("last_observed_hard_top20", {})
    val = subset.get("minFDE_K_px", run.get("val_metrics", {}).get("minFDE_K_px"))
    return float(val) if val is not None else float("inf")


def _select_run(runs: dict[str, dict[str, Any]], prefix: str, seeds: tuple[int, ...] = (42, 123, 456, 789, 2026)) -> dict[str, Any] | None:
    candidates = [runs.get(f"{prefix}_seed{seed}") for seed in seeds]
    candidates = [r for r in candidates if r]
    if not candidates:
        return None
    return sorted(candidates, key=_val_score)[0]


def _attach_v29_fields(rows: list[dict[str, Any]], combo: str, tags_by_uid: dict[tuple[str, str], list[str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        logical = logical_uid_from_row(row)
        tags = list(tags_by_uid.get((combo, logical), []))
        if not tags:
            continue
        x = dict(row)
        x["v29_in_manifest"] = True
        for tag in ["last_visible_hard", "anti_prior_motion", "nonlinear_large_disp", "occlusion_reappearance", "semantic_confuser", "extraction_uncertainty"]:
            x[f"v29_{tag}"] = tag in tags
        if x.get("minFDE_K_px") is not None:
            fde = float(x["minFDE_K_px"])
            x["MissRate_128px"] = float(fde > 128.0)
            x["threshold_auc_endpoint_16_32_64_128"] = float(np.mean([1.0 - float(fde > thr) for thr in (16.0, 32.0, 64.0, 128.0)]))
        else:
            x["MissRate_128px"] = None
            x["threshold_auc_endpoint_16_32_64_128"] = None
        x.setdefault("BestOfK_PCK_64px", None)
        x.setdefault("relative_deformation_layout_error_px", None)
        out.append(x)
    return out


def _subset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    subsets = {
        "all": aggregate_extended_rows(rows),
        "last_visible_hard": aggregate_extended_rows(rows, subset_key="v29_last_visible_hard"),
        "anti_prior_motion": aggregate_extended_rows(rows, subset_key="v29_anti_prior_motion"),
        "nonlinear_large_disp": aggregate_extended_rows(rows, subset_key="v29_nonlinear_large_disp"),
        "occlusion_reappearance": aggregate_extended_rows(rows, subset_key="v29_occlusion_reappearance"),
        "semantic_confuser": aggregate_extended_rows(rows, subset_key="v29_semantic_confuser"),
        "extraction_uncertainty": aggregate_extended_rows(rows, subset_key="v29_extraction_uncertainty"),
    }
    return subsets


def _by_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {ds: aggregate_extended_rows(rows, dataset=ds) for ds in sorted({r["dataset"] for r in rows})}


def _prior_rows_for_combo(combo: str, manifest_maps: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows, proto_centers, _, _ = build_v28_rows(combo, seed=42)
    samples = rows["test"]
    val_gamma, _ = choose_visibility_aware_gamma_on_val(rows["val"], proto_centers)
    priors = evaluate_prior_suite(samples, proto_centers, visibility_gamma=val_gamma)
    tags_by_uid = manifest_maps["test"]["tags_by_combo_uid"]
    return {name: _attach_v29_fields(priors[name]["item_rows"], combo, tags_by_uid) for name in PRIOR_NAMES}


def _bootstrap(boot: dict[str, Any], key: str, rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str, higher_better: bool, subset: str | None = None) -> None:
    boot[key] = paired_bootstrap_from_rows_v26(rows_a, rows_b, metric=metric, higher_better=higher_better, subset_key=subset)
    boot[key]["metric"] = metric
    boot[key]["subset_key"] = subset


def main() -> int:
    manifest_maps = _manifest_maps()
    runs = _load_runs()
    selected_runs = {
        "V28_H32_best_available": ("M128_H32", _select_run(runs, "v28_lastobs_m128_h32")),
        "V28_H64_best_available": ("M128_H64", _select_run(runs, "v28_lastobs_m128_h64")),
        "V28_M512_H32_available": ("M512_H32", runs.get("v28_lastobs_m512_h32_seed42")),
        "V28_H64_wo_dense_points": ("M128_H64", _select_run(runs, "v28_lastobs_m128_h64_wo_dense_points", (42, 123, 456))),
        "V28_H64_wo_semantic_memory": ("M128_H64", _select_run(runs, "v28_lastobs_m128_h64_wo_semantic_memory", (42, 123, 456))),
        "V28_H64_wo_residual_modes": ("M128_H64", _select_run(runs, "v28_lastobs_m128_h64_wo_residual_modes", (42, 123, 456))),
    }
    eval_payload: dict[str, Any] = {
        "eval_name": "stwm_ostf_v29_antiprior_existing_eval",
        "generated_at_utc": utc_now(),
        "manifest_dir": str(V29_MANIFEST_DIR.relative_to(ROOT)),
        "metric_schema_note": (
            "Analytic priors include MissRate@128, endpoint threshold-AUC, PCK@64 and relative layout error. "
            "Existing V28 item-score reports do not store raw predictions, so V28 PCK@64/layout error are null; "
            "V28 MissRate@128 and threshold-AUC are derived from stored minFDE."
        ),
        "priors": {},
        "existing_v28_models": {},
    }
    boot: dict[str, Any] = {
        "bootstrap_name": "stwm_ostf_v29_antiprior_existing_bootstrap",
        "generated_at_utc": utc_now(),
        "comparisons": {},
    }
    prior_rows_by_combo = {combo: _prior_rows_for_combo(combo, manifest_maps) for combo in COMBOS}
    for combo, prior_rows in prior_rows_by_combo.items():
        eval_payload["priors"][combo] = {
            name: {
                "subset_metrics": _subset_metrics(rows),
                "metrics_by_dataset": _by_dataset(rows),
            }
            for name, rows in prior_rows.items()
        }
    for label, (combo, run) in selected_runs.items():
        if not run:
            eval_payload["existing_v28_models"][label] = {"missing": True, "combo": combo}
            continue
        rows = _attach_v29_fields(run["item_scores"], combo, manifest_maps["test"]["tags_by_combo_uid"])
        eval_payload["existing_v28_models"][label] = {
            "missing": False,
            "combo": combo,
            "experiment_name": run["experiment_name"],
            "selection_rule": "best validation last_observed_hard_top20 minFDE among completed seeds; no test selection",
            "val_score_used": _val_score(run),
            "subset_metrics": _subset_metrics(rows),
            "metrics_by_dataset": _by_dataset(rows),
        }
        if combo in prior_rows_by_combo:
            for prior in ["last_visible_copy", "visibility_aware_damped", "last_observed_copy", "median_object_anchor_copy"]:
                base_rows = prior_rows_by_combo[combo][prior]
                prefix = f"{label}_vs_{prior}"
                _bootstrap(boot["comparisons"], f"{prefix}_all_minFDE", rows, base_rows, "minFDE_K_px", False)
                _bootstrap(boot["comparisons"], f"{prefix}_last_visible_hard_minFDE", rows, base_rows, "minFDE_K_px", False, "v29_last_visible_hard")
                _bootstrap(boot["comparisons"], f"{prefix}_anti_prior_motion_minFDE", rows, base_rows, "minFDE_K_px", False, "v29_anti_prior_motion")
                _bootstrap(boot["comparisons"], f"{prefix}_all_MissRate64", rows, base_rows, "MissRate_64px", False)
                _bootstrap(boot["comparisons"], f"{prefix}_all_threshold_auc", rows, base_rows, "threshold_auc_endpoint_16_32_64_128", True)
    # Prior hierarchy bootstrap on H64/H32 manifests.
    for combo in ["M128_H32", "M128_H64"]:
        rows = prior_rows_by_combo.get(combo, {})
        if "last_visible_copy" in rows and "visibility_aware_damped" in rows:
            _bootstrap(
                boot["comparisons"],
                f"{combo}_last_visible_vs_visibility_aware_damped_hard_minFDE",
                rows["last_visible_copy"],
                rows["visibility_aware_damped"],
                "minFDE_K_px",
                False,
                "v29_last_visible_hard",
            )
    dump_json(REPORT_PATH, eval_payload)
    dump_json(BOOT_PATH, boot)
    write_doc(
        DOC_PATH,
        "STWM OSTF V29 Anti-Prior Existing Eval",
        {
            "manifest_dir": eval_payload["manifest_dir"],
            "prior_combos": list(eval_payload["priors"].keys()),
            "existing_v28_models": list(eval_payload["existing_v28_models"].keys()),
            "bootstrap_path": str(BOOT_PATH.relative_to(ROOT)),
            "metric_schema_note": eval_payload["metric_schema_note"],
        },
        ["manifest_dir", "prior_combos", "existing_v28_models", "bootstrap_path", "metric_schema_note"],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
