#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _exists(path: str) -> bool:
    return Path(path).exists()


def _load(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    c_reports = {
        "C32": "reports/stwm_mixed_fullscale_v2_mixed_test_eval_complete_20260428.json",
        "C64": "reports/stwm_mixed_fullscale_v2_val_eval_c64_complete_20260428.json",
        "C16": "reports/stwm_fstf_scaling_c16_v9_eval_20260501.json",
        "C128": "reports/stwm_fstf_scaling_c128_v9_eval_20260501.json",
    }
    horizon_cache = {
        "H8": "reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json",
        "H16": "reports/stwm_fstf_h16_target_pool_v9_20260501.json",
        "H24": "reports/stwm_fstf_h24_target_pool_v9_20260501.json",
    }
    density_cache = {
        "K8": "reports/stwm_mixed_fullscale_v2_splits_20260428.json",
        "K16": "reports/stwm_fstf_k16_target_pool_v9_20260501.json",
        "K32": "reports/stwm_fstf_k32_target_pool_v9_20260501.json",
    }
    model_scaling_reports = {
        "small_current": "reports/stwm_mixed_fullscale_v2_mixed_test_eval_complete_20260428.json",
        "base_d512": "reports/stwm_fstf_model_scale_base_v9_eval_20260501.json",
        "large_d768": "reports/stwm_fstf_model_scale_large_v9_eval_20260501.json",
    }
    missing = {
        "prototype": [k for k, p in c_reports.items() if not _exists(p)],
        "horizon": [k for k, p in horizon_cache.items() if not _exists(p)],
        "trace_density": [k for k, p in density_cache.items() if not _exists(p)],
        "model_size": [k for k, p in model_scaling_reports.items() if not _exists(p)],
    }
    c32 = _load(c_reports["C32"]).get("best_metrics", {})
    c64 = _load(c_reports["C64"]).get("best_metrics", {})
    report = {
        "audit_name": "stwm_fstf_scaling_v9_gate",
        "scaling_completed": False,
        "prototype_scaling": {
            "requested": ["C16", "C32", "C64", "C128"],
            "available_reports": {k: _exists(p) for k, p in c_reports.items()},
            "current_available_metrics": {
                "C32_changed_gain": c32.get("changed_subset_gain_over_copy"),
                "C64_changed_gain": c64.get("changed_subset_gain_over_copy"),
            },
            "prototype_scaling_positive": None,
        },
        "horizon_scaling": {
            "requested": ["H8", "H16", "H24"],
            "available_target_or_eval_reports": {k: _exists(p) for k, p in horizon_cache.items()},
            "horizon_scaling_positive": None,
        },
        "trace_density_scaling": {
            "requested": ["K8", "K16", "K32"],
            "available_target_or_eval_reports": {k: _exists(p) for k, p in density_cache.items()},
            "trace_density_scaling_positive": None,
        },
        "model_size_scaling": {
            "requested": ["small/current", "base d_model512", "large d_model768"],
            "available_reports": {k: _exists(p) for k, p in model_scaling_reports.items()},
            "model_size_scaling_positive": None,
        },
        "missing_scaling_points": missing,
        "whether_dense_trace_field_claim_allowed": False,
        "whether_long_horizon_world_model_claim_allowed": False,
        "next_required_action": "materialize_missing_H16_H24_K16_K32_and_train_scaling_seeds",
    }
    out = Path("reports/stwm_fstf_scaling_v9_20260501.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    doc = Path("docs/STWM_FSTF_SCALING_V9_20260501.md")
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        "\n".join(
            [
                "# STWM FSTF Scaling V9 Gate",
                "",
                "- scaling_completed: `False`",
                "- whether_dense_trace_field_claim_allowed: `False`",
                "- whether_long_horizon_world_model_claim_allowed: `False`",
                f"- missing_scaling_points: `{missing}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[scaling-v9] report={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
