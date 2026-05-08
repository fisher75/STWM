#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


MODEL_PATH = ROOT / "code/stwm/modules/ostf_external_gt_world_model_v30.py"
TRAIN_PATH = ROOT / "code/stwm/tools/train_ostf_external_gt_v30_20260508.py"
REPORT_PATH = ROOT / "reports/stwm_ostf_v31_v30_field_compression_audit_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V31_V30_FIELD_COMPRESSION_AUDIT_20260508.md"


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is not None


def main() -> int:
    model_text = MODEL_PATH.read_text(encoding="utf-8")
    train_text = TRAIN_PATH.read_text(encoding="utf-8") if TRAIN_PATH.exists() else ""
    pools_to_object = all(
        [
            "DensityAwarePointSetEncoderV30" in model_text,
            "_masked_mean" in model_text or "point_token.mean(dim=1)" in model_text,
            _has(r"pooled,\s*density_stats\s*=\s*self\.density_pooler\(point_token", model_text),
            _has(r"obj\s*=\s*self\.object_encoder\(torch\.cat\(\[pooled,\s*sem\]", model_text),
        ]
    )
    object_temporal_rollout = all(
        [
            _has(r"step_tokens\s*=\s*obj\[:,\s*None,\s*:\]\s*\+\s*self\.time_embed", model_text),
            _has(r"step_hidden\s*=\s*self\.rollout\(step_tokens\)", model_text),
        ]
    )
    point_decoder_residual = _has(r"step_hidden\[:,\s*None,\s*:,\s*:\]\s*\+\s*point_token\[:,\s*:,\s*None,\s*:\]", model_text)
    output_field_like = all(key in model_text for key in ["point_hypotheses", "top1_point_pred", "visibility_logits", "semantic_logits"])
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "audited_files": [str(MODEL_PATH.relative_to(ROOT)), str(TRAIN_PATH.relative_to(ROOT))],
        "v30_pools_point_tokens_to_object_token": bool(pools_to_object),
        "v30_temporal_rollout_object_token_time_token": bool(object_temporal_rollout),
        "v30_per_point_future_decoder_residual": bool(point_decoder_residual),
        "v30_output_shape_field_like": bool(output_field_like),
        "v30_main_rollout_state_field_preserving": False if pools_to_object and object_temporal_rollout else None,
        "v30_can_be_used_as_object_token_rollout_baseline": bool(pools_to_object and object_temporal_rollout and output_field_like),
        "m512_m1024_density_failure_consistent_with_field_compression": bool(pools_to_object),
        "needs_v31_field_preserving_rollout": bool(pools_to_object and object_temporal_rollout),
        "answers": {
            "1_point_tokens_pooled_to_object_token": (
                "yes: V30 computes point_token [B,M,D], then density_pooler returns pooled [B,D]."
                if pools_to_object
                else "not confirmed"
            ),
            "2_rollout_state": (
                "yes: rollout consumes step_tokens = obj[:,None,:] + time_embed, so temporal state is object/time token."
                if object_temporal_rollout
                else "not confirmed"
            ),
            "3_per_point_future": (
                "yes: per-point residual decoder adds step_hidden back to each point_token; points are decoded but not rolled out as the primary state."
                if point_decoder_residual
                else "not confirmed"
            ),
            "4_density_failure_alignment": "consistent: higher M can change pooled statistics but cannot create independent field rollout states.",
            "5_field_output_vs_field_state": "output is field-shaped, but the main rollout state is not field-preserving.",
            "6_v30_role": "V30 should be reported as an object-token rollout baseline, not as strict field-level rollout.",
            "7_v31_need": "V31 is required to test the original trace-field world-model hypothesis.",
        },
        "non_modifying_audit": True,
        "train_script_mentions_density_pooling_args": "--density-aware-pooling" in train_text,
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V31 V30 Field Compression Audit",
        payload,
        [
            "v30_pools_point_tokens_to_object_token",
            "v30_temporal_rollout_object_token_time_token",
            "v30_per_point_future_decoder_residual",
            "v30_main_rollout_state_field_preserving",
            "v30_can_be_used_as_object_token_rollout_baseline",
            "needs_v31_field_preserving_rollout",
            "answers",
        ],
    )
    print(json.dumps({"report": str(REPORT_PATH.relative_to(ROOT)), "doc": str(DOC_PATH.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
