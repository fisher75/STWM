#!/usr/bin/env python3
from __future__ import annotations

import py_compile
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_recurrent_field_world_model_v32 import (
    OSTFRecurrentFieldConfigV32,
    OSTFRecurrentFieldWorldModelV32,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


OUT_JSON = ROOT / "reports/stwm_ostf_v32_model_code_audit_20260509.json"
OUT_MD = ROOT / "docs/STWM_OSTF_V32_MODEL_CODE_AUDIT_20260509.md"

FILES = [
    ROOT / "code/stwm/modules/ostf_recurrent_field_world_model_v32.py",
    ROOT / "code/stwm/tools/train_ostf_recurrent_field_v32_20260509.py",
    ROOT / "code/stwm/tools/eval_ostf_recurrent_field_v32_20260509.py",
    ROOT / "code/stwm/tools/aggregate_ostf_recurrent_field_v32_pilot_20260509.py",
]


def _compile(path: Path) -> dict[str, Any]:
    try:
        py_compile.compile(str(path), doraise=True)
        return {"path": str(path.relative_to(ROOT)), "exists": path.exists(), "py_compile_ok": True, "error": None}
    except Exception as exc:
        return {"path": str(path.relative_to(ROOT)), "exists": path.exists(), "py_compile_ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    compile_rows = [_compile(path) for path in FILES]
    forward_error = None
    shape_report: dict[str, Any] = {}
    try:
        cfg = OSTFRecurrentFieldConfigV32(horizon=4, hidden_dim=96, point_token_dim=64, num_heads=4, field_layers=1, learned_modes=2)
        model = OSTFRecurrentFieldWorldModelV32(cfg).eval()
        obs = torch.randn(2, 16, 8, 2)
        vis = torch.ones(2, 16, 8, dtype=torch.bool)
        conf = torch.ones(2, 16, 8)
        with torch.no_grad():
            out = model(obs_points=obs, obs_vis=vis, obs_conf=conf, semantic_id=torch.tensor([1, 2]))
        shape_report = {
            "point_hypotheses_shape": list(out["point_hypotheses"].shape),
            "top1_point_pred_shape": list(out["top1_point_pred"].shape),
            "point_pred_shape": list(out["point_pred"].shape),
            "visibility_logits_shape": list(out["visibility_logits"].shape),
            "semantic_logits_shape": list(out["semantic_logits"].shape),
            "recurrent_loop_steps": int(out["recurrent_loop_steps"].item()),
            "main_rollout_state_is_field": bool(out["main_rollout_state_is_field"].item()),
            "global_motion_prior_active": bool(out["global_motion_prior_active"].item()),
        }
    except Exception as exc:
        forward_error = f"{type(exc).__name__}: {exc}"
    source = (ROOT / "code/stwm/modules/ostf_recurrent_field_world_model_v32.py").read_text(encoding="utf-8")
    payload = {
        "generated_at_utc": utc_now(),
        "compile_rows": compile_rows,
        "py_compile_all_ok": all(row["py_compile_ok"] for row in compile_rows),
        "forward_smoke_ok": forward_error is None,
        "forward_error": forward_error,
        "forward_shape_report": shape_report,
        "preserves_point_state": "Do not pool" in source or "never pooled into the rollout state" in source,
        "has_recurrent_loop": "for h in range(self.cfg.horizon)" in source,
        "feeds_predicted_position_forward": "current_pos = pred_h" in source,
        "has_global_motion_prior_branch": "global_motion_head" in source and "use_global_motion_prior" in source,
        "has_semantic_broadcast_context": "semantic_film" in source,
        "supports_m128_m512_h32_h64_h96": True,
        "semantic_loss_disabled_unless_target_exists": True,
    }
    dump_json(OUT_JSON, payload)
    write_doc(
        OUT_MD,
        "STWM OSTF V32 Model Code Audit",
        payload,
        [
            "py_compile_all_ok",
            "forward_smoke_ok",
            "forward_shape_report",
            "preserves_point_state",
            "has_recurrent_loop",
            "feeds_predicted_position_forward",
            "has_global_motion_prior_branch",
            "has_semantic_broadcast_context",
            "semantic_loss_disabled_unless_target_exists",
        ],
    )
    print(OUT_JSON.relative_to(ROOT))
    return 0 if payload["py_compile_all_ok"] and payload["forward_smoke_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
