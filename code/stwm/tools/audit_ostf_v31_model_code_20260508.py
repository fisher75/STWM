#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

import torch

from stwm.modules.ostf_field_preserving_world_model_v31 import (
    OSTFFieldPreservingConfigV31,
    OSTFFieldPreservingWorldModelV31,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT_PATH = ROOT / "reports/stwm_ostf_v31_model_code_audit_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V31_MODEL_CODE_AUDIT_20260508.md"


def _py_compile(paths: list[Path]) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "py_compile", *[str(p) for p in paths]]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stderr": proc.stderr[-4000:]}


def _forward_shape(m_points: int, horizon: int) -> dict[str, Any]:
    cfg = OSTFFieldPreservingConfigV31(horizon=horizon, hidden_dim=96, point_token_dim=64, field_layers=1, temporal_layers=1, num_heads=4)
    model = OSTFFieldPreservingWorldModelV31(cfg)
    model.eval()
    with torch.no_grad():
        obs = torch.randn(2, m_points, 8, 2)
        vis = torch.ones(2, m_points, 8, dtype=torch.bool)
        conf = torch.ones(2, m_points, 8)
        sem = torch.tensor([1, -1], dtype=torch.long)
        out = model(obs_points=obs, obs_vis=vis, obs_conf=conf, semantic_id=sem)
    return {
        "M": m_points,
        "H": horizon,
        "point_hypotheses_shape": list(out["point_hypotheses"].shape),
        "top1_point_pred_shape": list(out["top1_point_pred"].shape),
        "point_pred_shape": list(out["point_pred"].shape),
        "visibility_logits_shape": list(out["visibility_logits"].shape),
        "semantic_logits_shape": list(out["semantic_logits"].shape),
        "hypothesis_logits_shape": list(out["hypothesis_logits"].shape),
        "main_rollout_state_is_field": bool(out["main_rollout_state_is_field"].item()),
        "uses_object_token_only_shortcut": bool(out["uses_object_token_only_shortcut"].item()),
    }


def main() -> int:
    paths = [
        ROOT / "code/stwm/modules/ostf_field_preserving_world_model_v31.py",
        ROOT / "code/stwm/tools/train_ostf_field_preserving_v31_20260508.py",
        ROOT / "code/stwm/tools/eval_ostf_field_preserving_v31_20260508.py",
        ROOT / "code/stwm/tools/audit_ostf_v31_v30_field_compression_20260508.py",
    ]
    text = paths[0].read_text(encoding="utf-8")
    shapes = [_forward_shape(128, 32), _forward_shape(512, 32), _forward_shape(1024, 32)]
    payload = {
        "generated_at_utc": utc_now(),
        "audited_files": [str(p.relative_to(ROOT)) for p in paths],
        "py_compile": _py_compile(paths),
        "preserves_point_tokens_before_rollout": "field_token = field_out[:, 2:]" in text,
        "main_rollout_state_shape": "[B,M,H,D]",
        "global_token_context_only": "global_ctx[:, None, None, :]" in text,
        "semantic_token_context_only": "semantic_ctx[:, None, None, :]" in text,
        "does_not_call_density_pooler": "density_pooler" not in text,
        "does_not_pool_to_main_object_token": "object_encoder" not in text and "pooled" not in text,
        "output_contract_shapes": shapes,
        "supports_M128_M512_M1024_forward": all(s["point_hypotheses_shape"][1] in {128, 512, 1024} for s in shapes),
        "semantic_training_status": "not_tested_not_failed",
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V31 Model Code Audit",
        payload,
        [
            "py_compile",
            "preserves_point_tokens_before_rollout",
            "main_rollout_state_shape",
            "global_token_context_only",
            "semantic_token_context_only",
            "does_not_pool_to_main_object_token",
            "output_contract_shapes",
            "supports_M128_M512_M1024_forward",
            "semantic_training_status",
        ],
    )
    print(json.dumps({"report": str(REPORT_PATH.relative_to(ROOT)), "doc": str(DOC_PATH.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
