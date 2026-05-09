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


REPORT_PATH = ROOT / "reports/stwm_ostf_v31_ablation_code_audit_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V31_ABLATION_CODE_AUDIT_20260508.md"


def _compile(paths: list[Path]) -> dict[str, Any]:
    proc = subprocess.run([sys.executable, "-m", "py_compile", *map(str, paths)], cwd=ROOT, text=True, capture_output=True)
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stderr": proc.stderr[-4000:]}


def _probe(**kwargs: Any) -> dict[str, Any]:
    base = {
        "horizon": 8,
        "hidden_dim": 96,
        "point_token_dim": 64,
        "field_layers": 1,
        "temporal_layers": 1,
        "num_heads": 4,
    }
    base.update(kwargs)
    cfg = OSTFFieldPreservingConfigV31(**base)
    model = OSTFFieldPreservingWorldModelV31(cfg).eval()
    with torch.no_grad():
        obs = torch.randn(2, 32, 8, 2)
        vis = torch.ones(2, 32, 8, dtype=torch.bool)
        conf = torch.ones(2, 32, 8)
        out = model(obs_points=obs, obs_vis=vis, obs_conf=conf, semantic_id=torch.tensor([1, -1]))
    return {
        "point_hypotheses_shape": list(out["point_hypotheses"].shape),
        "disable_field_interaction": bool(out["disable_field_interaction"].item()),
        "disable_global_context": bool(out["disable_global_context"].item()),
        "disable_semantic_context": bool(out["disable_semantic_context"].item()),
        "object_token_fallback": bool(out["uses_object_token_only_shortcut"].item()),
        "main_rollout_state_is_field": bool(out["main_rollout_state_is_field"].item()),
    }


def main() -> int:
    paths = [
        ROOT / "code/stwm/modules/ostf_field_preserving_world_model_v31.py",
        ROOT / "code/stwm/tools/train_ostf_field_preserving_v31_20260508.py",
        ROOT / "code/stwm/tools/eval_ostf_field_preserving_v31_20260508.py",
    ]
    train_text = paths[1].read_text(encoding="utf-8")
    payload = {
        "generated_at_utc": utc_now(),
        "py_compile": _compile(paths),
        "flags_present": {
            "--field-interaction-layers": "--field-interaction-layers" in train_text,
            "--disable-field-interaction": "--disable-field-interaction" in train_text,
            "--disable-global-context": "--disable-global-context" in train_text,
            "--disable-semantic-context": "--disable-semantic-context" in train_text,
            "--temporal-rollout-layers": "--temporal-rollout-layers" in train_text,
            "--object-token-fallback": "--object-token-fallback" in train_text,
        },
        "full_probe": _probe(),
        "no_field_interaction_probe": _probe(field_layers=0),
        "no_global_context_probe": _probe(disable_global_context=True),
        "no_semantic_context_probe": _probe(disable_semantic_context=True),
        "object_token_fallback_probe": _probe(object_token_fallback=True),
        "object_token_fallback_available": True,
        "semantic_ablation_diagnostic_only": True,
    }
    payload["audit_passed"] = bool(payload["py_compile"]["ok"] and all(payload["flags_present"].values()))
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V31 Ablation Code Audit",
        payload,
        ["audit_passed", "flags_present", "full_probe", "no_field_interaction_probe", "object_token_fallback_available", "semantic_ablation_diagnostic_only"],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0 if payload["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
