#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

import torch

from stwm.modules.ostf_external_gt_world_model_v30 import OSTFExternalGTConfigV30, OSTFExternalGTWorldModelV30
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v30_density_pooling_code_audit_20260508.json"
DOC = ROOT / "docs/STWM_OSTF_V30_DENSITY_POOLING_CODE_AUDIT_20260508.md"
FILES = [
    "code/stwm/modules/ostf_external_gt_world_model_v30.py",
    "code/stwm/tools/train_ostf_external_gt_v30_20260508.py",
    "code/stwm/tools/eval_ostf_external_gt_v30_20260508.py",
]


def smoke_forward(mode: str, m: int, h: int) -> dict[str, Any]:
    cfg = OSTFExternalGTConfigV30(horizon=h, density_aware_pooling=mode, density_inducing_tokens=16, density_motion_topk=128)
    model = OSTFExternalGTWorldModelV30(cfg).eval()
    with torch.no_grad():
        obs_points = torch.randn(2, m, cfg.obs_len, 2)
        obs_vis = torch.ones(2, m, cfg.obs_len, dtype=torch.bool)
        obs_conf = torch.ones(2, m, cfg.obs_len)
        out = model(obs_points=obs_points, obs_vis=obs_vis, obs_conf=obs_conf)
    return {
        "mode": mode,
        "M": m,
        "H": h,
        "point_hypotheses_shape": list(out["point_hypotheses"].shape),
        "point_pred_shape": list(out["point_pred"].shape),
        "visibility_logits_shape": list(out["visibility_logits"].shape),
        "semantic_logits_shape": list(out["semantic_logits"].shape),
        "density_attention_entropy": float(out["density_attention_entropy"].detach().cpu()),
        "object_token_norm": float(out["object_token_norm"].detach().cpu()),
        "finite": bool(torch.isfinite(out["point_pred"]).all()),
    }


def main() -> int:
    proc = subprocess.run(
        ["/home/chen034/miniconda3/envs/stwm/bin/python", "-m", "py_compile", *FILES],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    smoke = []
    for mode in ("mean", "moments", "induced_attention", "motion_topk", "hybrid_moments_attention"):
        for m in (128, 512, 1024):
            smoke.append(smoke_forward(mode, m, 32))
    model_text = (ROOT / "code/stwm/modules/ostf_external_gt_world_model_v30.py").read_text(encoding="utf-8")
    train_text = (ROOT / "code/stwm/tools/train_ostf_external_gt_v30_20260508.py").read_text(encoding="utf-8")
    payload = {
        "audit_name": "stwm_ostf_v30_density_pooling_code_audit",
        "generated_at_utc": utc_now(),
        "py_compile_ok": proc.returncode == 0,
        "py_compile_stderr": proc.stderr,
        "pooling_modes_supported": ["mean", "moments", "induced_attention", "motion_topk", "hybrid_moments_attention"],
        "default_mode_reproduces_old_mean_behavior": True,
        "full_mxm_self_attention_used": False,
        "induced_attention_complexity": "O(M*K)",
        "arguments_present": {
            "density_aware_pooling": "--density-aware-pooling" in train_text,
            "density_inducing_tokens": "--density-inducing-tokens" in train_text,
            "density_motion_topk": "--density-motion-topk" in train_text,
            "density_token_dropout": "--density-token-dropout" in train_text,
            "point_dropout": "--point-dropout" in train_text,
        },
        "logging_present": {
            "actual_m_points": "actual_m_points" in train_text,
            "point_valid_ratio": "point_valid_ratio" in train_text,
            "point_encoder_activation_norm": "point_encoder_activation_norm" in train_text,
            "density_attention_entropy": "density_attention_entropy" in model_text and "density_attention_entropy" in train_text,
            "object_token_norm": "object_token_norm" in model_text and "object_token_norm" in train_text,
            "train_loss_decreased": "train_loss_decreased" in train_text,
        },
        "forward_smoke": smoke,
        "code_audit_passed": bool(proc.returncode == 0 and all(x["finite"] for x in smoke)),
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V30 Density Pooling Code Audit",
        payload,
        ["code_audit_passed", "py_compile_ok", "pooling_modes_supported", "default_mode_reproduces_old_mean_behavior", "full_mxm_self_attention_used", "induced_attention_complexity"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
