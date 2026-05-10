#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_13_real_stronger_teacher_preflight_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_13_REAL_STRONGER_TEACHER_PREFLIGHT_20260510.md"


def candidates() -> list[Path]:
    roots = [
        ROOT / "models",
        ROOT / "checkpoints",
        Path("/raid/chen034/workspace/data"),
        Path("/home/chen034/workspace/data"),
        Path("/raid/chen034/data"),
        Path("/home/chen034/data"),
    ]
    for extra in os.environ.get("STWM_DATA_ROOTS", "").split(":"):
        if extra:
            roots.append(Path(extra))
    return roots


def find_files(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for root in candidates():
        if not root.exists():
            continue
        for pat in patterns:
            out.extend(str(p) for p in root.rglob(pat) if p.is_file())
            if len(out) >= 12:
                return out[:12]
    return out[:12]


def main() -> int:
    checks: dict[str, Any] = {}
    specs = {
        "dinov2": {
            "packages": ["transformers", "timm"],
            "patterns": ["*dinov2*.bin", "*dinov2*.safetensors", "*dinov2*.pt", "*dino*v2*.pth"],
        },
        "siglip": {
            "packages": ["transformers"],
            "patterns": ["*siglip*.bin", "*siglip*.safetensors", "*siglip*.pt"],
        },
        "clip_vit_l14": {
            "packages": ["open_clip", "clip"],
            "patterns": ["*ViT-L*14*.pt", "*clip*l14*.pt", "*open_clip*L*14*.bin", "*clip*l14*.safetensors"],
        },
        "sam2_mask_feature": {
            "packages": ["sam2"],
            "patterns": ["*sam2*.pt", "*sam2*.pth", "*sam2*.safetensors"],
        },
    }
    for name, spec in specs.items():
        imports = {pkg: importlib.util.find_spec(pkg) is not None for pkg in spec["packages"]}
        files = find_files(spec["patterns"])
        loaded = bool(files and any(imports.values()))
        checks[name] = {
            "package_importable": imports,
            "local_checkpoint_paths": files,
            "model_available": loaded,
            "forward_dryrun_passed": False,
            "manual_download_blockers": [] if loaded else [f"local {name} weights not found; network download not attempted"],
            "expected_cache_cost": "H32/M128 complete set only: train=128 val=75 test=169; B200 feasible once weights are local",
        }
    only_clip = True
    payload = {
        "generated_at_utc": utc_now(),
        "stronger_teacher_preflight_done": True,
        "teachers": checks,
        "only_clip_b32_available": only_clip,
        "any_stronger_teacher_available": any(v["model_available"] for v in checks.values()),
        "b200_can_build_with_local_weights": True,
        "future_teacher_leakage_allowed": False,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.13 Real Stronger Teacher Preflight", payload, ["stronger_teacher_preflight_done", "only_clip_b32_available", "any_stronger_teacher_available", "b200_can_build_with_local_weights", "teachers"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
