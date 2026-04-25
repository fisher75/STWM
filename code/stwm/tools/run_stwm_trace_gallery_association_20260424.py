#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    import setproctitle  # type: ignore
except Exception:
    setproctitle = None

if setproctitle is not None:
    try:
        setproctitle.setproctitle("python")
    except Exception:
        pass

for candidate in [Path("/raid/chen034/workspace/stwm/code"), Path("/home/chen034/workspace/stwm/code")]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from stwm.tools import run_stage2_state_identifiability_eval_20260415 as evalcore
from stwm.tools import run_stwm_trace_prototype_association_20260424 as impl


ROOT = Path("/raid/chen034/workspace/stwm")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"


def parse_args() -> Any:
    parser = ArgumentParser(description="Run STWM trace-conditioned prototype/gallery association readout.")
    parser.add_argument("--mode", default="all", choices=["audit", "eval", "bootstrap_decision", "all"])
    parser.add_argument("--audit-json", default=str(REPORTS / "stwm_trace_gallery_feasibility_audit_20260424.json"))
    parser.add_argument("--audit-md", default=str(DOCS / "STWM_TRACE_GALLERY_FEASIBILITY_AUDIT_20260424.md"))
    parser.add_argument("--eval-json", default=str(REPORTS / "stwm_trace_gallery_eval_20260424.json"))
    parser.add_argument("--eval-md", default=str(DOCS / "STWM_TRACE_GALLERY_EVAL_20260424.md"))
    parser.add_argument("--bootstrap-json", default=str(REPORTS / "stwm_trace_gallery_bootstrap_20260424.json"))
    parser.add_argument("--bootstrap-md", default=str(DOCS / "STWM_TRACE_GALLERY_BOOTSTRAP_20260424.md"))
    parser.add_argument("--decision-json", default=str(REPORTS / "stwm_trace_gallery_decision_20260424.json"))
    parser.add_argument("--decision-md", default=str(DOCS / "STWM_TRACE_GALLERY_DECISION_20260424.md"))
    parser.add_argument("--dense-protocol-json", default=str(REPORTS / "stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--extended-protocol-json", default=str(REPORTS / "stage2_protocol_v3_extended_evalset_20260420.json"))
    parser.add_argument("--source-shards", default=",".join([
        str(impl.SHARDS / "tusb_all_fixed.json"),
        str(impl.SHARDS / "legacysem.json"),
        str(impl.SHARDS / "calibration.json"),
        str(impl.SHARDS / "cropenc.json"),
    ]))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=12.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=4.0)
    parser.add_argument("--audit-sample-items", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    evalcore._apply_process_title_normalization()
    args = parse_args()
    if args.mode == "audit":
        impl._build_feasibility_audit(args)
    elif args.mode == "eval":
        impl._build_eval(args)
    elif args.mode == "bootstrap_decision":
        impl._build_bootstrap_decision(args)
    else:
        result = impl._build_eval(args)
        impl._build_bootstrap_decision(args, result["eval"])


if __name__ == "__main__":
    main()
