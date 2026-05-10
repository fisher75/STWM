#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


TRAIN = ROOT / "reports/stwm_ostf_v33_8_ablation_safe_train_summary_20260510.json"
EVAL = ROOT / "reports/stwm_ostf_v33_8_ablation_safe_eval_summary_20260510.json"
DECISION = ROOT / "reports/stwm_ostf_v33_8_ablation_safe_eval_decision_20260510.json"
COVERAGE = ROOT / "reports/stwm_ostf_v33_8_complete_h32_m128_target_coverage_20260510.json"
TRAIN_SCRIPT = ROOT / "code/stwm/tools/train_ostf_v33_8_ablation_safe_identity_semantic_20260510.py"
EVAL_SCRIPT = ROOT / "code/stwm/tools/eval_ostf_v33_8_ablation_safe_identity_semantic_20260510.py"
REPORT = ROOT / "reports/stwm_ostf_v33_9_v33_8_training_truth_audit_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_9_V33_8_TRAINING_TRUTH_AUDIT_20260510.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def rel_or_none(path: Path) -> str | None:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def ckpt_args(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        ck = torch.load(path, map_location="cpu")
    except Exception:
        return {}
    return dict(ck.get("args", {})) if isinstance(ck, dict) else {}


def main() -> int:
    train = load_json(TRAIN)
    coverage = load_json(COVERAGE)
    candidates = list(train.get("candidates", []))
    skipped_count = sum(1 for c in candidates if bool(c.get("skipped_existing")))
    roots: dict[str, str] = {}
    rows = []
    for cand in candidates:
        ckpt_path = ROOT / str(cand.get("checkpoint_path", ""))
        args = ckpt_args(ckpt_path)
        root = ckpt_path.parent
        roots[str(cand.get("name"))] = rel_or_none(root) or ""
        semantic_root = str(args.get("semantic_identity_sidecar_root", ""))
        global_root = str(args.get("global_identity_label_root", ""))
        visual_root = str(args.get("visual_teacher_root", ""))
        proto_root = str(args.get("semantic_prototype_target_root", ""))
        uses_complete_root = all("stwm_ostf_v33_8_complete_h32_m128" in x for x in [semantic_root, global_root, visual_root, proto_root] if x)
        rows.append(
            {
                "name": cand.get("name"),
                "skipped_existing": bool(cand.get("skipped_existing")),
                "checkpoint_path": cand.get("checkpoint_path"),
                "checkpoint_root": roots[str(cand.get("name"))],
                "checkpoint_args_exist": bool(args),
                "uses_v33_8_complete_target_root": uses_complete_root,
                "train_sample_count": args.get("train_sample_count"),
                "semantic_identity_sidecar_root": semantic_root,
                "global_identity_label_root": global_root,
                "visual_teacher_root": visual_root,
                "semantic_prototype_target_root": proto_root,
            }
        )
    all_skipped = bool(candidates) and skipped_count == len(candidates)
    fresh_proven = bool(candidates) and skipped_count == 0 and all(r["uses_v33_8_complete_target_root"] and r.get("train_sample_count") == 128 for r in rows)
    script_text = TRAIN_SCRIPT.read_text(encoding="utf-8") if TRAIN_SCRIPT.exists() else ""
    payload = {
        "generated_at_utc": utc_now(),
        "inputs": {
            "train_summary": str(TRAIN.relative_to(ROOT)),
            "eval_summary": str(EVAL.relative_to(ROOT)),
            "eval_decision": str(DECISION.relative_to(ROOT)),
            "coverage": str(COVERAGE.relative_to(ROOT)),
            "train_script": str(TRAIN_SCRIPT.relative_to(ROOT)),
            "eval_script": str(EVAL_SCRIPT.relative_to(ROOT)),
        },
        "v33_8_training_not_fresh": all_skipped,
        "v33_8_fresh_training_proven": fresh_proven,
        "skipped_existing_candidate_count": skipped_count,
        "candidate_count": len(candidates),
        "candidate_checkpoint_roots": roots,
        "candidate_audit": rows,
        "complete_train_sample_count": coverage.get("complete_train_sample_count"),
        "train_script_has_skip_existing": "--skip-existing" in script_text,
        "exact_risk": "V33.8 train summary shows all candidates skipped_existing=true, so current V33.8 metrics are expanded evaluation on pre-existing V33.6/V33.7 checkpoints, not fresh expanded-coverage training."
        if all_skipped
        else "No all-skipped pattern detected.",
        "recommended_fix": "Run V33.9 fresh retrain into outputs/checkpoints/stwm_ostf_v33_9_fresh_expanded_h32_m128 with skip-existing disabled and checkpoint args proving V33.8 complete target roots.",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.9 V33.8 Training Truth Audit",
        payload,
        ["v33_8_training_not_fresh", "v33_8_fresh_training_proven", "skipped_existing_candidate_count", "candidate_checkpoint_roots", "exact_risk", "recommended_fix"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
