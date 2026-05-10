#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
import setproctitle
# 将进程名修改为 "python"
setproctitle.setproctitle("python")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


SUMMARY = ROOT / "reports/stwm_ostf_v33_8_ablation_safe_train_summary_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_8_ABLATION_SAFE_TRAIN_SUMMARY_20260510.md"
COMPLETE = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128"
MASK_ROOT = ROOT / "manifests/ostf_v33_8_split_matched_hard_identity_semantic"


def selected_k(default: int = 32) -> int:
    report = ROOT / "reports/stwm_ostf_v33_8_complete_h32_m128_target_coverage_20260510.json"
    if report.exists():
        return int(json.loads(report.read_text(encoding="utf-8")).get("selected_K", default))
    return default


def candidate_specs(k: int) -> list[dict[str, Any]]:
    roots = {
        "--semantic-identity-sidecar-root": COMPLETE / "semantic_identity_targets/pointodyssey",
        "--global-identity-label-root": COMPLETE / "global_identity_labels/pointodyssey",
        "--visual-teacher-root": COMPLETE / "visual_teacher_prototypes/pointodyssey/clip_vit_b32_local",
        "--semantic-prototype-target-root": COMPLETE / f"semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K{k}",
        "--prototype-vocab-path": ROOT / f"outputs/cache/stwm_ostf_v33_8_semantic_prototypes/pointodyssey/clip_vit_b32_local/K{k}/prototype_vocab.npz",
    }
    mask = MASK_ROOT / "H32_M128_seed42.json"
    common_roots: list[str] = []
    for flag, path in roots.items():
        common_roots += [flag, str(path)]
    return [
        {
            "name": "v33_8_v33_6_global_contrastive_baseline_seed42",
            "script": ROOT / "code/stwm/tools/train_ostf_v33_6_identity_contrastive_repair_20260509.py",
            "checkpoint": ROOT / "outputs/checkpoints/stwm_ostf_v33_6_identity_contrastive_repair/v33_8_v33_6_global_contrastive_baseline_seed42_best.pt",
            "args": ["--experiment-name", "v33_8_v33_6_global_contrastive_baseline_seed42", "--hard-subset-manifest", str(mask), *common_roots],
        },
        {
            "name": "v33_8_v33_7_full_identity_belief_seed42",
            "script": ROOT / "code/stwm/tools/train_ostf_v33_7_identity_belief_calibration_20260509.py",
            "checkpoint": ROOT / "outputs/checkpoints/stwm_ostf_v33_7_identity_belief_calibration/v33_8_v33_7_full_identity_belief_seed42_best.pt",
            "args": ["--experiment-name", "v33_8_v33_7_full_identity_belief_seed42", "--hard-train-mask-manifest", str(mask), *common_roots],
        },
        {
            "name": "v33_8_v33_7_no_fused_logits_seed42",
            "script": ROOT / "code/stwm/tools/train_ostf_v33_7_identity_belief_calibration_20260509.py",
            "checkpoint": ROOT / "outputs/checkpoints/stwm_ostf_v33_7_identity_belief_calibration/v33_8_v33_7_no_fused_logits_seed42_best.pt",
            "args": ["--experiment-name", "v33_8_v33_7_no_fused_logits_seed42", "--disable-fused-logits", "--hard-train-mask-manifest", str(mask), *common_roots],
        },
        {
            "name": "v33_8_v33_7_no_hard_bce_seed42",
            "script": ROOT / "code/stwm/tools/train_ostf_v33_7_identity_belief_calibration_20260509.py",
            "checkpoint": ROOT / "outputs/checkpoints/stwm_ostf_v33_7_identity_belief_calibration/v33_8_v33_7_no_hard_bce_seed42_best.pt",
            "args": ["--experiment-name", "v33_8_v33_7_no_hard_bce_seed42", "--disable-hard-bce", "--hard-train-mask-manifest", str(mask), *common_roots],
        },
        {
            "name": "v33_8_v33_7_no_embedding_similarity_seed42",
            "script": ROOT / "code/stwm/tools/train_ostf_v33_7_identity_belief_calibration_20260509.py",
            "checkpoint": ROOT / "outputs/checkpoints/stwm_ostf_v33_7_identity_belief_calibration/v33_8_v33_7_no_embedding_similarity_seed42_best.pt",
            "args": ["--experiment-name", "v33_8_v33_7_no_embedding_similarity_seed42", "--disable-embedding-similarity-logits", "--hard-train-mask-manifest", str(mask), *common_roots],
        },
    ]


def run_candidate(py: str, spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_existing and Path(spec["checkpoint"]).exists():
        return {
            "name": spec["name"],
            "skipped_existing": True,
            "completed": True,
            "checkpoint_path": str(Path(spec["checkpoint"]).relative_to(ROOT)),
            "returncode": 0,
        }
    cmd = [
        py,
        str(spec["script"]),
        *spec["args"],
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--contrastive-max-tokens",
        str(args.contrastive_max_tokens),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'code'}:{env.get('PYTHONPATH', '')}"
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {
        "name": spec["name"],
        "skipped_existing": False,
        "completed": proc.returncode == 0 and Path(spec["checkpoint"]).exists(),
        "checkpoint_path": str(Path(spec["checkpoint"]).relative_to(ROOT)),
        "returncode": proc.returncode,
        "command": " ".join(cmd),
        "stdout_tail": proc.stdout[-5000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--contrastive-max-tokens", type=int, default=2048)
    parser.add_argument("--candidate", default="all", help="all or comma-separated candidate names")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    py = os.environ.get("STWM_PYTHON", sys.executable)
    k = selected_k()
    specs = candidate_specs(k)
    wanted = None if args.candidate == "all" else {x.strip() for x in args.candidate.split(",") if x.strip()}
    rows = []
    for spec in specs:
        if wanted is not None and spec["name"] not in wanted:
            continue
        rows.append(run_candidate(py, spec, args))
    payload = {
        "generated_at_utc": utc_now(),
        "selected_K": k,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "target_root": str(COMPLETE.relative_to(ROOT)),
        "hard_mask_manifest": str((MASK_ROOT / "H32_M128_seed42.json").relative_to(ROOT)),
        "candidate_count": len(rows),
        "completed_candidate_count": sum(1 for r in rows if r.get("completed")),
        "all_candidates_completed": all(bool(r.get("completed")) for r in rows),
        "candidates": rows,
    }
    dump_json(SUMMARY, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.8 Ablation-Safe Train Summary",
        payload,
        ["selected_K", "steps", "batch_size", "target_root", "candidate_count", "completed_candidate_count", "all_candidates_completed"],
    )
    print(SUMMARY.relative_to(ROOT))
    return 0 if payload["all_candidates_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
