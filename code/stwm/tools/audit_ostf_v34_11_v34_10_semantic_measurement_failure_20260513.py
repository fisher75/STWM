#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


TRAIN_SRC = ROOT / "code/stwm/tools/train_ostf_v34_10_trace_contract_oracle_residual_probe_20260512.py"
EVAL_SRC = ROOT / "code/stwm/tools/eval_ostf_v34_10_trace_contract_oracle_residual_probe_20260512.py"
MODEL_SRC = ROOT / "code/stwm/modules/ostf_v34_8_causal_assignment_bound_residual_memory.py"
TOKENIZER_SRC = ROOT / "code/stwm/modules/ostf_v34_2_dual_source_semantic_trace_units.py"
DECISION = ROOT / "reports/stwm_ostf_v34_10_decision_20260512.json"
TRAIN_SUMMARY = ROOT / "reports/stwm_ostf_v34_10_trace_contract_oracle_residual_probe_train_summary_20260512.json"
EVAL_SUMMARY = ROOT / "reports/stwm_ostf_v34_10_trace_contract_oracle_residual_probe_eval_summary_20260512.json"
EVAL_DECISION = ROOT / "reports/stwm_ostf_v34_10_trace_contract_oracle_residual_probe_decision_20260512.json"
BANK_DOC = ROOT / "docs/STWM_OSTF_V34_9_TRACE_PRESERVING_SEMANTIC_MEASUREMENT_BANK_20260512.md"
TARGET_DOC = ROOT / "docs/STWM_OSTF_V34_9_CAUSAL_ASSIGNMENT_RESIDUAL_TARGET_BUILD_20260512.md"
MEAS_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_11_v34_10_semantic_measurement_failure_audit_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_11_V34_10_SEMANTIC_MEASUREMENT_FAILURE_AUDIT_20260513.md"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def scalar(x: Any) -> Any:
    arr = np.asarray(x)
    return arr.item() if arr.shape == () else arr.reshape(-1)[0]


def sample_measurement_stats() -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    all_var: list[float] = []
    all_conf: list[np.ndarray] = []
    teacher_names: set[str] = set()
    dims: set[int] = set()
    for split in ("train", "val", "test"):
        files = sorted((MEAS_ROOT / split).glob("*.npz"))
        split_cov: list[float] = []
        split_var: list[float] = []
        split_nan = 0
        split_count = 0
        for path in files[: min(len(files), 64)]:
            z = np.load(path, allow_pickle=True)
            sem = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
            mask = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
            conf = np.asarray(z["obs_measurement_confidence"], dtype=np.float32)
            if "obs_measurement_teacher_name" in z.files:
                teacher_names.add(str(scalar(z["obs_measurement_teacher_name"])))
            dims.add(int(sem.shape[-1]))
            split_cov.append(float(mask.mean()))
            valid_sem = sem[mask]
            if valid_sem.size:
                split_var.append(float(np.nanmean(np.nanvar(valid_sem, axis=0))))
                all_var.append(split_var[-1])
            split_nan += int(np.isnan(sem).sum())
            split_count += int(sem.size)
            all_conf.append(conf.reshape(-1))
        by_split[split] = {
            "sample_count_checked": len(files[: min(len(files), 64)]),
            "coverage": None if not split_cov else float(np.mean(split_cov)),
            "mean_feature_variance": None if not split_var else float(np.mean(split_var)),
            "nan_ratio": None if split_count == 0 else float(split_nan / split_count),
        }
    conf_cat = np.concatenate(all_conf) if all_conf else np.asarray([], dtype=np.float32)
    conf_stats = {
        "min": None if conf_cat.size == 0 else float(conf_cat.min()),
        "max": None if conf_cat.size == 0 else float(conf_cat.max()),
        "mean": None if conf_cat.size == 0 else float(conf_cat.mean()),
        "std": None if conf_cat.size == 0 else float(conf_cat.std()),
        "all_one": bool(conf_cat.size > 0 and np.allclose(conf_cat, 1.0)),
    }
    return {
        "teacher_name": sorted(teacher_names)[0] if teacher_names else None,
        "embedding_dims": sorted(dims),
        "by_split": by_split,
        "confidence": conf_stats,
        "semantic_measurements_have_variance": bool(all_var and float(np.mean(all_var)) > 1e-8),
        "measurement_confidence_degenerate": bool(conf_stats["all_one"] or (conf_stats["std"] is not None and conf_stats["std"] < 1e-6)),
    }


def find_code_locations() -> dict[str, list[str]]:
    train = read_text(TRAIN_SRC)
    model = read_text(MODEL_SRC)
    tok = read_text(TOKENIZER_SRC)
    locations: dict[str, list[str]] = {
        "teacher_agreement_in_train": [],
        "mean_semantic_pooling": [],
        "global_usage_loss": [],
        "usage_scores_computed_not_applied": [],
    }
    for i, line in enumerate(train.splitlines(), 1):
        if "teacher_agreement" in line:
            locations["teacher_agreement_in_train"].append(f"{TRAIN_SRC.relative_to(ROOT)}:{i}: {line.strip()}")
        if "masked_cos(" in line or "sem_usage =" in line or "assign_contrast =" in line:
            locations["global_usage_loss"].append(f"{TRAIN_SRC.relative_to(ROOT)}:{i}: {line.strip()}")
    for i, line in enumerate(tok.splitlines(), 1):
        if "sem_pooled" in line and ("sum(dim=2)" in line or "sem_mask.sum" in line):
            locations["mean_semantic_pooling"].append(f"{TOKENIZER_SRC.relative_to(ROOT)}:{i}: {line.strip()}")
    for i, line in enumerate(model.splitlines(), 1):
        if "semantic_measurement_usage_score" in line or "assignment_usage_score" in line or "final_sem =" in line:
            locations["usage_scores_computed_not_applied"].append(f"{MODEL_SRC.relative_to(ROOT)}:{i}: {line.strip()}")
    return locations


def main() -> int:
    train = load_json(TRAIN_SUMMARY)
    eval_summary = load_json(EVAL_SUMMARY)
    eval_decision = load_json(EVAL_DECISION)
    decision = load_json(DECISION)
    stats = sample_measurement_stats()
    train_src = read_text(TRAIN_SRC)
    model_src = read_text(MODEL_SRC)
    tok_src = read_text(TOKENIZER_SRC)
    eval_dec = eval_summary.get("decision", eval_decision)
    zero_delta = eval_dec.get("zero_semantic_measurements_metric_delta", {})
    shuffle_delta = eval_dec.get("shuffle_semantic_measurements_metric_delta", {})
    assign_delta = eval_dec.get("shuffle_assignment_metric_delta", {})
    zero_bad = max(abs(float(zero_delta.get("val", 0.0) or 0.0)), abs(float(zero_delta.get("test", 0.0) or 0.0))) <= 0.003 or min(float(zero_delta.get("val", 0.0) or 0.0), float(zero_delta.get("test", 0.0) or 0.0)) <= 0.0
    shuf_bad = max(abs(float(shuffle_delta.get("val", 0.0) or 0.0)), abs(float(shuffle_delta.get("test", 0.0) or 0.0))) <= 0.003 or min(float(shuffle_delta.get("val", 0.0) or 0.0), float(shuffle_delta.get("test", 0.0) or 0.0)) <= 0.0
    assign_bad = max(abs(float(assign_delta.get("val", 0.0) or 0.0)), abs(float(assign_delta.get("test", 0.0) or 0.0))) <= 0.003 or min(float(assign_delta.get("val", 0.0) or 0.0), float(assign_delta.get("test", 0.0) or 0.0)) <= 0.0
    teacher_agreement_used = bool(re.search(r"teacher_agreement(_score)?", train_src))
    pooling_global = "sem_pooled" in tok_src and "sum(dim=2)" in tok_src and "sem_mask.sum(dim=2" in tok_src
    usage_global = "def masked_cos" in train_src and "normal_cos = masked_cos" in train_src
    final_line = re.search(r"final_sem\s*=\s*F\.normalize\((.+?)\)", model_src)
    score_not_used = "semantic_measurement_usage_score" in model_src and (not final_line or "semantic_measurement_usage_score" not in final_line.group(0))
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.10 已修复 trace contract 并激活 usage/assignment loss，但语义 measurement 仍不是因果必要路径；当前 tokenizer 的时间均值 pooling 和全局 scalar usage loss 是主要可疑点。",
        "semantic_measurement_bank_teacher_name": stats["teacher_name"],
        "semantic_measurement_stats": stats,
        "train_summary_loss": {
            "semantic_measurement_usage_loss_first": train.get("semantic_measurement_usage_loss_first"),
            "semantic_measurement_usage_loss_last": train.get("semantic_measurement_usage_loss_last"),
            "semantic_measurement_usage_loss_mean": train.get("semantic_measurement_usage_loss_mean"),
            "assignment_contrastive_loss_first": train.get("assignment_contrastive_loss_first"),
            "assignment_contrastive_loss_last": train.get("assignment_contrastive_loss_last"),
            "assignment_contrastive_loss_mean": train.get("assignment_contrastive_loss_mean"),
        },
        "intervention_deltas": {
            "zero_semantic_measurements_metric_delta": zero_delta,
            "shuffle_semantic_measurements_metric_delta": shuffle_delta,
            "shuffle_assignment_metric_delta": assign_delta,
        },
        "semantic_measurements_have_variance": stats["semantic_measurements_have_variance"],
        "measurement_confidence_degenerate": stats["measurement_confidence_degenerate"],
        "teacher_agreement_used_in_training": teacher_agreement_used,
        "semantic_pooling_too_global": bool(pooling_global),
        "usage_loss_too_global": bool(usage_global),
        "semantic_usage_score_not_used_in_residual": bool(score_not_used),
        "semantic_measurement_not_load_bearing_confirmed": bool(not decision.get("semantic_measurements_load_bearing_on_residual", False) and zero_bad and shuf_bad),
        "assignment_not_load_bearing_confirmed": bool(not decision.get("assignment_load_bearing_on_residual", False) and assign_bad),
        "exact_code_locations": find_code_locations(),
        "source_docs_checked": [str(BANK_DOC.relative_to(ROOT)), str(TARGET_DOC.relative_to(ROOT))],
        "recommended_fix": "先运行 semantic measurement quality probe；若 measurement 本身有 hard/changed 预测力，则把 usage loss 改为局部逐点逐 horizon，并把 usage score 接入 residual magnitude；否则重建 measurement bank。",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.11 V34.10 semantic measurement 因果失败审计中文报告",
        payload,
        [
            "中文结论",
            "semantic_measurement_bank_teacher_name",
            "semantic_measurements_have_variance",
            "measurement_confidence_degenerate",
            "teacher_agreement_used_in_training",
            "semantic_pooling_too_global",
            "usage_loss_too_global",
            "semantic_usage_score_not_used_in_residual",
            "semantic_measurement_not_load_bearing_confirmed",
            "assignment_not_load_bearing_confirmed",
            "recommended_fix",
        ],
    )
    print(f"已写出 V34.11 semantic measurement 失败审计: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
