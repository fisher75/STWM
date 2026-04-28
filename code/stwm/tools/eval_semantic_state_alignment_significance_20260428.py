#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import random


SUBSETS = ["long_gap_persistence", "occlusion_reappearance", "crossing_ambiguity", "OOD_hard", "appearance_change"]
METRICS = ["top1", "MRR", "AP", "AUROC"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM Semantic-State Alignment Significance",
        "",
        f"- version_tag: `{payload.get('version_tag')}`",
        f"- bootstrap_unit: `{payload.get('bootstrap_unit')}`",
        f"- split: `{payload.get('split')}`",
        f"- bootstrap_samples: `{payload.get('bootstrap_samples')}`",
    ]
    for name, comp in payload.get("comparisons", {}).items():
        ap = comp.get("metrics", {}).get("AP", {})
        lines.append(f"- {name} AP delta: `{ap.get('mean_delta')}` CI95=[`{ap.get('ci95_low')}`, `{ap.get('ci95_high')}`] zero_excluded=`{ap.get('zero_excluded')}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = (len(xs) - 1) * float(q)
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    frac = idx - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def binary_ap(scores: list[float], labels: list[int]) -> float | None:
    if not scores or 1 not in labels:
        return None
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    positives = 0
    precision_sum = 0.0
    total_pos = sum(1 for y in labels if int(y) == 1)
    for rank, idx in enumerate(order, 1):
        if int(labels[idx]) == 1:
            positives += 1
            precision_sum += positives / float(rank)
    return precision_sum / float(total_pos) if total_pos else None


def binary_auc(scores: list[float], labels: list[int]) -> float | None:
    n = len(scores)
    pos = sum(1 for y in labels if int(y) == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(n), key=lambda i: float(scores[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and float(scores[order[j]]) == float(scores[order[i]]):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    pos_rank_sum = sum(ranks[i] for i, y in enumerate(labels) if int(y) == 1)
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def item_metrics(items: list[dict[str, Any]], mode: str) -> tuple[list[float], list[float], list[float], list[int]]:
    top1: list[float] = []
    rr: list[float] = []
    flat_scores: list[float] = []
    flat_labels: list[int] = []
    for item in items:
        labels = [int(x) for x in item.get("labels", [])]
        scores = [float(x) for x in (item.get("scores", {}) or {}).get(mode, [])]
        if len(labels) != len(scores) or 1 not in labels or 0 not in labels:
            continue
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        gt = labels.index(1)
        top1.append(1.0 if labels[order[0]] == 1 else 0.0)
        rr.append(1.0 / float(order.index(gt) + 1))
        flat_scores.extend(scores)
        flat_labels.extend(labels)
    return top1, rr, flat_scores, flat_labels


def metrics_for_items(items: list[dict[str, Any]], mode: str) -> dict[str, float | None]:
    top1, rr, scores, labels = item_metrics(items, mode)
    return {
        "top1": mean(top1),
        "MRR": mean(rr),
        "AP": binary_ap(scores, labels),
        "AUROC": binary_auc(scores, labels),
        "item_count": len(top1),
        "candidate_record_count": len(labels),
    }


def diff_metrics(items: list[dict[str, Any]], a: str, b: str) -> dict[str, float | None]:
    ma = metrics_for_items(items, a)
    mb = metrics_for_items(items, b)
    out: dict[str, float | None] = {}
    for metric in METRICS:
        va = ma.get(metric)
        vb = mb.get(metric)
        out[metric] = None if va is None or vb is None else float(va) - float(vb)
    return out


def bootstrap_compare(items: list[dict[str, Any]], a: str, b: str, samples: int, seed: int) -> dict[str, Any]:
    rng = random.Random(int(seed))
    observed = diff_metrics(items, a, b)
    deltas: dict[str, list[float]] = {m: [] for m in METRICS}
    if not items:
        return {"observed_delta": observed, "metrics": {}}
    for _ in range(int(samples)):
        sample = [items[rng.randrange(len(items))] for _ in range(len(items))]
        diff = diff_metrics(sample, a, b)
        for metric in METRICS:
            value = diff.get(metric)
            if value is not None:
                deltas[metric].append(float(value))
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        xs = deltas[metric]
        lo = percentile(xs, 0.025)
        hi = percentile(xs, 0.975)
        obs = observed.get(metric)
        zero_excluded = bool(lo is not None and hi is not None and (lo > 0.0 or hi < 0.0))
        metrics[metric] = {
            "mean_delta": mean(xs),
            "observed_delta": obs,
            "ci95_low": lo,
            "ci95_high": hi,
            "zero_excluded": zero_excluded,
            "bootstrap_win_rate": (sum(1 for x in xs if x > 0.0) / len(xs)) if xs else None,
            "count": len(items),
            "p_value_permutation": None,
        }
    return {"mode_a": a, "mode_b": b, "observed_delta": observed, "metrics": metrics}


def _comparisons_for_version(version_tag: str) -> dict[str, tuple[str, str]]:
    if str(version_tag) == "v7":
        return {
            "posterior_v7_vs_posterior_v7_no_predicted_state": ("posterior_v7", "posterior_v7_no_predicted_state"),
            "posterior_v7_vs_distance_only": ("posterior_v7", "distance_only"),
            "posterior_v7_vs_target_candidate_appearance_frozen": ("posterior_v7", "target_candidate_appearance_frozen"),
            "aligned_predicted_semantic_identity_v7_vs_distance_only": ("aligned_predicted_semantic_identity_v7", "distance_only"),
        }
    return {
        "posterior_v6_vs_posterior_v6_no_predicted_state": ("posterior_v6", "posterior_v6_no_predicted_state"),
        "posterior_v6_vs_distance_only": ("posterior_v6", "distance_only"),
        "posterior_v6_vs_appearance_only": ("posterior_v6", "appearance_only"),
        "aligned_listwise_semantic_identity_vs_distance_only": ("aligned_listwise_semantic_identity", "distance_only"),
    }


def run(score_table: Path, output: Path, doc: Path, split: str, samples: int, seed: int, version_tag: str = "v6") -> dict[str, Any]:
    table = load_json(score_table)
    records = [r for r in table.get("records", []) if isinstance(r, dict)]
    if split != "all":
        records = [r for r in records if r.get("split") == split]
    comparisons = _comparisons_for_version(str(version_tag))
    comp_payload = {name: bootstrap_compare(records, a, b, samples, seed + idx * 17) for idx, (name, (a, b)) in enumerate(comparisons.items())}
    subset_payload: dict[str, Any] = {}
    for subset in SUBSETS:
        sub = [r for r in records if bool((r.get("subset_tags") or {}).get(subset))]
        subset_payload[subset] = {
            name: bootstrap_compare(sub, a, b, samples, seed + 1000 + idx * 17)
            for idx, (name, (a, b)) in enumerate(comparisons.items())
        }
    payload = {
        "generated_at_utc": now_iso(),
        "score_table": str(score_table),
        "version_tag": str(version_tag),
        "bootstrap_unit": "item",
        "split": split,
        "bootstrap_samples": int(samples),
        "item_count": len(records),
        "comparisons": comp_payload,
        "subset_comparisons": subset_payload,
        "v5_item_level_scores_available": False,
        "v5_signal_significant": "unclear",
        "v5_missing_item_level_reason": "Original V5 report was aggregate-only after raw export cleanup; significance is computed from the compact current score table.",
    }
    write_json(output, payload)
    write_doc(doc, payload)
    return payload


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--score-table", default="reports/stwm_semantic_state_alignment_v6_score_table_20260428.json")
    parser.add_argument("--output", default="reports/stwm_semantic_state_alignment_v6_significance_v1_20260428.json")
    parser.add_argument("--doc", default="docs/STWM_SEMANTIC_STATE_ALIGNMENT_V6_SIGNIFICANCE_V1_20260428.md")
    parser.add_argument("--split", default="heldout", choices=["heldout", "dev", "all"])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--version-tag", default="v6", choices=["v6", "v7"])
    args = parser.parse_args()
    run(
        Path(args.score_table),
        Path(args.output),
        Path(args.doc),
        str(args.split),
        int(args.bootstrap_samples),
        int(args.seed),
        str(args.version_tag),
    )


if __name__ == "__main__":
    main()
