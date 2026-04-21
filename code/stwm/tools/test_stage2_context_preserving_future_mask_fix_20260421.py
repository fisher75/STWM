#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List
import json
import os

from stwm.tools import run_stage2_state_identifiability_eval_20260415 as prev
from stwm.tools import run_stage2_state_identifiability_eval_v3_20260416 as eval_v3


ROOT = Path("/raid/chen034/workspace/stwm")


def _apply_process_title_normalization(default_title: str = "python") -> None:
    mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).strip().lower()
    if mode != "generic":
        return
    title = str(os.environ.get("STWM_PROC_TITLE", default_title)).strip() or default_title
    lowered = title.lower()
    if "stwm" in lowered or "tracewm" in lowered or "/home/" in lowered or "/raid/" in lowered:
        title = default_title
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        pass


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> Any:
    parser = ArgumentParser(description="Smoke-test the context-preserving future-mask fix on 5 previously skipped items")
    parser.add_argument("--protocol-json", default=str(ROOT / "reports/stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--audit-json", default=str(ROOT / "reports/stage2_v3p1_dualpanel_context_audit_20260420.json"))
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--output-json", default=str(ROOT / "reports/stage2_context_preserving_future_mask_fix_smoketest_20260421.json"))
    return parser.parse_args()


def _select_mixed_samples(skipped_rows: List[Dict[str, Any]], sample_count: int) -> List[Dict[str, Any]]:
    if sample_count <= 0:
        return []
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for row in skipped_rows:
        key = str(row.get("dataset", "")).strip().upper()
        by_dataset.setdefault(key, []).append(row)
    preferred = ["BURST", "VIPSEG"]
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()

    # First pass: ensure we cover both BURST and VIPSeg when available.
    for dataset in preferred:
        if len(selected) >= sample_count:
            break
        rows = by_dataset.get(dataset, [])
        if rows:
            row = rows.pop(0)
            pid = str(row.get("protocol_item_id", ""))
            if pid not in selected_ids:
                selected.append(row)
                selected_ids.add(pid)

    # Second pass: round-robin across preferred datasets.
    while len(selected) < sample_count:
        progress = False
        for dataset in preferred:
            if len(selected) >= sample_count:
                break
            rows = by_dataset.get(dataset, [])
            while rows:
                row = rows.pop(0)
                pid = str(row.get("protocol_item_id", ""))
                if pid in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(pid)
                progress = True
                break
        if not progress:
            break

    # Final fallback: anything left.
    if len(selected) < sample_count:
        for row in skipped_rows:
            if len(selected) >= sample_count:
                break
            pid = str(row.get("protocol_item_id", ""))
            if pid in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(pid)
    return selected


def main() -> None:
    _apply_process_title_normalization()
    args = parse_args()
    protocol = _read_json(Path(args.protocol_json))
    audit = _read_json(Path(args.audit_json))
    protocol_items = {str(item.get("protocol_item_id", "")): item for item in protocol.get("items", []) if isinstance(item, dict)}
    skipped_rows = list(((audit.get("densified_200_context_preserving") or {}).get("skipped_protocol_items")) or [])
    sample_rows = _select_mixed_samples(skipped_rows, max(int(args.sample_count), 1))

    results: List[Dict[str, Any]] = []
    failures: List[str] = []
    for row in sample_rows:
        protocol_item_id = str(row.get("protocol_item_id", ""))
        item = protocol_items.get(protocol_item_id)
        if item is None:
            failures.append(f"missing_protocol_item:{protocol_item_id}")
            continue
        target_id = str(item.get("target_id", ""))
        context_entity_ids = prev._protocol_observed_context_candidate_ids(item, max_context_entities=8)
        if not context_entity_ids or context_entity_ids[0] != target_id:
            context_entity_ids = [target_id] + [str(x) for x in context_entity_ids if str(x) != target_id]

        # Prove the target itself is readable.
        _, _, target_future_masks, target_future_mask = prev._extract_entity_masks(
            item,
            entity_id=target_id,
            require_future_mask=True,
        )
        if target_future_mask is None:
            failures.append(f"target_future_mask_missing_after_fix:{protocol_item_id}")
            continue

        # Reproduce the old failure mode: at least one context entity would fail if
        # we incorrectly demanded a future-step mask for every context entity.
        old_failure_entities: List[str] = []
        for entity_id in context_entity_ids[1:]:
            try:
                prev._extract_entity_masks(item, entity_id=entity_id, require_future_mask=True)
            except RuntimeError:
                old_failure_entities.append(str(entity_id))

        if not old_failure_entities:
            failures.append(f"no_old_failure_context_entity_found:{protocol_item_id}")
            continue

        batch, fixed_target_future_mask, future_masks = eval_v3._build_context_preserving_item_batch_v3(
            item,
            temporal_window=5,
            max_context_entities=8,
        )

        obs_state_shape = tuple(int(x) for x in batch["obs_state"].shape)
        if fixed_target_future_mask is None:
            failures.append(f"fixed_builder_missing_target_future_mask:{protocol_item_id}")
            continue
        if obs_state_shape[2] < 2:
            failures.append(f"context_entity_count_too_small:{protocol_item_id}:{obs_state_shape}")
            continue

        results.append(
            {
                "protocol_item_id": protocol_item_id,
                "dataset": str(item.get("dataset", "")),
                "target_id": target_id,
                "context_entity_ids": [str(x) for x in context_entity_ids],
                "old_failure_context_entity_ids": old_failure_entities,
                "target_future_mask_shape": [int(x) for x in fixed_target_future_mask.shape],
                "future_candidate_count": int(len(future_masks)),
                "obs_state_shape": [int(x) for x in obs_state_shape],
                "protocol_eval_context_entity_count": int(batch["meta"][0].get("protocol_eval_context_entity_count", 0)),
            }
        )

    payload = {
        "generated_at_utc": prev.now_iso(),
        "sample_count_requested": int(args.sample_count),
        "sample_count_executed": int(len(sample_rows)),
        "all_passed": bool(not failures and len(results) == len(sample_rows)),
        "failures": failures,
        "results": results,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    if failures:
        raise SystemExit(1)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
