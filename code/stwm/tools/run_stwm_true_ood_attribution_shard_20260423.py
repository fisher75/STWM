from __future__ import annotations

import gc
import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

try:
    import setproctitle  # type: ignore
except Exception:
    setproctitle = None

if setproctitle is not None:
    try:
        setproctitle.setproctitle("python")
    except Exception:
        pass

import torch

from stwm.tools import run_stage2_state_identifiability_eval_20260415 as evalcore
from stwm.tools import run_stwm_tusb_light_readout_eval_20260422 as lighteval
from stwm.tools import run_stwm_true_ood_attribution_20260423 as full
from stwm.tools import run_stwm_true_ood_eval_20260420 as oodcore


ROOT = Path("/raid/chen034/workspace/stwm")
REPORTS = ROOT / "reports"

GROUP_TO_METHOD = {
    "tusb": full.OFFICIAL_TUSB,
    "calibration": full.CAL,
    "cropenc": full.CROP,
    "legacysem": full.LEGACY,
}


def _parse_seed_list(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def build_shard(args: Any) -> Dict[str, Any]:
    split_json = str(args.split_materialization_json or (REPORTS / f"tmp_true_ood_split_materialization_{args.group}_{args.tag}.json"))
    split_md = str(args.split_materialization_md or (REPORTS / f"tmp_true_ood_split_materialization_{args.group}_{args.tag}.md"))
    split_args = SimpleNamespace(
        dense_protocol_json=str(args.dense_protocol_json),
        extended_protocol_json=str(args.extended_protocol_json),
        split_audit_json=split_json,
        split_audit_md=split_md,
    )
    materialization, panel_item_ids, extended_lookup = full._prepare_split_materialization(split_args)
    selected_ids = set().union(*(panel_item_ids[name] for name in full.OOD_SPLITS))
    prepared_items, skipped_reasons = oodcore._prepare_selected_items(extended_lookup, selected_ids)
    checkpoint_map = full._checkpoint_map(args)
    official_weights = full._official_weights()

    raw_rows: List[Dict[str, Any]] = []
    eval_started_at = full._now_iso()
    wall_start = time.time()
    device, device_info = evalcore._select_eval_device(args)
    print(f"[{full._now_iso()}] shard_device_ready group={args.group} mode={device_info.get('mode', '')} device={device}", flush=True)
    try:
        if args.group == "tusb":
            method_name = full.OFFICIAL_TUSB
            for seed in _parse_seed_list(args.seed_list):
                entry = checkpoint_map[method_name][int(seed)]
                print(f"[{full._now_iso()}] shard_eval_start group=tusb method={method_name} seed={seed}", flush=True)
                spec = evalcore.MethodSpec(
                    name=method_name,
                    run_name=str(entry["run_name"]),
                    method_type="stage2",
                    checkpoint_path=str(entry["checkpoint_path"]),
                )
                method = evalcore._load_method(spec, device=device)
                try:
                    total_items = len(prepared_items)
                    for index, protocol_item_id in enumerate(sorted(prepared_items), start=1):
                        prepared = prepared_items[protocol_item_id]
                        item = prepared["item"]
                        payload = evalcore._evaluate_tusb_light_readout_payload(
                            method=method,
                            item=item,
                            batch=prepared["batch"],
                            target_future_mask=prepared["target_future_mask"],
                            future_masks=prepared["future_masks"],
                            candidate_inputs=prepared["candidate_inputs"],
                            device=device,
                        )
                        coord_result = dict(payload.get("coord_result", {}))
                        coord_scores = dict(payload.get("coord_scores", {}))
                        unit_scores = dict(payload.get("unit_identity_scores", {}))
                        semantic_scores = dict(payload.get("semantic_teacher_scores", {}))
                        subset_tags = list(item.get("subset_tags", []))
                        dataset = str(item.get("dataset", ""))
                        clip_id = str(item.get("clip_id", ""))
                        ctx_count = int(prepared.get("protocol_eval_context_entity_count", 0))
                        raw_rows.append(full._row(protocol_item_id, seed, method_name, "coord_only", subset_tags, dataset, clip_id, ctx_count, coord_result))
                        raw_rows.append(
                            full._row(
                                protocol_item_id,
                                seed,
                                method_name,
                                "unit_identity_only",
                                subset_tags,
                                dataset,
                                clip_id,
                                ctx_count,
                                lighteval._compose_score_result(
                                    base_result=coord_result,
                                    score_map=unit_scores,
                                    target_id=str(item.get("target_id", "")),
                                    target_future_mask=prepared["target_future_mask"],
                                    future_masks=prepared["future_masks"],
                                    scoring_mode="unit_identity_only",
                                    unit_scores=unit_scores,
                                    semantic_scores=semantic_scores,
                                ),
                            )
                        )
                        raw_rows.append(
                            full._row(
                                protocol_item_id,
                                seed,
                                method_name,
                                "semantic_teacher_only",
                                subset_tags,
                                dataset,
                                clip_id,
                                ctx_count,
                                lighteval._compose_score_result(
                                    base_result=coord_result,
                                    score_map=semantic_scores,
                                    target_id=str(item.get("target_id", "")),
                                    target_future_mask=prepared["target_future_mask"],
                                    future_masks=prepared["future_masks"],
                                    scoring_mode="semantic_teacher_only",
                                    unit_scores=unit_scores,
                                    semantic_scores=semantic_scores,
                                ),
                            )
                        )
                        coord_plus_teacher = evalcore._build_hybrid_scores(
                            coord_scores=coord_scores,
                            unit_scores={},
                            semantic_scores=semantic_scores,
                            alpha=float(official_weights["alpha"]),
                            beta=0.0,
                            gamma=float(official_weights["gamma"]),
                        )
                        raw_rows.append(
                            full._row(
                                protocol_item_id,
                                seed,
                                method_name,
                                "coord_plus_teacher",
                                subset_tags,
                                dataset,
                                clip_id,
                                ctx_count,
                                lighteval._compose_score_result(
                                    base_result=coord_result,
                                    score_map=coord_plus_teacher,
                                    target_id=str(item.get("target_id", "")),
                                    target_future_mask=prepared["target_future_mask"],
                                    future_masks=prepared["future_masks"],
                                    scoring_mode="coord_plus_teacher",
                                    selected_weights={"alpha": float(official_weights["alpha"]), "beta": 0.0, "gamma": float(official_weights["gamma"])},
                                    unit_scores={},
                                    semantic_scores=semantic_scores,
                                ),
                            )
                        )
                        coord_plus_unit = evalcore._build_hybrid_scores(
                            coord_scores=coord_scores,
                            unit_scores=unit_scores,
                            semantic_scores={},
                            alpha=float(official_weights["alpha"]),
                            beta=float(official_weights["beta"]),
                            gamma=0.0,
                        )
                        raw_rows.append(
                            full._row(
                                protocol_item_id,
                                seed,
                                method_name,
                                "coord_plus_unit",
                                subset_tags,
                                dataset,
                                clip_id,
                                ctx_count,
                                lighteval._compose_score_result(
                                    base_result=coord_result,
                                    score_map=coord_plus_unit,
                                    target_id=str(item.get("target_id", "")),
                                    target_future_mask=prepared["target_future_mask"],
                                    future_masks=prepared["future_masks"],
                                    scoring_mode="coord_plus_unit",
                                    selected_weights={"alpha": float(official_weights["alpha"]), "beta": float(official_weights["beta"]), "gamma": 0.0},
                                    unit_scores=unit_scores,
                                    semantic_scores={},
                                ),
                            )
                        )
                        hybrid_scores = evalcore._build_hybrid_scores(
                            coord_scores=coord_scores,
                            unit_scores=unit_scores,
                            semantic_scores=semantic_scores,
                            alpha=float(official_weights["alpha"]),
                            beta=float(official_weights["beta"]),
                            gamma=float(official_weights["gamma"]),
                        )
                        raw_rows.append(
                            full._row(
                                protocol_item_id,
                                seed,
                                method_name,
                                "hybrid_light",
                                subset_tags,
                                dataset,
                                clip_id,
                                ctx_count,
                                lighteval._compose_score_result(
                                    base_result=coord_result,
                                    score_map=hybrid_scores,
                                    target_id=str(item.get("target_id", "")),
                                    target_future_mask=prepared["target_future_mask"],
                                    future_masks=prepared["future_masks"],
                                    scoring_mode="hybrid_light",
                                    selected_weights=official_weights,
                                    unit_scores=unit_scores,
                                    semantic_scores=semantic_scores,
                                ),
                            )
                        )
                        if index % 50 == 0 or index == total_items:
                            print(f"[{full._now_iso()}] shard_eval_progress group=tusb seed={seed} items={index}/{total_items}", flush=True)
                finally:
                    evalcore._release_method(method)
                print(f"[{full._now_iso()}] shard_eval_done group=tusb seed={seed}", flush=True)
        else:
            method_name = GROUP_TO_METHOD[str(args.group)]
            for seed in _parse_seed_list(args.seed_list):
                entry = checkpoint_map[method_name][int(seed)]
                print(f"[{full._now_iso()}] shard_eval_start group={args.group} method={method_name} seed={seed}", flush=True)
                spec = evalcore.MethodSpec(
                    name=method_name,
                    run_name=str(entry["run_name"]),
                    method_type="stage2",
                    checkpoint_path=str(entry["checkpoint_path"]),
                )
                method = evalcore._load_method(spec, device=device)
                try:
                    total_items = len(prepared_items)
                    for index, protocol_item_id in enumerate(sorted(prepared_items), start=1):
                        prepared = prepared_items[protocol_item_id]
                        item = prepared["item"]
                        result = evalcore._evaluate_item(
                            method=method,
                            item=item,
                            batch=prepared["batch"],
                            target_future_mask=prepared["target_future_mask"],
                            future_masks=prepared["future_masks"],
                            device=device,
                            scoring_mode="coord_only",
                            candidate_inputs=prepared["candidate_inputs"],
                        )
                        raw_rows.append(
                            full._row(
                                protocol_item_id,
                                seed,
                                method_name,
                                "coord_only",
                                list(item.get("subset_tags", [])),
                                str(item.get("dataset", "")),
                                str(item.get("clip_id", "")),
                                int(prepared.get("protocol_eval_context_entity_count", 0)),
                                result,
                            )
                        )
                        if index % 50 == 0 or index == total_items:
                            print(f"[{full._now_iso()}] shard_eval_progress group={args.group} seed={seed} items={index}/{total_items}", flush=True)
                finally:
                    evalcore._release_method(method)
                print(f"[{full._now_iso()}] shard_eval_done group={args.group} seed={seed}", flush=True)
    finally:
        lease_id = str(device_info.get("lease_id", "")).strip()
        if lease_id:
            try:
                evalcore.release_lease(lease_id=lease_id, lease_path=str(args.lease_path))
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "generated_at_utc": full._now_iso(),
        "group": str(args.group),
        "seed_list": _parse_seed_list(args.seed_list),
        "eval_started_at": eval_started_at,
        "eval_finished_at": full._now_iso(),
        "wall_time_seconds": float(time.time() - wall_start),
        "official_weights": official_weights,
        "split_item_ids": {name: sorted(panel_item_ids[name]) for name in full.OOD_SPLITS},
        "split_meta": {
            "heldout_burst_heavy_context_preserving": dict(materialization.get("split_a_vipseg_history_to_burst_heldout", {})),
            "heldout_scene_category_video_context_preserving": dict(materialization.get("split_b_scene_category_video_heldout", {})),
        },
        "prepared_item_meta": {
            item_id: {"protocol_eval_context_entity_count": int(prepared["protocol_eval_context_entity_count"])}
            for item_id, prepared in prepared_items.items()
        },
        "skipped_reasons": dict(skipped_reasons),
        "per_item_results_hash": full._sha256_json(raw_rows),
        "raw_rows": raw_rows,
    }
    _write_json(Path(args.output_json), payload)
    return payload


def main() -> None:
    parser = ArgumentParser(description="Run a shard of STWM true-OOD attribution 20260423.")
    parser.add_argument("--group", required=True, choices=["tusb", "calibration", "cropenc", "legacysem"])
    parser.add_argument("--seed-list", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--tag", default="shard")
    parser.add_argument("--dense-protocol-json", default=str(REPORTS / "stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--extended-protocol-json", default=str(REPORTS / "stage2_protocol_v3_extended_evalset_20260420.json"))
    parser.add_argument("--main-checkpoint-audit", default=str(REPORTS / "stwm_postfix_matched6seed_checkpoint_audit_20260421.json"))
    parser.add_argument("--sidecar-checkpoint-audit", default=str(REPORTS / "stwm_sidecar_checkpoint_audit_20260422.json"))
    parser.add_argument("--split-materialization-json", default="")
    parser.add_argument("--split-materialization-md", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=40.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=8.0)
    args = parser.parse_args()
    build_shard(args)


if __name__ == "__main__":
    main()
