#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List
import json

try:
    import setproctitle  # type: ignore
except Exception:  # pragma: no cover
    setproctitle = None

if setproctitle is not None:
    try:
        setproctitle.setproctitle("python")
    except Exception:
        pass

ROOT = Path("/raid/chen034/workspace/stwm")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}", ""]
    body.extend(list(lines))
    path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")


def _rows_from_final_eval(path: Path, panel_name: str) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    panels = payload.get("panels", {}) if isinstance(payload.get("panels", {}), dict) else {}
    panel = panels.get(panel_name, {}) if isinstance(panels.get(panel_name, {}), dict) else {}
    rows = panel.get("per_item_results", []) if isinstance(panel.get("per_item_results", []), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(len(vals), 1))


def _aggregate(rows: List[Dict[str, Any]], method_name: str) -> Dict[str, float]:
    scored = [row for row in rows if str(row.get("method_name")) == method_name]
    if not scored:
        return {
            "count": 0.0,
            "top1": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "loc_error": 0.0,
            "mask_iou_at_top1": 0.0,
        }
    return {
        "count": float(len(scored)),
        "top1": _mean(row.get("query_future_top1_acc", 0.0) for row in scored),
        "hit_rate": _mean(row.get("query_future_hit_rate", 0.0) for row in scored),
        "mrr": _mean(row.get("mrr", 0.0) for row in scored),
        "loc_error": _mean(row.get("query_future_localization_error", 0.0) for row in scored),
        "mask_iou_at_top1": _mean(row.get("future_mask_iou_at_top1", 0.0) for row in scored),
    }


def build_lightreadout_ood_eval(
    *,
    report_path: Path,
    doc_path: Path,
    final_eval_report: Path,
    panel_name: str = "protocol_v3_extended_600_context_preserving",
    official_tusb_method: str = "TUSB-v3.1::official(best_semantic_hard.pt+hybrid_light)",
    calibration_method: str = "calibration-only::best.pt",
    cropenc_method: str = "cropenc::best.pt",
    legacysem_method: str = "legacysem::best.pt",
) -> Dict[str, Any]:
    rows = _rows_from_final_eval(final_eval_report, panel_name)
    def _dataset(row: Dict[str, Any]) -> str:
        dataset = str(row.get("dataset", "")).strip()
        if dataset:
            return dataset
        protocol_item_id = str(row.get("protocol_item_id", "")).strip().lower()
        if protocol_item_id.startswith("burst::"):
            return "BURST"
        if protocol_item_id.startswith("vipseg::"):
            return "VIPSeg"
        return ""

    burst = [row for row in rows if _dataset(row) == "BURST"]
    vipseg = [row for row in rows if _dataset(row) == "VIPSeg"]

    burst_tusb = _aggregate(burst, official_tusb_method)
    burst_cal = _aggregate(burst, calibration_method)
    burst_crop = _aggregate(burst, cropenc_method)
    burst_legacy = _aggregate(burst, legacysem_method)
    vip_tusb = _aggregate(vipseg, official_tusb_method)
    vip_cal = _aggregate(vipseg, calibration_method)
    vip_crop = _aggregate(vipseg, cropenc_method)
    vip_legacy = _aggregate(vipseg, legacysem_method)

    def _beats(left_a: Dict[str, float], left_b: Dict[str, float], right_a: Dict[str, float], right_b: Dict[str, float]) -> bool:
        return bool(float(left_a["top1"]) > float(right_a["top1"]) and float(left_b["top1"]) > float(right_b["top1"]))

    payload = {
        "generated_at_utc": _now_iso(),
        "source_final_eval_report": str(final_eval_report),
        "panel_name": panel_name,
        "setting_a_vipseg_to_burst_heavy": {
            "official_tusb": burst_tusb,
            "calibration_only": burst_cal,
            "cropenc_baseline": burst_crop,
            "legacysem": burst_legacy,
        },
        "setting_b_burst_to_vipseg_heavy": {
            "official_tusb": vip_tusb,
            "calibration_only": vip_cal,
            "cropenc_baseline": vip_crop,
            "legacysem": vip_legacy,
        },
        "setting_c_conservative_heldout_split": {
            "supported": False,
            "reason": "current live repo still has no materialized true held-out scene/category/video split in this runner",
        },
        "ood_improved_vs_calibration": _beats(burst_tusb, vip_tusb, burst_cal, vip_cal),
        "ood_improved_vs_cropenc": _beats(burst_tusb, vip_tusb, burst_crop, vip_crop),
        "ood_improved_vs_legacysem": _beats(burst_tusb, vip_tusb, burst_legacy, vip_legacy),
        "ood_claim_ready": False,
        "proxy_only_vs_true_ood_boundary": "current rerun only refreshes the existing proxy domain split under official light readout; true held-out OOD is still unsupported here",
    }
    write_json(report_path, payload)
    write_md(
        doc_path,
        "STWM Light Readout OOD Eval 20260422",
        [
            f"- panel_name: {panel_name}",
            f"- ood_improved_vs_calibration: {payload['ood_improved_vs_calibration']}",
            f"- ood_improved_vs_cropenc: {payload['ood_improved_vs_cropenc']}",
            f"- ood_improved_vs_legacysem: {payload['ood_improved_vs_legacysem']}",
            f"- ood_claim_ready: {payload['ood_claim_ready']}",
            f"- proxy_only_vs_true_ood_boundary: {payload['proxy_only_vs_true_ood_boundary']}",
        ],
    )
    return payload


def main() -> None:
    parser = ArgumentParser(description="Run STWM OOD evaluation assets.")
    parser.add_argument("--final-eval-report", default="")
    parser.add_argument("--output-report", default=str(REPORTS / "stwm_true_ood_eval_20260420.json"))
    parser.add_argument("--output-doc", default=str(DOCS / "STWM_TRUE_OOD_EVAL_20260420.md"))
    parser.add_argument("--panel-name", default="protocol_v3_extended_600_context_preserving")
    parser.add_argument("--official-tusb-method", default="TUSB-v3.1::official(best_semantic_hard.pt+hybrid_light)")
    parser.add_argument("--calibration-method", default="calibration-only::best.pt")
    parser.add_argument("--cropenc-method", default="cropenc::best.pt")
    parser.add_argument("--legacysem-method", default="legacysem::best.pt")
    args = parser.parse_args()
    if str(args.final_eval_report).strip():
        build_lightreadout_ood_eval(
            report_path=Path(args.output_report),
            doc_path=Path(args.output_doc),
            final_eval_report=Path(args.final_eval_report),
            panel_name=str(args.panel_name),
            official_tusb_method=str(args.official_tusb_method),
            calibration_method=str(args.calibration_method),
            cropenc_method=str(args.cropenc_method),
            legacysem_method=str(args.legacysem_method),
        )
        return

    from run_stwm_decisive_validation_20260420 import (  # noqa: E402
        build_true_ood_eval_assets,
    )

    payload, md = build_true_ood_eval_assets()
    write_json(Path(args.output_report), payload)
    write_md(Path(args.output_doc), "STWM True OOD Eval 20260420", md)


if __name__ == "__main__":
    main()
