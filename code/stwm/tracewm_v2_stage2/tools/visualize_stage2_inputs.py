#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json

from PIL import Image, ImageDraw

from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import Stage2SemanticDataset, Stage2SemanticDatasetConfig


def parse_args() -> Any:
    p = ArgumentParser(description="Visualize Stage2 semantic bootstrap inputs")
    p.add_argument(
        "--stage2-contract-path",
        default="/home/chen034/workspace/stwm/reports/stage2_bootstrap_data_contract_20260408.json",
    )
    p.add_argument("--dataset-name", default="")
    p.add_argument("--split", default="train")
    p.add_argument("--output-image", default="/home/chen034/workspace/stwm/outputs/visualizations/stage2_bootstrap_inputs_20260408.png")
    p.add_argument("--report-json", default="/home/chen034/workspace/stwm/reports/stage2_bootstrap_visualization_20260408.json")
    p.add_argument("--obs-len", type=int, default=8)
    p.add_argument("--fut-len", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=1)
    return p.parse_args()


def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"json payload must be dict: {p}")
    return payload


def _pick_dataset(contract: Dict[str, Any], explicit_name: str) -> str:
    if explicit_name.strip():
        return explicit_name.strip().lower()

    datasets = contract.get("datasets", [])
    if isinstance(datasets, list):
        for ds in datasets:
            if not isinstance(ds, dict):
                continue
            if bool(ds.get("used_in_bootstrap_train", False)) and not bool(ds.get("not_in_current_bootstrap", False)):
                return str(ds.get("dataset_name", "")).strip().lower()
    return "vspw"


def _build_dataset(contract_path: str, dataset_name: str, split: str, args: Any) -> Stage2SemanticDataset:
    return Stage2SemanticDataset(
        Stage2SemanticDatasetConfig(
            dataset_names=[dataset_name],
            split=split,
            contract_path=str(contract_path),
            obs_len=int(args.obs_len),
            fut_len=int(args.fut_len),
            max_tokens=int(args.max_tokens),
            max_samples_per_dataset=max(int(args.max_samples), 1),
            semantic_patch_radius=12,
            semantic_frame_index=0,
        )
    )


def main() -> None:
    args = parse_args()
    contract = _read_json(args.stage2_contract_path)

    dataset_name = _pick_dataset(contract=contract, explicit_name=str(args.dataset_name))

    try:
        ds = _build_dataset(
            contract_path=str(args.stage2_contract_path),
            dataset_name=dataset_name,
            split=str(args.split),
            args=args,
        )
    except Exception:
        fallback_split = "val" if str(args.split).lower() == "train" else "train"
        ds = _build_dataset(
            contract_path=str(args.stage2_contract_path),
            dataset_name=dataset_name,
            split=fallback_split,
            args=args,
        )

    sample = ds[0]
    frame_path = str(sample.get("semantic_frame_path", ""))
    img = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    boxes = sample["semantic_boxes"].detach().cpu().numpy()
    mask = sample["semantic_mask"].detach().cpu().numpy().astype(bool)

    drawn = 0
    for i in range(min(len(boxes), 24)):
        if not mask[i]:
            continue
        x0, y0, x1, y1 = [int(float(v)) for v in boxes[i]]
        draw.rectangle((x0, y0, x1, y1), outline=(255, 128, 32), width=2)
        drawn += 1

    out_img = Path(args.output_image)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_img)

    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": str(sample.get("meta", {}).get("dataset", "")),
        "clip_id": str(sample.get("meta", {}).get("clip_id", "")),
        "frame_path": frame_path,
        "output_image": str(out_img),
        "token_boxes_drawn": int(drawn),
        "semantic_source_mode": str(sample.get("semantic_source_mode", "")),
        "semantic_source_summary": dict(sample.get("semantic_source_summary", {})),
    }

    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(f"[stage2-visualize] output_image={out_img}")
    print(f"[stage2-visualize] report_json={report_json}")


if __name__ == "__main__":
    main()
