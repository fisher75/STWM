from __future__ import annotations

import contextlib
import io
import json
import math
import os
import shutil
import sys
import time
import traceback
import urllib.request
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

try:
    import torch
except Exception:  # pragma: no cover - reported in source audit if it happens
    torch = None

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stwm.tools.external_baselines.common_io import (  # noqa: E402
    BASELINES,
    CHECKPOINTS,
    DOCS,
    OUTPUTS,
    REPORTS,
    REPOS,
    ROOT,
    load_json,
    sha256_json,
    write_json,
    write_markdown,
)


MANIFEST = REPORTS / "stwm_external_baseline_item_manifest_20260426.json"
VISUAL_SANITY = REPORTS / "stwm_external_baseline_manifest_visual_sanity_20260426.json"
PAYLOAD_DECISION = REPORTS / "stwm_external_baseline_payload_readiness_decision_20260426.json"
ENV_AUDIT = REPORTS / "stwm_external_baseline_env_audit_20260426.json"
CLONE_AUDIT = REPORTS / "stwm_external_baseline_clone_audit_20260426.json"

MAX_ITEMS = int(os.environ.get("STWM_EXTERNAL_SMOKE_MAX_ITEMS", "20"))
MAX_SIDE = int(os.environ.get("STWM_EXTERNAL_SMOKE_MAX_SIDE", "512"))
MIN_SUCCESS_FOR_PASS = 5
COTRACKER_CKPT_URL = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
OUTPUT_PHASE = os.environ.get("STWM_EXTERNAL_OUTPUT_PHASE", "smoke")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))[:180]


def read_json_report(path: Path) -> tuple[bool, dict[str, Any] | None, str | None]:
    if not path.exists():
        return False, None, f"missing:{path}"
    try:
        return True, load_json(path), None
    except Exception as exc:
        return True, None, f"invalid_json:{type(exc).__name__}:{exc}"


def bool_field(data: dict[str, Any] | None, key: str) -> bool:
    return bool((data or {}).get(key))


def write_simple_md(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = ["```json", json.dumps(payload, indent=2, sort_keys=True), "```"]
    write_markdown(path, title, lines)


def build_source_audit() -> dict[str, Any]:
    manifest_exists, manifest, manifest_error = read_json_report(MANIFEST)
    visual_exists, visual, visual_error = read_json_report(VISUAL_SANITY)
    payload_exists, payload, payload_error = read_json_report(PAYLOAD_DECISION)
    env_exists, env, env_error = read_json_report(ENV_AUDIT)
    clone_exists, clone, clone_error = read_json_report(CLONE_AUDIT)

    items = (manifest or {}).get("items") or []
    audit = {
        "created_at": now(),
        "manifest_exists": manifest_exists,
        "manifest_valid_json": manifest_exists and manifest_error is None,
        "manifest_error": manifest_error,
        "materialized_items": (manifest or {}).get("materialized_items", len(items)),
        "cutie_ready_items": (manifest or {}).get("cutie_ready_items", 0),
        "sam2_ready_items": (manifest or {}).get("sam2_ready_items", 0),
        "cotracker_ready_items": (manifest or {}).get("cotracker_ready_items", 0),
        "visual_sanity_exists": visual_exists,
        "visual_sanity_valid_json": visual_exists and visual_error is None,
        "visual_sanity_passed": bool_field(visual, "visual_sanity_passed"),
        "payload_readiness_exists": payload_exists,
        "payload_readiness_valid_json": payload_exists and payload_error is None,
        "payload_can_enter_smoke": bool_field(payload, "can_enter_external_baseline_smoke"),
        "clone_audit_exists": clone_exists,
        "clone_audit_valid_json": clone_exists and clone_error is None,
        "env_audit_exists": env_exists,
        "env_audit_valid_json": env_exists and env_error is None,
        "cutie_import_ok": bool_field(env, "cutie_import_ok"),
        "sam2_import_ok": bool_field(env, "sam2_import_ok"),
        "cotracker_import_ok": bool_field(env, "cotracker_import_ok"),
        "cutie_checkpoint_ready": bool_field(env, "cutie_checkpoint_ready"),
        "sam2_checkpoint_ready": bool_field(env, "sam2_checkpoint_ready"),
        "cotracker_checkpoint_ready": bool_field(env, "cotracker_checkpoint_ready"),
        "errors": {
            "visual_sanity": visual_error,
            "payload_readiness": payload_error,
            "env_audit": env_error,
            "clone_audit": clone_error,
        },
        "audit_passed": bool(
            manifest_exists
            and manifest_error is None
            and visual_exists
            and visual_error is None
            and bool_field(visual, "visual_sanity_passed")
        ),
    }
    write_json(REPORTS / "stwm_external_baseline_smoke_source_audit_20260426.json", audit)
    write_simple_md(DOCS / "STWM_EXTERNAL_BASELINE_SMOKE_SOURCE_AUDIT_20260426.md", "STWM External Baseline Smoke Source Audit 20260426", audit)
    return audit


def subset_flags(item: dict[str, Any]) -> dict[str, bool]:
    tags = item.get("subset_tags") or {}
    if isinstance(tags, list):
        tags = {k: True for k in tags}
    source_protocol = " ".join(item.get("source_protocol") or [])
    return {
        "occlusion_reappearance": bool(tags.get("occlusion_reappearance")),
        "long_gap_persistence": bool(tags.get("long_gap_persistence")),
        "crossing_ambiguity": bool(tags.get("crossing_ambiguity") or tags.get("ambiguity")),
        "OOD_hard": bool(tags.get("OOD_hard") or "heldout" in source_protocol.lower()),
        "appearance_change": bool(tags.get("appearance_change")),
    }


def item_payload_ready(item: dict[str, Any]) -> bool:
    frame_paths = item.get("frame_paths") or []
    observed = item.get("observed_target") or {}
    future_candidates = item.get("future_candidates") or []
    future_idx = item.get("future_frame_index")
    if not frame_paths or future_idx is None or future_idx >= len(frame_paths):
        return False
    if not all(Path(p).exists() for p in frame_paths):
        return False
    if not (observed.get("mask_rle") or observed.get("mask_path") or observed.get("bbox") or observed.get("point_prompt")):
        return False
    if not future_candidates or not item.get("gt_candidate_id"):
        return False
    return all(c.get("mask_rle") or c.get("mask_path") or c.get("bbox") for c in future_candidates)


def select_smoke_items(items: list[dict[str, Any]], max_items: int = MAX_ITEMS) -> dict[str, Any]:
    runnable = [x for x in items if item_payload_ready(x)]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_item(item: dict[str, Any]) -> None:
        if len(selected) >= max_items:
            return
        item_id = item.get("item_id")
        if item_id not in selected_ids:
            selected.append(item)
            selected_ids.add(item_id)

    for subset in ["occlusion_reappearance", "long_gap_persistence", "crossing_ambiguity", "OOD_hard", "appearance_change"]:
        count = 0
        for item in runnable:
            if subset_flags(item).get(subset):
                add_item(item)
                if item.get("item_id") in selected_ids:
                    count += 1
                if count >= 3 or len(selected) >= max_items:
                    break

    for item in runnable:
        if len(selected) >= max_items:
            break
        add_item(item)

    per_subset = Counter()
    for item in selected:
        for name, present in subset_flags(item).items():
            if present:
                per_subset[name] += 1

    checks = {
        "frame_path_exists_check": all(all(Path(p).exists() for p in item.get("frame_paths", [])) for item in selected),
        "prompt_exists_check": all(bool(item.get("observed_target") or {}) for item in selected),
        "future_candidate_exists_check": all(bool(item.get("future_candidates")) for item in selected),
    }
    report = {
        "created_at": now(),
        "selection_policy": {
            "max_items": max_items,
            "target_subsets": ["occlusion_reappearance", "long_gap_persistence", "crossing_ambiguity", "OOD_hard", "appearance_change"],
            "target_per_subset_if_available": 3,
            "easy_case_only": False,
        },
        "selected_item_count": len(selected),
        "per_subset_counts": dict(per_subset),
        "selected_item_ids": [x.get("item_id") for x in selected],
        **checks,
        "items": selected,
    }
    write_json(REPORTS / "stwm_external_baseline_smoke_item_selection_20260426.json", report)
    md = [
        f"- selected_item_count: `{len(selected)}`",
        f"- per_subset_counts: `{dict(per_subset)}`",
        f"- frame_path_exists_check: `{checks['frame_path_exists_check']}`",
        f"- prompt_exists_check: `{checks['prompt_exists_check']}`",
        f"- future_candidate_exists_check: `{checks['future_candidate_exists_check']}`",
        "",
        "| # | item_id | subsets |",
        "|---:|---|---|",
    ]
    for idx, item in enumerate(selected, start=1):
        present = [k for k, v in subset_flags(item).items() if v]
        md.append(f"| {idx} | `{item.get('item_id')}` | {', '.join(present)} |")
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_SMOKE_ITEM_SELECTION_20260426.md", "STWM External Baseline Smoke Item Selection 20260426", md)
    return report


def decode_rle(rle: dict[str, Any] | None) -> np.ndarray | None:
    if not rle:
        return None
    rr = dict(rle)
    if isinstance(rr.get("counts"), str):
        rr["counts"] = rr["counts"].encode("utf-8")
    decoded = mask_utils.decode(rr)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded.astype(bool)


def mask_from_bbox(bbox: list[float] | None, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if not bbox:
        return mask
    x1, y1, x2, y2 = [int(round(float(x))) for x in bbox]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h, y2))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def resize_mask(mask: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    if mask.shape[:2] == (h, w):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def scaled_bbox(bbox: list[float] | None, sx: float, sy: float) -> list[float] | None:
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return [float(x1) * sx, float(y1) * sy, float(x2) * sx, float(y2) * sy]


def bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def read_resize_frame(path: str, max_side: int = MAX_SIDE) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_h, new_w = h, w
    return image, (h, w), (new_w / w, new_h / h)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image.copy()
    if mask is None or mask.size == 0:
        return out
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"overlay mask must be 2-D after squeeze, got shape={mask.shape}")
    if mask.shape[:2] != out.shape[:2]:
        mask = resize_mask(mask.astype(bool), out.shape[:2])
    color_arr = np.array(color, dtype=np.uint8)
    mask = mask.astype(bool)
    out[mask] = (out[mask].astype(np.float32) * (1 - alpha) + color_arr.astype(np.float32) * alpha).astype(np.uint8)
    return out


def draw_bbox(image: np.ndarray, bbox: list[float] | None, color: tuple[int, int, int], label: str | None = None, width: int = 2) -> np.ndarray:
    out = image.copy()
    if not bbox:
        return out
    x1, y1, x2, y2 = [int(round(float(x))) for x in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, width)
    if label:
        cv2.putText(out, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def write_title(image: np.ndarray, title: str) -> np.ndarray:
    pad = 34
    out = np.zeros((image.shape[0] + pad, image.shape[1], 3), dtype=np.uint8)
    out[:pad] = (245, 245, 245)
    out[pad:] = image
    cv2.putText(out, title[:130], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def horizontal_stack(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    h = max(img.shape[0] for img in images)
    padded = []
    for img in images:
        if img.shape[0] < h:
            pad = np.full((h - img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)
    return np.hstack(padded)


class PreparedItem:
    def __init__(self, item: dict[str, Any], baseline: str):
        self.item = item
        self.baseline = baseline
        self.item_id = item.get("item_id")
        self.gt_candidate_id = str(item.get("gt_candidate_id"))
        self.future_frame_index = int(item.get("future_frame_index"))
        self.observed_prompt_frame_index = int(item.get("observed_prompt_frame_index", 0))
        self.start_index = self.observed_prompt_frame_index
        self.end_index = self.future_frame_index
        self.local_future_index = self.future_frame_index - self.start_index
        self.out_dir = OUTPUTS / baseline / OUTPUT_PHASE / safe_name(self.item_id)
        self.frames_dir = self.out_dir / "frames"
        self.overlay_dir = self.out_dir / "overlays"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir.mkdir(parents=True, exist_ok=True)

        paths = item.get("frame_paths") or []
        if self.start_index < 0 or self.end_index >= len(paths) or self.end_index < self.start_index:
            raise ValueError(f"invalid frame window start={self.start_index} end={self.end_index} len={len(paths)}")

        self.local_frames_bgr: list[np.ndarray] = []
        self.local_frame_paths: list[str] = []
        self.original_shape: tuple[int, int] | None = None
        self.scale_xy: tuple[float, float] | None = None
        for local_idx, src_idx in enumerate(range(self.start_index, self.end_index + 1)):
            frame, orig_shape, scale_xy = read_resize_frame(paths[src_idx], MAX_SIDE)
            if self.original_shape is None:
                self.original_shape = orig_shape
                self.scale_xy = scale_xy
            out_path = self.frames_dir / f"{local_idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            self.local_frame_paths.append(str(out_path))
            self.local_frames_bgr.append(frame)

        self.height, self.width = self.local_frames_bgr[0].shape[:2]
        sx, sy = self.scale_xy or (1.0, 1.0)
        observed = item.get("observed_target") or {}
        observed_mask = decode_rle(observed.get("mask_rle"))
        if observed_mask is None and observed.get("mask_path") and Path(observed["mask_path"]).exists():
            observed_mask = cv2.imread(observed["mask_path"], cv2.IMREAD_GRAYSCALE) > 0
        if observed_mask is None:
            observed_mask = mask_from_bbox(observed.get("bbox"), self.original_shape or (self.height, self.width))
        self.observed_mask = resize_mask(observed_mask, (self.height, self.width))
        self.observed_bbox = scaled_bbox(observed.get("bbox"), sx, sy) or bbox_from_mask(self.observed_mask)
        self.observed_point = None
        if observed.get("point_prompt"):
            self.observed_point = [float(observed["point_prompt"][0]) * sx, float(observed["point_prompt"][1]) * sy]

        self.future_candidates: list[dict[str, Any]] = []
        for cand in item.get("future_candidates") or []:
            cand_mask = decode_rle(cand.get("mask_rle"))
            if cand_mask is None and cand.get("mask_path") and Path(cand["mask_path"]).exists():
                cand_mask = cv2.imread(cand["mask_path"], cv2.IMREAD_GRAYSCALE) > 0
            if cand_mask is None:
                cand_mask = mask_from_bbox(cand.get("bbox"), self.original_shape or (self.height, self.width))
            cand_mask = resize_mask(cand_mask, (self.height, self.width))
            cand_bbox = scaled_bbox(cand.get("bbox"), sx, sy) or bbox_from_mask(cand_mask)
            self.future_candidates.append(
                {
                    "candidate_id": str(cand.get("candidate_id")),
                    "mask": cand_mask,
                    "bbox": cand_bbox,
                }
            )
        if not self.future_candidates:
            raise ValueError("no future candidates in prepared item")

    @property
    def future_frame_bgr(self) -> np.ndarray:
        return self.local_frames_bgr[self.local_future_index]

    def save_input_overlays(self) -> dict[str, str]:
        obs = overlay_mask(self.local_frames_bgr[0], self.observed_mask, (0, 180, 0), 0.4)
        obs = draw_bbox(obs, self.observed_bbox, (0, 220, 0), "observed prompt")
        obs = write_title(obs, f"{self.baseline}: observed prompt")
        obs_path = self.overlay_dir / "observed_prompt_overlay.png"
        cv2.imwrite(str(obs_path), obs)

        future = self.future_frame_bgr.copy()
        for cand in self.future_candidates:
            color = (0, 200, 0) if cand["candidate_id"] == self.gt_candidate_id else (80, 80, 230)
            future = overlay_mask(future, cand["mask"], color, 0.22)
            future = draw_bbox(future, cand["bbox"], color, f"cand {cand['candidate_id']}")
        future = write_title(future, f"{self.baseline}: future candidates (green=GT)")
        future_path = self.overlay_dir / "future_candidate_overlay.png"
        cv2.imwrite(str(future_path), future)
        return {"observed_prompt_overlay": str(obs_path), "future_candidate_overlay": str(future_path)}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def rank_from_scores(scores: dict[str, float], gt_candidate_id: str) -> dict[str, Any]:
    ordered = sorted(scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
    top5 = [cid for cid, _ in ordered[:5]]
    pred = top5[0] if top5 else None
    rank = None
    for idx, (cid, _) in enumerate(ordered, start=1):
        if cid == str(gt_candidate_id):
            rank = idx
            break
    return {
        "predicted_candidate_id": pred,
        "gt_candidate_id": str(gt_candidate_id),
        "top1_correct": bool(pred == str(gt_candidate_id)),
        "candidate_scores": {str(k): float(v) for k, v in scores.items()},
        "top5_candidates": top5,
        "mrr": float(1.0 / rank) if rank else 0.0,
        "gt_rank": rank,
    }


def score_mask_against_candidates(pred_mask: np.ndarray, prepared: PreparedItem) -> dict[str, Any]:
    pred_mask = np.squeeze(pred_mask)
    if pred_mask.ndim != 2:
        raise ValueError(f"predicted mask must be 2-D after squeeze, got shape={pred_mask.shape}")
    if pred_mask.shape[:2] != (prepared.height, prepared.width):
        pred_mask = resize_mask(pred_mask.astype(bool), (prepared.height, prepared.width))
    scores = {cand["candidate_id"]: iou(pred_mask, cand["mask"]) for cand in prepared.future_candidates}
    return rank_from_scores(scores, prepared.gt_candidate_id)


def save_prediction_overlay(prepared: PreparedItem, pred_mask: np.ndarray, ranking: dict[str, Any]) -> dict[str, str]:
    pred_mask = np.squeeze(pred_mask)
    if pred_mask.ndim != 2:
        raise ValueError(f"predicted mask must be 2-D after squeeze, got shape={pred_mask.shape}")
    if pred_mask.shape[:2] != (prepared.height, prepared.width):
        pred_mask = resize_mask(pred_mask.astype(bool), (prepared.height, prepared.width))
    image = prepared.future_frame_bgr.copy()
    image = overlay_mask(image, pred_mask.astype(bool), (255, 180, 0), 0.45)
    for cand in prepared.future_candidates:
        if cand["candidate_id"] == ranking.get("predicted_candidate_id"):
            color = (0, 190, 0) if ranking.get("top1_correct") else (0, 0, 255)
            label = f"pred {cand['candidate_id']}"
            image = draw_bbox(image, cand["bbox"], color, label, 3)
        elif cand["candidate_id"] == prepared.gt_candidate_id:
            image = draw_bbox(image, cand["bbox"], (0, 190, 0), f"GT {cand['candidate_id']}", 2)
    title = f"{prepared.baseline}: pred={ranking.get('predicted_candidate_id')} gt={prepared.gt_candidate_id} correct={ranking.get('top1_correct')}"
    image = write_title(image, title)
    path = prepared.overlay_dir / "predicted_vs_gt_overlay.png"
    cv2.imwrite(str(path), image)

    contact = horizontal_stack(
        [
            cv2.imread(str(prepared.overlay_dir / "observed_prompt_overlay.png")),
            cv2.imread(str(prepared.overlay_dir / "future_candidate_overlay.png")),
            image,
        ]
    )
    contact_path = prepared.overlay_dir / "smoke_contact_sheet.png"
    cv2.imwrite(str(contact_path), contact)
    return {"predicted_vs_gt_overlay": str(path), "smoke_contact_sheet": str(contact_path)}


def base_item_record(prepared: PreparedItem, started: float) -> dict[str, Any]:
    return {
        "item_id": prepared.item_id,
        "protocol_item_id": prepared.item.get("protocol_item_id"),
        "dataset": prepared.item.get("dataset"),
        "source_protocol": prepared.item.get("source_protocol"),
        "subset_tags": prepared.item.get("subset_tags"),
        "gt_candidate_id": prepared.gt_candidate_id,
        "runtime_seconds": round(time.time() - started, 4),
    }


def exception_payload(exc: BaseException) -> dict[str, str]:
    tb = traceback.format_exc()
    return {
        "exact_error": f"{type(exc).__name__}: {exc}",
        "stderr_excerpt": tb[-4000:],
    }


class BaselineSmokeRunner:
    name = "baseline"

    def __init__(self) -> None:
        self.model = None
        self.model_loaded = False
        self.model_load_error: dict[str, Any] | None = None
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        self.model_loaded = True

    def run_item(self, item: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def prepare(self, item: dict[str, Any]) -> PreparedItem:
        prepared = PreparedItem(item, self.name)
        prepared.save_input_overlays()
        return prepared


class CutieSmokeRunner(BaselineSmokeRunner):
    name = "cutie"

    def load(self) -> None:
        try:
            if torch is None or not torch.cuda.is_available():
                raise RuntimeError("Cutie get_default_model() requires CUDA in the official quick-start path.")
            repo = REPOS / "Cutie"
            sys.path.insert(0, str(repo))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                from cutie.inference.inference_core import InferenceCore
                from cutie.utils.get_default_model import get_default_model

                model = get_default_model()
                processor = InferenceCore(model, cfg=model.cfg)
                processor.max_internal_size = min(MAX_SIDE, 480)
            self.model = (model, processor)
            self.model_loaded = True
        except Exception as exc:
            self.model_load_error = {
                **exception_payload(exc),
                "adapter_stage_failed": "load_cutie_model",
                "possible_fix": "Check CUDA availability/free memory and Cutie weights under baselines/checkpoints/cutie or baselines/repos/Cutie/weights.",
            }
            self.model_loaded = False

    def run_item(self, item: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        try:
            if not self.model_loaded:
                raise RuntimeError("cutie_model_not_loaded")
            prepared = self.prepare(item)
            model, processor = self.model
            processor.clear_memory()
            from torchvision import transforms

            to_tensor = transforms.ToTensor()
            final_mask = None
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=True):
                    for idx, frame_path in enumerate(prepared.local_frame_paths):
                        image = Image.open(frame_path).convert("RGB")
                        image_t = to_tensor(image).cuda().float()
                        if idx == 0:
                            mask_t = torch.from_numpy(prepared.observed_mask.astype(np.int64)).cuda()
                            output_prob = processor.step(image_t, mask_t, objects=[1])
                        else:
                            output_prob = processor.step(image_t)
                        if idx == prepared.local_future_index:
                            final_mask = (processor.output_prob_to_mask(output_prob).detach().cpu().numpy() == 1)
                            break
            if final_mask is None:
                raise RuntimeError("cutie_no_future_prediction")
            ranking = score_mask_against_candidates(final_mask, prepared)
            overlays = save_prediction_overlay(prepared, final_mask, ranking)
            result = {
                **base_item_record(prepared, started),
                "baseline_name": self.name,
                "success": True,
                **ranking,
                "output_mask_path": None,
                "visual_overlays": {k: rel(v) for k, v in overlays.items()},
                "failure_reason_if_any": None,
            }
            return result
        except Exception as exc:
            payload = exception_payload(exc)
            return {
                "item_id": item.get("item_id"),
                "baseline_name": self.name,
                "success": False,
                "runtime_seconds": round(time.time() - started, 4),
                "predicted_candidate_id": None,
                "gt_candidate_id": str(item.get("gt_candidate_id")),
                "top1_correct": False,
                "candidate_scores": {},
                "top5_candidates": [],
                "mrr": 0.0,
                "output_mask_path": None,
                "failure_reason_if_any": payload["exact_error"],
                "exact_error": payload["exact_error"],
                "stderr_excerpt": payload["stderr_excerpt"],
                "adapter_stage_failed": "cutie_item_inference",
                "possible_fix": "Inspect item frame/mask payload and Cutie CUDA memory; try smaller STWM_EXTERNAL_SMOKE_MAX_SIDE if OOM.",
            }


class SAM2SmokeRunner(BaselineSmokeRunner):
    name = "sam2"

    def load(self) -> None:
        try:
            if torch is None:
                raise RuntimeError("torch_unavailable")
            repo = REPOS / "sam2"
            sys.path.insert(0, str(repo))
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            from hydra.utils import instantiate
            from omegaconf import OmegaConf

            ckpt = CHECKPOINTS / "sam2" / "sam2.1_hiera_tiny.pt"
            if not ckpt.exists():
                candidates = sorted((CHECKPOINTS / "sam2").glob("*.pt"))
                if not candidates:
                    raise FileNotFoundError("no SAM2 checkpoint under baselines/checkpoints/sam2")
                ckpt = candidates[0]

            config_dir = repo / "sam2" / "configs"
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                cfg = compose(
                    config_name="sam2.1/sam2.1_hiera_t.yaml",
                    overrides=[
                        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
                        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                        "++model.binarize_mask_from_pts_for_mem_enc=true",
                        "++model.fill_hole_area=8",
                    ],
                )
                OmegaConf.resolve(cfg)
                model = instantiate(cfg.model, _recursive_=True)
            state = torch.load(str(ckpt), map_location="cpu")
            if "model" in state:
                state = state["model"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            model = model.to(self.device).eval()
            self.model = {
                "predictor": model,
                "checkpoint": str(ckpt),
                "predictor_class_or_api": "sam2.sam2_video_predictor.SAM2VideoPredictor",
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            }
            self.model_loaded = True
        except Exception as exc:
            self.model_load_error = {
                **exception_payload(exc),
                "adapter_stage_failed": "load_sam2_video_predictor",
                "possible_fix": "Check SAM2 repo import path, Hydra config availability, checkpoint compatibility, and CUDA/CPU memory.",
            }
            self.model_loaded = False

    def run_item(self, item: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        try:
            if not self.model_loaded:
                raise RuntimeError("sam2_model_not_loaded")
            prepared = self.prepare(item)
            predictor = self.model["predictor"]
            inference_state = predictor.init_state(str(prepared.frames_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
            prompt_type = "box" if prepared.observed_bbox else "point"
            with torch.inference_mode():
                if prompt_type == "box":
                    predictor.add_new_points_or_box(
                        inference_state,
                        frame_idx=0,
                        obj_id=1,
                        box=np.array(prepared.observed_bbox, dtype=np.float32),
                    )
                else:
                    point = np.array([prepared.observed_point], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)
                    predictor.add_new_points_or_box(
                        inference_state,
                        frame_idx=0,
                        obj_id=1,
                        points=point,
                        labels=labels,
                    )
                final_mask = None
                for out_frame_idx, _obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    if out_frame_idx == prepared.local_future_index:
                        final_mask = (out_mask_logits[0].detach().cpu().numpy() > 0)
                        break
            if final_mask is None:
                raise RuntimeError("sam2_no_future_prediction")
            ranking = score_mask_against_candidates(final_mask, prepared)
            overlays = save_prediction_overlay(prepared, final_mask, ranking)
            return {
                **base_item_record(prepared, started),
                "baseline_name": self.name,
                "success": True,
                **ranking,
                "prompt_type_used": prompt_type,
                "checkpoint_used": rel(self.model.get("checkpoint")),
                "predictor_class_or_api": self.model.get("predictor_class_or_api"),
                "visual_overlays": {k: rel(v) for k, v in overlays.items()},
                "failure_stage_if_any": None,
                "failure_reason_if_any": None,
            }
        except Exception as exc:
            payload = exception_payload(exc)
            return {
                "item_id": item.get("item_id"),
                "baseline_name": self.name,
                "success": False,
                "runtime_seconds": round(time.time() - started, 4),
                "predicted_candidate_id": None,
                "gt_candidate_id": str(item.get("gt_candidate_id")),
                "top1_correct": False,
                "candidate_scores": {},
                "top5_candidates": [],
                "mrr": 0.0,
                "prompt_type_used": None,
                "checkpoint_used": rel(self.model.get("checkpoint")) if isinstance(self.model, dict) else None,
                "predictor_class_or_api": "sam2.sam2_video_predictor.SAM2VideoPredictor",
                "failure_reason_if_any": payload["exact_error"],
                "exact_error": payload["exact_error"],
                "stderr_excerpt": payload["stderr_excerpt"],
                "failure_stage_if_any": "sam2_item_inference",
                "possible_fix": "Inspect SAM2 video predictor API, prompt payload, and GPU/CPU memory. Try tiny checkpoint and smaller max side.",
            }


def ensure_cotracker_checkpoint() -> tuple[Path | None, dict[str, Any]]:
    ckpt_dir = CHECKPOINTS / "cotracker"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(ckpt_dir.glob("*.pth"))
    if existing:
        return existing[0], {"download_attempted": False, "checkpoint_path": str(existing[0])}
    target = ckpt_dir / "scaled_offline.pth"
    started = time.time()
    try:
        urllib.request.urlretrieve(COTRACKER_CKPT_URL, target)
        if target.exists() and target.stat().st_size > 1024:
            return target, {
                "download_attempted": True,
                "checkpoint_path": str(target),
                "url": COTRACKER_CKPT_URL,
                "success": True,
                "wall_time_seconds": round(time.time() - started, 3),
            }
        return None, {
            "download_attempted": True,
            "url": COTRACKER_CKPT_URL,
            "success": False,
            "exact_error": "download_created_empty_or_too_small_file",
            "wall_time_seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        return None, {
            "download_attempted": True,
            "url": COTRACKER_CKPT_URL,
            "success": False,
            **exception_payload(exc),
            "wall_time_seconds": round(time.time() - started, 3),
        }


def sample_points_from_mask(mask: np.ndarray, bbox: list[float] | None, max_points: int = 64) -> np.ndarray:
    if mask is not None and mask.any():
        ys, xs = np.where(mask)
        if xs.size <= max_points:
            return np.stack([xs, ys], axis=1).astype(np.float32)
        step = max(1, int(math.floor(xs.size / max_points)))
        idx = np.arange(0, xs.size, step)[:max_points]
        return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    if not bbox:
        return np.zeros((0, 2), dtype=np.float32)
    x1, y1, x2, y2 = [float(x) for x in bbox]
    gx = np.linspace(x1, x2, num=8)
    gy = np.linspace(y1, y2, num=8)
    pts = np.array([[x, y] for y in gy for x in gx], dtype=np.float32)
    return pts[:max_points]


class CoTrackerSmokeRunner(BaselineSmokeRunner):
    name = "cotracker"

    def load(self) -> None:
        try:
            if torch is None:
                raise RuntimeError("torch_unavailable")
            ckpt, download = ensure_cotracker_checkpoint()
            if ckpt is None:
                raise RuntimeError(f"cotracker_checkpoint_unavailable:{download}")
            repo = REPOS / "co-tracker"
            sys.path.insert(0, str(repo))
            from cotracker.predictor import CoTrackerPredictor

            predictor = CoTrackerPredictor(checkpoint=str(ckpt), offline=True, v2=False, window_len=60).to(self.device).eval()
            self.model = {"predictor": predictor, "checkpoint": str(ckpt), "download": download}
            self.model_loaded = True
        except Exception as exc:
            self.model_load_error = {
                **exception_payload(exc),
                "adapter_stage_failed": "load_cotracker_predictor_or_checkpoint",
                "exact_checkpoint_blocker": f"checkpoint_missing_or_unloadable; attempted_url={COTRACKER_CKPT_URL}",
                "suggested_fix": "Manually place scaled_offline.pth in baselines/checkpoints/cotracker/ or resolve Hugging Face download/network access, then rerun smoke.",
            }
            self.model_loaded = False

    def run_item(self, item: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        try:
            if not self.model_loaded:
                raise RuntimeError("cotracker_checkpoint_or_model_not_loaded")
            prepared = self.prepare(item)
            predictor = self.model["predictor"]
            points_xy = sample_points_from_mask(prepared.observed_mask, prepared.observed_bbox, max_points=64)
            if points_xy.size == 0:
                raise RuntimeError("no_points_sampled_from_observed_prompt")
            queries_np = np.concatenate([np.zeros((points_xy.shape[0], 1), dtype=np.float32), points_xy], axis=1)
            frames_rgb = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in prepared.local_frames_bgr]
            video_np = np.stack(frames_rgb, axis=0)
            video = torch.from_numpy(video_np).permute(0, 3, 1, 2).float().unsqueeze(0).to(self.device)
            queries = torch.from_numpy(queries_np).float().unsqueeze(0).to(self.device)
            with torch.inference_mode():
                tracks, visibility = predictor(video, queries=queries)
            future_pts = tracks[0, prepared.local_future_index].detach().cpu().numpy()
            future_vis = visibility[0, prepared.local_future_index].detach().cpu().numpy().astype(bool)
            visible_pts = future_pts[future_vis]
            scores: dict[str, float] = {}
            for cand in prepared.future_candidates:
                if visible_pts.size == 0:
                    inside_ratio = 0.0
                else:
                    xs = np.clip(np.round(visible_pts[:, 0]).astype(int), 0, prepared.width - 1)
                    ys = np.clip(np.round(visible_pts[:, 1]).astype(int), 0, prepared.height - 1)
                    inside_ratio = float(cand["mask"][ys, xs].mean())
                centroid_score = 0.0
                if cand["bbox"] and visible_pts.size:
                    x1, y1, x2, y2 = cand["bbox"]
                    cand_c = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
                    pts_c = visible_pts.mean(axis=0)
                    dist = float(np.linalg.norm(pts_c - cand_c))
                    centroid_score = 1.0 / (1.0 + dist / max(prepared.width, prepared.height))
                scores[cand["candidate_id"]] = inside_ratio + 0.001 * centroid_score
            ranking = rank_from_scores(scores, prepared.gt_candidate_id)
            pred_mask = np.zeros((prepared.height, prepared.width), dtype=np.uint8)
            for x, y in visible_pts:
                xi = int(np.clip(round(x), 0, prepared.width - 1))
                yi = int(np.clip(round(y), 0, prepared.height - 1))
                cv2.circle(pred_mask, (xi, yi), 2, 1, -1)
            overlays = save_prediction_overlay(prepared, pred_mask.astype(bool), ranking)
            return {
                **base_item_record(prepared, started),
                "baseline_name": self.name,
                "success": True,
                **ranking,
                "sampled_point_count": int(points_xy.shape[0]),
                "visible_point_count": int(future_vis.sum()),
                "candidate_point_inside_ratio": {k: float(v) for k, v in scores.items()},
                "centroid_tiebreak_score": "0.001 * inverse normalized centroid distance",
                "checkpoint_used": rel(self.model.get("checkpoint")),
                "visual_overlays": {k: rel(v) for k, v in overlays.items()},
                "failure_reason_if_any": None,
            }
        except Exception as exc:
            payload = exception_payload(exc)
            return {
                "item_id": item.get("item_id"),
                "baseline_name": self.name,
                "success": False,
                "runtime_seconds": round(time.time() - started, 4),
                "sampled_point_count": 0,
                "visible_point_count": 0,
                "candidate_point_inside_ratio": {},
                "centroid_tiebreak_score": None,
                "predicted_candidate_id": None,
                "gt_candidate_id": str(item.get("gt_candidate_id")),
                "top1_correct": False,
                "candidate_scores": {},
                "top5_candidates": [],
                "mrr": 0.0,
                "failure_reason_if_any": payload["exact_error"],
                "exact_error": payload["exact_error"],
                "stderr_excerpt": payload["stderr_excerpt"],
                "adapter_stage_failed": "cotracker_item_inference",
                "possible_fix": "Inspect CoTracker checkpoint/API, prompt point sampling, and GPU memory. This report does not fake inference if checkpoint/model fails.",
            }


def summarize_baseline(name: str, results: list[dict[str, Any]], runner: BaselineSmokeRunner, attempted_items: int) -> dict[str, Any]:
    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]
    avg_runtime = mean([float(r.get("runtime_seconds", 0.0)) for r in results]) if results else None
    top1 = mean([1.0 if r.get("top1_correct") else 0.0 for r in successes]) if successes else 0.0
    mrr = mean([float(r.get("mrr") or 0.0) for r in successes]) if successes else 0.0
    failure_reasons = Counter()
    for r in failures:
        failure_reasons[r.get("failure_reason_if_any") or r.get("exact_error") or "unknown_failure"] += 1
    visual_paths = []
    for r in successes:
        visual_paths.extend((r.get("visual_overlays") or {}).values())
    load_error = runner.model_load_error
    return {
        "baseline_name": name,
        "attempted_items": attempted_items if runner.model_loaded else 0,
        "successful_items": len(successes),
        "failed_items": len(failures) if runner.model_loaded else 0,
        "smoke_pass": len(successes) >= MIN_SUCCESS_FOR_PASS,
        "average_runtime": avg_runtime,
        "top1_smoke": top1,
        "mrr_smoke": mrr,
        "example_outputs": results[:5],
        "visual_output_paths": visual_paths[:20],
        "common_failure_reasons": dict(failure_reasons),
        "model_load_error": load_error,
        "exact_blocking_reason": load_error.get("exact_error") if load_error else None,
    }


def write_baseline_report(name: str, report: dict[str, Any]) -> None:
    path = REPORTS / f"stwm_external_baseline_{name}_smoke_20260426.json"
    write_json(path, report)
    title_name = {"cutie": "CUTIE", "sam2": "SAM2", "cotracker": "CoTracker"}[name]
    lines = [
        f"- smoke_pass: `{report['summary']['smoke_pass']}`",
        f"- attempted_items: `{report['summary']['attempted_items']}`",
        f"- successful_items: `{report['summary']['successful_items']}`",
        f"- failed_items: `{report['summary']['failed_items']}`",
        f"- exact_blocking_reason: `{report['summary'].get('exact_blocking_reason')}`",
        "",
        "| item_id | success | pred | gt | mrr | failure |",
        "|---|---:|---|---|---:|---|",
    ]
    for r in report.get("per_item_results", [])[:20]:
        lines.append(
            f"| `{r.get('item_id')}` | `{r.get('success')}` | `{r.get('predicted_candidate_id')}` | `{r.get('gt_candidate_id')}` | {float(r.get('mrr') or 0):.3f} | {str(r.get('failure_reason_if_any') or '')[:160]} |"
        )
    write_markdown(DOCS / f"STWM_EXTERNAL_BASELINE_{title_name.upper()}_SMOKE_20260426.md", f"STWM External Baseline {title_name} Smoke 20260426", lines)


def run_baseline(name: str, runner: BaselineSmokeRunner, selected_items: list[dict[str, Any]], env_ok: bool, ckpt_ready: bool) -> dict[str, Any]:
    out_dir = OUTPUTS / name / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    if not env_ok:
        runner.model_load_error = {
            "exact_error": f"{name}_import_not_ready_from_env_audit",
            "stderr_excerpt": "",
            "adapter_stage_failed": "source_audit_precheck",
            "possible_fix": "Fix external repo imports before rerunning smoke.",
        }
    else:
        # CoTracker is allowed to start with checkpoint_ready=false because this smoke
        # must attempt official checkpoint download before falling back to dry-run.
        if not ckpt_ready and name != "cotracker":
            runner.model_load_error = {
                "exact_error": f"{name}_checkpoint_not_ready_from_env_audit",
                "stderr_excerpt": "",
                "adapter_stage_failed": "source_audit_precheck",
                "possible_fix": "Download the official checkpoint into baselines/checkpoints before rerunning smoke.",
            }
        else:
            runner.load()
    if runner.model_loaded:
        for item in selected_items:
            results.append(runner.run_item(item))
            # Smoke pass is established at five successes, but continue a little
            # more to keep subset coverage without accidentally becoming full eval.
            if len([r for r in results if r.get("success")]) >= MIN_SUCCESS_FOR_PASS and len(results) >= min(10, len(selected_items)):
                break
    summary = summarize_baseline(name, results, runner, len(results))
    report = {
        "created_at": now(),
        "baseline_name": name,
        "smoke_policy": {
            "max_selected_items": MAX_ITEMS,
            "minimum_successful_items_for_pass": MIN_SUCCESS_FOR_PASS,
            "full_eval_executed": False,
            "paper_result": False,
        },
        "per_item_results": results,
        "summary": summary,
        "per_item_results_hash": sha256_json(results),
    }
    write_baseline_report(name, report)
    return report


def write_unified_and_decision(reports: dict[str, dict[str, Any]], source_audit: dict[str, Any], selection: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    summaries = {name: report["summary"] for name, report in reports.items()}
    common_failures = Counter()
    visual_paths: list[str] = []
    for summary in summaries.values():
        common_failures.update(summary.get("common_failure_reasons") or {})
        visual_paths.extend(summary.get("visual_output_paths") or [])
    unified = {
        "created_at": now(),
        "source_audit": "reports/stwm_external_baseline_smoke_source_audit_20260426.json",
        "item_selection": "reports/stwm_external_baseline_smoke_item_selection_20260426.json",
        "cutie_attempted_items": summaries["cutie"]["attempted_items"],
        "cutie_successful_items": summaries["cutie"]["successful_items"],
        "cutie_smoke_pass": summaries["cutie"]["smoke_pass"],
        "sam2_attempted_items": summaries["sam2"]["attempted_items"],
        "sam2_successful_items": summaries["sam2"]["successful_items"],
        "sam2_smoke_pass": summaries["sam2"]["smoke_pass"],
        "cotracker_attempted_items": summaries["cotracker"]["attempted_items"],
        "cotracker_successful_items": summaries["cotracker"]["successful_items"],
        "cotracker_smoke_pass": summaries["cotracker"]["smoke_pass"],
        "per_baseline_average_runtime": {k: v.get("average_runtime") for k, v in summaries.items()},
        "per_baseline_top1_smoke": {k: v.get("top1_smoke") for k, v in summaries.items()},
        "per_baseline_mrr_smoke": {k: v.get("mrr_smoke") for k, v in summaries.items()},
        "common_failure_reasons": dict(common_failures),
        "visual_output_paths": visual_paths[:60],
        "baseline_reports": {
            "cutie": "reports/stwm_external_baseline_cutie_smoke_20260426.json",
            "sam2": "reports/stwm_external_baseline_sam2_smoke_20260426.json",
            "cotracker": "reports/stwm_external_baseline_cotracker_smoke_20260426.json",
        },
        "full_eval_executed": False,
        "paper_result": False,
    }
    write_json(REPORTS / "stwm_external_baseline_smoke_20260426.json", unified)
    lines = [
        "| baseline | attempted | successful | smoke_pass | top1 smoke | MRR smoke | blocker |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in ["cutie", "sam2", "cotracker"]:
        s = summaries[name]
        lines.append(
            f"| {name} | {s['attempted_items']} | {s['successful_items']} | `{s['smoke_pass']}` | {float(s.get('top1_smoke') or 0):.3f} | {float(s.get('mrr_smoke') or 0):.3f} | {str(s.get('exact_blocking_reason') or '')[:180]} |"
        )
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_SMOKE_20260426.md", "STWM External Baseline Smoke 20260426", lines)

    pass_names = [name for name, summary in summaries.items() if summary.get("smoke_pass")]
    if summaries["cutie"]["smoke_pass"] and summaries["sam2"]["smoke_pass"] and summaries["cotracker"]["smoke_pass"]:
        next_step = "run_external_baseline_full_eval_next"
    elif summaries["cutie"]["smoke_pass"] or summaries["sam2"]["smoke_pass"]:
        next_step = "run_cutie_sam2_full_eval_only"
    elif not summaries["cotracker"]["smoke_pass"] and summaries["cotracker"].get("model_load_error"):
        next_step = "fix_cotracker_checkpoint_then_smoke"
    else:
        next_step = "stop_external_baseline_due_to_adapter_blocker"

    priority = None
    if pass_names:
        priority = max(pass_names, key=lambda n: (summaries[n].get("successful_items") or 0, summaries[n].get("mrr_smoke") or 0))
    decision = {
        "created_at": now(),
        "cutie_enter_full_eval": bool(summaries["cutie"]["smoke_pass"]),
        "sam2_enter_full_eval": bool(summaries["sam2"]["smoke_pass"]),
        "cotracker_enter_full_eval": bool(summaries["cotracker"]["smoke_pass"]),
        "priority_full_eval_baseline": priority,
        "needs_adapter_fix": {name: not bool(summary.get("smoke_pass")) for name, summary in summaries.items()},
        "next_step_choice": next_step,
        "decision_rules": {
            "minimum_successful_items_for_pass": MIN_SUCCESS_FOR_PASS,
            "cutie_or_sam2_pass_does_not_wait_for_cotracker": True,
        },
        "summaries": summaries,
    }
    write_json(REPORTS / "stwm_external_baseline_smoke_decision_20260426.json", decision)
    write_simple_md(DOCS / "STWM_EXTERNAL_BASELINE_SMOKE_DECISION_20260426.md", "STWM External Baseline Smoke Decision 20260426", decision)
    return unified, decision


def main() -> None:
    source_audit = build_source_audit()
    if not source_audit["audit_passed"]:
        empty_selection = {"items": [], "selected_item_count": 0}
        write_json(REPORTS / "stwm_external_baseline_smoke_item_selection_20260426.json", empty_selection)
        write_simple_md(DOCS / "STWM_EXTERNAL_BASELINE_SMOKE_ITEM_SELECTION_20260426.md", "STWM External Baseline Smoke Item Selection 20260426", empty_selection)
        reports = {}
        for name in ["cutie", "sam2", "cotracker"]:
            runner = BaselineSmokeRunner()
            runner.name = name
            runner.model_load_error = {
                "exact_error": "source_audit_failed_manifest_or_visual_sanity",
                "stderr_excerpt": json.dumps(source_audit, indent=2)[-3000:],
                "adapter_stage_failed": "source_audit",
                "possible_fix": "Repair manifest/visual sanity before running external smoke.",
            }
            report = {
                "created_at": now(),
                "baseline_name": name,
                "per_item_results": [],
                "summary": summarize_baseline(name, [], runner, 0),
                "per_item_results_hash": sha256_json([]),
            }
            write_baseline_report(name, report)
            reports[name] = report
        write_unified_and_decision(reports, source_audit, empty_selection)
        return

    manifest = load_json(MANIFEST)
    selection = select_smoke_items(manifest.get("items") or [], MAX_ITEMS)
    selected_items = selection.get("items") or []
    reports = {
        "cutie": run_baseline(
            "cutie",
            CutieSmokeRunner(),
            selected_items,
            bool(source_audit.get("cutie_import_ok")),
            bool(source_audit.get("cutie_checkpoint_ready")),
        ),
        "sam2": run_baseline(
            "sam2",
            SAM2SmokeRunner(),
            selected_items,
            bool(source_audit.get("sam2_import_ok")),
            bool(source_audit.get("sam2_checkpoint_ready")),
        ),
        "cotracker": run_baseline(
            "cotracker",
            CoTrackerSmokeRunner(),
            selected_items,
            bool(source_audit.get("cotracker_import_ok")),
            bool(source_audit.get("cotracker_checkpoint_ready")),
        ),
    }
    write_unified_and_decision(reports, source_audit, selection)


if __name__ == "__main__":
    main()
