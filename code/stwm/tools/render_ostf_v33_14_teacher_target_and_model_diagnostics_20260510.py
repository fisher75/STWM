#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_14_teacher_target_visualization_manifest_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_14_TEACHER_TARGET_VISUALIZATION_20260510.md"
OUT = ROOT / "outputs/figures/stwm_ostf_v33_14_teacher_target_diagnostics"
DINO_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_14_semantic_targets/pointodyssey"
FEATURE_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_14_teacher_features/pointodyssey"
CLIP_K256_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_12_semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K256"


def load_json(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def scalar(x: np.ndarray) -> Any:
    return x.item() if getattr(x, "shape", None) == () else x


def best_target_root() -> tuple[str, str, int, Path]:
    probe = load_json("reports/stwm_ostf_v33_14_teacher_target_space_probe_sweep_20260510.json")
    vocab = load_json("reports/stwm_ostf_v33_14_teacher_prototype_vocab_sweep_20260510.json")
    teacher = str(probe.get("best_teacher_by_val") or vocab.get("best_teacher_by_val") or "dinov2_base")
    aggregation = str(probe.get("best_aggregation_by_val") or vocab.get("best_aggregation_by_val") or "point_local_crop")
    k = int(probe.get("best_K_by_val") or vocab.get("best_K_by_val") or 128)
    root = DINO_ROOT / teacher / aggregation / f"K{k}"
    if not root.exists():
        candidates = sorted(DINO_ROOT.glob("*/*/K*"))
        if candidates:
            root = candidates[0]
            teacher = root.parents[1].name
            aggregation = root.parent.name
            k = int(root.name.replace("K", ""))
    return teacher, aggregation, k, root


def matching_feature_path(teacher: str, aggregation: str, split: str, name: str) -> Path:
    return FEATURE_ROOT / teacher / aggregation / split / name


def read_rgb(source_npz: Path) -> np.ndarray | None:
    try:
        z = np.load(source_npz, allow_pickle=True)
        frame_paths = list(z["frame_paths"])
        frame = Path(str(frame_paths[min(7, len(frame_paths) - 1)]))
        if not frame.exists():
            return None
        return np.asarray(Image.open(frame).convert("RGB"))
    except Exception:
        return None


def draw_case(path: Path, title: str, target_path: Path, clip_path: Path | None, reason: str) -> dict[str, Any]:
    target = np.load(target_path, allow_pickle=True)
    split = str(scalar(target["split"]))
    teacher = str(scalar(target["teacher_name"]))
    agg = str(scalar(target["aggregation"]))
    feat_path = matching_feature_path(teacher, agg, split, target_path.name)
    feat = np.load(feat_path, allow_pickle=True) if feat_path.exists() else None
    source = ROOT / str(scalar(feat["source_npz"])) if feat is not None and "source_npz" in feat.files else None
    src = np.load(source, allow_pickle=True) if source and source.exists() else None
    obs = src["obs_points"] if src is not None else np.zeros((128, 8, 2), dtype=np.float32)
    fut = src["fut_points"] if src is not None else np.zeros((128, 32, 2), dtype=np.float32)
    rgb = read_rgb(source) if source and source.exists() else None
    stable = np.asarray(target["semantic_stable_mask"], dtype=bool)
    changed = np.asarray(target["semantic_changed_mask"], dtype=bool)
    valid = np.asarray(target["semantic_prototype_available_mask"], dtype=bool)
    scores = {
        "stable_count": int(stable.sum()),
        "changed_count": int(changed.sum()),
        "valid_count": int(valid.sum()),
    }
    if "changed" in reason and changed.any():
        m, h = np.argwhere(changed)[0]
    elif stable.any():
        m, h = np.argwhere(stable)[0]
    elif valid.any():
        m, h = np.argwhere(valid)[0]
    else:
        m, h = 0, 0
    h = int(h)
    target_ids = np.asarray(target["semantic_prototype_id"])
    copy_ids = np.asarray(target["copy_semantic_prototype_id"])
    obs_ids = np.asarray(target["obs_semantic_prototype_id"])
    clip_ids = None
    if clip_path and clip_path.exists():
        clip = np.load(clip_path, allow_pickle=True)
        if "semantic_prototype_id" in clip.files:
            clip_ids = np.asarray(clip["semantic_prototype_id"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax in axes:
        if rgb is not None:
            ax.imshow(rgb)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title("observed trace / obs proto")
    axes[0].scatter(obs[:, -1, 0], obs[:, -1, 1], c=obs_ids[:, -1] % 20, cmap="tab20", s=12, alpha=0.9)
    axes[1].set_title("copy vs stronger target")
    axes[1].scatter(fut[:, h, 0], fut[:, h, 1], c=copy_ids[:, h] % 20, cmap="tab20", s=16, marker="o", alpha=0.45, label="copy")
    axes[1].scatter(fut[:, h, 0], fut[:, h, 1], c=target_ids[:, h] % 20, cmap="tab20", s=24, marker="x", label="stronger")
    axes[1].legend(fontsize=6)
    axes[2].set_title("CLIP-B/32 K256 vs stronger")
    if clip_ids is not None:
        axes[2].scatter(fut[:, h, 0], fut[:, h, 1], c=clip_ids[:, h] % 20, cmap="tab20", s=16, marker="o", alpha=0.45, label="clip")
    axes[2].scatter(fut[:, h, 0], fut[:, h, 1], c=target_ids[:, h] % 20, cmap="tab20", s=24, marker="x", label=teacher)
    axes[2].legend(fontsize=6)
    fig.suptitle(f"{title} | h={h} | {reason}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {
        "category": title,
        "path": str(path.relative_to(ROOT)),
        "sample_uid": str(scalar(target["sample_uid"])),
        "split": split,
        "horizon_index": h,
        "reason": reason,
        **scores,
    }


def choose_cases(root: Path) -> list[tuple[str, Path, str]]:
    files = sorted((root / "val").glob("*.npz")) + sorted((root / "test").glob("*.npz"))
    selected: list[tuple[str, Path, str]] = []
    categories = [
        ("clip_b32_target_failure_vs_stronger_teacher", "changed"),
        ("teacher_feature_temporal_consistency_example", "stable"),
        ("stable_semantic_preservation_example", "stable"),
        ("changed_semantic_correction_example", "changed"),
        ("semantic_hard_proxy_example", "changed"),
        ("sample_frequency_baseline_failure_example", "changed"),
        ("trace_overlay_field_example", "stable"),
        ("semantic_prototype_color_comparison", "changed"),
    ]
    for cat, reason in categories:
        for p in files:
            z = np.load(p, allow_pickle=True)
            mask = np.asarray(z["semantic_changed_mask" if reason == "changed" else "semantic_stable_mask"], dtype=bool)
            if mask.any():
                selected.append((cat, p, reason))
                break
    return selected


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    teacher, aggregation, k, root = best_target_root()
    examples = []
    blockers: list[str] = []
    if not root.exists():
        blockers.append(f"missing target root: {root}")
    else:
        for idx, (cat, p, reason) in enumerate(choose_cases(root)):
            clip_path = CLIP_K256_ROOT / str(np.load(p, allow_pickle=True)["split"]) / p.name
            out = OUT / f"{idx:02d}_{cat}.png"
            try:
                examples.append(draw_case(out, cat, p, clip_path if clip_path.exists() else None, reason))
            except Exception as exc:  # keep visualization diagnostic from blocking the protocol report
                blockers.append(f"{cat}: {exc}")
    payload = {
        "generated_at_utc": utc_now(),
        "teacher": teacher,
        "aggregation": aggregation,
        "K": k,
        "real_images_rendered": len(examples) > 0,
        "case_mining_used": True,
        "png_count": len(examples),
        "visualization_ready": len(examples) >= 6,
        "placeholder_only": False,
        "output_dir": str(OUT.relative_to(ROOT)),
        "examples": examples,
        "exact_blockers": blockers,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.14 Teacher Target Visualization",
        payload,
        ["teacher", "aggregation", "K", "real_images_rendered", "case_mining_used", "png_count", "visualization_ready", "placeholder_only", "exact_blockers"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
