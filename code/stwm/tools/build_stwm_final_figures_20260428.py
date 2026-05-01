#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FIG_DIR = Path("outputs/figures/stwm_final")


def load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def svg_bar(path: Path, title: str, labels: list[str], copy_vals: list[float], stwm_vals: list[float]) -> None:
    width, height = 760, 420
    left, top, bar_h, gap = 210, 70, 30, 36
    max_v = max(copy_vals + stwm_vals + [1e-6])
    rows = []
    for i, label in enumerate(labels):
        y = top + i * (2 * bar_h + gap)
        cw = int((copy_vals[i] / max_v) * 450)
        sw = int((stwm_vals[i] / max_v) * 450)
        rows.append(f'<text x="20" y="{y+22}" font-size="16" fill="#17202a">{label}</text>')
        rows.append(f'<rect x="{left}" y="{y}" width="{cw}" height="{bar_h}" fill="#9aa6b2"/>')
        rows.append(f'<rect x="{left}" y="{y+bar_h+5}" width="{sw}" height="{bar_h}" fill="#176b5b"/>')
        rows.append(f'<text x="{left+cw+8}" y="{y+21}" font-size="13" fill="#17202a">copy {copy_vals[i]:.3f}</text>')
        rows.append(f'<text x="{left+sw+8}" y="{y+bar_h+26}" font-size="13" fill="#17202a">STWM {stwm_vals[i]:.3f}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f1e8"/>
<text x="20" y="36" font-size="24" font-family="Georgia, serif" fill="#1f2d2b">{title}</text>
<text x="20" y="{height-28}" font-size="13" fill="#52615f">Free-rollout semantic trace field; selected on validation only.</text>
{''.join(rows)}
</svg>
'''
    write(path, svg)


def svg_method(path: Path) -> None:
    boxes = [
        (40, 120, "Observed video / trace", "#f2c078"),
        (245, 120, "Observed semantic memory", "#d98f89"),
        (470, 90, "Stage1 trace rollout", "#8fb9aa"),
        (470, 165, "Stage2 copy-gated residual", "#79a7d3"),
        (270, 285, "Future trace field + semantic prototype field", "#176b5b"),
    ]
    rects = []
    for x, y, text, color in boxes:
        rects.append(f'<rect x="{x}" y="{y}" width="190" height="62" rx="14" fill="{color}" opacity="0.92"/>')
        rects.append(f'<text x="{x+16}" y="{y+37}" font-size="15" fill="#10221f">{text}</text>')
    arrows = [
        '<path d="M230 151 L245 151" stroke="#17202a" stroke-width="3" marker-end="url(#arrow)"/>',
        '<path d="M435 151 L470 121" stroke="#17202a" stroke-width="3" marker-end="url(#arrow)"/>',
        '<path d="M435 151 L470 188" stroke="#17202a" stroke-width="3" marker-end="url(#arrow)"/>',
        '<path d="M565 155 C560 240 470 278 430 292" stroke="#17202a" stroke-width="3" fill="none" marker-end="url(#arrow)"/>',
    ]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="420" viewBox="0 0 760 420">
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#17202a"/></marker></defs>
<rect width="100%" height="100%" fill="#eef2e6"/>
<text x="38" y="50" font-size="25" font-family="Georgia, serif" fill="#1f2d2b">STWM semantic trace world model</text>
{''.join(rects + arrows)}
<text x="40" y="382" font-size="13" fill="#52615f">Not a tracker plugin: rollout input remains observed video/trace/semantic memory only.</text>
</svg>
'''
    write(path, svg)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mixed = load("reports/stwm_mixed_fullscale_v2_mixed_test_eval_complete_20260428.json")
    vspw = load("reports/stwm_mixed_fullscale_v2_vspw_test_eval_complete_20260428.json")
    vipseg = load("reports/stwm_mixed_fullscale_v2_vipseg_test_eval_complete_20260428.json")
    payloads = {"mixed": mixed, "VSPW": vspw, "VIPSeg": vipseg}
    labels = list(payloads.keys())
    copy_changed = [float(p.get("best_metrics", {}).get("copy_changed_subset_top5", 0.0)) for p in payloads.values()]
    stwm_changed = [float(p.get("best_metrics", {}).get("changed_subset_top5", 0.0)) for p in payloads.values()]
    copy_overall = [float(p.get("best_metrics", {}).get("copy_proto_top5", 0.0)) for p in payloads.values()]
    stwm_overall = [float(p.get("best_metrics", {}).get("proto_top5", 0.0)) for p in payloads.values()]

    svg_method(FIG_DIR / "figure_method_semantic_trace_world_model.svg")
    svg_bar(FIG_DIR / "figure_changed_subset_top5.svg", "Changed semantic states: copy prior vs STWM residual", labels, copy_changed, stwm_changed)
    svg_bar(FIG_DIR / "figure_dataset_top5_breakdown.svg", "Dataset breakdown: free-rollout top-5", labels, copy_overall, stwm_overall)

    manifest = {
        "audit_name": "stwm_final_figure_manifest",
        "figure_dir": str(FIG_DIR),
        "figures": [
            {"id": "motivation", "status": "planned", "description": "Trace has dynamics but lacks semantic identity; STWM adds semantic trace field output."},
            {"id": "method", "status": "generated_svg", "path": str(FIG_DIR / "figure_method_semantic_trace_world_model.svg")},
            {"id": "main_result_plot", "status": "generated_svg", "path": str(FIG_DIR / "figure_changed_subset_top5.svg")},
            {"id": "stable_changed_visualization", "status": "planned_from_eval_rows", "description": "Use item_scores to choose stable preserved and changed corrected examples."},
            {"id": "dataset_breakdown", "status": "generated_svg", "path": str(FIG_DIR / "figure_dataset_top5_breakdown.svg")},
            {"id": "qualitative_examples", "status": "planned", "required": "4 successes, 2 failures with trace + semantic field overlay."},
            {"id": "appendix_semantic_state_diagnostic", "status": "planned_optional", "description": "Use V7 negative semantic-state branch as limitation/diagnostic if space permits."},
        ],
        "no_candidate_scorer": True,
        "free_rollout": True,
    }
    write("reports/stwm_final_figure_manifest_20260428.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    lines = ["# STWM Final Figure Manifest", ""]
    for fig in manifest["figures"]:
        lines.append(f"- {fig['id']}: `{fig['status']}`" + (f" ({fig.get('path')})" if fig.get("path") else ""))
    write("docs/STWM_FINAL_FIGURE_MANIFEST_20260428.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
