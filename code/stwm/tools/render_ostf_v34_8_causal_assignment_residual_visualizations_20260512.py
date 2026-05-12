#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import setproctitle
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_8_causal_assignment_bound_residual_memory import CausalAssignmentBoundResidualMemoryV348
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_8_causal_assignment_oracle_residual_probe_20260512 import (
    SUMMARY as TRAIN_SUMMARY,
    CausalAssignmentResidualDataset,
    collate_v348,
    compose,
)


OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v34_8_causal_assignment_residual"
REPORT = ROOT / "reports/stwm_ostf_v34_8_causal_assignment_residual_visualization_manifest_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_8_CAUSAL_ASSIGNMENT_RESIDUAL_VISUALIZATION_20260512.md"


def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.nan_to_num(x / np.maximum(n, 1e-8))


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[CausalAssignmentBoundResidualMemoryV348 | None, argparse.Namespace | None, dict[str, Any]]:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    if not train.get("oracle_residual_probe_ran"):
        return None, None, train
    ck = torch.load(ROOT / train["checkpoint_path"], map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    model = CausalAssignmentBoundResidualMemoryV348(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs, train


def render_case(path: Path, pack: dict[str, Any], name: str, reason: str) -> dict[str, Any]:
    bd = pack["bd"]
    out = pack["out"]
    final = pack["final"]
    shuf = pack["shuffled"]
    zero_sem = pack["zero_sem"]
    bi, mi, hi = pack["idx"]
    obs = bd["obs_points"][bi].numpy()
    fut = out["point_pred"][bi, :, hi].numpy()
    target = bd["fut_teacher_embedding"][bi, :, hi].numpy()
    point = out["pointwise_semantic_belief"][bi, :, hi].numpy()
    fin = final[bi, :, hi].numpy()
    sh = shuf[bi, :, hi].numpy()
    zs = zero_sem[bi, :, hi].numpy()
    units = out["point_to_unit_assignment"][bi].argmax(dim=-1).numpy()
    causal = bd["causal_assignment_residual_semantic_mask"][bi, :, hi].numpy().astype(float)
    sims = {
        "pointwise": (_norm(point) * _norm(target)).sum(axis=-1),
        "causal": (_norm(fin) * _norm(target)).sum(axis=-1),
        "shuffle_assign": (_norm(sh) * _norm(target)).sum(axis=-1),
        "zero_sem": (_norm(zs) * _norm(target)).sum(axis=-1),
    }
    fig, ax = plt.subplots(1, 6, figsize=(21, 4))
    ax[0].scatter(obs[:, :, 0].reshape(-1), obs[:, :, 1].reshape(-1), c=np.repeat(units, obs.shape[1]), s=8, cmap="tab20")
    ax[0].scatter(obs[mi, :, 0], obs[mi, :, 1], c="red", s=20)
    ax[0].set_title("obs/unit")
    for i, key in enumerate(["pointwise", "causal", "shuffle_assign", "zero_sem"], start=1):
        ax[i].scatter(fut[:, 0], fut[:, 1], c=sims[key], s=16, cmap="viridis", vmin=-1, vmax=1)
        ax[i].scatter(fut[mi, 0], fut[mi, 1], c="red", s=32)
        ax[i].set_title(key)
    ax[5].scatter(fut[:, 0], fut[:, 1], c=causal, s=18, cmap="magma", vmin=0, vmax=1)
    ax[5].scatter(fut[mi, 0], fut[mi, 1], c="cyan", s=32)
    ax[5].set_title("causal target")
    for a in ax:
        a.invert_yaxis()
        a.set_aspect("equal", adjustable="box")
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{name}: {reason}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"category": name, "path": str(path.relative_to(ROOT)), "case_selection_reason": reason}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, train = load_model(args, device)
    if model is None or ckargs is None:
        payload = {"generated_at_utc": utc_now(), "中文结论": "V34.8 oracle residual probe 未训练，无法生成真实可视化。", "real_images_rendered": False, "case_mining_used": False, "png_count": 0, "placeholder_only": False, "visualization_ready": False, "exact_blockers": [train.get("skip_reason", "train_not_run")]}
        dump_json(REPORT, payload)
        write_doc(DOC, "V34.8 causal assignment residual 可视化中文报告", payload, ["中文结论", "real_images_rendered", "case_mining_used", "png_count", "visualization_ready", "exact_blockers"])
        print(f"已写出可视化跳过报告: {REPORT.relative_to(ROOT)}")
        return 0
    ds = CausalAssignmentResidualDataset("test", ckargs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_v348)
    specs = {
        "assignment_load_bearing_success": ("shuffle_delta", True),
        "assignment_not_load_bearing_failure": ("shuffle_delta", False),
        "semantic_measurement_load_bearing_success": ("sem_delta", True),
        "semantic_measurement_failure": ("sem_delta", False),
        "shuffled_assignment_destroys_correction": ("shuffle_delta", True),
        "zero_semantic_measurement_destroys_correction": ("sem_delta", True),
        "semantic_hard_success": ("hard_gain", True),
        "semantic_hard_failure": ("hard_gain", False),
        "stable_preservation_success": ("stable_abs", False),
        "m128_future_trace_causal_assignment_overlay": ("causal_gain", True),
    }
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            inputs = {
                "obs_points": bd["obs_points"],
                "obs_vis": bd["obs_vis"],
                "obs_conf": bd["obs_conf"],
                "obs_semantic_measurements": bd["obs_semantic_measurements"],
                "obs_semantic_measurement_mask": bd["obs_semantic_measurement_mask"],
                "semantic_id": bd["semantic_id"],
            }
            out = model(**inputs, intervention="force_gate_zero")
            shuf_out = model(**inputs, intervention="shuffle_assignment")
            zero_sem_out = model(**inputs, intervention="zero_semantic_measurements")
            final = compose(out, bd)
            shuffled = compose(shuf_out, bd)
            zero_sem = compose(zero_sem_out, bd)
            target = bd["fut_teacher_embedding"].cpu().numpy()
            point = out["pointwise_semantic_belief"].cpu().numpy()
            fin = final.cpu().numpy()
            sh = shuffled.cpu().numpy()
            zs = zero_sem.cpu().numpy()
            base_gain = (_norm(fin) * _norm(target)).sum(axis=-1) - (_norm(point) * _norm(target)).sum(axis=-1)
            sh_gain = (_norm(sh) * _norm(target)).sum(axis=-1) - (_norm(point) * _norm(target)).sum(axis=-1)
            zs_gain = (_norm(zs) * _norm(target)).sum(axis=-1) - (_norm(point) * _norm(target)).sum(axis=-1)
            score_map = {
                "causal_gain": base_gain,
                "shuffle_delta": base_gain - sh_gain,
                "sem_delta": base_gain - zs_gain,
                "hard_gain": base_gain,
                "stable_abs": np.abs(base_gain),
            }
            masks = {
                "causal_gain": bd["causal_assignment_residual_semantic_mask"].cpu().numpy().astype(bool),
                "shuffle_delta": bd["causal_assignment_residual_semantic_mask"].cpu().numpy().astype(bool),
                "sem_delta": bd["causal_assignment_residual_semantic_mask"].cpu().numpy().astype(bool),
                "hard_gain": bd["semantic_hard_mask"].cpu().numpy().astype(bool),
                "stable_abs": bd["stable_suppress_mask"].cpu().numpy().astype(bool),
            }
            cpu_pack = {
                "bd": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in bd.items()},
                "out": {k: v.detach().cpu() for k, v in out.items() if torch.is_tensor(v)},
                "final": final.detach().cpu(),
                "shuffled": shuffled.detach().cpu(),
                "zero_sem": zero_sem.detach().cpu(),
            }
            for name, (kind, high) in specs.items():
                idxs = np.argwhere(masks[kind])
                if idxs.size == 0:
                    continue
                vals = score_map[kind][tuple(idxs.T)]
                pos = int(np.nanargmax(vals) if high else np.nanargmin(vals))
                score = float(vals[pos])
                cur = best.get(name)
                if cur is None or (score > cur[0] if high else score < cur[0]):
                    best[name] = (score, {**cpu_pack, "idx": tuple(map(int, idxs[pos]))})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = []
    for name, (score, pack) in best.items():
        examples.append(render_case(OUT_DIR / f"{len(examples):02d}_{name}.png", pack, name, f"score={score:.4f}"))
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.8 可视化从 eval batch 中按 intervention delta / hard / stable 指标挖掘真实 case。",
        "real_images_rendered": bool(examples),
        "case_mining_used": True,
        "placeholder_only": False,
        "png_count": len(examples),
        "visualization_ready": bool(examples),
        "examples": examples,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "V34.8 causal assignment residual 可视化中文报告", payload, ["中文结论", "real_images_rendered", "case_mining_used", "placeholder_only", "png_count", "visualization_ready", "examples"])
    print(f"已写出 V34.8 可视化 manifest: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
