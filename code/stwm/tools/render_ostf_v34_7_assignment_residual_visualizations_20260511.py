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

from stwm.modules.ostf_v34_7_assignment_bound_residual_memory import AssignmentBoundResidualMemoryV347
from stwm.tools.eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import _norm
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_7_assignment_oracle_residual_probe_20260511 import SUMMARY as TRAIN_SUMMARY, AssignmentAwareResidualDataset, collate_v347, compose


OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v34_7_assignment_residual"
REPORT = ROOT / "reports/stwm_ostf_v34_7_assignment_residual_visualization_manifest_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_7_ASSIGNMENT_RESIDUAL_VISUALIZATION_20260511.md"


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[AssignmentBoundResidualMemoryV347 | None, argparse.Namespace | None]:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    if not train.get("assignment_oracle_residual_probe_ran"):
        return None, None
    ck = torch.load(ROOT / train["checkpoint_path"], map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    model = AssignmentBoundResidualMemoryV347(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs


def render(path: Path, bd: dict[str, torch.Tensor], out: dict[str, torch.Tensor], final: torch.Tensor, shuffled: torch.Tensor, idx: tuple[int, int, int], name: str, reason: str) -> dict[str, Any]:
    bi, mi, hi = idx
    obs = bd["obs_points"][bi].detach().cpu().numpy()
    fut = out["point_pred"][bi, :, hi].detach().cpu().numpy()
    target = bd["fut_teacher_embedding"][bi, :, hi].detach().cpu().numpy()
    point = out["pointwise_semantic_belief"][bi, :, hi].detach().cpu().numpy()
    fin = final[bi, :, hi].detach().cpu().numpy()
    shuf = shuffled[bi, :, hi].detach().cpu().numpy()
    units = out["point_to_unit_assignment"][bi].argmax(dim=-1).detach().cpu().numpy()
    utility = bd["assignment_aware_residual_semantic_mask"][bi, :, hi].detach().cpu().numpy().astype(float)
    point_sim = (_norm(point) * _norm(target)).sum(axis=-1)
    final_sim = (_norm(fin) * _norm(target)).sum(axis=-1)
    shuf_sim = (_norm(shuf) * _norm(target)).sum(axis=-1)
    fig, ax = plt.subplots(1, 5, figsize=(18, 4))
    ax[0].scatter(obs[:, :, 0].reshape(-1), obs[:, :, 1].reshape(-1), c=np.repeat(units, obs.shape[1]), s=8, cmap="tab20")
    ax[0].scatter(obs[mi, :, 0], obs[mi, :, 1], c="red", s=20)
    ax[0].set_title("observed/unit")
    ax[1].scatter(fut[:, 0], fut[:, 1], c=point_sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[1].scatter(fut[mi, 0], fut[mi, 1], c="red", s=32)
    ax[1].set_title("pointwise")
    ax[2].scatter(fut[:, 0], fut[:, 1], c=final_sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[2].scatter(fut[mi, 0], fut[mi, 1], c="red", s=32)
    ax[2].set_title("assignment residual")
    ax[3].scatter(fut[:, 0], fut[:, 1], c=shuf_sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[3].scatter(fut[mi, 0], fut[mi, 1], c="red", s=32)
    ax[3].set_title("shuffled assignment")
    ax[4].scatter(fut[:, 0], fut[:, 1], c=utility, s=18, cmap="magma", vmin=0, vmax=1)
    ax[4].scatter(fut[mi, 0], fut[mi, 1], c="cyan", s=32)
    ax[4].set_title("assignment target")
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
    model, ckargs = load_model(args, device)
    if model is None or ckargs is None:
        payload = {"generated_at_utc": utc_now(), "real_images_rendered": False, "case_mining_used": False, "png_count": 0, "visualization_ready": False, "placeholder_only": False, "exact_blockers": ["assignment_oracle_train_not_run"]}
        dump_json(REPORT, payload)
        write_doc(DOC, "STWM OSTF V34.7 Assignment Residual Visualization", payload, ["real_images_rendered", "case_mining_used", "png_count", "visualization_ready", "exact_blockers"])
        print(REPORT.relative_to(ROOT))
        return 0
    ds = AssignmentAwareResidualDataset("test", ckargs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_v347)
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    specs = {
        "assignment_bound_residual_success": ("assign", True),
        "assignment_bound_residual_failure": ("assign", False),
        "shuffled_assignment_destroys_residual_correction": ("shuffle_delta", True),
        "assignment_not_useful_case": ("shuffle_delta", False),
        "semantic_hard_success": ("hard", True),
        "semantic_hard_failure": ("hard", False),
        "stable_preservation_success": ("stable_abs", False),
        "changed_success": ("changed", True),
        "changed_failure": ("changed", False),
        "m128_future_trace_assignment_overlay": ("assign", True),
    }
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"], intervention="force_gate_zero")
            shuf_out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"], intervention="permute_unit_assignment")
            final = compose(out, bd)
            shuffled = compose(shuf_out, bd)
            target = bd["fut_teacher_embedding"].detach().cpu().numpy()
            point = out["pointwise_semantic_belief"].detach().cpu().numpy()
            fin = final.detach().cpu().numpy()
            sh = shuffled.detach().cpu().numpy()
            gain = (_norm(fin) * _norm(target)).sum(axis=-1) - (_norm(point) * _norm(target)).sum(axis=-1)
            sh_gain = (_norm(sh) * _norm(target)).sum(axis=-1) - (_norm(point) * _norm(target)).sum(axis=-1)
            score_map = {
                "assign": gain,
                "shuffle_delta": gain - sh_gain,
                "hard": gain,
                "changed": gain,
                "stable_abs": np.abs(gain),
            }
            masks = {
                "assign": bd["assignment_aware_residual_semantic_mask"].detach().cpu().numpy().astype(bool),
                "hard": (bd["semantic_hard_mask"] if "semantic_hard_mask" in bd else bd["semantic_hard_train_mask"]).detach().cpu().numpy().astype(bool),
                "changed": bd["changed_mask"].detach().cpu().numpy().astype(bool),
                "stable_abs": bd["stable_suppress_mask"].detach().cpu().numpy().astype(bool),
                "shuffle_delta": bd["assignment_aware_residual_semantic_mask"].detach().cpu().numpy().astype(bool),
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
                    best[name] = (score, {"bd": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in bd.items()}, "out": {k: v.detach().cpu() for k, v in out.items() if torch.is_tensor(v)}, "final": final.detach().cpu(), "shuffled": shuffled.detach().cpu(), "idx": tuple(map(int, idxs[pos]))})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = []
    for name, (score, pack) in best.items():
        examples.append(render(OUT_DIR / f"{len(examples):02d}_{name}.png", pack["bd"], pack["out"], pack["final"], pack["shuffled"], pack["idx"], name, f"mined_score={score:.4f}"))
    payload = {"generated_at_utc": utc_now(), "real_images_rendered": bool(examples), "case_mining_used": True, "placeholder_only": False, "png_count": len(examples), "visualization_ready": bool(examples), "examples": examples}
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V34.7 Assignment Residual Visualization", payload, ["real_images_rendered", "case_mining_used", "placeholder_only", "png_count", "visualization_ready", "examples"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
