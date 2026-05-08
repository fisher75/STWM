#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SETPROCTITLE_STATUS: dict[str, object] = {
    "requested_title": "python",
    "setproctitle_ok": False,
    "exact_error": None,
}
try:
    import setproctitle  # type: ignore

    setproctitle.setproctitle("python")
    SETPROCTITLE_STATUS["setproctitle_ok"] = True
except Exception as exc:  # pragma: no cover - environment dependent.
    SETPROCTITLE_STATUS["exact_error"] = f"{type(exc).__name__}: {exc}"

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_external_gt_v30_20260508 import build_model, evaluate, make_loader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--point-dim", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--learned-modes", type=int, default=4)
    p.add_argument("--damped-gamma", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-eval-items", type=int, default=None)
    p.add_argument("--wo-semantic", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--point-dropout", type=float, default=0.0)
    p.add_argument("--density-aware-pooling", choices=("none", "valid_weighted", "local_attention"), default="none")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(args).to(device)
    ckpt = torch.load(ROOT / args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loader = make_loader("test", args, shuffle=False, max_items=args.max_eval_items)
    metrics, rows = evaluate(model, test_loader, device)
    payload = {
        "eval_name": args.experiment_name,
        "generated_at_utc": utc_now(),
        "checkpoint": args.checkpoint,
        "setproctitle_status": SETPROCTITLE_STATUS,
        "test_metrics": metrics,
        "test_item_rows": rows,
    }
    path = ROOT / f"reports/stwm_ostf_v30_external_gt_runs/{args.experiment_name}_eval.json"
    dump_json(path, payload)
    write_doc(
        ROOT / f"docs/STWM_OSTF_V30_EXTERNAL_GT_EVAL_{args.experiment_name}.md",
        f"STWM OSTF V30 Eval {args.experiment_name}",
        payload,
        ["checkpoint", "test_metrics"],
    )
    print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
