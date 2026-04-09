#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict
import json
import os
import subprocess
import sys

import numpy as np


DEFAULT_TAPNET_PYTHON = "/home/chen034/workspace/data/.venv_tapvid3d_repair_20260406_py310/bin/python"


def _default_tapnet_python() -> str:
    env_val = str(os.environ.get("TRACEWM_TAPNET_PYTHON", "")).strip()
    if env_val:
        return env_val
    return DEFAULT_TAPNET_PYTHON


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _tail(s: str, limit: int = 4000) -> str:
    txt = str(s)
    if len(txt) <= limit:
        return txt
    return txt[-limit:]


def run_official_tapvid_eval(
    *,
    tap_payload_npz: str | Path,
    output_json: str | Path | None = None,
    tapnet_python: str = DEFAULT_TAPNET_PYTHON,
    query_mode: str = "first",
) -> Dict[str, Any]:
    payload_path = Path(tap_payload_npz)
    if not payload_path.exists():
        raise FileNotFoundError(f"tap payload not found: {payload_path}")

    probe_code = r"""
import json
import numpy as np
import sys
from tapnet.tapvid import evaluation_datasets as eval_ds

arr = np.load(sys.argv[1], allow_pickle=False)
metrics = eval_ds.compute_tapvid_metrics(
    query_points=arr["query_points"],
    gt_occluded=arr["gt_occluded"],
    gt_tracks=arr["gt_tracks"],
    pred_occluded=arr["pred_occluded"],
    pred_tracks=arr["pred_tracks"],
    query_mode=sys.argv[2],
)
out = {
    "official_tapvid_evaluator_connected": True,
    "tapvid_module_path": str(eval_ds.__file__),
    "query_mode": str(sys.argv[2]),
    "metric_means": {k: float(np.mean(np.asarray(v))) for k, v in metrics.items()},
    "metric_shapes": {k: list(np.asarray(v).shape) for k, v in metrics.items()},
}
print(json.dumps(out, ensure_ascii=True))
"""

    proc = subprocess.run(
        [str(tapnet_python), "-c", probe_code, str(payload_path), str(query_mode)],
        text=True,
        capture_output=True,
    )

    result: Dict[str, Any] = {
        "tap_payload_npz": str(payload_path),
        "tapnet_python": str(tapnet_python),
        "query_mode": str(query_mode),
        "returncode": int(proc.returncode),
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
        "official_evaluator_invoked": bool(proc.returncode == 0),
        "official_tapvid_evaluator_connected": False,
    }

    if proc.returncode == 0:
        stdout = str(proc.stdout).strip()
        parsed = json.loads(stdout)
        if not isinstance(parsed, dict):
            raise RuntimeError("official TAP eval stdout did not decode to dict")
        result.update(parsed)
    else:
        result["exact_blocking_reason"] = "official TAP-Vid evaluator subprocess failed"
        result["exact_missing_component"] = "working official tapvid metric runtime invocation"

    if output_json:
        _write_json(output_json, result)
    return result


def parse_args() -> Any:
    p = ArgumentParser(description="Run the official TAP-Vid metric on an exported Stage2 TAP payload")
    p.add_argument("--tap-payload-npz", required=True)
    p.add_argument("--output-json", default="")
    p.add_argument("--tapnet-python", default=_default_tapnet_python())
    p.add_argument("--query-mode", default="first")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_official_tapvid_eval(
        tap_payload_npz=args.tap_payload_npz,
        output_json=args.output_json or None,
        tapnet_python=str(args.tapnet_python),
        query_mode=str(args.query_mode),
    )
    json.dump(result, sys.stdout, ensure_ascii=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
