from __future__ import annotations

from pathlib import Path

DATE_TAG = "20260408"

DATA_ROOT = Path("/home/chen034/workspace/data")
WORK_ROOT = Path("/home/chen034/workspace/stwm")

TRACE_CACHE_ROOT = DATA_ROOT / "_cache" / "tracewm_stage1_v2"
TRACE_CONTRACT_PATH = DATA_ROOT / "_manifests" / f"stage1_v2_trace_cache_contract_{DATE_TAG}.json"
TRACE_AUDIT_REPORT_PATH = WORK_ROOT / "reports" / f"stage1_v2_trace_cache_audit_{DATE_TAG}.json"

FEATURE_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "visibility": 3,
    "vx": 4,
    "vy": 5,
    "rx": 6,
    "ry": 7,
}
STATE_DIM = 8

DATASET_ID_MAP = {
    "pointodyssey": 0,
    "kubric": 1,
}
