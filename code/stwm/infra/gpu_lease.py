from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
import fcntl
import json
import os
import socket
import uuid

from stwm.tracewm_v2.constants import DATE_TAG, WORK_ROOT


DEFAULT_LEASE_PATH = WORK_ROOT / "reports" / f"stage1_v2_gpu_lease_{DATE_TAG}.json"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(ts: datetime) -> str:
    return ts.isoformat()


def _parse_iso(raw: str) -> datetime:
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return now_utc() - timedelta(days=3650)


@contextmanager
def _locked_json(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o664)
    try:
        with os.fdopen(fd, "r+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                content = fh.read().strip()
                payload = json.loads(content) if content else {"leases": []}
            except Exception:
                payload = {"leases": []}

            if not isinstance(payload, dict):
                payload = {"leases": []}
            if not isinstance(payload.get("leases", []), list):
                payload["leases"] = []

            yield payload

            fh.seek(0)
            fh.truncate(0)
            fh.write(json.dumps(payload, ensure_ascii=False, indent=2))
            fh.flush()
            os.fsync(fh.fileno())
    finally:
        pass


def _cleanup_expired(leases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now = now_utc()
    out: List[Dict[str, Any]] = []
    for lease in leases:
        if not isinstance(lease, dict):
            continue
        expires_at = _parse_iso(str(lease.get("expires_at_utc", "")))
        if expires_at <= now:
            continue
        out.append(lease)
    return out


def list_active_leases(lease_path: str | Path = DEFAULT_LEASE_PATH) -> List[Dict[str, Any]]:
    path = Path(lease_path)
    if not path.exists():
        return []
    with _locked_json(path) as payload:
        leases = _cleanup_expired(list(payload.get("leases", [])))
        payload["leases"] = leases
        return leases


def is_gpu_leased(gpu_id: int, lease_path: str | Path = DEFAULT_LEASE_PATH) -> bool:
    gpu_id = int(gpu_id)
    for lease in list_active_leases(lease_path=lease_path):
        if int(lease.get("gpu_id", -1)) == gpu_id:
            return True
    return False


def acquire_lease(
    gpu_id: int,
    owner: str,
    ttl_seconds: int = 6 * 3600,
    lease_path: str | Path = DEFAULT_LEASE_PATH,
    allow_shared: bool = False,
) -> Dict[str, Any]:
    path = Path(lease_path)
    now = now_utc()
    expires = now + timedelta(seconds=max(int(ttl_seconds), 60))

    with _locked_json(path) as payload:
        leases = _cleanup_expired(list(payload.get("leases", [])))
        if not bool(allow_shared):
            for lease in leases:
                if int(lease.get("gpu_id", -1)) == int(gpu_id):
                    raise RuntimeError(f"gpu {gpu_id} already leased")

        rec = {
            "lease_id": str(uuid.uuid4()),
            "gpu_id": int(gpu_id),
            "owner": str(owner),
            "host": str(socket.gethostname()),
            "pid": int(os.getpid()),
            "acquired_at_utc": _to_iso(now),
            "expires_at_utc": _to_iso(expires),
            "allow_shared": bool(allow_shared),
        }
        leases.append(rec)
        payload["leases"] = leases
        return rec


def release_lease(lease_id: str, lease_path: str | Path = DEFAULT_LEASE_PATH) -> bool:
    path = Path(lease_path)
    if not path.exists():
        return False

    released = False
    with _locked_json(path) as payload:
        leases = _cleanup_expired(list(payload.get("leases", [])))
        kept: List[Dict[str, Any]] = []
        for rec in leases:
            if str(rec.get("lease_id", "")) == str(lease_id):
                released = True
                continue
            kept.append(rec)
        payload["leases"] = kept

    return released
