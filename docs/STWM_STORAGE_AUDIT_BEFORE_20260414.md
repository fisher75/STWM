# STWM Storage Audit Before 20260414

- root: `/raid/chen034/workspace/stwm`
- total: 674.48 GB
- active_process_detected: True
- active_run_names: stage2_final_evidence_closure_20260414

## Top-Level Sizes

| path | size |
|---|---:|
| `data` | 623.71 GB |
| `outputs` | 38.74 GB |
| `models` | 10.52 GB |
| `third_party` | 718.85 MB |
| `logs` | 498.14 MB |
| `tmp` | 144.09 MB |
| `manifests` | 104.37 MB |
| `.git` | 28.90 MB |
| `yolov8s-worldv2.pt` | 24.72 MB |
| `reports` | 9.34 MB |
| `code` | 5.68 MB |
| `handover` | 3.23 MB |
| `docs` | 744.52 KB |
| `scripts` | 636.49 KB |
| `env` | 6.18 KB |
| `.gitignore` | 2.35 KB |
| `.pytest_cache` | 917.00 B |
| `README.md` | 698.00 B |
| `.gitattributes` | 205.00 B |

## Safety Classification

- safe_delete: Python/editor/system/tmp caches outside protected top-level paths; tmp/bak/swp older than 3 days
- safe_compress: old logs/jsonl/out/err only when no active run is detected
- safe_move_to_quarantine: intermediate checkpoints and old output queues only when no active run is detected
- review_required: large protected assets, unclear files over 1GB, any uncertain checkpoint
- protected_never_delete: ['code', 'configs', 'data', 'docs', 'manifests', 'models', 'reports', 'third_party']
