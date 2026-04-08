# TRACEWM Stage1 v2 Trace Cache Protocol (2026-04-08)

## 1. Purpose
Define canonical real-trace cache contract for Stage1 v2 datasets.

## 2. Cache Root
Default cache root:
- `/home/chen034/workspace/data/_cache/tracewm_stage1_v2`

## 3. Cache Unit
One cache file equals one clip sample.

Required metadata:
- `dataset` (string)
- `split` (string)
- `clip_id` (string)
- `source_ref` (string)
- `track_source` (string, must not contain deterministic synthetic marker)

Required tensors/arrays:
- `tracks_2d`: `[T,K,2]` float32
- `tracks_3d`: `[T,K,3]` float32
- `valid`: `[T,K]` bool
- `visibility`: `[T,K]` bool
- `point_ids`: `[K]` int64

Optional arrays:
- `intrinsics`: `[T,3,3]` float32 or empty
- `extrinsics`: `[T,4,4]` float32 or empty

## 4. Dataset-Specific Sources
PointOdyssey:
- source files: `anno.npz` (+ optional `scene_info.json` and RGB dimensions).
- real fields: `trajs_2d`, `trajs_3d`, `valids`, `visibs`.

Kubric:
- source file: `metadata.json`.
- real fields from instances:
  - `image_positions` (2D trajectory)
  - `positions` (3D trajectory)
  - `visibility` (pixel visibility count)

## 5. Validation Rules
1. Shape consistency across all required fields.
2. Finite ratio check for coordinates.
3. Non-empty token count (`K >= 1`).
4. Temporal length check (`T == obs_len + fut_len`).
5. Boolean masks are aligned with `tracks_2d`/`tracks_3d` dimensions.

## 6. Contract Manifest
Manifest path:
- `/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json`

Manifest must include:
- generation timestamp
- schema version
- feature layout description
- per-dataset index path
- per-split sample counts
- audit status

## 7. Audit Report
Audit path:
- `/home/chen034/workspace/stwm/reports/stage1_v2_trace_cache_audit_20260408.json`

Audit must include:
- sampled files
- pass/fail by dataset
- finite ratio summary
- missing-field summary
- deterministic synthetic marker scan result
