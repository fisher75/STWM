# STWM Object-Dense Trace Teacher Audit V15

## TraceAnything
- teacher_available: `True`
- checkpoint_path: ``
- expected_point_count: `interactive/sparse-to-dense depending adapter; no audited local checkpoint in this repo`
- gpu_memory_estimate: `unknown; must audit before full use`
- speed_estimate: `unknown`
- license_status: `third_party source present; official checkpoint/license not audited for OSTF target generation`
- exact_blocker: ``

## CoTracker_or_CoTracker3
- teacher_available: `True`
- checkpoint_path: `baselines/checkpoints/cotracker/scaled_offline.pth`
- expected_point_count: `128-2048 query points feasible per object after adapter work`
- gpu_memory_estimate: `likely B200-feasible for batched clips; exact memory not profiled in V15`
- speed_estimate: `adapter exists for external baseline; OSTF mask-grid target generation not yet wired`
- license_status: `external baseline used previously; OSTF teacher use needs separate protocol note`
- exact_blocker: ``

## TAPIR
- teacher_available: `False`
- checkpoint_path: ``
- expected_point_count: `not available locally`
- gpu_memory_estimate: `not audited`
- speed_estimate: `not audited`
- license_status: `not audited`
- exact_blocker: `TAPIR repo/checkpoint not present locally`

## PointOdyssey_GT
- teacher_available: `False`
- checkpoint_path: `GT trajectories if dataset exists`
- expected_point_count: `dense GT possible only if local PointOdyssey data exists`
- gpu_memory_estimate: `CPU data read`
- speed_estimate: `not applicable`
- license_status: `dataset not found locally`
- exact_blocker: `PointOdyssey data not present under data/external or data/raw`

## internal_stage1_or_stage2_trace_cache
- teacher_available: `True`
- checkpoint_path: `predecoded entity boxes/masks, not object-internal physical point tracks`
- expected_point_count: `M1/M128/M512 pseudo point targets can be derived from masks/bboxes`
- gpu_memory_estimate: `CPU materialization; no teacher GPU required`
- speed_estimate: `minutes for mixed split depending M`
- license_status: `derived from VSPW/VIPSeg local masks`
- exact_blocker: `not a true physical point tracker; must label as mask_bbox_relative_pseudo_track`

## Decision
- physical_point_teacher_ready_for_full_OSTF: `True`
- true_GT_dense_trajectory_dataset_available: `False`
- phase1_cache_source: `mask_bbox_relative_pseudo_track from internal Stage2 predecode masks/boxes`
- can_claim_physical_dense_trace_GT: `False`
- can_build_object_internal_point_supervision_now: `True`
- required_next_teacher_step: `wire CoTracker/TraceAnything teacher if physical long-range point supervision is required`
