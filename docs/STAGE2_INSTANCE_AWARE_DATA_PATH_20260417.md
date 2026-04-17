# Stage2 Instance-Aware Data Path 20260417

- generated_at_utc: 2026-04-17T13:47:06.229937+00:00
- VIPSeg: true instance-aware supervision path preserved
- VSPW: fallback only; no fake high-quality instance ids are invented
- BURST: true instance continuity available for protocol/eval, not pulled back into main training binding
- predecode_cache_compatible_with_instance_aware_fields: False
- predecode_cache_blocking_reason: existing 20260416 predecode cache was built before TUSB instance-aware fields; dataset now falls back to raw decode when cached payload misses semantic_instance_id_* or semantic_objectness_score
- training_priority: use approved existing binding, but instance-aware unit supervision should exploit real continuity whenever present
