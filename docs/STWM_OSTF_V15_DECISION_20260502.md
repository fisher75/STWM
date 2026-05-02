# STWM OSTF V15 Decision

- object_dense_trace_field_claim_allowed: `False`
- hierarchical_semantic_trace_unit_story_supported: `partial_cache_foundation_only_not_model_claim`
- physical_point_teacher_used: `False`
- next_step_choice: `fix_trace_teacher_or_cache`

## Point Counts
- objects_per_scene_mean: `6.392369020501139`
- points_per_object: `{'M1': 1, 'M128': 128, 'M512': 512}`
- points_per_scene_mean: `{'M1': 6.392369020501139, 'M128': 818.2232346241458, 'M512': 3272.892938496583}`
- total_points: `{'M1': 11225, 'M128': 1436800, 'M512': 5747200}`

## Result
- M128_beats_M1: `False`
- M512_beats_M1: `False`
- boundary: Phase-1 created object-internal pseudo point trace caches and visualization, but physical teacher/GT is not wired and M128/M512 pilot did not beat M1.
