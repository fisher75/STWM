# Stage2 Semantic Hard Subsets

- generated_at_utc: 2026-04-10T08:33:03.992295+00:00

## occlusion_reappearance
- build_rule: top area_range clips, using sampled mask-derived box area range as occlusion/reappearance proxy
- clip_count: 24
- per_dataset_count: {'VSPW': 24}
- exact_source_fields_used: ['obs_state[...,6:8]', 'fut_state[...,6:8]', 'mask_paths via dataset box extraction']

## crossing_or_interaction_ambiguity
- build_rule: top center_interaction score, using high trajectory motion near image center as ambiguity proxy
- clip_count: 24
- per_dataset_count: {'VSPW': 24}
- exact_source_fields_used: ['obs_state[...,0:2]', 'fut_state[...,0:2]']

## small_object_or_low_area
- build_rule: lowest average normalized mask/box area
- clip_count: 24
- per_dataset_count: {'VSPW': 24}
- exact_source_fields_used: ['obs_state[...,6:8]', 'fut_state[...,6:8]']

## appearance_change_or_semantic_shift
- build_rule: top color_std_mean + area_range proxy over object/mask crop statistics
- clip_count: 24
- per_dataset_count: {'VSPW': 24}
- exact_source_fields_used: ['semantic_features[3:6]', 'obs_state[...,6:8]', 'fut_state[...,6:8]']

## burst_persistence_stress
- build_rule: top frame_count BURST validation clips as optional persistence/long-gap stress panel
- clip_count: 24
- per_dataset_count: {'BURST': 24}
- exact_source_fields_used: ['meta.frame_count_total', 'BURST val frame paths']
