# STWM OSTF V30 Training Readiness Audit

- readiness_passed: `True`
- PointOdyssey_cache_path: `outputs/cache/stwm_ostf_v30_external_gt/pointodyssey`
- total_raw_cache_items: `6809`
- H32_H64_H96_split_counts: `{'H32_motion_test': 352, 'H64_motion_test': 450, 'H96_motion_test': 523}`
- M128_M512_M1024_availability: `{'M1024_H32': {'test': 238, 'train': 191, 'val': 113}, 'M1024_H64': {'test': 234, 'train': 189, 'val': 111}, 'M1024_H96': {'test': 232, 'train': 185, 'val': 107}, 'M128_H32': {'test': 241, 'train': 190, 'val': 117}, 'M128_H64': {'test': 236, 'train': 190, 'val': 114}, 'M128_H96': {'test': 234, 'train': 187, 'val': 110}, 'M512_H32': {'test': 239, 'train': 191, 'val': 117}, 'M512_H64': {'test': 236, 'train': 190, 'val': 112}, 'M512_H96': {'test': 235, 'train': 187, 'val': 110}}`
- two_d_fields_available: `True`
- three_d_fields_available: `False`
- obs_fut_visibility_fields_available: `True`
- train_val_test_video_level_leakage_check: `{'checked_video_uid_count': 1284, 'leakage_detected': False, 'leakage_examples': []}`
- external_GT_benchmark_main_ready: `True`
- existing_V28_incompatible: `True`
- V30_new_model_training_required: `True`
- exact_training_combos_recommended: `['M128_H32_seed42', 'M128_H64_seed42', 'M128_H32_wo_semantic_seed42', 'M128_H64_wo_semantic_seed42', 'M512_H32_seed42_optional', 'M512_H64_seed42_optional']`
