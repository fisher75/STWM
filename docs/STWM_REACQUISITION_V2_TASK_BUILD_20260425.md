# STWM Reacquisition V2 Task Build 20260425

Task: identify the same future candidate after occlusion/long-gap, with confuser-controlled breakdowns.

| group | item_count |
|---|---:|
| all_reacquisition_items | 240 |
| occlusion_reappearance | 240 |
| long_gap_persistence | 113 |
| occlusion_and_long_gap | 113 |
| appearance_similar_confuser | 134 |
| spatially_close_confuser | 170 |
| crossing_or_overlap_confuser | 111 |
| OOD_confuser | 136 |

- leakage_check_passed = `True`
- candidate_count_stats = `{'count': 1440, 'mean': 6.304166666666666, 'min': 2.0, 'max': 8.0, 'std': 1.9819979748279821}`
- gap_length_stats = `{'available': False, 'exact_missing_reason': 'No numeric gap length field exists in source per-item rows.', 'proxy_by_subset_tag': {'occlusion_reappearance': 240, 'long_gap_persistence': 113, 'occlusion_and_long_gap': 113}}`
