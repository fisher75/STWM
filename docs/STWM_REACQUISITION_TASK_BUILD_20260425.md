# STWM Reacquisition Task Build 20260425

Source: `/home/chen034/workspace/stwm/reports/stwm_trace_belief_eval_20260424.json`

Task rows are filtered to items with `occlusion_reappearance` or `long_gap_persistence` tags. Split uses stable SHA256 over `protocol_item_id`.

| split | item_count | hash |
|---|---:|---|
| train | 15 | `3731177d12ebc7039604c4ae5f46a3d1b2ef2f56d876aa88ded3bb56ef9f1350` |
| val | 21 | `480f42bdb714ad2e082b29629504e9288a931b668064db4b62982f6a2c8b75de` |
| test | 91 | `e41ab7b55bc0f5506241372d625cea35d117cb10fc9a1e0c9ed56ae73d046710` |


- total unique item_count: `127`
- leakage_check_passed: `True`
- candidate_count_stats: `{'count': 1440, 'mean': 6.304166666666666, 'min': 2.0, 'max': 8.0, 'std': 1.9819979748279821}`
- gap_length_stats: `{'available': False, 'exact_missing_reason': 'No numeric gap length field exists in per-item rows.', 'proxy_breakdown_by_subset_tag': {'occlusion_and_long_gap': 678, 'occlusion_only': 762}}`
- stage1 frozen availability: `No matching per-item reacquisition rows in stwm_trace_belief_eval_20260424.json for this method/scoring_mode.`
