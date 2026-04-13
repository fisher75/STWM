# Stage2 Semantic Objective Redesign V7 Results

- generated_at_utc: 2026-04-13T07:35:21.343103+00:00
- redesign_v7_status: 0_running_8_completed_0_failed

| run_name | family | combo | gate_family | gpu | batch | steps | status | declared_persist | declared_but_inactive | best_endpoint_l2 | gate_ratio | guaranteed_pair_count | valuable_pair_ratio | strict_pair_ratio | fallback_trigger_rate | fallback_pair_ratio |
|---|---|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage2_semobjv7_alignonly_topk1_seed42_20260413 | calibration_only_family | v7_calibration_only_topk1_alignment | hard_topk_query_gating_v2 | 1 | 8 | 3360 | completed | False | False | 0.00113695 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stage2_semobjv7_alignonly_qcap15_seed42_20260413 | calibration_only_family | v7_calibration_only_qcap15_alignment | capped_quantile_sparse_gating_v2 | 5 | 8 | 3360 | completed | False | False | 0.00113695 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stage2_semobjv7_alignpersist_topk1_seed42_20260413 | calibration_plus_active_persistence_family | v7_calibration_plus_active_persistence_topk1 | hard_topk_query_gating_v2 | 2 | 8 | 3360 | completed | True | True | 0.00113695 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stage2_semobjv7_alignpersist_qcap15_seed42_20260413 | calibration_plus_active_persistence_family | v7_calibration_plus_active_persistence_qcap15 | capped_quantile_sparse_gating_v2 | 5 | 8 | 3360 | completed | True | True | 0.00113695 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stage2_semobjv7_alignonly_topk1_seed123_20260413 | calibration_only_family | v7_calibration_only_topk1_alignment | hard_topk_query_gating_v2 | 7 | 8 | 8360 | completed | False | False | 0.00080269 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stage2_semobjv7_alignonly_qcap15_seed123_20260413 | calibration_only_family | v7_calibration_only_qcap15_alignment | capped_quantile_sparse_gating_v2 | 3 | 8 | 8360 | completed | False | False | 0.00080269 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stage2_semobjv7_alignpersist_topk1_seed123_20260413 | calibration_plus_active_persistence_family | v7_calibration_plus_active_persistence_topk1 | hard_topk_query_gating_v2 | 4 | 8 | 8360 | completed | True | True | 0.00080269 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stage2_semobjv7_alignpersist_qcap15_seed123_20260413 | calibration_plus_active_persistence_family | v7_calibration_plus_active_persistence_qcap15 | capped_quantile_sparse_gating_v2 | 6 | 8 | 8360 | completed | True | True | 0.00080269 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
