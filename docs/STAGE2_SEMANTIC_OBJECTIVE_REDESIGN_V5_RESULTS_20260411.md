# Stage2 Semantic Objective Redesign V5 Results

- generated_at_utc: 2026-04-11T17:57:59.561838+00:00
- redesign_v5_status: 0_running_6_completed_0_failed

| run_name | combo | gate_family | gpu | batch | steps | status | best_endpoint_l2 | gate_ratio | valuable_pair_ratio | final_aux_weight | same_ckpt | sidecar_diverged |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| stage2_semobjv5_topk1_persistdelay_seed42_20260411 | hard_topk_query_gating_v2+high_value_sparse_persistence+conservative_delayed_aux_schedule | hard_topk_query_gating_v2 | 6 | 8 | 3360 | completed | 0.00113695 | 0.1250 | 0.0000 | 0.00007458 | False | True |
| stage2_semobjv5_topk2_persistdelay_seed42_20260411 | hard_topk_query_gating_v2_k2+high_value_sparse_persistence+conservative_delayed_aux_schedule | hard_topk_query_gating_v2 | 2 | 8 | 3360 | completed | 0.00113695 | 0.2500 | 1.0000 | 0.00007458 | False | True |
| stage2_semobjv5_qcap15_persistdelay_seed42_20260411 | capped_quantile_sparse_gating_v2_cap15+high_value_sparse_persistence+conservative_delayed_aux_schedule | capped_quantile_sparse_gating_v2 | 1 | 8 | 3360 | completed | 0.00113695 | 0.1250 | 0.0000 | 0.00007458 | False | True |
| stage2_semobjv5_qcap25_persistdelay_seed42_20260411 | capped_quantile_sparse_gating_v2_cap25+high_value_sparse_persistence+conservative_delayed_aux_schedule | capped_quantile_sparse_gating_v2 | 4 | 8 | 3360 | completed | 0.00113695 | 0.2500 | 1.0000 | 0.00007458 | False | True |
| stage2_semobjv5_topk1_persistdelay_seed123_20260411 | hard_topk_query_gating_v2+high_value_sparse_persistence+conservative_delayed_aux_schedule | hard_topk_query_gating_v2 | 6 | 8 | 8360 | completed | 0.00080269 | 0.1250 | 0.0000 | 0.00007458 | False | True |
| stage2_semobjv5_qcap15_persistdelay_seed123_20260411 | capped_quantile_sparse_gating_v2_cap15+high_value_sparse_persistence+conservative_delayed_aux_schedule | capped_quantile_sparse_gating_v2 | 2 | 8 | 8360 | completed | 0.00080269 | 0.1250 | 0.0000 | 0.00007458 | False | True |
