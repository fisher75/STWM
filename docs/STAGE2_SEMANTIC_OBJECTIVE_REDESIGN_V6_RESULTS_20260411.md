# Stage2 Semantic Objective Redesign V6 Results

- generated_at_utc: 2026-04-13T03:05:26.574447+00:00
- redesign_v6_status: 0_running_6_completed_0_failed

| run_name | combo | gate_family | gpu | batch | steps | status | best_endpoint_l2 | gate_ratio | guaranteed_pair_count | strict_pair_ratio | fallback_trigger_rate | fallback_pair_ratio | same_ckpt | sidecar_diverged |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| stage2_semobjv6_topk1_g2_fb1_seed42_20260411 | hard_topk_query_gating_v6_k1+guaranteed_sparse_persistence_g2+two_level_pair_mining+conservative_delayed_aux_schedule | hard_topk_query_gating_v2 | 5 | 8 | 3360 | completed | 0.00113695 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True |
| stage2_semobjv6_topk2_g2_fb1_seed42_20260411 | hard_topk_query_gating_v6_k2+guaranteed_sparse_persistence_g2+two_level_pair_mining+conservative_delayed_aux_schedule | hard_topk_query_gating_v2 | 1 | 8 | 3360 | completed | 0.00113695 | 0.2500 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | False | True |
| stage2_semobjv6_qcap15_g2_fb1_seed42_20260411 | capped_quantile_sparse_gating_v6_cap15+guaranteed_sparse_persistence_g2+two_level_pair_mining+conservative_delayed_aux_schedule | capped_quantile_sparse_gating_v2 | 3 | 8 | 3360 | completed | 0.00113695 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True |
| stage2_semobjv6_qcap25_g2_fb1_seed42_20260411 | capped_quantile_sparse_gating_v6_cap25+guaranteed_sparse_persistence_g2+two_level_pair_mining+conservative_delayed_aux_schedule | capped_quantile_sparse_gating_v2 | 6 | 8 | 3360 | completed | 0.00113695 | 0.2500 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | False | True |
| stage2_semobjv6_topk1_g2_fb1_seed123_20260411 | hard_topk_query_gating_v6_k1+guaranteed_sparse_persistence_g2+two_level_pair_mining+conservative_delayed_aux_schedule | hard_topk_query_gating_v2 | 2 | 8 | 8360 | completed | 0.00080269 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True |
| stage2_semobjv6_qcap15_g2_fb0_seed123_20260411 | capped_quantile_sparse_gating_v6_cap15+guaranteed_sparse_persistence_g2+single_level_pair_mining+conservative_delayed_aux_schedule | capped_quantile_sparse_gating_v2 | 4 | 8 | 8360 | completed | 0.00080269 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True |
