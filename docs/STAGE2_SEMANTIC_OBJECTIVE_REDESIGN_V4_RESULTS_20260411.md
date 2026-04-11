# Stage2 Semantic Objective Redesign V4 Results

- generated_at_utc: 2026-04-11T07:22:51.687480+00:00
- redesign_v4_status: 0_running_6_completed_0_failed

| run_name | combo | gate_family | gpu | batch | steps | status | best_endpoint_l2 | hard_sidecar | conf_gate_ratio | sparse_gate_ratio | high_value_pair_ratio |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| stage2_semobjv4_quantile_align_seed42_20260411 | quantile_sparse_gating+confidence_gated_readout_alignment | quantile_sparse_gating | 5 | 8 | 3300 | completed | 0.00113695 | 0.00112013 | 1.0000 | 1.0000 | 0.2500 |
| stage2_semobjv4_topk_align_seed42_20260411 | topk_query_gating+confidence_gated_readout_alignment | topk_query_gating | 1 | 8 | 3300 | completed | 0.00113695 | 0.00111903 | 1.0000 | 1.0000 | 0.2500 |
| stage2_semobjv4_quantile_persistdelay_seed42_20260411 | quantile_sparse_gating+high_value_sparse_persistence+strong_delayed_aux_schedule | quantile_sparse_gating | 3 | 8 | 3300 | completed | 0.00113695 | 0.00111468 | 1.0000 | 1.0000 | 0.2500 |
| stage2_semobjv4_quantile_persistdelay_seed123_20260411 | quantile_sparse_gating+high_value_sparse_persistence+strong_delayed_aux_schedule | quantile_sparse_gating | 2 | 8 | 8300 | completed | 0.00080269 | 0.00063077 | 1.0000 | 1.0000 | 0.2512 |
| stage2_semobjv4_topk_persistdelay_seed42_20260411 | topk_query_gating+high_value_sparse_persistence+strong_delayed_aux_schedule | topk_query_gating | 4 | 8 | 3300 | completed | 0.00113695 | 0.00111550 | 1.0000 | 1.0000 | 0.2500 |
| stage2_semobjv4_topk_persistdelay_seed123_20260411 | topk_query_gating+high_value_sparse_persistence+strong_delayed_aux_schedule | topk_query_gating | 7 | 8 | 8300 | completed | 0.00080269 | 0.00063748 | 1.0000 | 1.0000 | 0.2512 |
