# Stage2 Semantic Objective Redesign V3 Results

- generated_at_utc: 2026-04-11T05:52:59.025103+00:00
- redesign_v3_status: 0_running_5_completed_0_failed

| run_name | combo | gpu | batch | steps | status | best_endpoint_l2 | hard_sidecar | conf_gate_ratio |
|---|---|---:|---:|---:|---|---:|---:|---:|
| stage2_semobjv3_confalign_seed42_20260410 | confidence_gated_readout_alignment | 5 | 8 | 3300 | completed | 0.00113695 | 1000000000.00000000 | 1.0000 |
| stage2_semobjv3_confpersist_seed42_20260410 | confidence_gated_readout_alignment+sparse_persistence_contrastive_loss | 7 | 8 | 3300 | completed | 0.00113695 | 1000000000.00000000 | 1.0000 |
| stage2_semobjv3_confpersistdelay_seed42_20260410 | confidence_gated_readout_alignment+sparse_persistence_contrastive_loss+delayed_aux_loss_schedule | 4 | 8 | 3300 | completed | 0.00113695 | 1000000000.00000000 | 1.0000 |
| stage2_semobjv3_confpersistdelay_seed123_20260410 | confidence_gated_readout_alignment+sparse_persistence_contrastive_loss+delayed_aux_loss_schedule | 2 | 8 | 8300 | completed | 0.00080269 | 1000000000.00000000 | 1.0000 |
| stage2_semobjv3_confhardsidecar_seed42_20260410 | confidence_gated_readout_alignment+sparse_persistence_contrastive_loss+delayed_aux_loss_schedule+semantic_hard_best_sidecar_selection | 3 | 8 | 3300 | completed | 0.00113695 | 0.00110980 | 1.0000 |
