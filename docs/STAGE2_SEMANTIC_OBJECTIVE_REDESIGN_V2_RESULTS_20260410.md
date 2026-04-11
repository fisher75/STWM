# Stage2 Semantic Objective Redesign V2 Results

- generated_at_utc: 2026-04-10T18:36:58.923495+00:00
- redesign_v2_status: 0_running_4_completed_0_failed
- next_step_choice_internal: summarize_redesign_v2_after_completion

| run_name | combo | gpu | batch | steps | status | best_endpoint_l2 | latest_endpoint_l2 |
|---|---|---:|---:|---:|---|---:|---:|
| stage2_semobjv2_readoutalign_seed42_20260410 | readout_semantic_alignment_head | 0 | 8 | 3200 | completed | 0.00113695 | 0.00538051 |
| stage2_semobjv2_readoutpersist_seed42_20260410 | readout_semantic_alignment_head+persistence_contrastive_or_ranking_loss | 1 | 8 | 3200 | completed | 0.00113695 | 0.00537094 |
| stage2_semobjv2_readouthard_seed42_20260410 | readout_semantic_alignment_head+auxiliary_subset_weighting_only | 2 | 8 | 3200 | completed | 0.00113695 | 0.00538113 |
| stage2_semobjv2_readoutpersist_seed123_20260410 | readout_semantic_alignment_head+persistence_contrastive_or_ranking_loss | 4 | 8 | 8200 | completed | 0.00080269 | 0.00129164 |
