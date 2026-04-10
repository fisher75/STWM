# Stage2 Semantic Objective Redesign V1 Results

- generated_at_utc: 2026-04-10T15:59:50.550953+00:00
- redesign_v1_status: 0_running_4_completed_0_failed
- next_step_choice_internal: summarize_redesign_v1_after_completion

| run_name | combo | gpu | batch | steps | status | best_endpoint_l2 | latest_endpoint_l2 |
|---|---|---:|---:|---:|---|---:|---:|
| stage2_semobjv1_align_seed42_20260410 | semantic_alignment_loss | 5 | 8 | 3300 | completed | 0.00113695 | 0.00378193 |
| stage2_semobjv1_alignpersist_seed42_20260410 | semantic_alignment_loss+query_persistence_consistency_loss | 1 | 8 | 3300 | completed | 0.00113695 | 0.00224524 |
| stage2_semobjv1_alignhard_seed42_20260410 | semantic_alignment_loss+semantic_hard_curriculum_or_weighting | 2 | 8 | 3300 | completed | 0.00113695 | 0.00202349 |
| stage2_semobjv1_alignpersist_seed123_20260410 | semantic_alignment_loss+query_persistence_consistency_loss | 4 | 8 | 8300 | completed | 0.00080269 | 0.00192716 |

## Diagnosis

- chosen_real_bootstrap_backend: local_clip_vit_b32_mask_crop_visual_teacher
- real_bootstrap_backend_usable: true
- cache_build_blocked: false
- true_new_best_not_warm_start_inherited: false
- semantic_hard_positive_signal: false
- improved_vs_current_cropenc_baseline: false
- narrowed_or_won_vs_legacysem: false
- full_validation_non_catastrophic: false
- best_v1_objective_combo: semantic_alignment_loss+query_persistence_consistency_loss
- best_v1_full_validation_endpoint_l2: 0.0018621101247760367
- current_cropenc_fullscale_mean_endpoint_l2: 0.0009458137729957648
- legacysem_fullscale_mean_endpoint_l2: 0.0007597727700389118
- next_step_choice: redesign_stage2_semantic_objective_v2
