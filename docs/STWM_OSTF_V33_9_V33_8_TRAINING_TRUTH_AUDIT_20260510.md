# STWM OSTF V33.9 V33.8 Training Truth Audit

- v33_8_training_not_fresh: `True`
- v33_8_fresh_training_proven: `False`
- skipped_existing_candidate_count: `5`
- candidate_checkpoint_roots: `{'v33_8_v33_6_global_contrastive_baseline_seed42': 'outputs/checkpoints/stwm_ostf_v33_6_identity_contrastive_repair', 'v33_8_v33_7_full_identity_belief_seed42': 'outputs/checkpoints/stwm_ostf_v33_7_identity_belief_calibration', 'v33_8_v33_7_no_fused_logits_seed42': 'outputs/checkpoints/stwm_ostf_v33_7_identity_belief_calibration', 'v33_8_v33_7_no_hard_bce_seed42': 'outputs/checkpoints/stwm_ostf_v33_7_identity_belief_calibration', 'v33_8_v33_7_no_embedding_similarity_seed42': 'outputs/checkpoints/stwm_ostf_v33_7_identity_belief_calibration'}`
- exact_risk: `V33.8 train summary shows all candidates skipped_existing=true, so current V33.8 metrics are expanded evaluation on pre-existing V33.6/V33.7 checkpoints, not fresh expanded-coverage training.`
- recommended_fix: `Run V33.9 fresh retrain into outputs/checkpoints/stwm_ostf_v33_9_fresh_expanded_h32_m128 with skip-existing disabled and checkpoint args proving V33.8 complete target roots.`
