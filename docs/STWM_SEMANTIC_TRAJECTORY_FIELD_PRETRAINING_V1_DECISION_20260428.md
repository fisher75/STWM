# STWM Semantic Trajectory Field Pretraining V1 Decision

- future_semantic_feature_targets_available: `True`
- feature_target_quality: `future_gt_bbox_crop_frozen_clip`
- target_valid_ratio: `0.6796875`
- feature_head_training_successful: `False`
- free_rollout_feature_prediction_signal: `False`
- retrieval_signal_positive: `False`
- trace_regression_detected: `False`
- world_model_branch_status: `hidden_lacks_feature_signal`
- paper_world_model_claimable: `False`
- recommended_next_step_choice: `improve_feature_targets`

The world-model direction is now represented by a direct future semantic feature prediction path, but this smoke did not yet show a reliable feature retrieval signal.

Training was stable, but feature-head learning was not successful by the stricter loss/retrieval criterion.
