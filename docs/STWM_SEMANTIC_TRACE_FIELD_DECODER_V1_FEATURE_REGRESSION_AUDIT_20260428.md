# STWM Semantic Trace Field Decoder V1 Feature Regression Audit

- current_feature_target: `future GT bbox crop feature extracted with local OpenAI CLIP ViT-B/32`
- regresses_high_dim_clip_crop_feature: `True`
- feature_head_training_successful: `False`
- target_valid_ratio: `0.6796875`

CLIP crop features are retained only for prototype / pseudo-label construction. The main output should be a structured semantic prototype field, not raw CLIP regression.
