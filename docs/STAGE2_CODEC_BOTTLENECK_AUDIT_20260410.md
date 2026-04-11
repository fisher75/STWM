# Stage2 Codec Bottleneck Audit

- semantic_crop_encoder_params: 438144
- semantic_branch_total_trainable_params: 3463298
- frozen_stage1_backbone_params: 207615754
- trainable_to_frozen_param_ratio: 0.016681
- bottleneck_judgment: likely_not_bottleneck
- judgment_reason: Capacity may matter as a secondary issue if CLIP side-probe wins, but v1/v2 objective failures do not prove the 3M crop branch is the primary bottleneck.
