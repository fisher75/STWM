# Stage2 Bootstrap Smoke Results

- generated_at_utc: 2026-04-08T16:33:54.997509+00:00
- run_name: stage2_bootstrap_smoke_20260408
- stage2_bootstrap_ready: True
- next_step_choice: start_stage2_small_train

## Required Answers
- stage1_frozen_backbone_loadable: True
- semantic_branch_accepts_inputs: True
- fusion_forward_working: True
- core_datasets_provide_stage2_inputs: True
- stage2_bootstrap_ready: True

## Core Dataset Inputs
- ready: True
- details: {'VSPW': {'train_required': True, 'eval_required': True, 'train_sample_count': 6, 'val_sample_count': 3, 'ready': True}, 'VIPSEG': {'train_required': True, 'eval_required': True, 'train_sample_count': 6, 'val_sample_count': 3, 'ready': True}}

## Freeze Boundary
- stage1_trainable_parameter_count: 0
- semantic_trainable_parameter_count: 3025154
- stage1_grad_detected_after_backward: False
- semantic_grad_norm: 11.650684375044397
