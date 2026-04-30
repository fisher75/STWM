# STWM Fullscale Semantic Trace World Model V1 Single Train

- audit_name: `stwm_fullscale_semantic_trace_world_model_v1_single_train`
- prototype_count: `64`
- seed: `123`
- steps: `5000`
- lr: `3e-05`
- residual_scale: `0.25`
- device: `cuda`
- cuda_visible_devices: `1`
- train_batch_count: `760`
- val_batch_count: `164`
- stage1_trainable_param_count: `0`
- trace_backbone_trainable: `False`
- dynamic_trainable_params: `0`
- candidate_scorer_used: `False`
- future_candidate_leakage: `False`
- loss_finite_ratio: `1.0`
- trace_regression_detected: `False`
- checkpoint_path: `outputs/checkpoints/stwm_mixed_fullscale_v2_20260428/c64_seed123_final.pt`
