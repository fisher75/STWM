# STWM Semantic-State Feedback V1 Train Summary

- training_completed: true
- train_steps: 120
- checkpoint: outputs/checkpoints/stage2_tusb_v3p1_semantic_state_feedback_v1_20260427/latest.pt
- trainable_param_count_total: 1049295
- stage1_trainable_param_count: 0
- trace_backbone_trainable: false
- feedback_adapter_trainable_params: 142153
- loss_finite_ratio: 1.0
- output_valid_ratio: 1.0
- feedback_gate_mean: 0.01798769757442642
- feedback_delta_norm: 0.005309318647050532
- trace_rollout_regression_detected: false

This was a controlled feasibility run: Stage1 and trace/TUSB trunk stayed frozen, with only the FutureSemanticTraceState head, semantic projection/readout slice, and lightweight feedback adapter trainable.
