# STWM Reappearance/Visibility Joint V1 Summary

- training_completed: `True`
- train_steps_this_invocation: `150`
- checkpoint_path: `outputs/checkpoints/stage2_tusb_v3p1_reappearance_visibility_joint_v1_20260427_run2/latest.pt`
- loss_finite_ratio: `1.0`
- output_valid_ratio: `1.0`
- future_reappearance_event_loss_mean: `0.28574199199676514`
- future_reappearance_loss_mean: `0.3504013995329539`
- future_visibility_loss_mean: `0.39241068681081137`
- trainable_param_count_total: `907142`
- stage1_trainable_param_count: `0`
- trace_backbone_trainable: `False`
- trace_rollout_regression_detected: `False`

The controlled joint run trained only the minimal FutureSemanticTraceState head plus semantic projection/readout slice. Stage1 and the trace backbone remained frozen.
