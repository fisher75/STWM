# STWM World Model Upgrade V2 Smalltrain Summary 20260427

- tmux_session: `stwm_wm_v2_smalltrain_20260427`
- smalltrain_only: `true`
- learning_rate: `1e-07`
- train_step_count: `4`
- loss_finite_ratio: `1.0`
- future_semantic_state_output_valid_ratio: `1.0`
- trace_rollout_regression_detected: `False`
- checkpoint_path: `outputs/checkpoints/stage2_tusb_v3p1_worldmodel_v2_smalltrain_lr1e7_20260427/latest.pt`

The first lr=1e-4 attempt completed but showed trace proxy degradation. The retained V2 smalltrain summary uses a conservative lr=1e-7 relaunch and treats historical resumed best metrics as non-comparable to the one-batch V2 smoke window.
