# Stage2 TUSB Eval Liveness 20260418

- current_calibration_only_best: stage2_calonly_topk1_seed123_longconfirm_v2_20260414
- current_tusb_best: stage2_tusb_lite_seed123_20260417
- eval_path_tusb_liveness_passed: false
- protocol_v3_sees_tusb_modules: false
- previous_metric_flatness_likely_eval_blind_artifact: true
- blocking_reason: checkpoint_missing_tusb_state_dicts
- note: old TUSB-lite checkpoint is missing trace-unit state dicts, so protocol-v3 parity with calibration-only is not scientifically meaningful
