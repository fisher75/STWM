# STWM Reappearance Visibility Training V1 Head Only Summary

- generated_at_utc: `2026-04-27T17:21:57Z`
- training_completed: `True`
- training_status: `completed`
- tmux_session: `stwm_reappearance_visibility_headonly_v1_20260427`
- train_steps: `100`
- checkpoint_path: `outputs/checkpoints/stage2_tusb_v3p1_reappearance_visibility_headonly_v1_20260427/latest.pt`
- best_checkpoint_path: `outputs/checkpoints/stage2_tusb_v3p1_reappearance_visibility_headonly_v1_20260427/best.pt`
- trainable_param_audit: `{"freeze_non_future_semantic_head_during_warmup": true, "future_semantic_head_only_warmup": true, "future_semantic_head_only_warmup_steps": 100, "future_semantic_state_head_trainable_params": 605315, "head_only_boundary_ok": true, "non_future_semantic_head_trainable_params": 0, "non_future_semantic_head_trainable_params_by_module": {"readout_head": 0, "semantic_encoder": 0, "semantic_fusion": 0, "semantic_rescue_heads": 0, "stage1_model": 0, "trace_unit_broadcast": 0, "trace_unit_factorized_state": 0, "trace_unit_handshake": 0, "trace_unit_tokenizer": 0}, "total_trainable_params": 605315}`
- visibility_loss_mean: `0.4280238655209541`
- reappearance_loss_mean: `1.1968115150928498`
- reappearance_pos_weight_mean: `43.01338950872421`
- reappearance_positive_rate_mean: `0.02309970238095238`
- loss_finite_ratio: `1.0`
- output_valid_ratio: `1.0`
- trace_rollout_regression_detected: `False`
- raw_summary: `reports/stwm_reappearance_visibility_training_v1_head_only_summary_raw_20260427.json`
- exact_command_line: `#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
CUDA_VISIBLE_DEVICES=5 STWM_PROC_TITLE=python PYTHONPATH=code /home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tracewm_v2_stage2/trainers/train_tracewm_stage2_smalltrain.py \
  --resume-from outputs/checkpoints/stage2_tusb_v3p1_medium_semantic_state_v1_20260427/latest.pt \
  --skip-resume-optimizer \
  --predecode-cache-path /raid/chen034/workspace/stwm/data/processed/stage2_tusb_v3_predecode_cache_20260418 \
  --teacher-semantic-cache-path /raid/chen034/workspace/stwm/data/processed/stage2_teacher_semantic_cache_v4_appearance_20260418 \
  --stage2-structure-mode trace_unit_semantic_binding \
  --trace-unit-use-instance-prior-bias \
  --enable-future-semantic-state-head \
  --future-semantic-head-only-warmup \
  --future-semantic-head-only-warmup-steps 100 \
  --freeze-non-future-semantic-head-during-warmup \
  --future-semantic-loss-weight 0.01 \
  --future-visibility-loss-weight 0.03 \
  --future-reappearance-loss-weight 0.08 \
  --future-reappearance-pos-weight auto \
  --future-reappearance-pos-weight-max 50 \
  --future-identity-belief-loss-weight 0.005 \
  --future-uncertainty-loss-weight 0.002 \
  --future-hypothesis-count 1 \
  --future-hypothesis-loss-weight 0.0 \
  --lr 1e-7 \
  --max-samples-train 64 \
  --max-samples-val 32 \
  --batch-size 1 \
  --train-steps 11144 \
  --eval-interval 50 \
  --eval-max-batches 4 \
  --save-every-n-steps 50 \
  --output-dir outputs/checkpoints/stage2_tusb_v3p1_reappearance_visibility_headonly_v1_20260427 \
  --run-name stage2_tusb_v3p1_reappearance_visibility_headonly_v1_20260427 \
  --run-summary-json reports/stwm_reappearance_visibility_training_v1_head_only_summary_raw_20260427.json \
  --progress-json reports/stwm_reappearance_visibility_training_v1_head_only_progress_20260427.json \
  --seed 20260427
`
