# STWM Reappearance Visibility Training V1 Head Only Warmup Audit

- generated_at_utc: `2026-04-27T16:46:14Z`
- support_added_in_trainer: `True`
- new_cli_args: `["--future-semantic-head-only-warmup", "--future-semantic-head-only-warmup-steps", "--freeze-non-future-semantic-head-during-warmup"]`
- start_checkpoint: `outputs/checkpoints/stage2_tusb_v3p1_medium_semantic_state_v1_20260427/latest.pt`
- start_checkpoint_global_step: `11044`
- missing_reappearance_head_weights_in_start_checkpoint: `True`
- unexpected_keys: `[]`
- planned_future_semantic_head_only_warmup: `True`
- planned_warmup_steps: `100`
- planned_freeze_non_future_semantic_head_during_warmup: `True`
- total_trainable_params: `605315`
- future_semantic_state_head_trainable_params: `605315`
- non_future_semantic_head_trainable_params: `0`
- head_only_boundary_ok: `True`
- boundary_fail_policy: `trainer raises RuntimeError if non_future_semantic_head_trainable_params > 0 during head-only warmup`
