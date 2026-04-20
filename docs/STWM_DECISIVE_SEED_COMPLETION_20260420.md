# STWM Decisive Seed Completion 20260420

## Status
- real_completion_started: `False`
- training_jobs_launched: `[]`
- matched_seed_completion_ready: `False`

## Missing Coverage
- missing_after_launch: `[{'method': 'TUSB-v3.1', 'seed': 654, 'if_missing_exact_reason': 'checkpoint_missing_in_live_repo'}, {'method': 'TUSB-v3.1', 'seed': 789, 'if_missing_exact_reason': 'checkpoint_missing_in_live_repo'}, {'method': 'TUSB-v3.1', 'seed': 321, 'if_missing_exact_reason': 'checkpoint_missing_in_live_repo'}, {'method': 'cropenc baseline', 'seed': 654, 'if_missing_exact_reason': 'checkpoint_missing_in_live_repo'}, {'method': 'cropenc baseline', 'seed': 321, 'if_missing_exact_reason': 'checkpoint_missing_in_live_repo'}, {'method': 'legacysem baseline', 'seed': 654, 'if_missing_exact_reason': 'checkpoint_missing_in_live_repo'}, {'method': 'legacysem baseline', 'seed': 789, 'if_missing_exact_reason': 'checkpoint_missing_in_live_repo'}, {'method': 'legacysem baseline', 'seed': 321, 'if_missing_exact_reason': 'checkpoint_missing_in_live_repo'}]`

## Blocking
- live repo still lacks materialized matched-seed checkpoints for TUSB-v3.1/cropenc/legacysem; this decisive pass does not fabricate completion
