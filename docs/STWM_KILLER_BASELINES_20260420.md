# STWM Killer Baselines 20260420

## Killer Baseline Status
- killer_baselines_passed: `False`
- blocking_reason: teacher-only and object-slot/no-trace killer baselines are not materialized; trace-only TUSB is only available for seed123

## Available
- trace_only_tusb: `{'available': True, 'seed_coverage': [123], 'additional_seed_available': False, 'run_name': 'stage2_tusb_v2_no_teacher_prior_seed123_20260418', 'reason_if_incomplete': 'only seed123 materialized in live repo'}`
- stage1_frozen_trace_only_readout: `{'available': True, 'run_name': 'stage1_frozen_baseline'}`
- identity_binding_ablation_support: `{'available': True, 'run_name': 'stage2_tusb_v3_no_identity_binding_seed123_20260418'}`

## Missing
- teacher_only_semantic_only_retrieval: `{'available': False, 'reason': 'no materialized teacher-only checkpoint or standalone semantic-only evaluation asset in live repo'}`
- object_slot_no_trace: `{'available': False, 'reason': 'no materialized object-slot/no-trace baseline checkpoint or eval asset in live repo'}`
