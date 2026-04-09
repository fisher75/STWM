# Stage2 Data Evidence Bundle

- generated_at_utc: 2026-04-09T09:07:15.499793+00:00
- core_binding_zero_missing: True
- datasets_bound_for_train: ['vspw', 'vipseg']
- datasets_bound_for_eval: ['vspw', 'vipseg']

## VSPW
- role_in_stage2: core
- completeness_status: core_ready
- evidence_source_path: /home/chen034/workspace/data/_manifests/stage2_dataset_audit_20260408.json
- evidence_source_exists: True
- regenerated_in_this_round: True
- warning_reason: some clips are shorter than 16 frames or contain frame/mask name irregularities, but the current Stage2 loader tolerates these via resampling and optional mask fallback

## VIPSeg
- role_in_stage2: core
- completeness_status: core_ready
- evidence_source_path: /home/chen034/workspace/data/_manifests/stage2_dataset_audit_20260408.json
- evidence_source_exists: True
- regenerated_in_this_round: True
- warning_reason: many clips are shorter than 16 frames, but the current Stage2 loader still admits them because rollout indices are resampled from any clip with >=2 readable frames

## BURST
- role_in_stage2: optional_extension
- completeness_status: optional_extension_ready
- evidence_source_path: /home/chen034/workspace/data/_manifests/stage2_dataset_audit_20260408.json
- evidence_source_exists: True
- regenerated_in_this_round: False

## TAO
- role_in_stage2: optional_extension
- completeness_status: access_ready
- evidence_source_path: /home/chen034/workspace/data/_manifests/stage2_dataset_audit_20260408.json
- evidence_source_exists: True
- regenerated_in_this_round: False

## VISOR
- role_in_stage2: manual_gate_extension
- completeness_status: manual_gate
- evidence_source_path: /home/chen034/workspace/data/_manifests/stage2_dataset_audit_20260408.json
- evidence_source_exists: True
- regenerated_in_this_round: False
- blocking_reason: EPIC-KITCHENS dependency remains manual-gated

