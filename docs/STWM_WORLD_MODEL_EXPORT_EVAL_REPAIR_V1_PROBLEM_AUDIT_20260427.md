# STWM World Model Export/Eval Repair V1 Problem Audit 20260427

- generated_at_utc: `2026-04-27T14:15:47.646302+00:00`
- current_export_loads_v2_smalltrain_checkpoint: `True`
- current_checkpoint_path: `outputs/checkpoints/stage2_tusb_v3p1_worldmodel_v2_smalltrain_lr1e7_20260427/latest.pt`
- current_checkpoint_exists: `True`
- current_checkpoint_loaded: `True`
- current_future_semantic_state_head_key_count: `16`
- current_export_calls_model_forward_or_free_rollout: `False`
- current_export_forward_scope: `future_semantic_state_head_checkpoint_forward_with_manifest_surrogate_features`
- current_export_saves_raw_tensor_or_summary: `summary_only_norm_mean_coord_list`
- current_eval_reads_export_file: `True`
- current_eval_still_reads_old_association_report: `False`
- current_degeneracy_audit_directly_based_on_raw_export: `False`
- safe_for_medium_training_false_reason: `V2 export lacks raw future_semantic_embedding/future_identity_embedding tensors; only norm summaries are available, so unit/horizon embedding degeneracy cannot be ruled out.`

## Missing Fields
- future_semantic_embedding
- future_identity_embedding
- future_uncertainty

## Minimum Fix Path
- export raw-output-derived shape/stat/variance for semantic/identity/visibility/uncertainty
- add strict raw export consuming eval mode with no old-report fallback
- make degeneracy audit require raw export schema
- rerun 32-item validation from V2 smalltrain checkpoint
