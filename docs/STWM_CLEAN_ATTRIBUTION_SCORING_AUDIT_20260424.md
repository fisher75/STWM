# STWM Clean Attribution Scoring Audit 20260424

- semantic_teacher_only_should_rename: True
- external_teacher_only_clean: False
- frozen_external_teacher_only_clean: True
- semantic_target_tiebreak_effective: False
- exact_breakpoint: semantic_teacher_only enters _teacher_forced_predict and semantic_tokens[0]; external_teacher_only routes through _external_teacher_score_map with method.semantic_encoder; semantic_target_tiebreak reason: coord veto ineffective; selected tie-break behavior matched tusb_semantic_target on all comparable rows
