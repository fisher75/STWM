# STWM Clean Attribution Audit 20260423

- semantic_teacher_only_should_rename_to_tusb_semantic_target: True
- current_external_teacher_only_is_clean: False
- frozen_external_teacher_only_implemented: True
- semantic_target_tiebreak_effective: False
- exact_breakpoint: external_teacher_only calls _external_teacher_score_map, which calls method.semantic_encoder; semantic_target_tiebreak matched tusb_semantic_target because selected coord_tiebreak_weight=0.0 and coord_veto_penalty=0.0
