# STWM True OOD Attribution Audit 20260423

- official_tusb_checkpoint: best_semantic_hard.pt
- official_tusb_scoring_mode: hybrid_light
- attribution_scoring_modes: ["coord_only", "unit_identity_only", "semantic_teacher_only", "coord_plus_teacher", "coord_plus_unit", "hybrid_light"]
- baselines_scoring_modes: {"calibration-only::best.pt": "coord_only", "cropenc::best.pt": "coord_only", "legacysem::best.pt": "coord_only"}
- audit_passed: True
