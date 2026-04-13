# Stage2 Semantic Objective Redesign V7 Protocol

- generated_at_utc: 2026-04-13T07:03:02.263205+00:00
- stage1_mutation_allowed: false
- main_task: future trace / future state generation
- teacher_as_mainline_semantic_source: false
- v7_goal: final disambiguation between calibration-only and calibration-plus-active-persistence under unchanged Stage1 backbone.
- objective_families: calibration_only_family vs calibration_plus_active_persistence_family.
- hard_persistence_activation_rule: if persistence is declared, guaranteed_pair_count_mean must be >= 1.0 and valuable_pair_ratio_mean must be > 0.0, otherwise mark declared_but_inactive.
- activation_audit_required_before_launch: true
- persistence_mining_focus: strict/fallback telemetry + guaranteed pair activation check
- selective_supervision_position: readout-side only, never overwriting frozen trace dynamics.
- forbidden: teacher semantic token replacement; external-eval work; Stage1 rollback; codec/VAE upgrade; full-scale long train; batch/lr sweep; DDP retrofit.
