# Stage2 Semantic Objective Redesign V6 Protocol

- generated_at_utc: 2026-04-12T15:15:45.475343+00:00
- stage1_mutation_allowed: false
- main_task: future trace / future state generation
- teacher_as_mainline_semantic_source: false
- v5_failure_summary: sparse gating activated, but persistence mining was frequently inactive and sidecar gains did not transfer to overall best.
- v6_core_principles: keep sparse query gating active; enforce guaranteed persistence pairs; add two-level fallback mining; keep delayed auxiliary schedule; sidecar remains independent.
- activation_audit_required_before_launch: true
- persistence_mining_repair_focus: guaranteed_min_pairs + strict/fallback pair ratio telemetry
- selective_supervision_position: readout-side only, never overwriting frozen trace dynamics.
- forbidden: teacher semantic token replacement; external-eval work; Stage1 rollback; codec/VAE upgrade; full-scale long train; batch/lr sweep; DDP retrofit.
