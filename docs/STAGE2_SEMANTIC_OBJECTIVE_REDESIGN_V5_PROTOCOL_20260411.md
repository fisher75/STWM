# Stage2 Semantic Objective Redesign V5 Protocol

- generated_at_utc: 2026-04-11T17:35:31.164006+00:00
- stage1_mutation_allowed: false
- main_task: future trace / future state generation
- teacher_as_mainline_semantic_source: false
- v4_failure_summary: gating remained effectively saturated and freeze logic was too permissive despite no true new best or semantic-hard improvement.
- v5_core_principles: real sparse query-level gating only; high-value persistence pairs only; longer delayed auxiliary schedule; sidecar stays independent from overall best.
- selective_supervision_position: readout-side only, never overwriting frozen trace dynamics.
- forbidden: teacher semantic token replacement; external-eval work; Stage1 rollback; codec/VAE upgrade; full-scale long train; batch/lr sweep; DDP retrofit.
