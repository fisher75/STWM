# Stage2 Semantic Objective Redesign V3 Protocol

- generated_at_utc: 2026-04-11T05:36:35.991733+00:00
- stage1_mutation_allowed: false
- main_task: future trace / future state generation
- teacher_as_mainline_semantic_source: false
- v1_failure_summary: directly stacking semantic rescue losses hurt rollout optimum.
- v2_summary: readout-side alignment plus persistence ranking was directionally right but did not create a true new global best.
- v3_core_principles: semantics supervise and calibrate, not overwrite dynamics; semantic intervention is selective, confidence-aware, and hard-case-focused; semantic-hard evaluation is a first-class criterion.
- forbidden: teacher semantic token replacement; stronger fusion trick; codec/VAE mainline replacement; full-scale long train; batch/lr sweep; DDP retrofit.
