# Stage2 Semantic Objective Redesign V4 Protocol

- generated_at_utc: 2026-04-11T07:11:31.656572+00:00
- stage1_mutation_allowed: false
- main_task: future trace / future state generation
- teacher_as_mainline_semantic_source: false
- v3_failure_summary: confidence-gated affected ratio saturated near 1.0 and failed to produce a true new best beyond warm-start inheritance.
- v4_core_principles: sparse and selective gating only; two switchable sparse gate families (quantile_sparse_gating, topk_query_gating); high-value pair filtering for persistence term; stronger delayed/ramped auxiliary schedule.
- sidecar_policy: semantic-hard sidecar stays independent from overall best checkpoint selection.
- forbidden: teacher semantic token replacement; stronger fusion trick; codec/VAE mainline replacement; full-scale long train; batch/lr sweep; DDP retrofit.
