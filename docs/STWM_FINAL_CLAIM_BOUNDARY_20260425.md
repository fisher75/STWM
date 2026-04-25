# STWM Final Claim Boundary 20260425

## Strong Claims Allowed
- Official STWM / TUSB-v3.1 + trace_belief_assoc improves over calibration-only, cropenc, and legacysem under the frozen best_semantic_hard.pt official setting.
- Trace-conditioned belief readout improves over frozen_external_teacher_only and legacysem on ID densified setting.
- semantic_teacher_only should be treated as TUSB semantic target / trace-conditioned semantic target, not as clean external teacher-only.
- Matched 6-seed context-preserving evaluation supports moving to main submission assets under the official belief readout.

## Moderate Claims Allowed
- True OOD evidence is positive and claim-ready in current materialized splits, but should be framed as held-out validation rather than universal OOD dominance.
- Downstream utility supports representation usefulness, but should not be expanded into a sweeping planning/general intelligence claim.
- Mechanism evidence is supportive/diagnostic, not a standalone strong cross-seed mechanism proof.
- Oral/spotlight should be aspirational only until writing quality and reviewer-facing clarity are tested.

## Claims Not Allowed
- appearance-change solved
- universal OOD dominance
- oral/spotlight guaranteed
- teacher-only baseline fully defeated in every setting
- hybrid_light is the official method
- clean_residual_v2 or trace_gallery_assoc is the official mainline
- Stage1 was updated or retrained
- new protocol v4 or new TUSB-v3.4/v3.5 structure exists
