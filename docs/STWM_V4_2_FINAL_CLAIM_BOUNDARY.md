# STWM V4.2 Final Claim Boundary

## Scope

Final claim boundary for paperization mode.

Paper structure:

1. Semantic Trajectory State on dense 4D trajectory fields
2. Instance-Grounded Future State Identification
3. Identity/reconnect as boundary or secondary analysis only

## 1) Claims Allowed In Title And Abstract

Allowed:

1. A dense 4D trajectory-field based semantic trajectory state formulation.
2. A state-identifiability protocol demonstrating instance-grounded future-state identification.
3. Harder-protocol evidence that query behavior remains decoupled from old-v1 proxy-collapse signature.

Not allowed in title/abstract:

1. Any positive identity/reconnect superiority claim.
2. Any statement implying universal dominance over all baselines under all protocols.

## 2) Claims Allowed In Main Body (But Not Headline)

Allowed in results/discussion:

1. `full_v4_2` shows clearer mean edge vs `wo_object_bias_v4_2` on state-identifiability query metrics.
2. `full_v4_2` vs `wo_semantics_v4_2` is mixed under state-identifiability eval-only settings.
3. Decoupling advantage vs `wo_object_bias_v4_2` under harder protocol is meaningful.

Required caveat:

1. legacy old/current baseline row is contextual continuity, not strict same-pipeline apples-to-apples evidence.

## 3) Claims That Must Stay In Appendix

1. identity rescue round details and negative/neutral outcomes.
2. full per-seed per-type exhaustive tables.
3. boundary/failure case deep dives.

## 4) Required Wording For wo_object_bias_v4_2

Use this framing:

1. `wo_object_bias_v4_2` is a matched-budget internal representation control.
2. It keeps architecture and parameter budget fixed and only neutralizes object-bias inputs (`prior_features` and `teacher_objectness` bias channel).
3. It is not presented as a fully independent external baseline family.

Avoid this framing:

1. “strong standalone baseline”
2. “new SOTA baseline family”
3. any wording that implies independent architecture-level design.

## Final Safe Summary

1. Headline strength should be concentrated on semantic trajectory state plus instance-grounded future identification.
2. Identity/reconnect remains boundary evidence, not headline evidence.
3. Legacy baseline comparison is contextual and explicitly caveated.
