# STWM Semantic Memory World Model V2 Eval

## Scope
- V2 evaluates copy-gated residual semantic memory as structured semantic trace-field output, not a candidate scorer.
- Stage1 remains frozen; trace dynamic path remains frozen; no future candidate leakage is used.
- Current completed eval is heldout teacher-forced semantic-field eval over the runtime covered split. A separate free-rollout full eval was attempted but blocked by runtime/data loading and is not counted as a paper-level claim.

## C32
- Best seed: 456.
- Copy top5 overall/stable/changed: 0.683309 / 1.000000 / 0.528302.
- Residual top5 overall/stable/changed: 0.729508 / 0.995465 / 0.599334.
- Changed subset gain over copy: 0.071032.

## C64
- Best seed: 123.
- Copy top5 overall/stable/changed: 0.625931 / 1.000000 / 0.424971.
- Residual top5 overall/stable/changed: 0.687034 / 0.968017 / 0.536082.
- Changed subset gain over copy: 0.111111.

## Significance
- C32 seed-level overall top5 delta CI: [0.016947886408987728, 0.05110971486404241].
- C32 seed-level changed top5 delta CI: [0.035992390279325916, 0.07573533312386815].
- C64 seed-level overall top5 delta CI: [0.04877536941648097, 0.061507787293289436].
- C64 seed-level changed top5 delta CI: [0.09068633184987278, 0.11626289467135302].
- Item-level bootstrap is still required before any paper-level world-model claim.
