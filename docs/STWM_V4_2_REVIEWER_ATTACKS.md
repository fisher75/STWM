# STWM V4.2 Reviewer Attacks And Mitigations

## Scope

Final paperization-stage reviewer attack plan.

## Strongest Attack

Attack:

1. “`wo_object_bias_v4_2` is not a truly independent baseline; your gain may be an ablation artifact.”

Risk:

1. Reviewer may down-rank second contribution as under-controlled.

Rebuttal evidence:

1. Matched-budget design keeps architecture and parameter budget fixed.
2. Only object-bias input channels are neutralized.
3. Harder-protocol decoupling and identifiability deltas are consistent with representation-level effect.

Text-side mitigation:

1. Explicitly label it as internal matched-budget control (not standalone baseline).
2. Move “independent baseline” wording out of main text.
3. Keep claims bounded to “evidence for representation contribution under fixed-budget controls.”

## Second Strongest Attack

Attack:

1. “Semantics effect under state-identifiability is mixed; second contribution may not be robust.”

Risk:

1. Reviewer challenges whether two-contribution story is over-claimed.

Rebuttal evidence:

1. Two contributions are separated:
   - contribution 1: semantic trajectory state (base multiseed mainline)
   - contribution 2: instance-grounded identification (protocol + representation control + decoupling)
2. The strongest second-contribution evidence is full vs `wo_object_bias_v4_2`, not full vs `wo_semantics_v4_2`.

Text-side mitigation:

1. Avoid claiming large semantics gain inside identifiability protocol.
2. Present semantics there as supportive but not sole driver.
3. Put mixed slices in body/appendix with explicit caveat.

## Third Likely Attack

Attack:

1. “Legacy old/current baseline row is not fully apples-to-apples with v4.2.”

Risk:

1. Reviewer questions fairness of historical comparison.

Rebuttal evidence:

1. Legacy baseline is marked as context continuity only.
2. Core claims do not rely on legacy row for statistical superiority.

Text-side mitigation:

1. Add comparability warning directly in table caption.
2. Keep strict claims anchored to v4.2-internal comparisons.

## Red-Line Statements To Avoid

1. “Identity/reconnect is a strong positive contribution.”
2. “Our method dominates all baselines across all protocols.”
3. “wo_object_bias_v4_2 proves full independent baseline superiority.”
