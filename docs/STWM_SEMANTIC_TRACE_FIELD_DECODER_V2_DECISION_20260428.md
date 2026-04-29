# STWM Semantic Trace Field Decoder V2 Decision

## Decision

- Large targets built: `True`
- Selected prototype count: `128`
- Target item count: `640`
- Valid feature count: `25472`
- Semantic branch unfreeze executed: `True`
- Trainable params total: `1057030`
- Stage1 trainable params: `0`
- Trace backbone trainable: `False`
- Head-only free proto top5: `0.033854166666666664`
- Semantic-branch free proto top5: `0.033854166666666664`
- Frequency baseline top5: `0.1796875`
- Free-rollout semantic field signal: `false`
- Trace regression detected: `False`
- World-model output contract satisfied: `unclear`
- Paper world-model claimable: `false`
- Recommended next step: `improve_prototype_targets`

## Interpretation

V2 successfully moved from the 6-item smoke target to a 640-item structured semantic prototype target cache and ran a controlled semantic-branch unfreeze without Stage1 or trace-backbone drift. However, the semantic prototype field is not yet learned: free-rollout top5 remains below the frequency baseline. The result argues for improving prototype targets / representation, not for abandoning STWM or continuing blind training.
