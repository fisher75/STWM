# STWM Semantic-Only TUSB Unfreeze V1 Decision

## Outcome

- Boundary ok: `True`
- Training completed: `True`
- Best prototype count: `64`
- C64 free-rollout proto top5: `0.040690104166666664`
- C128 free-rollout proto top5: `0.033854166666666664`
- Best frequency baseline top5: `0.2490234375`
- Best top5 gain over frequency: `-0.20833333333333334`
- Semantic-only signal positive: `False`
- Trace regression detected: `False`
- World-model output contract satisfied: `unclear`
- Paper world-model claimable: `false`
- Recommended next step: `improve_prototype_targets`

## Interpretation

This run fixes the V2 naming bug: semantic prototype loss now reaches TUSB semantic state through factorized semantic parameters and broadcast semantic projection. The result is still not positive because both C64 and C128 remain below their frequency baselines. That means the next bottleneck is likely prototype target quality / semantic discretization, not merely missing semantic-branch gradients.
