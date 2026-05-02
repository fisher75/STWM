# STWM-FSTF Final Claim Boundary V13

## Allowed Strong Claims
- STWM predicts future semantic trace-unit fields over frozen video-derived trace/semantic states.
- STWM improves changed semantic prototype prediction over copy and strong copy-aware baselines while preserving stable semantic memory.
- Future rollout hidden is load-bearing at H8 and remains load-bearing at H16/H24 under V13 hidden-shuffle/random intervention audits.
- C32 is selected as the best prototype vocabulary tradeoff; C128 fails the stability/granularity tradeoff.

## Allowed Moderate Claims
- H16/H24 retain positive changed-subset gains under the frozen-cache FSTF protocol.
- K16/K32 are evaluated as trace-unit density stress tests, but current valid-unit coverage only supports semantic trace-unit field wording.
- Raw-frame rollout visualizations are available as system demonstrations, while training/evaluation uses frozen video-derived trace/semantic caches.

## Forbidden Claims
- Raw-video end-to-end training.
- Full RGB video generation world model.
- Dense semantic trace field, because K16/K32 valid-unit coverage is weak/inconclusive.
- Model-size scaling is positive, because base/large do not beat small under strict grouped rules.
- Future trace coordinate or temporal order is load-bearing.
- Universal OOD dominance or universal cross-dataset generalization.
- STWM beats SAM2/CoTracker overall external SOTA or treats SAM2/CoTracker as same-output FSTF baselines.

## Key Flags
- dense_trace_field_claim_allowed: `False`
- long_horizon_claim_allowed: `True`
- model_size_scaling_claim_allowed: `False`
- raw_video_end_to_end_training_claim_allowed: `False`
