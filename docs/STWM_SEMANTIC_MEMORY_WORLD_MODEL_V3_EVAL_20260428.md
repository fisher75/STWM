# STWM Semantic Memory World Model V3 Eval

## Protocol
- Fixed split report: train/val/test requested 512/128/128 if available; observed-memory coverage allowed 320 eligible items and runtime materialized 249 loadable items.
- Stage1 frozen, trace dynamic path frozen, no candidate scorer, no future candidate leakage.
- Completed eval is teacher-forced semantic-field heldout. Free-rollout robust eval is still required before paper-level claim.

## C32
- Best seed: 456.
- Copy top5 overall/stable/changed: 0.672937 / 1.000000 / 0.471569.
- Residual top5 overall/stable/changed: 0.801578 / 1.000000 / 0.679412.
- Seed mean changed gain over copy: 0.184706.

## C64
- Best seed: 456.
- Copy top5 overall/stable/changed: 0.663835 / 1.000000 / 0.415612.
- Residual top5 overall/stable/changed: 0.756068 / 0.972857 / 0.595992.
- Seed mean changed gain over copy: 0.178903.

## Significance
- Best C32 item-level overall top5 delta CI: [0.07424194117021911, 0.16987515246386034].
- Best C32 item-level changed top5 delta CI: [0.12968590740453112, 0.2815128559246659].
- Best C32 stable preservation drop CI: [0.0, 0.0].
