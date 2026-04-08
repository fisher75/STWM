# TRACEWM Stage1-v2 Perf Hardening Protocol (2026-04-08)

## Objective
This round is a perf result hardening and correctness fix round only.
No model scientific logic changes are allowed.

## Frozen Current Problems
1. `primary_bottleneck` is currently driven mainly by `debug_small` timing evidence, which is weak for final attribution. Attribution must be upgraded to use `prototype_220m` as the primary source.
2. The dataloader profiler reports worker-side `getitem/collate` values as `0.0` for `num_workers > 0`, which is not reliable. These fields must be marked unavailable instead of being interpreted as CPU/IO evidence.
3. Trainer/runtime defaults have not yet absorbed the current best perf configuration.
4. Runtime remains single-GPU only. Multi-GPU support must not be claimed.

## Hardening Scope
1. Fix perf summary attribution logic to prioritize `prototype_220m` evidence.
2. Fix dataloader profiler metric reliability semantics.
3. Backwrite recommended runtime defaults only.
4. Export a single-GPU recommended runtime artifact.
5. Run a very short `prototype_220m` confirmation only.

## Explicit Non-Goals
1. No new model experiment.
2. No Stage2 work.
3. No WAN or MotionCrafter VAE changes.
4. No model architecture or loss definition changes.

## Recommended Defaults Policy (Runtime Only)
1. `num_workers=8`
2. `pin_memory=true`
3. `persistent_workers=true`
4. `prefetch_factor=4`

These are recommended runtime defaults, not scientific variables.

## Single-GPU Constraint
1. All formal runs in this hardening round are single-GPU only.
2. GPU selection policy is based on the previous sampled single-GPU audit artifacts.