# STWM Delayed Router Mainline V1

Generated: 2026-04-04

## Context And Switch Rationale

Current evidence indicates simple scalar object-bias control is insufficient for mainline promotion:

- alpha050 is not cross-seed stable (seed42 positive, seed123 negative).
- gated is stronger than alpha050, but still not officially superior to `wo_object_bias` on seed123.
- warmup extension has no mainline rescue value and stays downgraded.

Therefore the next unique mainline is shifted from scalar alpha/gate tuning to staged structural control:

- temporal activation control (delayed-only),
- path-structured neutral-vs-biased routing (two-path residual),
- query/confidence-aware delayed residual router (combined).

## Design 1: Delayed-Only

Core idea:

- Keep full biased path unchanged at steady state.
- Disable object-bias impact during early training steps, then enable after a fixed delay.

Proxy implementation in current trainer knobs:

- `--object-bias-delay-steps 200`
- Keep default `--object-bias-alpha 1.0`
- No gated router enabled.

Unique question to validate:

- Is early optimization instability the primary cause of object-bias harm, and can pure temporal deferment recover full performance without reducing later bias capacity?

## Design 2: Two-Path Residual Neutral-vs-Biased Routing

Core idea:

- Build two conceptual paths:
  - neutral path: object-bias-suppressed signal,
  - biased path: object-bias-enhanced signal.
- Use residual blending and gating to route sample contributions between paths.

Proxy implementation in current trainer knobs (minimal approximation):

- `--object-bias-alpha 0.50` (residual blend strength)
- `--object-bias-gated --object-bias-gate-threshold 0.5`
- No delay.

Unique question to validate:

- Can structured path routing outperform static alpha scaling by preserving neutral fallback while retaining biased gains when useful?

## Design 3: Delayed + Query/Confidence-Aware Residual Router (Combined)

Core idea:

- Start with delayed activation to avoid early harmful coupling.
- After delayed phase, enable residual/gated routing.
- Treat router behavior as confidence-aware proxy: low-confidence regions should prefer neutral path, high-confidence regions can use biased path.

Proxy implementation in current trainer knobs (phase-1 approximation):

- `--object-bias-delay-steps 200`
- `--object-bias-alpha 0.50`
- `--object-bias-gated --object-bias-gate-threshold 0.5`

Unique question to validate:

- Is there a complementary effect where temporal deferment stabilizes early learning and residual routing improves late-stage selectivity beyond either mechanism alone?

## Mainline Diagnostic Matrix (Minimal, Seed42)

Three runs only:

1. `delayed_only_seed42_challenge_v1`
2. `two_path_residual_seed42_challenge_v1`
3. `delayed_residual_router_seed42_challenge_v1`

Shared constraints:

- frontend_cache default path
- protocol `v2_4_detached_frozen`
- detached queue/status/log/report workflow
- no extra branches, no concurrency increase, no warmup revival

## Decision Principle

The matrix is diagnostic-first, not direct promotion:

- rank all variants by official selection rule,
- compare each against current full baseline and `wo_object_bias` reference behavior,
- only if combined delayed-residual shows robust superiority should it proceed to next clean-matrix mainline stage.
