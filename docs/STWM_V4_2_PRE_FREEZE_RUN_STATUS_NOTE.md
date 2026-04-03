# STWM V4.2 Pre-Freeze Run Status Note

Date: 2026-04-03

## Scope

This note marks the current unfinished STWM queue/training state as pre-freeze exploratory execution.

## Policy Tag

From this freeze point onward:

- Unfinished runs in the current queue are tagged as pre-freeze exploratory runs.
- Their artifacts are retained for diagnosis and engineering traceability.
- They are not treated as default final evidence for protocol-frozen conclusions.

## Why This Tag Is Applied

1. Detached evaluator compatibility is already unblocked and executable on STWM V4.2 checkpoints.
2. Protocol metric hierarchy, protocol-best checkpoint rule, and validation split policy are being refactored/frozen.
3. Continuing long queue execution before these rules are frozen has low ROI and high re-train risk.

## Evidence Handling Rule

- Keep all produced checkpoints, logs, summaries, and detached eval outputs.
- Allow diagnostic analysis on pre-freeze artifacts.
- For final claims, prioritize runs generated under the frozen protocol/split/checkpoint policy.

## Operational Consequence

- Queue consumption is parked/stopped.
- Resume is allowed for specific diagnostic continuity.
- Publication-grade claim evidence should be regenerated or revalidated under the post-freeze policy.
