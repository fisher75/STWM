# STWM V4.2 Diagnostic Trigger Rules (D2, Definition Only)

Date: 2026-04-03
Status: DEFINITION ONLY (do not execute yet)
Scope: 220M protocol-frozen diagnostics after D1

## Inputs Required for Trigger Decisions

Decisions must be based on both:

1. Protocol metrics from detached evaluator on protocol_val_main_v1
   - query_localization_error
   - query_top1_acc
   - future_trajectory_l1
   - plus diagnostic axes (identity/occlusion/reconnect)
2. Gradient audit traces
   - ||g_traj||, ||g_query||, ||g_sem||
   - cos(g_sem, g_traj), cos(g_sem, g_query)

## Trigger for Uncertainty Weighting

Enable uncertainty weighting only if all conditions hold:

1. Best fixed-lambda candidate remains unstable across adjacent 500-step windows on protocol-main:
   - no net improvement in query_localization_error over at least 3 consecutive windows
   - and tie metrics do not compensate (query_top1_acc non-improving, future_trajectory_l1 non-improving)
2. Gradient scale imbalance persists:
   - median(||g_sem|| / max(||g_traj||, 1e-12)) outside [0.2, 5.0] for at least 30% audited points
3. Sign pattern does not indicate pure conflict-only regime:
   - median cos(g_sem, g_traj) > -0.2 and median cos(g_sem, g_query) > -0.2

Rationale: use uncertainty weighting first for scale mismatch before conflict projection.

## Trigger for GradNorm

Enable GradNorm only if all conditions hold:

1. Uncertainty weighting trial fails to recover protocol-main trend after at least one short diagnostic run
2. Persistent cross-task norm divergence remains:
   - p90/p10 of task norms among {g_traj, g_query, g_sem} > 6.0
3. Protocol-main still not improved versus fixed-lambda best_protocol_main baseline

Rationale: GradNorm is second-line normalization when simple uncertainty scaling is insufficient.

## Trigger for PCGrad

Enable PCGrad only if all conditions hold:

1. Fixed-lambda and normalization-family trials both fail to improve best_protocol_main baseline
2. Conflict signal is persistent and strong:
   - fraction of audited points with cos(g_sem, g_traj) < -0.2 is at least 40%
   - or fraction with cos(g_sem, g_query) < -0.2 is at least 40%
3. Negative alignment coincides with protocol-main stagnation or regression windows

Rationale: projection-based conflict surgery is reserved for demonstrated destructive interference, not used as default.

## Hard Constraints

1. No advanced trigger can bypass frozen official-best rule.
2. Every comparison must remain on detached protocol-main artifacts.
3. Trigger activation must be explicitly documented before execution.
