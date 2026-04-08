# TRACEWM Stage1 v2 Refactor Protocol (2026-04-08)

## 1. Objective
Stage1 v2 only targets trace-aware future-state backbone upgrade for Stage1.

Primary goal:
- Replace legacy synthetic-track + mean-5D GRU pipeline with real-trace cache + multi-token causal Transformer backbone.

## 2. Hard Scope Boundaries
Allowed:
- Stage1 trace/state modeling only.
- Real trace sources from PointOdyssey and Kubric.
- Multi-token state modeling and structured trajectory losses.

Disallowed:
- Stage2 semantics flow, WAN, MotionCrafter VAE, DynamicReplica, video reconstruction.
- Any new joint-rescue branch or continuation of old iter/fix/final_rescue lines.

## 3. Isolation Rule
All new implementation must live under:
- `/home/chen034/workspace/stwm/code/stwm/tracewm_v2/`

Legacy namespace `tracewm/` remains unchanged except optional import exposure.

## 4. Milestone Order (Strict)
Execution order must be:
1. P0 real trace cache integration and audit.
2. P1 multi-token state `[T,K,D]` integration.
3. P2 causal Transformer backbone (target near 220M config).
4. P3 structured loss stack (coord/vis/residual/vel/(optional endpoint)).

No P1/P2/P3 result can be declared valid before P0 passes.

## 5. Implementation Deliverables
Required docs:
- `/home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_V2_REFACTOR_PROTOCOL_20260408.md`
- `/home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_V2_ACCEPTANCE_CRITERIA_20260408.md`
- `/home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_V2_TRACE_CACHE_PROTOCOL_20260408.md`

Required P0 artifacts:
- `/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json`
- `/home/chen034/workspace/stwm/reports/stage1_v2_trace_cache_audit_20260408.json`

Required staged reports:
- `/home/chen034/workspace/stwm/reports/tracewm_stage1_v2_train_summary_20260408.json`
- `/home/chen034/workspace/stwm/reports/tracewm_stage1_v2_g1_g5_20260408.json`
- `/home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_V2_G1_G5_20260408.md`

## 6. Runtime Session Convention
Fixed background session and log target:
- tmux session: `tracewm_stage1_v2_20260408`
- log file: `/home/chen034/workspace/stwm/logs/tracewm_stage1_v2_20260408.log`

## 7. Ablation Matrix (G1-G5)
G1:
- P1 only sanity lane (multi-token state + minimal losses).

G2:
- G1 + P2 causal transformer enabled.

G3:
- G2 + visibility supervision.

G4:
- G3 + residual and velocity supervision.

G5:
- G4 + optional endpoint supervision.

## 8. Final Decision Rule
Mainline recommendation must come only from generated report fields.
No unsupported conclusions or implicit assumptions are allowed.
