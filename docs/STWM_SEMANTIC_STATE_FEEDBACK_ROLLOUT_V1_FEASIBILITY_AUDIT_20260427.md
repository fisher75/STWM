# STWM Semantic-State Feedback Rollout V1 Feasibility Audit

The current system produces FutureSemanticTraceState from `future_hidden` via a readout head; it does not feed semantic state back into rollout hidden before this V1. The safest V1 insertion point is a lightweight gated residual adapter in `readout_only` mode; `hidden_residual` is an ablation. Stage1 and the TUSB/trace trunk remain frozen.
