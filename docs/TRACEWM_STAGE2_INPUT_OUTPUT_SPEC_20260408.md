# TRACEWM Stage2 Input/Output Spec (2026-04-08)

## 1. Stage2 Inputs (Locked)

Stage2 input is fixed to two channels only:
1. frozen Stage1 trace/state tokens
2. semantic tokens or semantic embeddings

Input tensor contract:
- Stage1 tokens: [B, T, K, D_state]
- Raw semantic features: [B, K, D_sem_raw]
- Encoded semantic tokens: [B, K, D_sem]

The Stage1 token channel is provided to Stage2 with Stage1 backbone frozen.

## 2. Stage2 Outputs (Locked)

Stage2 output is fixed to:
1. enhanced future trace/state rollout
2. optional intermediate hidden diagnostics for audit

Output tensor contract:
- future rollout prediction: [B, T_fut, K, D_rollout]
- optional fused hidden diagnostics: [B, T, K, H]

## 3. Compatibility Requirement

Stage2 output must remain compatible with Stage1 evaluation protocol.
No new incompatible evaluator path is introduced in this bootstrap round.

## 4. Non-Ambiguity Rule

This document intentionally avoids mixed wording:
- Stage1 channel is frozen input channel.
- Stage2 semantic branch is trainable branch.
- No hidden or implicit backbone unfreeze is allowed.
