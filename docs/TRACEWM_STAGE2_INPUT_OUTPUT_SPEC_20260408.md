# TRACEWM Stage2 Input/Output Spec (2026-04-08)

## 1. Semantic Source Definition (Frozen)

Stage2 semantics must come from visual semantic state extracted from object regions / mask-crop context.

Mainline source policy:
- primary: object-region visual features from tracked object regions,
- optional: mask-crop visual features when mask resources are available,
- disallowed as mainline: fake hash labels,
- disallowed as mainline: CLIP teacher distillation as core training path.

## 2. Stage2 Inputs

Stage2 consumes:
1. frozen Stage1 trace/state tokens,
2. semantic tokens / semantic embeddings from object-region or mask-crop features.

Input tensor contract:
- Stage1 tokens: `[B, T, K, D_state]` (from Stage1-v2 cache/state contract)
- Semantic features: `[B, K, D_sem_raw]`
- Encoded semantic tokens: `[B, K, D_sem]`

## 3. Stage2 Outputs

Stage2 outputs enhanced future rollout states while preserving compatibility with Stage1 evaluation protocol.

Output contract:
- enhanced future coord rollout: `[B, T_fut, K, 2]`
- optional enhanced hidden state for diagnostics: `[B, T, K, H]`

Compatibility requirement:
- Stage2 output must remain directly mappable to Stage1 evaluation metrics interface.

## 4. Stage2 Training Strategy (Frozen)

- Stage1 backbone: frozen
- Stage2 semantic branch: trainable
- Stage2 semantic fusion/adapter: trainable
- optional lightweight readout head: trainable

This bootstrap round does not run full long-train; only smoke-level optimization is allowed.
