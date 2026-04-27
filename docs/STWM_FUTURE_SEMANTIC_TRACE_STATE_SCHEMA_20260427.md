# STWM Future Semantic Trace State Schema 20260427

## Required Tensors
- `future_trace_coord`: [B,H,K,2 or 3]
- `future_visibility_logit`: [B,H,K]
- `future_semantic_embedding`: [B,H,K,D_sem]
- `future_identity_embedding`: [B,H,K,D_id]
- `future_uncertainty`: [B,H,K]

## Optional Tensors
- `future_semantic_logits`: [B,H,K,C] optional
- `future_extent_box`: [B,H,K,4] optional
- `future_hypothesis_logits`: [B,M] optional
- `future_hypothesis_trace_coord`: [B,M,H,K,2 or 3] optional

## Default-Off Compatibility
- The head is not instantiated unless `--enable-future-semantic-state-head` is passed.
- All new loss weights default to 0.0.
