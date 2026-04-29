# STWM Semantic-Only TUSB Unfreeze V1 Trainability Audit

## Current V2 Naming Bug

The previous `semantic_branch_unfreeze` was head/proj/readout-only. It trained `future_semantic_state_head`, `semantic_fusion.semantic_proj`, and `readout_head`, but did not train the TUSB semantic state modules. Therefore V2 cannot be called a true semantic-branch unfreeze.

## Semantic/Dynamic Split

- `trace_unit_factorized_state` semantic params: `['sem_proj.weight', 'sem_proj.bias', 'sem_gate.weight', 'sem_gate.bias', 'norm_sem.weight', 'norm_sem.bias']`
- `trace_unit_factorized_state` dynamic params: `['dyn_gru.weight_ih_l0', 'dyn_gru.weight_hh_l0', 'dyn_gru.bias_ih_l0', 'dyn_gru.bias_hh_l0', 'norm_dyn.weight', 'norm_dyn.bias']`
- `trace_unit_broadcast` semantic params: `['sem_proj.weight', 'sem_proj.bias']`
- `trace_unit_broadcast` dynamic params: `['dyn_proj.weight']`
- `trace_unit_handshake` has semantic-source `k/v` projections, but writeback remains mixed, so handshake stays disabled in V1.

## Conclusion

A safe semantic-only parameter group is available: factorized semantic params + broadcast semantic params + future semantic head + semantic fusion projection + optional readout. Dynamic trace params, tokenizer, and Stage1 remain frozen.
