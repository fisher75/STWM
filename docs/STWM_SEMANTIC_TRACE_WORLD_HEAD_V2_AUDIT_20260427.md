# STWM Semantic Trace World Head V2 Audit 20260427

- audit_passed: `True`
- dynamic_coord_dim_supported: `True`
- coord_dim_policy: MultiHypothesisTraceHead slices a max-dim delta to the runtime coord dimension.
- target_support: semantic/identity targets `[B,K,D]` and `[B,H,K,D]`; confidence `[B]`, `[B,K]`, `[B,H,K]`.

| case | valid | coord_dim | hypothesis_shape |
|---|---:|---:|---|
| multi_hypothesis_dynamic_2d | True | 2 | `[2, 2, 3, 4, 2]` |
| multi_hypothesis_dynamic_3d | True | 3 | `[2, 2, 3, 4, 3]` |
