# STWM World Model Upgrade V3 Medium Train Summary 20260427

- medium_training_started: `false`
- checkpoint_path: `null`
- exact_blocking_reason: `V2 export does not include raw future_semantic_embedding/future_identity_embedding/future_uncertainty tensors; it includes norm/mean summaries, so embedding degeneracy cannot be fully audited.; V2 export forward_scope is head checkpoint forward with manifest surrogate features, not full Stage1/Stage2 feature-based export.; V2 export lacks raw future_semantic_embedding/future_identity_embedding tensors; only norm summaries are available, so unit/horizon embedding degeneracy cannot be ruled out.`

Medium training was intentionally blocked because V3 reality check did not pass the export/eval hardening gate.
