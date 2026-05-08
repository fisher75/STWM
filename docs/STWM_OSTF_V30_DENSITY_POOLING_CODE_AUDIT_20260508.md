# STWM OSTF V30 Density Pooling Code Audit

- code_audit_passed: `True`
- py_compile_ok: `True`
- pooling_modes_supported: `['mean', 'moments', 'induced_attention', 'motion_topk', 'hybrid_moments_attention']`
- default_mode_reproduces_old_mean_behavior: `True`
- full_mxm_self_attention_used: `False`
- induced_attention_complexity: `O(M*K)`
