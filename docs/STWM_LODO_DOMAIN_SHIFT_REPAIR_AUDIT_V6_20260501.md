# STWM LODO Domain Shift Repair Audit V6 20260501

## Status
- lodo_completed: `True`
- trusted_lodo_conclusion: `negative`
- lodo_domain_shift_diagnosed: `True`
- cache_bug_found: `False`
- cache_bug_fixed: `False`
- rerun_required: `False`

## Diagnosis
- crop_feature_norm_mean VSPW: `1.0`
- crop_feature_norm_mean VIPSEG: `0.0`
- C32 JS divergence: `0.16038782546361047`
- C64 JS divergence: `0.1693801967488865`

## Claim Boundary
- LODO is a domain-shift appendix/limitation. Do not claim universal cross-dataset generalization; do not reinterpret the negative as world-model failure because mixed free-rollout remains positive and trace regression stays false.
