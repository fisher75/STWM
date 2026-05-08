# STWM OSTF V29 Benchmark Decision

- v29_benchmark_main_ready: `False`
- h32_main_ready: `False`
- h64_main_ready: `False`
- h32_benchmark_main_ready: `False`
- h64_benchmark_main_ready: `False`
- v29_traceanything_benchmark_main_ready: `False`
- v29_external_gt_benchmark_main_ready: `False`
- h64_stress_only: `True`
- last_visible_prior_dominates_after_fix: `True`
- missrate32_saturated: `True`
- threshold_auc_needed: `True`
- dataset_balance_ok: `True`
- external_gt_dataset_needed: `True`
- recommended_next_step: `integrate_PointOdyssey_TAPVid3D_GT`
- decision_rationale: `H64 remains stress-only and/or last_visible_copy still dominates key anti-prior subsets; do not train a new OSTF-v2 model until the benchmark route is fixed with external GT or expanded motion clips.`
