# STWM Light Readout Utility Plumbing Audit 20260422

- official_final_eval_report: `/raid/chen034/workspace/stwm/reports/stwm_lightreadout_final_eval_20260422.json`
- why_probe_counts_zero: The previous light-readout utility report consumed the official final-eval rows with the old nested `row["methods"][method_name]` schema assumption. The live official final-eval report stores flat `per_item_results` rows keyed by `method_name` and `scoring_mode`, so every method lookup missed and every probe count collapsed to zero.
- row_schema_mismatch: True
- exact_breakpoint_1: `/raid/chen034/workspace/stwm/code/stwm/tools/run_stwm_downstream_utility_v2_20260420.py:76`
- exact_breakpoint_2: `/raid/chen034/workspace/stwm/code/stwm/tools/run_stwm_downstream_utility_v2_20260420.py:94`
- exact_breakpoint_3: `/raid/chen034/workspace/stwm/code/stwm/tools/run_stwm_top_tier_downstream_utility_20260420.py:75`
- current_row_count: 3576
- sample_method_name: `TUSB-v3.1::official(best_semantic_hard.pt+hybrid_light)`
- fix_strategy: read flat per_item_results rows, rebuild rankings from explicit score maps or ranked_candidate_ids, then run test-only utility probes and paired bootstrap
