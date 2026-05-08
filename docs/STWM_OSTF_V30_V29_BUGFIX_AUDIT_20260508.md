# STWM OSTF V30 V29 Bugfix Audit

- bugfix_audit_passed: `True`
- legacy_prefight_report_preserved: `True`
- correct_preflight_report_exists: `True`
- external_dataset_preflight_roots: `['/raid/chen034/workspace/stwm/data', '/home/chen034/workspace/stwm/data', '/raid/chen034/workspace/data', '/home/chen034/workspace/data', '/raid/chen034/data', '/home/chen034/data']`
- decision_logic_fixed_fields: `['h32_benchmark_main_ready', 'h64_benchmark_main_ready', 'v29_traceanything_benchmark_main_ready', 'v29_external_gt_benchmark_main_ready']`
- missrate32_saturation_rule: `hard subset last_visible and V28 both 0 or both 1 implies saturated; threshold_auc_needed=true`
- threshold_auc_metric_key: `threshold_auc_endpoint_16_32_64_128`
- external_gt_recursive_cache_discovery_fixed: `True`
- external_gt_cache_discovery_rule: `outputs/cache/stwm_ostf_v30_external_gt/<dataset>/<M_H>/<split>/*.npz via recursive rglob`
