# STWM OSTF V29 Anti-Prior Hardbench Protocol

- manifest_dir: `manifests/ostf_v29_antiprior`
- total_item_counts: `{'train': 795, 'val': 122, 'test': 97}`
- per_dataset_counts: `{'train': {'VIPSEG': 144, 'VSPW': 651}, 'val': {'VIPSEG': 22, 'VSPW': 100}, 'test': {'VIPSEG': 24, 'VSPW': 73}}`
- per_subset_counts: `{'test_h32_mixed': {'item_count': 90, 'by_dataset': {'VIPSEG': 24, 'VSPW': 66}, 'by_combo': {'M128_H32': 90}, 'by_subset': {'anti_prior_motion': 41, 'extraction_uncertainty': 30, 'last_visible_hard': 42, 'nonlinear_large_disp': 24, 'occlusion_reappearance': 60, 'semantic_confuser': 10}}, 'test_h64_motion': {'item_count': 2, 'by_dataset': {'VSPW': 2}, 'by_combo': {'M128_H64': 2}, 'by_subset': {'anti_prior_motion': 2, 'last_visible_hard': 2, 'occlusion_reappearance': 2}}, 'test_occlusion_reappearance': {'item_count': 64, 'by_dataset': {'VIPSEG': 17, 'VSPW': 47}, 'by_combo': {'M128_H32': 60, 'M128_H64': 4}, 'by_subset': {'anti_prior_motion': 36, 'extraction_uncertainty': 15, 'last_visible_hard': 37, 'nonlinear_large_disp': 18, 'occlusion_reappearance': 64, 'semantic_confuser': 4}}, 'test_semantic_confuser': {'item_count': 10, 'by_dataset': {'VSPW': 10}, 'by_combo': {'M128_H32': 10}, 'by_subset': {'anti_prior_motion': 4, 'extraction_uncertainty': 1, 'last_visible_hard': 4, 'occlusion_reappearance': 4, 'semantic_confuser': 10}}, 'test_nonlinear_large_disp': {'item_count': 24, 'by_dataset': {'VIPSEG': 4, 'VSPW': 20}, 'by_combo': {'M128_H32': 24}, 'by_subset': {'anti_prior_motion': 15, 'extraction_uncertainty': 9, 'last_visible_hard': 16, 'nonlinear_large_disp': 24, 'occlusion_reappearance': 18}}}`
- h32_main_ready: `False`
- h64_main_ready: `False`
- h64_stress_only: `True`
- v29_benchmark_main_ready: `False`
- main_ready_note: `TraceAnything anti-prior split is diagnostic until H32 count/balance improves or external GT benchmark is integrated.`
- no_test_selection_rule: `All thresholds are selected on train/val only; test is filtered once by the frozen rule.`
