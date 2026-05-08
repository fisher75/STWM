# STWM OSTF V30 H96 Readiness Audit

- h96_m128_ready: `True`
- test_h96_motion_item_count: `174`
- split_counts_H96_M128: `{'train': 187, 'val': 108, 'test': 231}`
- strongest_prior_H96: `last_observed_copy`
- missrate_saturation: `{'last_observed_copy': {'item_count': 231, 'motion_item_count': 174, 'motion_MissRate@32': 1.0, 'motion_MissRate@64': 1.0, 'motion_MissRate@128': 0.6494252873563219, 'motion_threshold_auc_endpoint_16_32_64_128': 0.08764367816091954}, 'last_visible_copy': {'item_count': 231, 'motion_item_count': 174, 'motion_MissRate@32': 1.0, 'motion_MissRate@64': 1.0, 'motion_MissRate@128': 0.6494252873563219, 'motion_threshold_auc_endpoint_16_32_64_128': 0.08764367816091954}, 'visibility_aware_damped': {'item_count': 231, 'motion_item_count': 174, 'motion_MissRate@32': 1.0, 'motion_MissRate@64': 1.0, 'motion_MissRate@128': 0.6494252873563219, 'motion_threshold_auc_endpoint_16_32_64_128': 0.08764367816091954}, 'visibility_aware_cv': {'item_count': 231, 'motion_item_count': 174, 'motion_MissRate@32': 1.0, 'motion_MissRate@64': 0.9770114942528736, 'motion_MissRate@128': 0.896551724137931, 'motion_threshold_auc_endpoint_16_32_64_128': 0.031609195402298854}, 'fixed_affine': {'item_count': 231, 'motion_item_count': 174, 'motion_MissRate@32': 1.0, 'motion_MissRate@64': 0.9712643678160919, 'motion_MissRate@128': 0.735632183908046, 'motion_threshold_auc_endpoint_16_32_64_128': 0.07327586206896551}, 'missrate32_saturated': True, 'threshold_auc_endpoint_16_32_64_128_required': True}`
- train_val_test_video_level_leakage_check: `{'overlap_counts': {'train_val': 0, 'train_test': 0, 'val_test': 0}, 'passed': True}`
- exact_blocker: `None`
