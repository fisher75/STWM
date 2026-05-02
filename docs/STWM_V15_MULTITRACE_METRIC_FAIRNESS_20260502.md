# STWM V15 MultiTrace Metric Fairness

- M1_L1_vs_M512_point_L1_comparable: `False`
- M512_improves_coverage: `False`
- M512_improves_anchor_or_extent: `improves_extent_coverage_not_anchor_metric`
- current_failure_due_to_metric_mismatch: `True`
- visibility_F1: `invalid_for_claim_pseudo_valid_mask_not_tracker_occlusion`
- interpretation: `M1 predicts an object anchor/centroid-like target while M512 averages many internal pseudo-points. Those losses are not a fair final dense evidence comparison, especially without a physical teacher.`
