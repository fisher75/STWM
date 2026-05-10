# STWM OSTF V33.7 Identity Training Forensics

- train_sample_count: `47`
- complete_samples_by_split: `{'train': 47, 'val': 42, 'test': 44}`
- training_coverage_bottleneck_detected: `True`
- same_instance_hard_bce_missing: `True`
- embedding_logit_mismatch_detected: `True`
- threshold_calibration_problem_detected: `True`
- recommended_fix: `Expand/record H32 complete target coverage, train BCE on balanced hard identity masks, add observed-anchor embedding similarity logits, and calibrate fused same-instance belief using validation threshold.`
