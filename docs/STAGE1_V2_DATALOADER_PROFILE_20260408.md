# Stage1-v2 Dataloader Profile

- generated_at_utc: 2026-04-08T09:32:40.676184+00:00
- best_batches_per_sec: 1400.8783
- best_config: {"num_workers": 8, "pin_memory": true, "persistent_workers": true, "prefetch_factor": 4}

| workers | pin_memory | persistent_workers | prefetch_factor | status | dataset_init_sec | mean_getitem_sec | mean_collate_sec | batches_per_sec |
|---:|---|---|---:|---|---:|---:|---:|---:|
| 0 | False | False | 0 | pass | 0.0010 | 0.000810 | 0.000102 | 584.9845 |
| 0 | True | False | 0 | pass | 0.0006 | 0.001922 | 0.000150 | 250.6163 |
| 4 | False | False | 2 | pass | 0.0010 | 0.000000 | 0.000000 | 643.3980 |
| 4 | False | False | 4 | pass | 0.0013 | 0.000000 | 0.000000 | 617.0133 |
| 4 | False | True | 2 | pass | 0.0009 | 0.000000 | 0.000000 | 881.3224 |
| 4 | False | True | 4 | pass | 0.0012 | 0.000000 | 0.000000 | 726.0557 |
| 4 | True | False | 2 | pass | 0.0017 | 0.000000 | 0.000000 | 369.5015 |
| 4 | True | False | 4 | pass | 0.0011 | 0.000000 | 0.000000 | 445.8339 |
| 4 | True | True | 2 | pass | 0.0010 | 0.000000 | 0.000000 | 705.6824 |
| 4 | True | True | 4 | pass | 0.0013 | 0.000000 | 0.000000 | 623.3516 |
| 8 | False | False | 2 | pass | 0.0013 | 0.000000 | 0.000000 | 658.7825 |
| 8 | False | False | 4 | pass | 0.0021 | 0.000000 | 0.000000 | 631.3831 |
| 8 | False | True | 2 | pass | 0.0011 | 0.000000 | 0.000000 | 1043.7997 |
| 8 | False | True | 4 | pass | 0.0009 | 0.000000 | 0.000000 | 735.8636 |
| 8 | True | False | 2 | pass | 0.0009 | 0.000000 | 0.000000 | 595.3352 |
| 8 | True | False | 4 | pass | 0.0011 | 0.000000 | 0.000000 | 562.3896 |
| 8 | True | True | 2 | pass | 0.0011 | 0.000000 | 0.000000 | 954.8538 |
| 8 | True | True | 4 | pass | 0.0022 | 0.000000 | 0.000000 | 1400.8783 |
| 16 | False | False | 2 | pass | 0.0013 | 0.000000 | 0.000000 | 355.4285 |
| 16 | False | False | 4 | pass | 0.0012 | 0.000000 | 0.000000 | 432.7054 |
| 16 | False | True | 2 | pass | 0.0012 | 0.000000 | 0.000000 | 672.9261 |
| 16 | False | True | 4 | pass | 0.0023 | 0.000000 | 0.000000 | 624.0684 |
| 16 | True | False | 2 | pass | 0.0010 | 0.000000 | 0.000000 | 416.5789 |
| 16 | True | False | 4 | pass | 0.0022 | 0.000000 | 0.000000 | 280.1907 |
| 16 | True | True | 2 | pass | 0.0046 | 0.000000 | 0.000000 | 723.4322 |
| 16 | True | True | 4 | pass | 0.0029 | 0.000000 | 0.000000 | 436.7294 |
