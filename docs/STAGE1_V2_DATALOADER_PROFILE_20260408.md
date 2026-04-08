# Stage1-v2 Dataloader Profile

- generated_at_utc: 2026-04-08T09:55:09.213238+00:00
- best_batches_per_sec: 771.8252
- best_config: {"num_workers": 4, "pin_memory": true, "persistent_workers": true, "prefetch_factor": 4}
- reliability: batches_per_sec and dataset_init_time_sec are reliable for all num_workers
- reliability: mean_getitem_time_sec and mean_collate_time_sec are only reliable when num_workers=0
- reliability: worker-side timing for num_workers>0 is explicitly marked unavailable

| workers | pin_memory | persistent_workers | prefetch_factor | status | worker_timing_status | dataset_init_sec | mean_getitem_sec | mean_collate_sec | batches_per_sec |
|---:|---|---|---:|---|---|---:|---|---|---:|
| 0 | False | False | 0 | pass | available_main_process_single_worker | 0.0013 | 0.001902 | 0.000268 | 232.1716 |
| 0 | True | False | 0 | pass | available_main_process_single_worker | 0.0009 | 0.001476 | 0.000112 | 552.9871 |
| 4 | False | False | 2 | pass | unavailable_multiprocess_workers | 0.0008 | unavailable | unavailable | 609.2496 |
| 4 | False | False | 4 | pass | unavailable_multiprocess_workers | 0.0011 | unavailable | unavailable | 438.3813 |
| 4 | False | True | 2 | pass | unavailable_multiprocess_workers | 0.0015 | unavailable | unavailable | 675.1357 |
| 4 | False | True | 4 | pass | unavailable_multiprocess_workers | 0.0010 | unavailable | unavailable | 609.4503 |
| 4 | True | False | 2 | pass | unavailable_multiprocess_workers | 0.0020 | unavailable | unavailable | 88.9514 |
| 4 | True | False | 4 | pass | unavailable_multiprocess_workers | 0.0045 | unavailable | unavailable | 97.3468 |
| 4 | True | True | 2 | pass | unavailable_multiprocess_workers | 0.0036 | unavailable | unavailable | 389.0683 |
| 4 | True | True | 4 | pass | unavailable_multiprocess_workers | 0.0018 | unavailable | unavailable | 771.8252 |
| 8 | False | False | 2 | pass | unavailable_multiprocess_workers | 0.0013 | unavailable | unavailable | 354.6115 |
| 8 | False | False | 4 | pass | unavailable_multiprocess_workers | 0.0010 | unavailable | unavailable | 109.8026 |
| 8 | False | True | 2 | pass | unavailable_multiprocess_workers | 0.0031 | unavailable | unavailable | 150.9953 |
| 8 | False | True | 4 | pass | unavailable_multiprocess_workers | 0.0024 | unavailable | unavailable | 263.2778 |
| 8 | True | False | 2 | pass | unavailable_multiprocess_workers | 0.0014 | unavailable | unavailable | 114.2950 |
| 8 | True | False | 4 | pass | unavailable_multiprocess_workers | 0.0010 | unavailable | unavailable | 601.2507 |
| 8 | True | True | 2 | pass | unavailable_multiprocess_workers | 0.0010 | unavailable | unavailable | 708.7326 |
| 8 | True | True | 4 | pass | unavailable_multiprocess_workers | 0.0021 | unavailable | unavailable | 516.6225 |
| 16 | False | False | 2 | pass | unavailable_multiprocess_workers | 0.0025 | unavailable | unavailable | 130.7739 |
| 16 | False | False | 4 | pass | unavailable_multiprocess_workers | 0.0038 | unavailable | unavailable | 83.6716 |
| 16 | False | True | 2 | pass | unavailable_multiprocess_workers | 0.0032 | unavailable | unavailable | 734.7431 |
| 16 | False | True | 4 | pass | unavailable_multiprocess_workers | 0.0032 | unavailable | unavailable | 589.8861 |
| 16 | True | False | 2 | pass | unavailable_multiprocess_workers | 0.0010 | unavailable | unavailable | 355.5127 |
| 16 | True | False | 4 | pass | unavailable_multiprocess_workers | 0.0010 | unavailable | unavailable | 211.6891 |
| 16 | True | True | 2 | pass | unavailable_multiprocess_workers | 0.0037 | unavailable | unavailable | 127.1289 |
| 16 | True | True | 4 | pass | unavailable_multiprocess_workers | 0.0054 | unavailable | unavailable | 183.0427 |
