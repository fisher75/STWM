# Stage1-v2 GPU Selection Audit

- selected_gpu_id: 6
- required_mem_gb: 40.0
- safety_margin_gb: 8.0
- sample_count: 12
- sample_interval_sec: 2.0
- lease_id: d1b3336f-f3c2-423b-a407-f3d434780f8f

| gpu_id | free_mem_gb | avg_gpu_util | avg_mem_util | active_compute_process_count | selected | selected_reason |
|---:|---:|---:|---:|---:|---|---|
| 0 | 126.16 | 92.50 | 13.83 | 4 | False | candidate_not_top_rank |
| 1 | 81.60 | 94.25 | 16.08 | 3 | False | candidate_not_top_rank |
| 2 | 72.62 | 93.33 | 16.67 | 4 | False | candidate_not_top_rank |
| 3 | 69.42 | 81.08 | 28.75 | 1 | False | candidate_not_top_rank |
| 4 | 132.99 | 92.25 | 5.00 | 3 | False | candidate_not_top_rank |
| 5 | 88.45 | 92.33 | 13.58 | 3 | False | candidate_not_top_rank |
| 6 | 144.75 | 8.42 | 0.00 | 2 | True | best_rank_after_window_sampling |
| 7 | 123.75 | 25.58 | 3.08 | 2 | False | candidate_not_top_rank |
