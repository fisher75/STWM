# Stage2 Auxiliary External Probe Batch

- scope: adapter-based TAP-style probe only; not official benchmark
- generated_at_utc: 2026-04-14T17:25:22.682448+00:00
- best_calibration_probe_target: calibration_only_wave1_best
- calibration_only_not_worse_than_cropenc_on_aux_probe: True
- calibration_only_not_worse_than_legacysem_on_aux_probe: True

| name | run_name | probe_status | tap_style_eval_status | average_jaccard | avg_pts_within_thresh | adapter_probe_only |
|---|---|---|---|---:|---:|---|
| stage1_frozen_baseline | stage1_frozen_baseline | unsupported |  | 1000000000.000000 | 1000000000.000000 | True |
| legacysem_best | stage2_fullscale_core_legacysem_seed456_wave2_20260409 | completed | partially_bridged | 0.200000 | 0.200000 | True |
| cropenc_baseline_best | stage2_fullscale_core_cropenc_seed456_20260409 | completed | partially_bridged | 1.000000 | 1.000000 | True |
| v7_alignment_only_best | stage2_semobjv7_alignonly_topk1_seed123_20260413 | completed | partially_bridged | 1.000000 | 1.000000 | True |
| calibration_only_wave1_best | stage2_calonly_topk1_seed123_wave1_20260413 | completed | partially_bridged | 1.000000 | 1.000000 | True |
| calibration_only_wave2_best | stage2_calonly_topk1_seed654_wave2_20260414 | completed | partially_bridged | 1.000000 | 1.000000 | True |
