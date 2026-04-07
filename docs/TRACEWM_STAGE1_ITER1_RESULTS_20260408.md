# TraceWM Stage 1 Iteration-1 Results (2026-04-08)

- generated_at_utc: 2026-04-07T17:56:54.589619+00:00
- protocol_doc: /home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_ITERATION1_PROTOCOL_20260408.md
- splits_doc: /home/chen034/workspace/stwm/docs/STAGE1_ITER1_SPLITS_20260408.md
- comparison_json: /home/chen034/workspace/stwm/reports/tracewm_stage1_iter1_comparison_20260408.json

## Run Metrics

| run | stability_score | val_total_loss | tf_free_gap | tapvid_free_endpoint_l2 | tapvid3d_free_endpoint_l2_limited |
|---|---:|---:|---:|---:|---:|
| pointodyssey_only | 0.000000 | 0.000000 | -0.000000 | 0.264460 | 13.606836 |
| kubric_only | 0.000000 | 0.000001 | 0.000000 | 0.207753 | 12.850231 |
| joint_po_kubric | 0.000002 | 0.000002 | -0.000000 | 0.260326 | 11.736500 |

## Required Answers

1. Stability winner (PointOdyssey-only vs Kubric-only): pointodyssey_only
2. Joint better than best single: False
3. Teacher-forced vs free gap: {'pointodyssey_only': -3.967649320202327e-09, 'kubric_only': 8.089216407825006e-09, 'joint_po_kubric': -7.36886249796953e-08}
4. Best on TAP-Vid: kubric_only
5. Best on TAPVid-3D limited eval: joint_po_kubric
6. Next-round decision: stage1_model_fix_round
