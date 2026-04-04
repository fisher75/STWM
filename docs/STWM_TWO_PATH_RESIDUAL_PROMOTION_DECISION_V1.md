# STWM Two Path Residual Promotion Decision V1

Generated: 2026-04-04 16:15:18

## A. Inputs

- delayed router seed42 report: /home/chen034/workspace/stwm/reports/stwm_delayed_router_mainline_seed42_report_v1.json
- replacement seed42 decision: /home/chen034/workspace/stwm/reports/stwm_replacement_clean_matrix_seed42_final_decision_v1.json
- seed42 gated reference (blindbox): /home/chen034/workspace/stwm/reports/stwm_seed42_objdiag_blindbox_readonly_v1.json
- seed123 gated optional reference: /home/chen034/workspace/stwm/reports/stwm_gated_challenge_seed123_final_decision_v1.json

## B. Official Rule Comparisons (qloc asc, qtop1 desc, l1 asc)

| comparison | two_path_beats_official | tie | d_qloc | d_qtop1 | d_l1 |
|---|---|---|---:|---:|---:|
| two vs delayed_only | True | False | -0.000934 | 0.022901 | -0.000713 |
| two vs delayed_residual_router | True | False | -0.000330 | 0.000000 | -0.000158 |
| two vs current_full_nowarm | True | False | -0.001634 | 0.030534 | -0.001355 |
| two vs alpha050_replacement | False | True | 0.000000 | 0.000000 | 0.000000 |
| two vs wo_object_bias_seed42 | False | False | 0.002803 | -0.022901 | 0.002753 |
| two vs seed42_gated_reference | False | False | 0.000323 | 0.020356 | 0.000592 |

## C. Required Answers

1) two_path wins current full_nowarm: True
2) two_path wins alpha050 replacement: False (tie=True)
3) two_path wins seed42 gated best reference: False (seed42 gated here is reference-only blindbox track)
4) two_path wins wo_object_bias: False
5) if not win wo_object_bias, gap is closer than alpha/gated: alpha=False, seed42_gated_ref=False

## D. Unique Promotion Verdict

- verdict: B
- verdict_text: two_path_residual 是当前最强新主线，但仍未足够 promotion；下一步只做最小 two_path_residual seed123 challenge

