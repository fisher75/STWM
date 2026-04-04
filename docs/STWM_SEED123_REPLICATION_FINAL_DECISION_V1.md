# STWM Seed123 Replication Final Decision V1

Generated: 2026-04-04 13:53:24
All complete: True

| role | run_name | state | selected_best_step | qloc | qtop1 | l1 | iou | id_consistency | id_switch_rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| current full baseline | full_v4_2_seed123_fixed_nowarm_lambda1_rerun_v2 | done | 2000 | 0.008337827849630788 | 0.8982188295165394 | 0.008263798046658057 | 0.16048144346295068 | 0.9990458015267175 | 0.0 |
| alpha050 replacement | full_v4_2_seed123_objbias_alpha050_replacement_v1 | done | 2000 | 0.009226071508482819 | 0.8727735368956743 | 0.009275155431062514 | 0.1604693212705619 | 0.9984096692111959 | 0.0 |
| wo_semantics control | wo_semantics_v4_2_seed123_control_v1 | done | 2000 | 0.006679702123612848 | 0.9287531806615776 | 0.006662510458673837 | 0.160484419504266 | 0.9990458015267175 | 0.0 |
| wo_object_bias control | wo_object_bias_v4_2_seed123_control_v1 | done | 2000 | 0.008151991562988922 | 0.9083969465648855 | 0.008287249180386388 | 0.16046858846114917 | 0.9990458015267175 | 0.0 |

Seed123 replacement beats baseline (official): False
Cross-seed consistent: False
Canonical promotion decision: hold_promotion_cross_seed_inconsistent

