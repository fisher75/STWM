# STWM Object Bias Autopsy V1

Date: 2026-04-04 00:27:12
Seed: 42
Full run: full_v4_2_seed42_fixed_nowarm_lambda1
WO object bias run: wo_object_bias_v4_2_seed42

## 1) Why wo_object_bias Beats full

- Paired clips: 393
- query_localization_error: full_worse_rate=0.984733, mean(full-wo)=0.004436
- future_trajectory_l1: full_worse_rate=0.982188, mean(full-wo)=0.004108
- query_top1_acc: full_worse_rate=0.058524, mean(full-wo)=-0.053435
- future_mask_iou / identity consistency are not degraded at the same magnitude, indicating the dominant failure is query-trajectory axis rather than broad collapse.

## 2) Over-Strong / Over-Early Evidence

- step_0001_0200: delta_query=0.051179, delta_traj=0.051322, delta_objectness_mean=0.248672
- step_0201_0600: delta_query=0.001156, delta_traj=0.001187, delta_objectness_mean=0.551844
- step_0601_1200: delta_query=0.000748, delta_traj=0.000919, delta_objectness_mean=0.718928
- step_1201_2000: delta_query=0.000183, delta_traj=0.000302, delta_objectness_mean=0.691095
- query gap early/late ratio: 279.279138 (large ratio supports over-early bias injection).
- query/traj delta coupling: corr=0.547155, share(query_worse & traj_not_worse)=0.002545

## 3) Most Suspicious Issue

- object_bias_strength_and_timing
- full path appears over-biased to objectness and injects this bias too early: query/trajectory are worse on almost all clips while objectness_mean remains much higher than wo_object_bias, and the largest query gap occurs in early steps.
- Key evidence: query_worse_rate=0.984733, traj_worse_rate=0.982188, query_early_late_ratio=279.279138, late_objectness_gap=0.691095
- Wrong-position signal check: share(query_worse but traj_not_worse)=0.002545; current evidence favors broad query+traj degradation rather than isolated query-anchor misplacement.

## 4) Representative Full-Worse Cases (query_localization_error)

- clip=2216_3Nq9zNAtI3s#0, full=0.036648, wo=0.012389, full-wo=0.024259
- clip=2249_DaeGaqASjeY#0, full=0.040477, wo=0.023541, full-wo=0.016937
- clip=231_-_w6ZFauJBI#0, full=0.115412, wo=0.098752, full-wo=0.016660
- clip=2245_CUJfZkWzx9Q#1, full=0.014063, wo=0.002421, full-wo=0.011643
- clip=1984_Km3UEW34R_Q#0, full=0.014322, wo=0.002775, full-wo=0.011547
- clip=516_5bqIhLCjTzE#0, full=0.017973, wo=0.006876, full-wo=0.011097
- clip=1010_kI0mOZirPGs#0, full=0.012131, wo=0.001634, full-wo=0.010497
- clip=2325_Tp33Bt_mc0c#0, full=0.012158, wo=0.003468, full-wo=0.008690
- clip=2271__5dMMZFgaN4#0, full=0.014928, wo=0.006495, full-wo=0.008433
- clip=1223_xD8VN2r_h2s#0, full=0.009660, wo=0.001270, full-wo=0.008389

## 5) Diagnostic Recommendation

- Priority variants: delayed200 and alpha050 (then alpha025).
- Keep protocol selection rule unchanged; compare at same short-mid endpoint first.
- Warmup is not the primary lever for this failure mode; object-bias timing/strength should be fixed first.

