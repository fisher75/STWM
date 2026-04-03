# STWM D1 Clean Matrix Final Report V1

Generated at: 2026-04-04 00:09:34

## A. Completion Validation

- full_v4_2_seed42_fixed_nowarm_lambda1: state=done, final_step=2000, latest.pt=True, best_protocol_main.pt=True, selection_sidecar=/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_default_v1/seed_42/full_v4_2_seed42_fixed_nowarm_lambda1/checkpoints/best_protocol_main_selection.json
- full_v4_2_seed42_fixed_warmup_lambda1: state=done, final_step=2000, latest.pt=True, best_protocol_main.pt=True, selection_sidecar=/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_default_v1/seed_42/full_v4_2_seed42_fixed_warmup_lambda1/checkpoints/best_protocol_main_selection.json
- wo_semantics_v4_2_seed42: state=done, final_step=2000, latest.pt=True, best_protocol_main.pt=True, selection_sidecar=/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_default_v1/seed_42/wo_semantics_v4_2_seed42/checkpoints/best_protocol_main_selection.json
- wo_object_bias_v4_2_seed42: state=done, final_step=2000, latest.pt=True, best_protocol_main.pt=True, selection_sidecar=/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_default_v1/seed_42/wo_object_bias_v4_2_seed42/checkpoints/best_protocol_main_selection.json

## B. Gradient Audit

### nowarm
- ||g_traj||: first=0.000343280524, median=0.000145139966, last=9.17723228e-05, min=6.80395024e-05, max=0.000343280524, n=20
- ||g_sem||: first=9.31142807e-08, median=6.28601633e-08, last=1.3360907e-08, min=1.3360907e-08, max=7.5974117e-07, n=20
- cos(g_sem, g_traj): first=-0.0119660108, median=0.00126880514, last=0.00137739864, min=-0.0143364455, max=0.0164339832, n=20

### warmup
- ||g_traj||: first=0.00158592535, median=0.000153922228, last=9.89930486e-05, min=5.86195965e-05, max=0.00158592535, n=20
- ||g_sem||: first=0.00017125126, median=1.66911007e-09, last=3.85353555e-12, min=3.85353555e-12, max=0.00017125126, n=20
- cos(g_sem, g_traj): first=-0.0788646873, median=-0.00106845183, last=-1.50241404e-09, min=-0.0788646873, max=0.0120544446, n=20

- warmup持续缓解冲突: False
- query修复后非零: True (qpath_g_query_norm median is non-zero for both nowarm and warmup)

## C. Selection Scorecard

- full_v4_2_seed42_fixed_nowarm_lambda1: step=None, q_loc=0.0066951184934028836, q_top1=0.926208651399491, fut_l1=0.006538258073969955, fut_iou=None, id_cons=None, id_sw=None
- full_v4_2_seed42_fixed_warmup_lambda1: step=None, q_loc=0.008714631579001137, q_top1=0.9007633587786259, fut_l1=0.008738707040102427, fut_iou=None, id_cons=None, id_sw=None
- wo_semantics_v4_2_seed42: step=None, q_loc=0.008400639429043875, q_top1=0.8956743002544529, fut_l1=0.00847303056876168, fut_iou=None, id_cons=None, id_sw=None
- wo_object_bias_v4_2_seed42: step=None, q_loc=0.0022589355403837053, q_top1=0.9796437659033079, fut_l1=0.0024298117105059952, fut_iou=None, id_cons=None, id_sw=None

- full优于wo_sem/wo_object_bias: False
- warmup相对nowarm净收益: False

## D. Efficiency

- full_v4_2_seed42_fixed_nowarm_lambda1: mean_step=3.7966, mean_data=0.8541, mean_wait=0.2292, recent500_step=3.6742, recent500_wait=0.2309, p50=3.6717, p95=4.0778
- full_v4_2_seed42_fixed_warmup_lambda1: mean_step=1.6251, mean_data=0.3459, mean_wait=0.2247, recent500_step=1.4148, recent500_wait=0.2278, p50=1.4974, p95=1.9113
- wo_semantics_v4_2_seed42: mean_step=3.6670, mean_data=0.8353, mean_wait=0.2307, recent500_step=3.5677, recent500_wait=0.2293, p50=3.6188, p95=4.0065
- wo_object_bias_v4_2_seed42: mean_step=1.4651, mean_data=0.3220, mean_wait=0.2293, recent500_step=1.3793, recent500_wait=0.2309, p50=1.3719, p95=1.6858

- frontend_cache在2000-step是否持续显著提速: True
- 提速档位: 3x (factor=3.7153)
- data_wait_ratio是否可接受: False

## E. Final 5 Conclusions

1) 四个run完整成功结束: True
2) warmup vs nowarm梯度冲突: warmup未呈现持续缓解
3) query gradient修复后是否正常: True
4) full是否赢下wo_semantics/wo_object_bias: False
5) frontend_cache默认主线是否成立: False (提速档位=3x)
