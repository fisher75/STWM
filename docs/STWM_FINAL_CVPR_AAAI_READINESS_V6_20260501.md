# STWM Final CVPR AAAI Readiness V6 20260501

## Status
- ready_for_cvpr_aaai_main: `unclear`
- ready_for_overleaf: `True`
- next_step_choice: `run_missing_baselines`
- strongest_table: `mixed_free_rollout_main_result`
- weakest_table: `same_output_baseline_suite`

## Remaining Risks
- same-output baseline suite is still incomplete
- model-size scaling is incomplete
- horizon scaling is incomplete
- trace-density scaling is incomplete
- paper-ready 8-step rollout video/figure pack is incomplete
- LODO is negative and must be positioned as domain shift rather than universal generalization

## Required Additional Runs
- same-output baseline suite: trace-only / semantic-only / semantic+trace / SlotFormer-like / DINO-WM-like
- prototype scaling with missing C16 controlled mixed runs
- model-size scaling small/base/large
- horizon scaling H16/H24 cache rebuild + retrain/eval
- trace-density scaling K16/K32 cache rebuild + retrain/eval
- actual 8-step rollout visualization generation
