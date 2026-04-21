# STWM Final Credibility Truth Ledger 20260421

这份账本只基于当前 live repo 的已有 logs / reports / code path 审计，不包含任何新的训练或评测。

## 1. 确实 fresh 跑过的

- `stwm_final_credibility_asset_audit_20260420.json`
  这是运行时对 live filesystem 的 fresh 审计。
- `stwm_final_credibility_live_manifest_20260420.json`
  这是运行时对 code/docs/reports/scripts/configs 的 fresh manifest。
- `stage2_v3p1_dualpanel_context_audit_20260420.json::densified_200_context_preserving`
  这部分是真实 panel eval。log 记录了：
  - `panel_eval_start mode=densified_200_context_preserving items=200 methods=3`
  - `panel_eval_done mode=densified_200_context_preserving valid_items=85 skipped=115`
- `stage2_protocol_v3_extended_evalset_20260420.json::context_preserving_eval`
  这部分也是真实 panel eval。log 记录了：
  - `panel_eval_start mode=protocol_v3_extended_600_context_preserving items=600 methods=3`
  - `panel_eval_done mode=protocol_v3_extended_600_context_preserving valid_items=323 skipped=277`
- `stage2_final_bootstrap_ci_20260420.json`
  它是基于上面 fresh eval 的 per-item results 现算出来的。
- `stage2_v3p1_downstream_utility_20260420.json`
  它是基于上面 fresh eval 中新增的 candidate ranking 现算出来的 retrieval-style probe，不是新的大训练。
- `stwm_final_paper_position_decision_20260420.json`
- `stwm_final_credibility_utility_summary_20260420.json`
- `stwm_final_credibility_utility_diagnosis_20260420.json`
  这三个是 final run 当场写出的 fresh decision / summary 资产。

## 2. 基于现有资产重新聚合的

- `stage2_v3p1_dualpanel_context_audit_20260420.json::legacy_85_context_preserving`
  复用了 `stage2_tusb_v3p3_dualpanel_judge_20260419.json`
- `stage2_v3p1_dualpanel_context_audit_20260420.json::densified_200_single_target`
  也复用了 `stage2_tusb_v3p3_dualpanel_judge_20260419.json`
- `stage2_v3p1_matched_6seed_dualpanel_20260420.json`
  本质是 checkpoint coverage 审计 + 从已有 multiseed 资产抽已有 seed 行，不是 fresh 6-seed eval。
- `stage2_v3p1_mechanism_6seed_20260420.json`
  是从现有 `stage2_tusb_v3p1_seed{42,123,456}_20260418_final.json` 重聚合出来的，不是 fresh 6-seed mechanism run。
- `stage2_final_appearance_plumbing_fix_audit_20260420.json`
  是从 `stage2_v3p1_appearance_plumbing_audit_20260420.json` 再聚合出来的。
- `stwm_final_credibility_protocol_20260420.json`
  是总结性 protocol 文档，不是 fresh 实验结果。
- `stwm_final_credibility_utility_launch_20260420.json`
  只是 launcher metadata。它写着 `asset_based_conservative_completion` 和 `gpu_tasks_materialized=false`，但这不能单独拿来证明“什么都没跑”，因为 log 里确实有 fresh eval。

## 3. 仍然没有 fresh 重算、不能当强证据的

- `matched 6-seed` 主结论
  不能当强证据。原因是主表缺 matched seeds：
  - TUSB-v3.1 缺 `654, 789, 321`
  - cropenc 缺 `654, 321`
  - legacysem 缺 `654, 789, 321`
- `mechanism robustness across 6 seeds`
  不能当强证据。只有 `42, 123, 456` 三个 seed 的 final json 真正在 live repo 里可用。
- `densified_200_context_preserving` 的“完整 200-item hard judge”
  不能当强证据。fresh run 只 materialize 了 `85` 个 valid items，`115` 个被 skip。
- `protocol_v3_extended_600_context_preserving` 的“完整 600-item judge”
  也不能当强证据。fresh run 只 materialize 了 `323` 个 valid items，`277` 个被 skip。
- `appearance_change 已经解决`
  不能当强证据。offline drift 非零，但 batch/loss 路径仍然没有真正激活。

## densified_200_context_preserving 的 skipped=115：具体原因

### 归一化后的主原因

- `future target mask missing`: `115`

这不是“很多种原因各占一点”，而是当前 live repo 里这 `115` 个 skip 全部都归一化到同一个主原因。

### dataset 分布

- `BURST`: `84`
- `VIPSeg`: `31`

### subset tags 在 skipped items 中的覆盖

- `small_object`: `95`
- `appearance_change`: `86`
- `occlusion_reappearance`: `82`
- `crossing_ambiguity`: `78`
- `long_gap_persistence`: `48`

### 最常出现的 clip

- `LaSOT/pool-15`: `6`
- `1102_c00Q1d_Sp6U`: `6`
- `YFCC100M/v_ebde549780bfa5773cf366571f4fb5c`: `5`
- `YFCC100M/v_e3919198858e7ac1a61aae1e486c7664`: `4`
- `YFCC100M/v_8d9acade6c74d4212bc148c54c2c3b20`: `3`

正式 JSON 在：
- `reports/stwm_final_credibility_truth_ledger_20260421.json`
