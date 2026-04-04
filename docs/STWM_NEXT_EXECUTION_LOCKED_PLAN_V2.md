# STWM Next Execution Locked Plan V2

Generated: 2026-04-04

## Inputs Used

- [docs/STWM_SEED123_REPLICATION_FINAL_DECISION_V1.md](docs/STWM_SEED123_REPLICATION_FINAL_DECISION_V1.md)
- [reports/stwm_seed123_replication_final_decision_v1.json](reports/stwm_seed123_replication_final_decision_v1.json)
- [docs/STWM_ALPHA050_CANONICAL_PROMOTION_DECISION_V1.md](docs/STWM_ALPHA050_CANONICAL_PROMOTION_DECISION_V1.md)
- [docs/STWM_POST_SEED123_MAINLINE_PLAN_V1.md](docs/STWM_POST_SEED123_MAINLINE_PLAN_V1.md)
- [reports/stwm_seed42_objdiag_blindbox_readonly_v1.json](reports/stwm_seed42_objdiag_blindbox_readonly_v1.json)

## Locked Judgment

- Seed123 finalizer gives `hold_promotion_cross_seed_inconsistent`.
- Therefore alpha050 does **not** pass cross-seed canonical promotion.
- Warmup path remains downgraded and out of mainline.
- Gated is a valid shadow challenger (seed42 objdiag beats `wo_object_bias` under official rule), but it is **not** yet canonical mainline evidence.

## Unique Mainline (No Branches)

Only one next action is allowed:

1. Stop canonical promotion of alpha050.
2. Launch exactly one minimal gated replacement clean-matrix challenge (single direction only, no extra branches, no concurrency increase).
3. Keep detached/queue/status/log/report workflow unchanged.

## Minimal Required Challenge Matrix

- Challenger run (new): `full_v4_2_seed123_objbias_gated_replacement_challenge_v1`
- Baseline comparator (reuse existing, no rerun): `full_v4_2_seed123_fixed_nowarm_lambda1_rerun_v2`
- Same frozen stack as seed123 clean matrix:
  - protocol: `v2_4_detached_frozen`
  - data mode: `frontend_cache`
  - queue: `outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train`
  - detached background only
- Only changed knob on challenger:
  - `--object-bias-gated`
  - `--object-bias-gate-threshold 0.5`

## Acceptance Gate For Promoting Gated

- Official selection rule beats seed123 full baseline.
- No new stability regressions (NaN/cache corruption/instability).
- If pass: promote gated to next canonical candidate.
- If fail: keep current full baseline and return to object-bias diagnosis.

## Single Command To Send Next To Codex

在 /home/chen034/workspace/stwm 执行且仅执行一个后台主线动作：按 seed123 clean matrix 同配置（frontend_cache + v2_4_detached_frozen + detached queue/status/log/report），发起 1 个新挑战 run `full_v4_2_seed123_objbias_gated_replacement_challenge_v1`（唯一改动 `--object-bias-gated --object-bias-gate-threshold 0.5`），复用已完成 baseline `full_v4_2_seed123_fixed_nowarm_lambda1_rerun_v2` 做 official rule 对比，产出最终挑战决策报告，不新增其他实验、不提高并发。