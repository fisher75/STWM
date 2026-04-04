# STWM QTSA Mainline V1

Generated: 2026-04-04

## 1) 宏观 idea 不变：Trace + Semantics World Model

主假设保持不变：STWM 的核心仍是 `Trace + Semantics` 的世界模型。
- `Trace backbone` 承担世界动态与轨迹演化。
- `Semantics` 提供可解释语义上下文与目标相关线索。
- 目标仍是提升 query localization、轨迹预测与关联稳定性。

## 2) Object 线定位：保留为负结果与动机来源

已有 clean evidence 表明：
- alpha050 canonical promotion 失败（跨 seed 不一致）。
- gated / two_path_residual 只给出局部正信号，尚不足以成为第二主创新主线。

因此 object 线保留为：
- 负结果与设计动机来源。
- 不再作为 headline 路线。

## 3) 当前 QSTR 结果说明的问题

QSTR seed42 clean 结果被 strongest non-object baseline 正式击败，说明：
- 将 semantics 直接注入 transition/state routing，会干扰当前最强 trace-centric 动力学主干。
- 在该实现形态下，语义“进状态转移”带来的副作用大于收益。

结论不是否定 semantics，而是否定“semantics 直接改写 transition”的这一路实现。

## 4) QTSA 核心（唯一新主线）

QTSA = Query-Conditioned Trace-Semantic Alignment / Readout。

核心原则：
- `trace-centric world dynamics backbone`：保持 trace 主干决定 latent transition。
- `query-conditioned trace-semantic alignment/readout`：语义主要在 readout/association 侧按 query 条件对齐。
- `semantics 服务 query localization / selection / association`：强调“读出增强”，而非“转移改写”。
- `不直接改写 latent transition`：语义不进入 transition 输入，避免伤害 strongest trace backbone。

## 5) QTSA + Temporal Semantic Consistency（轻量扩展）

在 QTSA readout-only 基础上增加轻量时序一致性：
- 仅约束 readout/association 语义分布的短时平滑一致性。
- 不引入复杂分叉、不改动 transition 主干。
- 目标是减少 query 条件读出抖动并提升稳定性。

## 6) 为什么当前 clean evidence 支持切到 QTSA

支持切换的证据链：
- 宏观 Trace + Semantics 假设仍成立，失败集中在 object-only 小修补与语义进 transition 的实现。
- QSTR 的失败不是“语义无用”，而是“语义进入 transition 的位置不对”。
- QTSA 将语义作用位置后移到 query-conditioned readout/association，更符合当前 strongest trace backbone 的稳定性约束。
- QTSA 可在现有 detached/frozen/frontend_cache protocol 下最小代价验证，风险可控、可复现。

## Seed42 最小诊断矩阵（本轮）

- `trace_sem_baseline_seed42_qtsa_control_v1`
  - strongest non-object baseline，对照组。
- `qtsa_readout_only_seed42_challenge_v1`
  - query-conditioned trace-semantic alignment/readout，仅 readout 侧生效。
- `qtsa_readout_temporal_consistency_seed42_challenge_v1`
  - 在 readout-only 上增加轻量 temporal semantic consistency。

仅执行以上 3 个 run，不扩展 object-only sweep，不新增 fancy 分叉。
