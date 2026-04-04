# STWM QSTR Mainline V1

Generated: 2026-04-04

## 1) Headline 仍是 Trace + Semantics World Model

当前主张不变：STWM 的核心仍然是 `Trace + Semantics` 的世界模型建模能力。
- `Trace backbone` 负责时序几何与动态轨迹。
- `Semantic state` 负责语义上下文与可解释 token 选择。
- 主目标仍是通过统一时空状态来提升 query 定位与未来轨迹质量。

## 2) Object 线定位：保留为负结果与动机，不再作为 headline

已有 clean evidence 显示：
- alpha050 canonical promotion 失败（跨 seed 不一致）。
- gated 与 two_path_residual 有价值，但不足以把 static object bias / simple gate 立为第二主创新。

因此 object-only 小修小补在论文叙事中转为：
- 负结果与动机来源（为什么简单 object bias 不足）。
- 不是下一阶段 headline 路线。

## 3) QSTR 核心结构（唯一新主线）

QSTR = Query-Conditioned Semantic Trace Routing。

核心组件：
- Trace backbone：保持当前 v4.2 主干与 memory 机制。
- Semantic state：保留语义通道，不做 disable。
- Query-conditioned semantic residual routing：
  - 使用 query 条件信号构造语义 residual 路由强度。
  - 在语义特征上做轻量 residual 注入，而不是替代主干。
- Neutral path 保留：
  - 始终保留原语义主路径（neutral path）。
  - QSTR 只作为 residual 增强，避免“全量改道”导致不稳定。

## 4) QSTR + Temporal Semantic Consistency（轻量扩展）

在 QSTR-only 基础上增加一个轻量时序一致性项：
- 对相邻时刻的语义分布加入平滑一致性正则。
- 目标是约束语义路由的短时抖动，避免 query 条件导致的瞬时语义漂移。
- 该项是最小增量，不引入复杂分叉或新分支训练流程。

## 5) 为什么当前 clean evidence 支持切换到 QSTR

切换理由：
- 主体 world model 假设仍有效，失败主要集中在 object-only 控制层而非主干。
- two_path_residual 在 seed42 三段式中表现最强，说明“保留 neutral path + 条件 residual”方向有效。
- 但它尚未在 seed123 完成 promotion gate，因此需要把“条件 residual”上升为更一般的 query-conditioned semantic routing 进行最小诊断验证。
- QSTR 能直接复用现有 detached/frozen/frontend_cache 实验栈，低风险、可追踪、可复现实证。

## Seed42 最小诊断矩阵（本轮）

- `trace_sem_baseline_seed42_qstr_control_v1`
  - 目标：当前最强 non-object baseline 对照。
- `qstr_only_seed42_challenge_v1`
  - 目标：验证 query-conditioned semantic residual routing + neutral path。
- `qstr_temporal_consistency_seed42_challenge_v1`
  - 目标：在 QSTR-only 上评估轻量 temporal semantic consistency 的增益。

仅这 3 个 run，不扩展 alpha/gated/delay sweep，不新增 fancy 分叉。
