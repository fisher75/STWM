# V34.24 claim boundary / gate over-open 风险审计

## 中文结论
V34.23 seed42/123/456 已经形成稳定的 residual probe positive，但 claim 边界必须收紧：可以说 hard/changed residual 与 semantic/assignment/unit memory 路径在 M128/H32 多 seed 上有正信号；不能说 integrated semantic field success。主要 blocker 是 stable gate over-open：stable 区域 gate 大量打开，虽然 stable over-update 很低，但这属于 calibration/sparsity 风险，必须单独修。

## 允许的阶段性 claim
- V34.23 在 V30 frozen、H32/M128、seed42/123/456 上复现了 hard/changed residual 正信号。
- V34.23 的 top-k evidence residual、activation-state gate probe、semantic/assignment/unit intervention 在复现实验中显示 load-bearing。
- V34.23 的 stable preservation 在 seed123/456 val/test 仍为 true，且 stable over-update 率很低。
- V34.23 可以被描述为“受控 residual probe 的多 seed positive”，不是完整 semantic field 成功。

## 禁止的 claim
- 不允许 claim integrated semantic field success。
- 不允许 claim integrated identity field success。
- 不允许 claim learned gate 已经 sparse/calibrated/production-ready。
- 不允许 claim video-to-semantic world model 已闭环；当前仍是 observed trace + observed semantic measurements 输入。
- 不允许把结果外推到 H64/H96、M512 dense 或 1B 规模。
- 不允许推荐写论文或 Overleaf。

## Gate 风险核心证据
- seed123: val stable gate mean=0.482938, test stable gate mean=0.582706, val stable over-open=0.924639, test stable over-open=0.958955, val/test stable over-update=0.000295/0.000442。
- seed456: val stable gate mean=0.475499, test stable gate mean=0.569689, val stable over-open=0.931602, test stable over-open=0.964145, val/test stable over-update=0.000295/0.000442。

## 阶段性分析
这轮不是继续修 bug，也不是进入 H64/H96，而是把跨 seed 复现后的 claim boundary 切干净。最新状态支持“residual memory 在 hard/changed 上有可复现的因果正信号”，但 gate 还没有成为可靠稀疏选择器。stable 区域 gate 大量打开却几乎不造成 over-update，说明主路径保护仍有效，但 gate 本身的选择性不足；如果现在 claim semantic field success，会把 probe positive 和 integrated calibrated field 混为一谈，这是不安全的。

## 论文相关问题解决方案参考
- Slot Attention / object-centric slots: slot/unit memory 必须用分配和干预证明 load-bearing，不能只看最终指标。 https://arxiv.org/abs/2006.15055
- XMem video memory: 视频 memory 需要明确的读写边界和选择策略；当前 gate 过开说明选择策略还不够稀疏。 https://arxiv.org/abs/2207.07115
- Perceiver IO: query-conditioned memory reading 是合理方向，但输出 query 的 gate 仍要做 calibration，而不是默认全读。 https://arxiv.org/abs/2107.14795
- FiLM / conditional modulation: 条件调制要约束作用强度；否则 residual gate 容易变成泛化的全局调制而非 hard-case correction。 https://arxiv.org/abs/1709.07871

## 精确阻塞点
- stable gate over-open 在 seed123/456 的 val/test 上重复出现；这说明 gate 虽然保持 stable 不退化，但没有学到足够稀疏的 hard/changed 选择边界。
- stable over-update 率很低，说明当前主要风险是 calibration / sparsity，而不是直接破坏 stable 输出。
- seed42 原始 summary 没有同等详细的 stable over-open audit；严格 claim 只能依赖 seed123/456 的详细 gate 风险诊断和 seed42 的 top-line 复现。
- 当前没有 identity field 复现主张，identity 仍不能被纳入 integrated claim。
- 当前没有 H64/H96/M512/视频输入闭环复现，仍需留在 frozen V30 M128 范围内。

## 最佳下一步方案
下一轮如果继续，应只做 sparse/calibrated gate 风险处理：引入 stable-negative focal/hinge、预算约束或 temperature/threshold 校准，并强制报告 gate over-open、stable over-update、semantic/assignment/unit intervention delta。不要扩大模型，不要跑 H64/H96，不要 claim semantic field success。

## 关键字段
- claim_boundary_audit_done: `True`
- cross_seed_semantic_hard_positive: `True`
- cross_seed_changed_positive: `True`
- cross_seed_stable_preserved: `True`
- semantic_measurement_load_bearing_replicated: `True`
- assignment_load_bearing_replicated: `True`
- unit_memory_load_bearing_replicated: `True`
- gate_order_replicated: `True`
- stable_gate_overopen_detected: `True`
- stable_overupdate_detected: `False`
- semantic_field_success_claim_allowed: `False`
- integrated_semantic_field_claim_allowed: `False`
- integrated_identity_field_claim_allowed: `False`
- recommended_next_step: `fix_gate_calibration_sparse_gate`
