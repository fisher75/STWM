# STWM Frontend Cache Pilot Plan V1

Date: 2026-04-03
Scope: Pilot-only plan, no full pre-extraction rollout

## 1) 目标

在不做全量预提取的前提下，先用小规模固定切片验证 frontend cache 是否真正改善：

1. `step_time_s`
2. `data_time_s`
3. `data_wait_ratio`
4. 稳定性（是否出现 cache 读损坏导致退出）

## 2) 实验设计（固定 slice / 同配置）

### 2.1 数据切片

固定从 `manifests/protocol_v2/train_v2.json` 按稳定顺序选取一个小切片，例如 256 clips（建议 4 x 64）。

要求:

1. 切片清单单独落盘（pilot manifest）
2. raw 与 frontend_cache 两组使用同一切片

### 2.2 A/B 组别

1. Raw 组: 现有路径（不启用 frontend cache 预提取）
2. Frontend-cache 组: 启用 frontend cache 读取（仅 pilot 切片）

### 2.3 控制变量

保持完全一致:

1. seed（例如 42）
2. model preset（220m）
3. micro-batch / grad-accum（2 / 8）
4. num-workers / prefetch / pin-memory / persistent-workers
5. bf16 / activation-checkpointing
6. 训练步数（建议 120-200 steps）

## 3) 观测指标与统计

主指标（来自 train_log）：

1. `step_time_s` 的均值 / p50 / p95
2. `data_time_s` 的均值 / p50 / p95
3. `data_wait_ratio` 的均值 / p50 / p95

稳定性指标:

1. 是否出现 `BadZipFile` / `EOFError` / 读缓存异常退出
2. 运行中断次数
3. 自动恢复是否成功

次指标:

1. 每小时有效 step（throughput）
2. cache 命中率（若可记录）

## 4) 判定标准（Go / No-Go）

建议 Go 条件（满足其一且稳定性不退化）：

1. `data_wait_ratio` 绝对下降 >= 0.15
2. `step_time_s` 下降 >= 20%

同时必须满足:

1. 不出现新增的 cache 可靠性问题
2. 训练不中断率不低于 raw 组

## 5) 执行顺序

1. 先合入 trace cache hardening（最小修复）
2. 再执行 pilot A/B
3. 最后决定是否扩大到更大切片或全量

原因:

- 若不先修 cache correctness，frontend cache 结论会被损坏重试/崩溃噪声污染。

## 6) 不做事项（本轮明确排除）

1. 不做全量数据预提取
2. 不在 active 生产任务上直接切换路径
3. 不同时叠加多项超参变更

## 7) 预期输出

1. pilot 运行日志与 summary（raw vs frontend_cache 对照）
2. 指标对照表（mean/p50/p95 + 稳定性）
3. 是否进入下一阶段扩容的结论
