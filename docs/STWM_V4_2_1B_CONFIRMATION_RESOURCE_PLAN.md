# STWM V4.2 1B Confirmation Resource Plan

更新时间：2026-04-01 13:03 +08

## C1. 4-8 卡需求是否应保持？

审计结论：不建议继续把 `stwm_1b_confirmation` 设为 `min_gpus=4`。

理由：

- 当前训练入口 `train_stwm_v4_2.py` 使用 `torch.device("cuda")` 单进程执行。
- 队列中的 `prefer/min` 仅影响 `CUDA_VISIBLE_DEVICES` 可见集合，不会自动变成 DDP。
- 因此在当前实现下，`min_gpus=4` 只会降低可启动概率，几乎不带来吞吐收益。

建议：

- confirmation 任务改为 `prefer_gpus=1, min_gpus=1`。

## C2. 共享 8xB200 下最合理启动策略

建议采用“分阶段执行”（推荐）：

1. 阶段 0（已完成）：
   - 先跑通 smoke，确认执行链可靠。
2. 阶段 1（保守起步）：
   - confirmation 仅跑 seed 42（base + identifiability），先验证长任务链路。
3. 阶段 2（扩展稳定性）：
   - 增加 seed 123。
4. 阶段 3（完整确认）：
   - 再补 seed 456 与可视化打包。

不建议的策略：

- 等 8 卡再跑：在当前共享占用下很可能长期不启动。
- 继续 4~8 卡阈值 + 近空闲条件：会把任务卡死在等待状态。

## C3. 若资源长期抢不到，现实可运行队列配置

面向共享高负载的现实配置（推荐）：

- `prefer-gpus=1`
- `min-gpus=1`
- `poll-seconds=30`
- `max-mem-used-mib=90000`
- `max-utilization=98`
- `candidate-gpus=0,1,3,7`（可按现场更新）

理由：

- 本轮观测到多数卡长期高利用率，严格空闲阈值（2000/20）几乎无触发机会。
- smoke 真实运行峰值约 90GiB；`max-mem-used-mib=90000` 可过滤接近满显存卡，同时允许可运行窗口。

更保守但可能更慢的配置：

- `max-mem-used-mib=80000`
- `max-utilization=95`

## C4. 当前队列状态下的直接建议

当前 `stwm_1b_confirmation` 仍为：

- `prefer=8, min=4, max_mem=2000, max_util=20`

建议下一步：

1. 将 confirmation 资源门槛改为单卡可启动。
2. 先跑阶段 1（seed 42）验证长链稳定。
3. 再按阶段扩展到完整 confirmation。
