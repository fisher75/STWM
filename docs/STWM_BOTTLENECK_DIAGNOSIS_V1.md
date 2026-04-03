# STWM Bottleneck Diagnosis V1

## 1) 执行摘要

在 D1 当前真实运行态中，核心问题不是“模型算不动”，而是“input pipeline + cache correctness”。

补充校正：共享集群上的整卡利用率（whole-GPU util）是全体用户进程加总，不能直接当成我方 STWM 进程利用率。基于 active train pid 的进程级采样，我方进程 SM 利用率很低，不支持“我方训练已吃满 GPU”的说法。

## 2) 证据链

### 2.1 实际并发状态

- 两个 run 仍 active（nowarm / wo_object_bias），两个 run failed（warmup / wo_semantics）。
- 失败日志均指向 trace cache 读取异常:
  - `BadZipFile: File is not a zip file`
  - `EOFError: No data left in file`

证据文件:

- `reports/d1_runtime_liveness_audit.json`
- `outputs/queue/stwm_protocol_v2/d1_train/queue_events.log`
- `outputs/queue/stwm_protocol_v2/d1_train/logs/1775200846981_wo_semantics_v4_2_seed42.log`
- `outputs/queue/stwm_protocol_v2/d1_train/logs/1775200846820_full_v4_2_seed42_fixed_warmup_lambda1.log`

### 2.2 时间分解证据

active run（recent30）:

- step_avg ≈ 37.3s
- data_avg ≈ 35.0-35.3s
- compute_avg ≈ 2.0-2.3s
- data_wait_ratio_avg ≈ 0.93-0.94

证据文件:

- `reports/d1_step_timing_breakdown.json`

说明:

- 该结论来自我方 train_log 的 step/data 统计，不依赖整卡 util。

### 2.2b GPU 解释校正（whole-GPU vs our-process）

来自 `reports/d1_process_gpu_util_correction.json`（active train pid，60 样本）：

- full_nowarm（pid=3936401）:
   - our-process SM util avg=0.67%，p95=9.00%
   - used_mem≈5074 MiB
   - host whole-GPU4 util avg=92.82%
- wo_object_bias（pid=3936697）:
   - our-process SM util avg=0.77%，p95=4.00%
   - used_mem≈5074 MiB
   - host whole-GPU3 util avg=55.62%

结论:

- 不能用整卡 util 推断“我方 run 很忙”；应以 train pid 的进程级指标为准。

### 2.3 代码级路径证据

`data_time_s` 在主循环中累加两段:

1. `next_micro_batch()`
2. `_build_features_for_sample(...)`

而 `_build_features_for_sample(...)` 内部执行:

- `trace_adapter.encode(...)`
- `semantic_adapter.encode(...)`
- 对每帧 mask 调用 `_read_mask_ratio(...)`（`Image.open`）
- 构建并搬运 tensor 到 device

这说明“数据等待”包含了大量主循环内前端特征工作，而非仅 DataLoader worker 排队。

### 2.4 数据组织与 I/O 形态

- 训练清单由大量 `.jpg` / `.png` 小文件组成（无视频流路径）。
- 这类组织形式通常更偏向小文件元数据访问 + 高频解码，容易压在 CPU 路径。

证据文件:

- `reports/d1_data_layout_audit.json`

### 2.5 worker 利用与 CPU affinity

- train 主进程 CPU 约 99%。
- `pt_data_worker` 子进程多数接近 0%。
- affinity 全部 `0-223`，无绑核导致的硬限制。

解释: 主要热点并未下沉到 DataLoader worker，而是留在训练主进程。

### 2.6 checkpoint/eval 干扰是否成立

- 触发周期 500 步，当前 active run ~113-114 步。
- checkpoint 目录尚无周期产物。

结论: 当前窗口不是 checkpoint/eval 干扰期。

## 3) 主瓶颈 Top 3（按影响排序）

1. 主循环内特征构建占比过高（高影响，高置信）
   - 证据: `data_wait_ratio` 近期约 0.93-0.94，`compute_avg` 仅约 2s。

2. trace cache 并发一致性不足导致 run 失败（高影响，高置信）
   - 证据: 两个 run 在 `trace_adapter` 读缓存时报 `BadZipFile` / `EOFError`。
   - 代码对比: semantic cache 已实现临时文件 + 原子替换 + 坏文件隔离；trace cache 尚未对齐此安全写入范式。

3. 小文件路径 + 每帧 mask 读取放大数据路径开销（中高影响，中高置信）
   - 证据: 238,614 张 jpg + 238,614 张 png，且每样本构建时逐帧读 mask。

注:

- 本排序不使用“整卡 util 高”作为我方算力利用证据。

## 4) frontend cache / feature pre-extraction 是否值得

结论: 值得，且优先级高。

理由:

- 当前 compute 时间只占小头，任何降低数据/前端耗时的手段都会直接提升吞吐。
- 仅调模型超参很难覆盖 30+s 的 data path 开销。
- 但 cache 必须具备并发安全，否则会以可靠性崩溃抵消收益。

建议顺序:

1. 先做 cache 正确性加固（原子写 + 损坏隔离 + 读失败重建）
2. 再做 feature pre-extraction/offline 预计算
3. 最后再调 worker/prefetch 等参数

## 5) 如果只能做一个优化，最优项

最优单项: 修复 trace cache 的并发安全与损坏恢复。

原因:

- 这是吞吐和稳定性的共同前提。
- 当前已有两条 run 因此失败，边际收益立刻可见（恢复有效并发 + 避免重复失败）。
- 实现范式可直接参考 semantic cache 已有实现（临时文件 + `os.replace` + quarantine）。

## 6) 先不要做的优化

1. 仅靠提升 `num_workers` / `prefetch_factor` 期待显著改善
   - 当前瓶颈主要在主循环 `_build_features_for_sample`，不是 worker 侧 `__getitem__`。

2. 仅做 GPU 级调度/并发策略微调
   - 现阶段主因是数据前端与 cache 可靠性，不是 GPU 选卡规则；且整卡 util 高不代表我方进程 util 高。

3. 在未修复 trace cache 前继续扩大并发
   - 并发会放大缓存竞争与损坏概率，可能导致更多 run 异常退出。

## 7) 风险与置信度

- 结论置信度: 高（有运行日志、队列状态、代码路径、5 分钟采样四类证据交叉支持）。
- 主要残余不确定性:
  - 本轮未对单 step 进行更细粒度算子级 profiler（例如 PyTorch profiler 的 op 时间树）。
  - 但在 data/compute 量级差（35s vs 2s）下，不影响主因判定。

## 8) 对后续实验设计的影响

- 下一轮性能实验应先保证 4/4 run 稳定活跃，再比较吞吐。
- 若稳定性未先解决，任何效率结论都会混入“任务提前退出”的偏差。