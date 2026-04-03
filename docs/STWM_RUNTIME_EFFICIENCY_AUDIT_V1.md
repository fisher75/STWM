# STWM Runtime Efficiency Audit V1

## 1) 审计范围与约束

- 审计对象: D1 诊断矩阵 4 个 run（seed=42）。
- 约束: 只读审计，不停止或修改任何正在运行训练。
- 审计窗口:
  - 连续系统采样窗口约 5 分钟（10 秒间隔，30-31 个样本）。
  - 训练日志统计截至本次审计生成时刻。

## 2) 证据来源

- 运行态快照: `reports/d1_runtime_snapshot.json`
- 连续采样原始数据: `reports/d1_resource_timeseries.json`
- 连续采样统计汇总: `reports/d1_resource_summary.json`
- 进程级 GPU 校正采样: `reports/d1_process_gpu_util_correction.json`
- 运行活跃/失败核验: `reports/d1_runtime_liveness_audit.json`
- step 分解统计: `reports/d1_step_timing_breakdown.json`
- 数据组织与缓存状态: `reports/d1_data_layout_audit.json`
- 采样脚本: `reports/d1_sampler_runtime.py`

## 3) A. Run 映射与活跃性核验

当前 D1 四任务并非都在有效训练。状态如下（以 queue status + 训练日志更新时间 + 错误日志三重核验）：

| run_name | job_id | queue_state | last_step | 现状 |
|---|---|---|---:|---|
| full_v4_2_seed42_fixed_nowarm_lambda1 | 20260403_152046_20358 | running | 113+ | active |
| wo_object_bias_v4_2_seed42 | 20260403_152047_16897 | running | 113+ | active |
| full_v4_2_seed42_fixed_warmup_lambda1 | 20260403_152046_21460 | failed | 48 | 已退出（EOFError） |
| wo_semantics_v4_2_seed42 | 20260403_152046_3719 | failed | 37 | 已退出（BadZipFile） |

失败证据:

- queue 失败记录:
  - `outputs/queue/stwm_protocol_v2/d1_train/queue_events.log`（job_failed at 15:35:24 / 15:41:34）
- 失败栈:
  - `outputs/queue/stwm_protocol_v2/d1_train/logs/1775200846981_wo_semantics_v4_2_seed42.log`
  - `outputs/queue/stwm_protocol_v2/d1_train/logs/1775200846820_full_v4_2_seed42_fixed_warmup_lambda1.log`
- 两者均在 `trace_adapter` 读取缓存时出错（BadZipFile / EOFError）。

## 4) B. 3-5 分钟连续采样结果（系统层 + 进程层校正）

来自 `reports/d1_resource_summary.json`：

- CPU 总利用率: 平均 56.31%，P95 74.32%。
- load average: load1 平均约 170.05（高负载）。
- 磁盘吞吐: 读平均 4.08 MB/s（P95 39.43），写平均 3.89 MB/s（P95 9.90）。
- GPU 利用（全机视角，whole-GPU，不等价于我方进程利用率）:
  - GPU4: util 平均 82.87%
  - GPU3: util 平均 45.13%

与 D1 run 的对应关系（同文件 `run_summary`）:

- full_nowarm: active_ratio=1.0, train_cpu_avg≈99.3, gpu_live_mode=4
- wo_object_bias: active_ratio=1.0, train_cpu_avg≈99.2, gpu_live_mode=3
- warmup / wo_semantics: active_ratio=0.0（在采样窗口内无活跃 train 主进程）

解释:

- 当前 D1 实际处于“2 训练活跃 + 2 已失败”的运行格局。

来自 `reports/d1_process_gpu_util_correction.json`（只看 active train pid，60 样本校正）：

| run | train pid | host gpu | our-process SM util avg | our-process SM util p95 | our-process used mem avg (MiB) | host whole-GPU util avg |
|---|---:|---:|---:|---:|---:|---:|
| full_nowarm | 3936401 | 4 | 0.67% | 9.00% | 5074 | 92.82% |
| wo_object_bias | 3936697 | 3 | 0.77% | 4.00% | 5074 | 55.62% |

校正结论:

- whole-GPU utilization 只能说明该卡上“所有用户进程总负载”。
- 我方 active train pid 的进程级 SM 利用率显著低于整卡利用率，不能再用整卡 util 推断“我方训练吃满 GPU”。

## 5) C. 单 step 时间分解（训练日志）

来自 `reports/d1_step_timing_breakdown.json`。定义:

- compute_time 近似 = `step_time_s - data_time_s`
- `data_wait_ratio = data_time_s / step_time_s`

重点看 active 两个 run 的 recent30（更接近当前稳态）：

| run | step_avg(s) | data_avg(s) | compute_avg(s) | data_wait_ratio_avg |
|---|---:|---:|---:|---:|
| full_nowarm | 37.31 | 35.02 | 2.29 | 0.932 |
| wo_object_bias | 37.32 | 35.31 | 2.01 | 0.942 |

结论:

- 当前 step 时间约 90%+ 落在 data/path 侧，而不是模型前向反向优化器本体。
- 上述结论来自我方 `train_log` 的 `step_time_s/data_time_s/data_wait_ratio`，不是来自整卡 util。

## 6) D. 数据组织方式审计

来自 `reports/d1_data_layout_audit.json`：

- clips: 3814
- frame_paths_total: 238,614（全部 `.jpg`）
- mask_paths_total: 238,614（全部 `.png`）
- `video_like_frame_paths = 0`

含义:

- 当前并非视频流解码路径，而是大量小文件路径读取。
- 每 step 涉及多帧图片/掩码的分散读取与解码，天然偏 CPU + 文件系统元数据路径。

## 7) E. DataLoader 与进程绑定审计

训练配置（来自 active run 最新 train_log 行）:

- micro_batch_per_gpu=2
- grad_accum=8
- effective_batch=16
- num_workers=12
- prefetch_factor=2
- persistent_workers=1
- pin_memory=1
- bf16=1
- activation_checkpointing=1

进程核验:

- active 两个 run 的 train 主进程 CPU 约 99%。
- 其下 `pt_data_worker` 子进程多数接近 0%。
- CPU affinity: train + dataloader worker 均为 `0-223`，无 core pinning。

含义:

- 主进程承担了主要数据特征构建开销，DataLoader worker 并未成为主计算承担者。

## 8) F. checkpoint/eval 干扰评估

配置:

- checkpoint_interval=500
- protocol_eval_interval=500

代码触发关系:

- protocol eval 触发前会强制 `maybe_save_checkpoint(..., force=True)`。

本次审计期实际状态:

- active run 当前 step 约 113-114，尚未到 500。
- 对应 checkpoint 目录尚无 `latest.pt` 等周期产物。

结论:

- 当前窗口中 checkpoint/eval 不是实测瓶颈；它是后续里程碑步点（500/1000/1500...）的潜在抖动来源。

## 9) G. 训练时间分解到阶段（证据映射）

代码路径显示 `data_time_s` 累加两段:

1. `next_micro_batch()`（DataLoader 取样）
2. `_build_features_for_sample(...)`（含 trace/semantic 编码、mask 读取、张量构建与 device 放置）

并且 `_build_features_for_sample` 在训练主循环内执行。

因此 `data_wait_ratio` 高，不等价于“纯 DataLoader 阻塞”，更准确是“样本取出 + 主循环特征构建”的合并耗时高。

## 10) H. 资源利用效率初步判断

- 计算侧（forward/backward/optimizer）在当前配置下只占 step 的小头（约 2s）。
- 数据与前端特征路径占主导（约 35s）。
- 进程级采样显示我方 train pid 的 SM 利用率低，说明当前不是“我方训练已吃满 GPU”，而是上游供应链（主循环特征构建）成为主瓶颈。

## 11) I. 本文结论

- 本次审计窗口内，D1 有效活跃 run 为 2 个，另 2 个已因 trace cache 读取错误失败。
- 资源效率问题的主线是“input pipeline + cache correctness”，不是“GPU 已被我方训练吃满”。
- checkpoint/eval 当前尚未进入触发区间，非本轮主因。

详尽瓶颈排序、收益优先级与“先不要做”的事项见: `docs/STWM_BOTTLENECK_DIAGNOSIS_V1.md`。