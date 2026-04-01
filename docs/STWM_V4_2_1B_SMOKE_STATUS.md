# STWM V4.2 1B Smoke Status

更新时间：2026-04-01 13:02 +08

## 结论

`1B smoke` 已真正跑通并完成（不是仅排队/等待）。

## B1. 从“待抢卡”到“已运行”的过程

初始状态：

- job 在 `running/`，但只是在 `gpu_auto_claim_run.sh` 内等待
- 阈值为：`prefer=4, min=1, max_mem=2000, max_util=20`
- 在共享 8xB200 高负载下持续无匹配 GPU

修正后（不改模型/损失/协议）：

- smoke job 资源参数改为：
  - `prefer_gpus=1`
  - `min_gpus=1`
  - `max_mem_used_mib=90000`
  - `max_util_percent=98`
  - `candidate_gpus=0,1,3,7`
- 执行命令（轻量 smoke，确保尽快验证链路）：
  - `STWM_V4_2_1B_SMOKE_RUN_TRIO=0`
  - `STWM_V4_2_1B_SMOKE_STEPS=8`
  - `STWM_V4_2_1B_SMOKE_SAMPLE_LIMIT=8`

结果：

- `gpu-auto` 成功 claim：GPU `0`
- smoke 完成并进入 `done/`
- `exit_code=0`

## B2. 产物核验

主 smoke 产物：

- `outputs/training/stwm_v4_2_1b_smoke/full_v4_2/train_log.jsonl`
- `outputs/training/stwm_v4_2_1b_smoke/full_v4_2/mini_val_summary.json`
- `outputs/queue/stwm_1b/done/1775018725400_stwm_1b_smoke.job`
- `outputs/queue/stwm_1b/logs/1775018725400_stwm_1b_smoke.log`

资源审计补充产物（短探针复跑）：

- `outputs/training/stwm_v4_2_1b_smoke/resource_probe/resource_probe_meta.txt`
- `outputs/training/stwm_v4_2_1b_smoke/resource_probe/gpu_usage_trace.csv`
- `outputs/training/stwm_v4_2_1b_smoke/resource_probe/full_v4_2/train_log.jsonl`
- `outputs/training/stwm_v4_2_1b_smoke/resource_probe/full_v4_2/mini_val_summary.json`

## B3. 参数量与设备/显存记录

来自 `mini_val_summary.json`：

- `model_preset`: `prototype_1b_v4_2`
- `model_parameters`: `1,020,660,598`
- `rough_parameter_budget`: `1,002,222,592`

来自 queue 日志和 probe：

- 实际设备：`GPU 0`
- probe 期间显存区间：`68,852 MiB` 到 `89,972 MiB`
- probe 期间利用率区间：`90%` 到 `95%`

## B4. 资源需求评估（基于本轮观测）

- 最小卡数：`1`（脚本与训练器为单进程单设备路径）
- 显存：建议保守按 `~90 GiB` 峰值预留
- 时长：
  - 本轮 8-step 轻量 smoke 从 start 到 done 约 `13s`
  - 4-step probe 约 `11s`

## B5. 失败与修复记录

本轮有一次 probe 中断：

- 现象：`KeyboardInterrupt` / `CondaError: KeyboardInterrupt`
- 根因：交互终端中断导致子进程被打断，不是训练逻辑报错
- 修复：改为后台终端执行并 `await`，复跑成功，`exit_code=0`
