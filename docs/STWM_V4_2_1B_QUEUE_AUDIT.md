# STWM V4.2 1B Queue Execution Correctness Audit

更新时间：2026-04-01 13:01 +08

## 审计范围

- 队列脚本：`scripts/gpu_queue_submit.sh`、`scripts/gpu_queue_worker.sh`、`scripts/start_gpu_queue_tmux.sh`
- 抢卡脚本：`scripts/gpu_auto_claim_run.sh`、`scripts/gpu_pick_idle.sh`
- 队列根目录：`outputs/queue/stwm_1b/`
- 任务：`stwm_1b_smoke`、`stwm_1b_confirmation`

## A1. 参数语义（基于脚本真实实现）

以下参数会写入 job 文件，worker 启动任务时透传给 `gpu_auto_claim_run.sh`：

- `prefer-gpus`
  - 含义：希望优先拿到的 GPU 数量上限。
  - 实现细节：`gpu_auto_claim_run.sh` 会从 `prefer` 递减尝试到 `min`，找到“满足阈值的最大可用集合”就启动。
  - 重要说明：这只是设置 `CUDA_VISIBLE_DEVICES`，并不自动启用多卡训练。

- `min-gpus`
  - 含义：可启动所需的最小 GPU 数量。
  - 实现细节：若当前满足阈值的 GPU 数量小于 `min`，任务持续等待（或超时退出）。

- `poll-seconds`
  - 含义：两次资源探测之间的轮询间隔秒数。
  - 作用范围：`gpu_auto_claim_run.sh` 内部循环，以及 `gpu_pick_idle.sh` 的扫描节奏。

- `max-mem-used-mib`
  - 含义：把 GPU 判定为“可用”时，允许的最大已用显存（MiB）。
  - 实现细节：来自 `nvidia-smi --query-gpu=memory.used`，需满足 `memory.used <= threshold`。

- `max-utilization`
  - 含义：把 GPU 判定为“可用”时，允许的最大利用率（%）。
  - 实现细节：来自 `nvidia-smi --query-gpu=utilization.gpu`，需满足 `util <= threshold`。

## A2. 队列目录结构检查

当前 `outputs/queue/stwm_1b/` 结构完整：

- `pending/`
- `running/`
- `done/`
- `failed/`
- `logs/`
- `queue_events.log`
- `.worker.lock`

结论：目录结构符合 FIFO worker 预期。

## A3. tmux 断连后持续运行性

证据：

- `tmux ls` 存在会话 `stwm_1b_queue`
- `tmux list-panes` 可看到 pane pid 存活
- `queue_events.log` 持续写入（worker_start、job_start、job_done）

结论：worker 放在 tmux 后，客户端断连不会中断任务。

## A4. worker 崩溃/退出是否可被发现

当前能力：

- 可通过 `tmux has-session -t stwm_1b_queue` 发现会话是否消失
- 可通过 `ps`/`pgrep` 判断 `gpu_queue_worker.sh` 是否还在
- 可通过 `queue_events.log` 是否持续增长判断是否“活着”

当前缺口：

- 没有内建 watchdog / 自动重启 / 告警推送
- `.worker.lock` 仅用于互斥，不用于健康恢复

结论：当前是“可人工发现”，不是“自动恢复”。

## A5. 当前任务状态（审计时刻）

- `stwm_1b_smoke`
  - 状态：`done`
  - 文件：`outputs/queue/stwm_1b/done/1775018725400_stwm_1b_smoke.job`
  - 退出码：0
  - 完成时间：2026-04-01 12:57:45 +08

- `stwm_1b_confirmation`
  - 状态：`running`（实为抢卡等待）
  - 文件：`outputs/queue/stwm_1b/running/1775018725815_stwm_1b_confirmation.job`
  - 当前阈值：`prefer=8, min=4, max_mem=2000, max_util=20`
  - 现象：日志持续 `no suitable GPU set yet`

## 审计结论

- 队列系统本身（提交、FIFO、worker、tmux 持久化）工作正常。
- `stwm_1b_smoke` 已从等待推进到成功完成，证明执行链可运行。
- 当前主要风险是确认任务阈值过严（4~8 卡 + 近空闲判定）导致长时间不启动。
