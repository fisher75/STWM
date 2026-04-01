# STWM V4.2 1B Queue Operations

更新时间：2026-04-01 13:04 +08

## D1. 查看当前队列/任务状态（命令清单）

在仓库根目录执行：

```bash
cd /home/chen034/workspace/stwm
```

查看四态目录：

```bash
for d in pending running done failed; do
  echo "[$d]"
  ls -1 "outputs/queue/stwm_1b/$d" 2>/dev/null || true
done
```

查看事件流：

```bash
tail -n 120 outputs/queue/stwm_1b/queue_events.log
```

查看 tmux worker：

```bash
tmux ls | grep stwm_1b_queue
tmux list-panes -t stwm_1b_queue -F '#{session_name} #{window_index}.#{pane_index} pid=#{pane_pid} cmd=#{pane_current_command}'
```

查看 GPU 快照：

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

## D2. 查看单个 job 日志入口

smoke：

```bash
tail -n 120 outputs/queue/stwm_1b/logs/1775018725400_stwm_1b_smoke.log
```

confirmation：

```bash
tail -n 120 outputs/queue/stwm_1b/logs/1775018725815_stwm_1b_confirmation.log
```

worker 总日志：

```bash
tail -n 200 logs/stwm_1b_queue.log
tmux capture-pane -pt stwm_1b_queue:0 | tail -n 200
```

## D3. worker 死掉时恢复步骤

1. 确认会话是否存在：

```bash
tmux has-session -t stwm_1b_queue
```

2. 若不存在或无响应，清理锁并重启：

```bash
rm -f outputs/queue/stwm_1b/.worker.lock
bash scripts/start_gpu_queue_tmux.sh \
  --session stwm_1b_queue \
  --queue-dir /home/chen034/workspace/stwm/outputs/queue/stwm_1b \
  --log-file /home/chen034/workspace/stwm/logs/stwm_1b_queue.log \
  --idle-sleep 20
```

3. 复核恢复结果：

```bash
tmux ls | grep stwm_1b_queue
tail -n 40 outputs/queue/stwm_1b/queue_events.log
```

## D4. 任务失败后的重试步骤

方式 A：重新提交（推荐，最干净）

```bash
bash scripts/gpu_queue_submit.sh \
  --queue-dir /home/chen034/workspace/stwm/outputs/queue/stwm_1b \
  --job-name stwm_1b_smoke_retry \
  --prefer-gpus 1 --min-gpus 1 \
  --max-mem-used-mib 90000 --max-utilization 98 \
  --candidate-gpus 0,1,3,7 \
  -- bash scripts/run_stwm_v4_2_1b_smoke.sh
```

方式 B：将失败 job 文件回推到 pending（需确认参数无误）

```bash
mv outputs/queue/stwm_1b/failed/<job>.job outputs/queue/stwm_1b/pending/<new_stamp>_<name>.job
```

## D5. 建议的日常巡检频率

- 高频阶段（有任务运行）：每 5-10 分钟查看一次 `queue_events.log` 和对应 job log。
- 夜间长跑：至少保留一个 tmux 会话监看，或定时脚本巡检会话存活与日志增量。
