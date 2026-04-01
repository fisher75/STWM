# STWM V4.2 1B Setup And Execution

## Scope

This setup is restricted to the approved boundary:

1. no new module graph
2. no loss/protocol invention
3. run only full / wo_semantics / wo_object_bias
4. no direct 3B training in this round

## New Assets

1. `code/stwm/configs/model_presets_v4_2_1b.json`
2. `scripts/run_stwm_v4_2_1b_smoke.sh`
3. `scripts/run_stwm_v4_2_1b_minival_multiseed.sh`
4. `scripts/run_stwm_v4_2_1b_state_identifiability.sh`
5. `scripts/run_stwm_v4_2_1b_confirmation_round.sh`
6. `scripts/gpu_queue_submit.sh`
7. `scripts/gpu_queue_worker.sh`
8. `scripts/start_gpu_queue_tmux.sh`

## 1B Preset

`prototype_1b_v4_2` in `model_presets_v4_2_1b.json`:

1. hidden size = 1792
2. seq layers = 13
3. token layers = 13
4. heads = 28
5. state tokens = 20
6. memory slots = 64

Rough parameter budget from current estimator is close to 1B while preserving V4.2 topology.

## Quick Smoke

```bash
cd /home/chen034/workspace/stwm
bash scripts/run_stwm_v4_2_1b_smoke.sh
```

Optional single-run smoke:

```bash
STWM_V4_2_1B_SMOKE_RUN_TRIO=0 bash scripts/run_stwm_v4_2_1b_smoke.sh
```

## Full Confirmation (Foreground)

```bash
cd /home/chen034/workspace/stwm
bash scripts/run_stwm_v4_2_1b_confirmation_round.sh
```

## FIFO Queue (Recommended On Shared 8xB200)

Start persistent worker in tmux:

```bash
cd /home/chen034/workspace/stwm
bash scripts/start_gpu_queue_tmux.sh \
  --session stwm_1b_queue \
  --queue-dir /home/chen034/workspace/stwm/outputs/queue/stwm_1b \
  --prefer-gpus 8 \
  --min-gpus 4
```

Submit jobs in order (first-come-first-run):

```bash
bash scripts/gpu_queue_submit.sh \
  --queue-dir /home/chen034/workspace/stwm/outputs/queue/stwm_1b \
  --job-name stwm_1b_smoke \
  --prefer-gpus 4 --min-gpus 1 -- \
  bash scripts/run_stwm_v4_2_1b_smoke.sh

bash scripts/gpu_queue_submit.sh \
  --queue-dir /home/chen034/workspace/stwm/outputs/queue/stwm_1b \
  --job-name stwm_1b_confirm \
  --prefer-gpus 8 --min-gpus 4 -- \
  bash scripts/run_stwm_v4_2_1b_confirmation_round.sh
```

Monitor:

```bash
tmux attach -t stwm_1b_queue
tail -f /home/chen034/workspace/stwm/logs/stwm_1b_queue.log
```
