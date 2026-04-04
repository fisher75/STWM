# STWM Next Clean Matrix Plan V1

Generated: 2026-04-04 10:55:15

## Scope & Constraints

- No new object-bias variants.
- No additional warmup expansion.
- No concurrency increase in this planning stage.
- This document provides submission plan/commands only; do not auto-run.

## Winner & Replacement

- winner: full_v4_2_seed42_objbias_alpha050_objdiag_v1
- replace current full with: --object-bias-alpha 0.50

## Wave 1: seed42 replacement clean matrix (planned)

- Run A: full_replacement_objbias_alpha050_seed42
- Run B: wo_semantics_seed42_control
- Run C: wo_object_bias_seed42_control

Command template (do not run now):
```bash
cd /home/chen034/workspace/stwm
QUEUE_DIR=/home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train
TRAIN=/home/chen034/workspace/stwm/code/stwm/trainers/train_stwm_v4_2_real.py
# A) replacement full (alpha050)
bash scripts/protocol_v2_queue_submit.sh --queue-dir "$QUEUE_DIR" --job-name "full_replacement_objbias_alpha050_seed42" --class-type B --workdir /home/chen034/workspace/stwm --notes "seed42 replacement clean matrix" --resume-hint "auto-resume" -- env PYTHONPATH=/home/chen034/workspace/stwm/code:${PYTHONPATH:-} conda run --no-capture-output -n stwm python "$TRAIN" --data-root /home/chen034/workspace/stwm/data/external --manifest /home/chen034/workspace/stwm/manifests/protocol_v2/train_v2.json --output-dir /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_clean_matrix_replacement_v1/seed_42/full_replacement_objbias_alpha050_seed42 --run-name full_replacement_objbias_alpha050_seed42 --seed 42 --steps 2000 --target-epochs 0 --min-optimizer-steps 0 --max-optimizer-steps 0 --sample-limit 0 --model-preset prototype_220m_v4_2 --preset-file /home/chen034/workspace/stwm/code/stwm/configs/model_presets_v4_2.json --use-teacher-priors --save-checkpoint --checkpoint-dir-name checkpoints --checkpoint-interval 500 --milestone-interval 0 --auto-resume --micro-batch-per-gpu 2 --grad-accum 8 --num-workers 12 --prefetch-factor 2 --persistent-workers --pin-memory --bf16 --activation-checkpointing --lambda-traj 1.0 --lambda-vis 0.25 --lambda-sem 0.5 --lambda-reid 0.25 --lambda-query 0.25 --lambda-reconnect 0.1 --protocol-eval-interval 500 --protocol-eval-manifest /home/chen034/workspace/stwm/manifests/protocol_v2/protocol_val_main_v1.json --protocol-eval-dataset all --protocol-eval-max-clips 0 --protocol-eval-seed 42 --protocol-eval-obs-steps 8 --protocol-eval-pred-steps 8 --protocol-eval-run-name protocol_val_main --protocol-diagnostics-manifest /home/chen034/workspace/stwm/manifests/protocol_v2/protocol_val_eventful_v1.json --protocol-diagnostics-dataset all --protocol-diagnostics-max-clips 0 --protocol-diagnostics-run-name protocol_val_eventful --protocol-version v2_4_detached_frozen --protocol-best-checkpoint-name best_protocol_main.pt --protocol-best-selection-name best_protocol_main_selection.json --data-mode frontend_cache --frontend-cache-dir /home/chen034/workspace/stwm/data/cache/frontend_cache_protocol_v2_full_v1 --frontend-cache-index /home/chen034/workspace/stwm/data/cache/frontend_cache_protocol_v2_full_v1/index.json --frontend-cache-max-shards-in-memory 8 --gradient-audit-interval 0 --object-bias-alpha 0.50
# B/C controls can reuse the same template with: --disable-semantics or --neutralize-object-bias and run-name/output-dir replacements.
```

## Wave 2: seed123 replication clean matrix (planned, after Wave 1 pass)

- Same run set as seed42, only --seed 123 and output root to seed_123.
- Launch gate: only after seed42 replacement confirms no unacceptable regression.

## Launch Recommendation

- Recommended: seed42 replacement first, then seed123 replication.
- Not recommended at this time: direct seed42+123 dual launch.

