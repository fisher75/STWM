# STWM Semantic-Hard Protocol V1

## 产物

1. Manifest: `/home/chen034/workspace/stwm/manifests/protocol_v2/semantic_hard_seed42_v1.json`
2. Clip IDs: `/home/chen034/workspace/stwm/manifests/protocol_v2/semantic_hard_seed42_v1_clip_ids.json`
3. Report: `/home/chen034/workspace/stwm/reports/stwm_semantic_hard_protocol_v1.json`

本次自动构建结果：`selected_count=96`，且 `selected_is_harder_than_full=true`。

## 自动构建规则（可复现）

输入：

1. 主协议清单 `protocol_val_main_v1.json`。
2. 既有 seed42 强基线评估摘要（QSTR control + QTSA control）的 `per_clip` 指标。

对每个 clip 聚合：

1. `avg_query_localization_error`
2. `avg_query_top1_acc`
3. `avg_query_hit_rate`
4. `avg_query_same_class_candidates`
5. 跨 run 不稳定性：`std(query_localization_error) + std(query_top1_acc)`

将以上量映射为全体 clip 的分位排名后，按权重合成难度分数：

1. `0.45 * qerr_rank`
2. `0.35 * fail_rank`（fail = 1 - top1）
3. `0.10 * miss_rank`（miss = 1 - hit_rate）
4. `0.08 * instability_rank`
5. `0.02 * ambiguity_rank`（same-class 候选数）

按 `difficulty_score` 降序选取前 96 个 clip，完全自动，无手工挑选。

## 复现命令

```bash
cd /home/chen034/workspace/stwm
conda run --no-capture-output -n stwm \
  python code/stwm/tools/build_semantic_hard_protocol_v1.py
```

如需改变规模：添加 `--target-size <N>`。
