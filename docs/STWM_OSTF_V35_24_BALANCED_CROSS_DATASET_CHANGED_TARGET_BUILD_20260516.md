# STWM OSTF V35.24 Balanced Cross-Dataset Changed Target Build

- balanced_cross_dataset_changed_targets_built: True
- sample_count: 325
- enough_changed_tokens_per_split_dataset: True
- semantic_id_shortcut_for_cross_dataset_eval_forbidden: true
- balanced_cross_dataset_changed_target_ready: True
- recommended_next_step: eval_balanced_cross_dataset_changed_predictability

## 中文总结
V35.24 没有引入新 teacher 或 future input，而是在 V35.21 目标上增加 balanced changed benchmark 元数据。跨数据集 changed eval 明确禁止 semantic-id shortcut，优先使用 trace/risk/measurement 的 ontology-agnostic features。
