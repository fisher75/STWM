# STWM OSTF V35.18 增量 observed semantic measurement cache

- total_trace_samples_seen: 325
- skipped_existing_sample_count: 289
- new_sample_count: 36
- future_teacher_embeddings_input_allowed: false
- leakage_safe: true

## 中文总结
V35.18 只为新增 video trace clips 增量构建 observed-frame CLIP measurement cache；已有 cache 不重复计算，future frames 未作为输入。
