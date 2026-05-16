# STWM OSTF V36 Past-Only Observed Trace Input Build

- sample_count: 325
- obs_only_input_built: True
- future_trace_teacher_target_available: True
- future_trace_teacher_input_allowed: false
- leakage_safe: true

## 中文总结
已从 V35.49 full-clip teacher trace 中只抽取 observed 段作为 V36 输入；future trace 仅保留为 target/upper-bound 对比，不允许作为模型输入。
