# V34.41 prototype mode generalization audit 中文报告

- 中文结论: `V34.41 prototype mode generalization audit 完成；拆分 codebook 上界与 observed-only prototype mode 选择泛化，避免继续盲训 writer。`
- oracle_codebook_has_upper_bound: `False`
- predicted_mode_generalizes: `False`
- recommended_next_step: `fix_shared_prototype_codebook_targets`
- 阶段性分析: `如果 oracle_label_codebook 在 val/test 有正增益而 predicted_prototype_codebook 没有，问题是 prototype mode selector；如果 oracle_label_codebook 自身没上界，问题是 codebook/target。`
- 论文相关问题解决方案参考: `该审计对应 VQ/codebook 与 MoE routing 的分解：先确认共享专家/codebook 本身有上界，再确认 observed-only router 是否能泛化，而不是把两者混成一个失败信号。`
