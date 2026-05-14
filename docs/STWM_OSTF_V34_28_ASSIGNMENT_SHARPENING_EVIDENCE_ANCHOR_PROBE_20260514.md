# V34.28 assignment sharpening evidence-anchor probe 中文报告

## 中文结论
V34.28 已在 evidence-anchor 系统上做 assignment sharpening/top-k 读出诊断。本轮不训练模型，只检查 assignment 不 load-bearing 是 assignment 过软，还是 unit memory correction 本身没有形成 assignment-bound 区分。

## 阶段性分析
V34.27 已经说明 evidence-anchor + residual 能小幅超过 copy_mean/top-k，但 assignment shuffle 几乎不伤结果。V34.28 进一步测试：如果 sharpen/top-1 assignment 可以恢复 delta，问题是 assignment 太软；如果仍不恢复，问题更可能是 unit memory slots 之间 correction 同质化，需要重新训练 assignment-discriminative residual，而不是继续调 gate。

## 论文相关问题解决方案参考
这对应 object-centric 表征中常见的 slot collapse / slot interchangeability 问题：Slot Attention、SAVi/STEVE、OCVP 一类工作通常需要 slot competition、permutation-aware matching、object-consistency loss 来避免所有 slot 学成可互换记忆。当前 probe 正是在确认 STWM unit memory 是否存在这种同质化。

## 关键结果
- probe_passed: `False`
- best_assignment_variant: `top1`
- assignment_load_bearing_restored: `False`
- unit_memory_load_bearing_on_system: `True`
- semantic_hard_signal: `{'val': True, 'test': True}`
- changed_semantic_signal: `{'val': True, 'test': True}`
- stable_preservation: `{'val': True, 'test': True}`
- integrated_semantic_field_claim_allowed: `False`
- integrated_identity_field_claim_allowed: `False`
- recommended_next_step: `fix_assignment_bound_residual_model`

## 最佳下一步方案
如果 assignment sharpening 通过，可以把 sharpened assignment 作为 V34.28 系统读出并进入 M512 可视化；如果不通过，下一步应训练 evidence-anchor-relative 的 assignment-discriminative unit residual，明确加入 shuffled-assignment contrast、slot diversity、unit-specific correction target。
