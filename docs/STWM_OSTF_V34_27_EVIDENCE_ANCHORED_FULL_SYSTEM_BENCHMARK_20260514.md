# V34.27 evidence-anchored full-system benchmark 中文报告

## 中文结论
V34.27 已按 full-system baseline gap 做 evidence-anchored composition benchmark：固定 V34.25/V34.20 权重，将 final semantic 从 pointwise+residual 改为 semantic_evidence_base+sparse_gate*unit_residual_correction，并用 val 选配置、test 确认。

## 阶段性分析
V34.26 暴露的问题不是 sparse gate 本身，而是系统分解把 observed semantic evidence 放在 residual 支路里，导致完整协议下 copy_mean/top-k evidence 这种非 oracle baseline 直接压过 pointwise+residual。V34.27 因此只改 composition：semantic base 先由 observed evidence 提供，unit memory 只负责结构化 correction。如果这个后验组合仍不能赢 copy/top-k，说明 residual 学到的 correction 还没有超出 semantic persistence / raw evidence transport。

## 论文相关问题解决方案参考
这个修法借鉴了 object-centric memory 与 query-conditioned retrieval 的常见结构：Slot Attention/OCVP 类方法先建立对象槽和 assignment，XMem/SAM2 类视频记忆方法把 memory read 作为主证据路径，Perceiver IO/DETR 类 cross-attention 用 future query 读 observed memory。对我们当前系统，最重要的启发是：semantic measurement 不应只是补丁 loss，而应成为可替代 copy/top-k baseline 的显式 evidence base；unit residual 必须在这个强 base 上继续提供 hard/changed 增益。

## 关键结果
- benchmark_passed: `False`
- best_evidence_anchor_method: `evidence_anchor_copy_mean_observed_w1.00_r0.25`
- best_evidence_anchor_beats_copy_topk: `True`
- unit_residual_improves_evidence_anchor: `True`
- semantic_measurements_load_bearing_on_system: `True`
- assignment_load_bearing_on_system: `False`
- unit_memory_load_bearing_on_system: `True`
- semantic_hard_signal: `{'val': True, 'test': True}`
- changed_semantic_signal: `{'val': True, 'test': True}`
- stable_preservation: `{'val': True, 'test': True}`
- m512_dense_ready: `False`
- integrated_semantic_field_claim_allowed: `False`
- integrated_identity_field_claim_allowed: `False`
- recommended_next_step: `fix_full_system_baseline_gap`

## 最佳下一步方案
如果 V34.27 打赢 copy/top-k，下一步才进入 M512 dense visualization；如果没打赢，继续修 full_system_baseline_gap，优先检查 hard/changed target 是否真正需要未来结构推理、以及 unit residual correction 是否被训练成相对 evidence base 的增量，而不是相对 pointwise base 的增量。
