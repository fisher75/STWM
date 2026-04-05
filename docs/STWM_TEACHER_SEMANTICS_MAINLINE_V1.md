# STWM Teacher-Grounded Semantics Mainline V1

## 目标与边界

本主线将 V4.2 的语义分支切换为 teacher-grounded 路径，聚焦两件事：

1. 查询读出端蒸馏（trace-centric distillation）。
2. 低置信/困难样本的置信度门控重排（confidence-gated rerank）。

同时保持已验证约束：语义不直接改写 latent transition（仍使用 `--qtsa-disable-semantic-transition` 控制）。

## 为什么切主线

1. 现有 object/QSTR/QTSA 变体已完成清洁矩阵与 seed42 对照，主结论明确，继续加同类变体的边际收益下降。
2. 语义项在 V4.2 中是核心组成，但“是否稳健提升”在主协议上仍高风险未决。
3. Teacher-grounded 路径可以把语义监督从 proxy 标签哈希转向更外部的视觉语义信号，提高可解释性与可审计性。
4. query/readout 侧蒸馏 + 置信度重排是最小侵入路径，不触碰已稳定的 transition 主干。

## 实施要点

1. 新增 `code/stwm/modules/semantic_adapter_teacher_v2.py`：teacher 优先级与严格模式能力缺口报告。
2. 训练脚本接入：
   - `--semantic-adapter-mode teacher_v2`
   - `--semteacher-distill-enable`
   - `--semteacher-distill-weight`
   - `--semteacher-association-temperature`
   - `--semteacher-confidence-rerank-enable`（仅影响评估时查询帧选择）
3. 评估脚本接入：
   - 低置信 + hard 子集门控重排（不改变 protocol 度量定义）。
4. 前端缓存 schema 升级：
   - `semantic_features_teacher`
   - `target_semantic_probs_teacher`
   - `cache_version`
   - `manifest_hash`
   - `frontend_hash`

## 去主线声明

从本版本开始，object-prior/QSTR/QTSA 不再作为主叙事 headline；它们保留为历史对照与诊断基线。当前主线是 teacher-grounded semantic state + trace-centric distillation/reranking。
