# STWM OSTF V35 Semantic State Head Train Summary

- semantic_state_head_training_ran: true
- checkpoint_path: `outputs/checkpoints/stwm_ostf_v35_8_identity_only_retrieval_finetune_h32_m128_seed456/v35_semantic_state_head_m128_h32_seed456_best.pt`
- best_val_loss: 8.914381551742554
- v30_backbone_frozen: true
- future_leakage_detected: false

## 中文总结
V35 semantic state head 已在 V35.1 fixed targets 上完成 seed42 训练；本训练只更新 semantic state head，不训练 V30 trajectory backbone。
