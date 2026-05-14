# V34.40 prototype-conditioned mixture unit_delta writer 决策中文报告

- 中文结论: `V34.40 prototype-conditioned mixture unit_delta writer 完成；不训练 gate、不跑 M512，只判断共享 prototype mode 是否让 learned writer 在 copy/top-k evidence anchor 上泛化。`
- prototype_conditioned_mixture_writer_built: `True`
- probe_passed: `False`
- selected_config_by_val: `{'gate_mode': 'predictable_oracle_mask', 'scale': 0.25, 'stable': True, 'val_gain_anchor': -0.00017329632181130978, 'val_gain_pointwise': 0.4751608331877029}`
- beats_copy_topk_baseline: `False`
- unit_residual_improves_evidence_anchor: `False`
- assignment_load_bearing_on_system: `False`
- unit_memory_load_bearing_on_system: `False`
- semantic_hard_signal: `{'test': True, 'val': True}`
- changed_semantic_signal: `{'test': True, 'val': True}`
- stable_preservation: `{'test': True, 'val': True}`
- recommended_next_step: `fix_prototype_conditioned_writer_generalization`
