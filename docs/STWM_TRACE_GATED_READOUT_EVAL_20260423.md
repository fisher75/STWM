# STWM Trace-Gated Readout Eval 20260423

- frozen_external_teacher_only_backend: clip_vit-b_16_frozen_crop_direct
- selected_gate_family: semantic_topk_coord_veto
- selected_gate_params: {"selected_gate_family": "semantic_topk_coord_veto", "top_k": 4, "coord_gate_threshold": 0.0, "semantic_tie_margin": 0.0, "coord_tiebreak_weight": 0.0, "veto_penalty": 1000.0, "val_selection_score": 0.3974979062009291, "selection_split": "val"}
- gate_activation_rate: 0.0
- gate_veto_rate: 0.0
- densified_200_context_preserving: valid_items=200 test_items=149 skipped_items=0 hash=8398a676076c3ea605e5543928c2721654d99c0107028530f2ecdecac83b1b6b
- heldout_burst_heavy_context_preserving: valid_items=306 test_items=225 skipped_items=0 hash=c93dc75e74c5a9bad5465ec8074f600932f29708920c7daeae515eb77ebba3c4
- heldout_scene_category_video_context_preserving: valid_items=303 test_items=206 skipped_items=0 hash=38f853f666d74c2f5ccb1abb46471209d0127b747eacfdc378921e9762e6ca7f
