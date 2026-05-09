# STWM OSTF V33.5 Protocol Rerun Decision

- manifest_full_coverage_ok: `True`
- available_ratio: `1.0`
- identity_hard_balanced: `True`
- split_shift_suspected: `False`
- hard_identity_ROC_AUC: `{'val': {'mean': 0.5504239424681407, 'std': 0.001734207146628656, 'worst': 0.5483504499741434}, 'test': {'mean': 0.6827690508041032, 'std': 0.0027841414354874082, 'worst': 0.6788716778712617}}`
- hard_identity_balanced_accuracy: `{'val': {'mean': 0.5323528424163058, 'std': 0.000816654008059861, 'worst': 0.5312279641382089}, 'test': {'mean': 0.6337252473076082, 'std': 0.0014022042906743132, 'worst': 0.6317619246280073}}`
- identity_retrieval_exclude_same_point_top1: `{'val': {'mean': 0.5670572916666666, 'std': 0.004192463174606529, 'worst': 0.561767578125}, 'test': {'mean': 0.572021484375, 'std': 0.0027907597980928135, 'worst': 0.569580078125}}`
- identity_retrieval_same_frame_top1: `{'val': {'mean': 0.42563887378757215, 'std': 0.003137189602547014, 'worst': 0.42121684867394693}, 'test': {'mean': 0.35280190068737133, 'std': 0.004161704020005976, 'worst': 0.3469281045751634}}`
- identity_retrieval_instance_pooled_top1: `{'val': {'mean': 0.818603515625, 'std': 0.008557700638257332, 'worst': 0.80908203125}, 'test': {'mean': 0.8307291666666666, 'std': 0.0028772248583436993, 'worst': 0.82666015625}}`
- semantic_proto_top1: `{'val': {'mean': 0.20435417971429423, 'std': 0.0, 'worst': 0.20435417971429423}, 'test': {'mean': 0.24708125702284872, 'std': 0.0008951568030607266, 'worst': 0.24595220197826062}}`
- semantic_proto_top5: `{'val': {'mean': 0.5474038949032238, 'std': 0.0, 'worst': 0.5474038949032238}, 'test': {'mean': 0.6171100150052369, 'std': 0.002008726712129325, 'worst': 0.6142749758384938}}`
- semantic_top1_copy_beaten: `False`
- semantic_top5_copy_beaten: `True`
- trajectory_degraded: `False`
- identity_signal_stable: `False`
- semantic_ranking_signal_stable: `True`
- val_test_agree: `True`
- recommended_next_step: `fix_identity_contrastive_loss`
