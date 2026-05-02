# STWM Current Trace-Unit Semantics Audit V15

- current_K_meaning: `max_entities_per_sample / semantic entity slots, not object-internal dense trajectories`
- current_can_claim_dense_trace_field: `False`
- object_internal_point_traces_exist: `False`
- recommended_wording: `semantic entity trace-unit field; object-dense trace field requires V15 object-internal point supervision`

## Entity Count
- count: `6298`
- mean: `5.286281359161639`
- median: `6.0`
- p90: `8.0`
- max: `8`

## K Scaling Valid Supervision
- K16: {'slot_count': 16, 'valid_joint_supervised_slots': 84512, 'new_valid_slots_added_vs_K8': 0, 'joint_supervision_coverage': 0.10483486821213084}
- K32: {'slot_count': 32, 'valid_joint_supervised_slots': 84512, 'new_valid_slots_added_vs_K8': 0, 'joint_supervision_coverage': 0.05241743410606542}
- K8: {'slot_count': 8, 'valid_joint_supervised_slots': 84512, 'new_valid_slots_added_vs_K8': None, 'joint_supervision_coverage': 0.20966973642426168}
