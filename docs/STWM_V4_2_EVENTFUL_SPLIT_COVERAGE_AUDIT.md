# STWM V4.2 Eventful Split Coverage Audit (Phase B)

Date: 2026-04-03
Source: manifests/protocol_v2/protocol_v2_split_audit.json
Status: FROZEN (diagnostic tier)

## Eventful Selection Scope

- Parent split: protocol_val_main_v1
- Eventful split: protocol_val_eventful_v1
- Event criterion: reappearance events detected from mask presence dynamics

## Coverage by Dataset

### VSPW

- val_clip_count: 249
- selected_eventful_clip_count: 80
- selected_with_reappearance: 80
- selected_reappearance_event_count: 157
- selected_avg_max_disappear_gap: 16.4125

### VIPSeg

- val_clip_count: 144
- selected_eventful_clip_count: 53
- selected_with_reappearance: 53
- selected_reappearance_event_count: 206
- selected_avg_max_disappear_gap: 9.3774

## Combined Coverage

- selected_eventful_clip_count: 133
- selected_with_reappearance: 133
- selected_reappearance_event_count: 363

## Interpretation

- Eventful split is non-empty and event-rich on both datasets.
- Reappearance coverage is present in 133/133 selected eventful clips.
- This split is suitable for identity/occlusion/reconnect diagnostics.

## Governance Boundary

- protocol_val_eventful_v1 remains diagnostics-only.
- Official model selection must use protocol_val_main_v1 only.
- internal_final_test_v1 remains locked for final reporting only.
