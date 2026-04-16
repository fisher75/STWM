# Stage2 State-Identifiability Protocol V2 20260416

- protocol_contribution: true
- rollout_error_proxy_only: false
- primary datasets: VIPSeg, BURST
- VSPW excluded from main instance-identifiability protocol
- selected_protocol_item_count: 180
- per_dataset_counts: {"BURST": 122, "VIPSeg": 58}

## Panel Counts

- full_identifiability_panel: 180
- occlusion_reappearance: 36
- crossing_ambiguity: 36
- small_object: 36
- appearance_change: 36
- long_gap_persistence: 36

## Selection Policy

- minimum_total_items: 120
- ideal_total_items: 180
- panel targets filled first, then global top-up from remaining highest-difficulty items
- no leakage: validation split only
- no overlap: unique protocol_item_id

## Shortage Notes

- none
