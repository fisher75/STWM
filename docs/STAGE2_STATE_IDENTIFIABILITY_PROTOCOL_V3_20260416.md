# Stage2 State-Identifiability Protocol V3 20260416

- protocol_contribution: true
- rollout_error_proxy_only: false
- primary datasets: VIPSeg, BURST
- VSPW excluded from main instance-identifiability protocol
- selected_protocol_item_count: 200
- per_dataset_counts: {"BURST": 132, "VIPSeg": 68}

## Panel Counts

- full_identifiability_panel: 200
- occlusion_reappearance: 40
- crossing_ambiguity: 40
- small_object: 40
- appearance_change: 40
- long_gap_persistence: 40

## Selection Policy

- minimum_total_items: 120
- ideal_total_items: 200
- panel targets filled first, then global top-up from remaining highest-difficulty items
- no leakage: validation split only
- no overlap: unique protocol_item_id

## Shortage Notes

- none
