# Stage2 State-Identifiability Protocol 20260415

- task: real future grounding with true instance continuity and future masks
- primary datasets: VIPSeg, BURST
- VSPW: supplemental only; excluded from main identifiability protocol
- full_identifiability_panel: 38
- occlusion_reappearance: 18
- crossing_ambiguity: 18
- small_object: 18
- appearance_change: 18
- long_gap_persistence: 10

## Notes

- Query frame uses the same semantic-frame convention as the current Stage2 mainline.
- Future recovery uses true instance identity / future mask continuity rather than rollout-L2 proxy scores.
