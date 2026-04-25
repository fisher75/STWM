# STWM Reacquisition V2 Bootstrap 20260425

Positive deltas mean full_trace_belief improvement. For false rates, positive is baseline false rate minus full false rate.

## full_trace_belief_vs_frozen_external_teacher_only

| group | metric | count | mean_delta | ci95_low | ci95_high | zero_excluded | win_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| all_reacquisition_items | top1 | 1038 | 0.0058 | -0.0279 | 0.0414 | False | 0.1715 |
| all_reacquisition_items | MRR | 1038 | 0.0020 | -0.0185 | 0.0227 | False | 0.2794 |
| all_reacquisition_items | false_confuser_rate | 1038 | 0.0058 | -0.0279 | 0.0414 | False | 0.1715 |
| all_reacquisition_items | false_reacquisition_rate | 1038 | 0.0058 | -0.0279 | 0.0414 | False | 0.1715 |
| occlusion_reappearance | top1 | 1038 | 0.0058 | -0.0279 | 0.0414 | False | 0.1715 |
| occlusion_reappearance | MRR | 1038 | 0.0020 | -0.0185 | 0.0227 | False | 0.2794 |
| occlusion_reappearance | false_confuser_rate | 1038 | 0.0058 | -0.0279 | 0.0414 | False | 0.1715 |
| occlusion_reappearance | false_reacquisition_rate | 1038 | 0.0058 | -0.0279 | 0.0414 | False | 0.1715 |
| long_gap_persistence | top1 | 498 | -0.0803 | -0.1225 | -0.0361 | True | 0.0803 |
| long_gap_persistence | MRR | 498 | -0.0453 | -0.0703 | -0.0217 | True | 0.2289 |
| long_gap_persistence | false_confuser_rate | 498 | -0.0803 | -0.1225 | -0.0361 | True | 0.0803 |
| long_gap_persistence | false_reacquisition_rate | 498 | -0.0803 | -0.1225 | -0.0361 | True | 0.0803 |
| occlusion_and_long_gap | top1 | 498 | -0.0803 | -0.1225 | -0.0361 | True | 0.0803 |
| occlusion_and_long_gap | MRR | 498 | -0.0453 | -0.0703 | -0.0217 | True | 0.2289 |
| occlusion_and_long_gap | false_confuser_rate | 498 | -0.0803 | -0.1225 | -0.0361 | True | 0.0803 |
| occlusion_and_long_gap | false_reacquisition_rate | 498 | -0.0803 | -0.1225 | -0.0361 | True | 0.0803 |
| appearance_similar_confuser | top1 | 624 | -0.0385 | -0.0849 | 0.0048 | False | 0.1458 |
| appearance_similar_confuser | MRR | 624 | -0.0322 | -0.0595 | -0.0051 | True | 0.2340 |
| appearance_similar_confuser | false_confuser_rate | 624 | -0.0385 | -0.0849 | 0.0048 | False | 0.1458 |
| appearance_similar_confuser | false_reacquisition_rate | 624 | -0.0385 | -0.0849 | 0.0048 | False | 0.1458 |
| spatially_close_confuser | top1 | 696 | 0.0489 | 0.0043 | 0.0963 | True | 0.2040 |
| spatially_close_confuser | MRR | 696 | 0.0269 | -0.0005 | 0.0554 | False | 0.3305 |
| spatially_close_confuser | false_confuser_rate | 696 | 0.0489 | 0.0043 | 0.0963 | True | 0.2040 |
| spatially_close_confuser | false_reacquisition_rate | 696 | 0.0489 | 0.0043 | 0.0963 | True | 0.2040 |
| crossing_or_overlap_confuser | top1 | 438 | 0.1050 | 0.0479 | 0.1644 | True | 0.2420 |
| crossing_or_overlap_confuser | MRR | 438 | 0.0618 | 0.0270 | 0.0968 | True | 0.4110 |
| crossing_or_overlap_confuser | false_confuser_rate | 438 | 0.1050 | 0.0479 | 0.1644 | True | 0.2420 |
| crossing_or_overlap_confuser | false_reacquisition_rate | 438 | 0.1050 | 0.0479 | 0.1644 | True | 0.2420 |
| OOD_confuser | top1 | 582 | -0.0155 | -0.0636 | 0.0344 | False | 0.1701 |
| OOD_confuser | MRR | 582 | -0.0132 | -0.0420 | 0.0150 | False | 0.2663 |
| OOD_confuser | false_confuser_rate | 582 | -0.0155 | -0.0636 | 0.0344 | False | 0.1701 |
| OOD_confuser | false_reacquisition_rate | 582 | -0.0155 | -0.0636 | 0.0344 | False | 0.1701 |

## full_trace_belief_vs_legacysem

| group | metric | count | mean_delta | ci95_low | ci95_high | zero_excluded | win_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| all_reacquisition_items | top1 | 1038 | 0.3420 | 0.3025 | 0.3786 | True | 0.4229 |
| all_reacquisition_items | MRR | 1038 | 0.2552 | 0.2301 | 0.2799 | True | 0.6339 |
| all_reacquisition_items | false_confuser_rate | 1038 | 0.3420 | 0.3025 | 0.3786 | True | 0.4229 |
| all_reacquisition_items | false_reacquisition_rate | 1038 | 0.3420 | 0.3025 | 0.3786 | True | 0.4229 |
| occlusion_reappearance | top1 | 1038 | 0.3420 | 0.3025 | 0.3786 | True | 0.4229 |
| occlusion_reappearance | MRR | 1038 | 0.2552 | 0.2301 | 0.2799 | True | 0.6339 |
| occlusion_reappearance | false_confuser_rate | 1038 | 0.3420 | 0.3025 | 0.3786 | True | 0.4229 |
| occlusion_reappearance | false_reacquisition_rate | 1038 | 0.3420 | 0.3025 | 0.3786 | True | 0.4229 |
| long_gap_persistence | top1 | 498 | 0.2952 | 0.2369 | 0.3494 | True | 0.3795 |
| long_gap_persistence | MRR | 498 | 0.1988 | 0.1644 | 0.2324 | True | 0.6104 |
| long_gap_persistence | false_confuser_rate | 498 | 0.2952 | 0.2369 | 0.3494 | True | 0.3795 |
| long_gap_persistence | false_reacquisition_rate | 498 | 0.2952 | 0.2369 | 0.3494 | True | 0.3795 |
| occlusion_and_long_gap | top1 | 498 | 0.2952 | 0.2369 | 0.3494 | True | 0.3795 |
| occlusion_and_long_gap | MRR | 498 | 0.1988 | 0.1644 | 0.2324 | True | 0.6104 |
| occlusion_and_long_gap | false_confuser_rate | 498 | 0.2952 | 0.2369 | 0.3494 | True | 0.3795 |
| occlusion_and_long_gap | false_reacquisition_rate | 498 | 0.2952 | 0.2369 | 0.3494 | True | 0.3795 |
| appearance_similar_confuser | top1 | 624 | 0.2740 | 0.2228 | 0.3253 | True | 0.3878 |
| appearance_similar_confuser | MRR | 624 | 0.1998 | 0.1672 | 0.2329 | True | 0.5865 |
| appearance_similar_confuser | false_confuser_rate | 624 | 0.2740 | 0.2228 | 0.3253 | True | 0.3878 |
| appearance_similar_confuser | false_reacquisition_rate | 624 | 0.2740 | 0.2228 | 0.3253 | True | 0.3878 |
| spatially_close_confuser | top1 | 696 | 0.2917 | 0.2457 | 0.3376 | True | 0.3736 |
| spatially_close_confuser | MRR | 696 | 0.2558 | 0.2259 | 0.2865 | True | 0.6638 |
| spatially_close_confuser | false_confuser_rate | 696 | 0.2917 | 0.2457 | 0.3376 | True | 0.3736 |
| spatially_close_confuser | false_reacquisition_rate | 696 | 0.2917 | 0.2457 | 0.3376 | True | 0.3736 |
| crossing_or_overlap_confuser | top1 | 438 | 0.2785 | 0.2215 | 0.3356 | True | 0.3562 |
| crossing_or_overlap_confuser | MRR | 438 | 0.2183 | 0.1774 | 0.2604 | True | 0.5639 |
| crossing_or_overlap_confuser | false_confuser_rate | 438 | 0.2785 | 0.2215 | 0.3356 | True | 0.3562 |
| crossing_or_overlap_confuser | false_reacquisition_rate | 438 | 0.2785 | 0.2215 | 0.3356 | True | 0.3562 |
| OOD_confuser | top1 | 582 | 0.3007 | 0.2491 | 0.3505 | True | 0.3883 |
| OOD_confuser | MRR | 582 | 0.2320 | 0.2001 | 0.2640 | True | 0.6048 |
| OOD_confuser | false_confuser_rate | 582 | 0.3007 | 0.2491 | 0.3505 | True | 0.3883 |
| OOD_confuser | false_reacquisition_rate | 582 | 0.3007 | 0.2491 | 0.3505 | True | 0.3883 |

## full_trace_belief_vs_calibration-only

| group | metric | count | mean_delta | ci95_low | ci95_high | zero_excluded | win_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| all_reacquisition_items | top1 | 1038 | 0.3622 | 0.3237 | 0.3998 | True | 0.4432 |
| all_reacquisition_items | MRR | 1038 | 0.2661 | 0.2415 | 0.2910 | True | 0.6532 |
| all_reacquisition_items | false_confuser_rate | 1038 | 0.3622 | 0.3237 | 0.3998 | True | 0.4432 |
| all_reacquisition_items | false_reacquisition_rate | 1038 | 0.3622 | 0.3237 | 0.3998 | True | 0.4432 |
| occlusion_reappearance | top1 | 1038 | 0.3622 | 0.3237 | 0.3998 | True | 0.4432 |
| occlusion_reappearance | MRR | 1038 | 0.2661 | 0.2415 | 0.2910 | True | 0.6532 |
| occlusion_reappearance | false_confuser_rate | 1038 | 0.3622 | 0.3237 | 0.3998 | True | 0.4432 |
| occlusion_reappearance | false_reacquisition_rate | 1038 | 0.3622 | 0.3237 | 0.3998 | True | 0.4432 |
| long_gap_persistence | top1 | 498 | 0.3414 | 0.2871 | 0.3976 | True | 0.4257 |
| long_gap_persistence | MRR | 498 | 0.2250 | 0.1904 | 0.2593 | True | 0.6807 |
| long_gap_persistence | false_confuser_rate | 498 | 0.3414 | 0.2871 | 0.3976 | True | 0.4257 |
| long_gap_persistence | false_reacquisition_rate | 498 | 0.3414 | 0.2871 | 0.3976 | True | 0.4257 |
| occlusion_and_long_gap | top1 | 498 | 0.3414 | 0.2871 | 0.3976 | True | 0.4257 |
| occlusion_and_long_gap | MRR | 498 | 0.2250 | 0.1904 | 0.2593 | True | 0.6807 |
| occlusion_and_long_gap | false_confuser_rate | 498 | 0.3414 | 0.2871 | 0.3976 | True | 0.4257 |
| occlusion_and_long_gap | false_reacquisition_rate | 498 | 0.3414 | 0.2871 | 0.3976 | True | 0.4257 |
| appearance_similar_confuser | top1 | 624 | 0.2853 | 0.2356 | 0.3365 | True | 0.3974 |
| appearance_similar_confuser | MRR | 624 | 0.2078 | 0.1750 | 0.2408 | True | 0.6010 |
| appearance_similar_confuser | false_confuser_rate | 624 | 0.2853 | 0.2356 | 0.3365 | True | 0.3974 |
| appearance_similar_confuser | false_reacquisition_rate | 624 | 0.2853 | 0.2356 | 0.3365 | True | 0.3974 |
| spatially_close_confuser | top1 | 696 | 0.2960 | 0.2500 | 0.3405 | True | 0.3793 |
| spatially_close_confuser | MRR | 696 | 0.2624 | 0.2302 | 0.2940 | True | 0.6695 |
| spatially_close_confuser | false_confuser_rate | 696 | 0.2960 | 0.2500 | 0.3405 | True | 0.3793 |
| spatially_close_confuser | false_reacquisition_rate | 696 | 0.2960 | 0.2500 | 0.3405 | True | 0.3793 |
| crossing_or_overlap_confuser | top1 | 438 | 0.2922 | 0.2374 | 0.3470 | True | 0.3699 |
| crossing_or_overlap_confuser | MRR | 438 | 0.2131 | 0.1718 | 0.2538 | True | 0.5479 |
| crossing_or_overlap_confuser | false_confuser_rate | 438 | 0.2922 | 0.2374 | 0.3470 | True | 0.3699 |
| crossing_or_overlap_confuser | false_reacquisition_rate | 438 | 0.2922 | 0.2374 | 0.3470 | True | 0.3699 |
| OOD_confuser | top1 | 582 | 0.3230 | 0.2715 | 0.3694 | True | 0.4038 |
| OOD_confuser | MRR | 582 | 0.2450 | 0.2111 | 0.2784 | True | 0.6237 |
| OOD_confuser | false_confuser_rate | 582 | 0.3230 | 0.2715 | 0.3694 | True | 0.4038 |
| OOD_confuser | false_reacquisition_rate | 582 | 0.3230 | 0.2715 | 0.3694 | True | 0.4038 |

## full_trace_belief_vs_belief_without_trace_prior

| group | metric | count | mean_delta | ci95_low | ci95_high | zero_excluded | win_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| all_reacquisition_items | top1 | 1038 | -0.0723 | -0.0896 | -0.0559 | True | 0.0067 |
| all_reacquisition_items | MRR | 1038 | -0.0463 | -0.0556 | -0.0373 | True | 0.0299 |
| all_reacquisition_items | false_confuser_rate | 1038 | -0.0723 | -0.0896 | -0.0559 | True | 0.0067 |
| all_reacquisition_items | false_reacquisition_rate | 1038 | -0.0723 | -0.0896 | -0.0559 | True | 0.0067 |
| occlusion_reappearance | top1 | 1038 | -0.0723 | -0.0896 | -0.0559 | True | 0.0067 |
| occlusion_reappearance | MRR | 1038 | -0.0463 | -0.0556 | -0.0373 | True | 0.0299 |
| occlusion_reappearance | false_confuser_rate | 1038 | -0.0723 | -0.0896 | -0.0559 | True | 0.0067 |
| occlusion_reappearance | false_reacquisition_rate | 1038 | -0.0723 | -0.0896 | -0.0559 | True | 0.0067 |
| long_gap_persistence | top1 | 498 | -0.0582 | -0.0823 | -0.0361 | True | 0.0060 |
| long_gap_persistence | MRR | 498 | -0.0389 | -0.0502 | -0.0271 | True | 0.0301 |
| long_gap_persistence | false_confuser_rate | 498 | -0.0582 | -0.0823 | -0.0361 | True | 0.0060 |
| long_gap_persistence | false_reacquisition_rate | 498 | -0.0582 | -0.0823 | -0.0361 | True | 0.0060 |
| occlusion_and_long_gap | top1 | 498 | -0.0582 | -0.0823 | -0.0361 | True | 0.0060 |
| occlusion_and_long_gap | MRR | 498 | -0.0389 | -0.0502 | -0.0271 | True | 0.0301 |
| occlusion_and_long_gap | false_confuser_rate | 498 | -0.0582 | -0.0823 | -0.0361 | True | 0.0060 |
| occlusion_and_long_gap | false_reacquisition_rate | 498 | -0.0582 | -0.0823 | -0.0361 | True | 0.0060 |
| appearance_similar_confuser | top1 | 624 | -0.0801 | -0.1026 | -0.0593 | True | 0.0000 |
| appearance_similar_confuser | MRR | 624 | -0.0513 | -0.0628 | -0.0399 | True | 0.0064 |
| appearance_similar_confuser | false_confuser_rate | 624 | -0.0801 | -0.1026 | -0.0593 | True | 0.0000 |
| appearance_similar_confuser | false_reacquisition_rate | 624 | -0.0801 | -0.1026 | -0.0593 | True | 0.0000 |
| spatially_close_confuser | top1 | 696 | -0.0632 | -0.0848 | -0.0431 | True | 0.0086 |
| spatially_close_confuser | MRR | 696 | -0.0455 | -0.0565 | -0.0345 | True | 0.0345 |
| spatially_close_confuser | false_confuser_rate | 696 | -0.0632 | -0.0848 | -0.0431 | True | 0.0086 |
| spatially_close_confuser | false_reacquisition_rate | 696 | -0.0632 | -0.0848 | -0.0431 | True | 0.0086 |
| crossing_or_overlap_confuser | top1 | 438 | -0.0616 | -0.0868 | -0.0365 | True | 0.0091 |
| crossing_or_overlap_confuser | MRR | 438 | -0.0386 | -0.0526 | -0.0244 | True | 0.0594 |
| crossing_or_overlap_confuser | false_confuser_rate | 438 | -0.0616 | -0.0868 | -0.0365 | True | 0.0091 |
| crossing_or_overlap_confuser | false_reacquisition_rate | 438 | -0.0616 | -0.0868 | -0.0365 | True | 0.0091 |
| OOD_confuser | top1 | 582 | -0.0790 | -0.1031 | -0.0567 | True | 0.0069 |
| OOD_confuser | MRR | 582 | -0.0501 | -0.0629 | -0.0374 | True | 0.0292 |
| OOD_confuser | false_confuser_rate | 582 | -0.0790 | -0.1031 | -0.0567 | True | 0.0069 |
| OOD_confuser | false_reacquisition_rate | 582 | -0.0790 | -0.1031 | -0.0567 | True | 0.0069 |

## full_trace_belief_vs_belief_with_shuffled_trace

| group | metric | count | mean_delta | ci95_low | ci95_high | zero_excluded | win_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| all_reacquisition_items | top1 | 1038 | 0.4162 | 0.3805 | 0.4518 | True | 0.4692 |
| all_reacquisition_items | MRR | 1038 | 0.5876 | 0.5599 | 0.6150 | True | 0.8902 |
| all_reacquisition_items | false_confuser_rate | 1038 | 0.4162 | 0.3805 | 0.4518 | True | 0.4692 |
| all_reacquisition_items | false_reacquisition_rate | 1038 | 0.4162 | 0.3805 | 0.4518 | True | 0.4692 |
| occlusion_reappearance | top1 | 1038 | 0.4162 | 0.3805 | 0.4518 | True | 0.4692 |
| occlusion_reappearance | MRR | 1038 | 0.5876 | 0.5599 | 0.6150 | True | 0.8902 |
| occlusion_reappearance | false_confuser_rate | 1038 | 0.4162 | 0.3805 | 0.4518 | True | 0.4692 |
| occlusion_reappearance | false_reacquisition_rate | 1038 | 0.4162 | 0.3805 | 0.4518 | True | 0.4692 |
| long_gap_persistence | top1 | 498 | 0.3534 | 0.3012 | 0.4076 | True | 0.4177 |
| long_gap_persistence | MRR | 498 | 0.5312 | 0.4905 | 0.5734 | True | 0.8434 |
| long_gap_persistence | false_confuser_rate | 498 | 0.3534 | 0.3012 | 0.4076 | True | 0.4177 |
| long_gap_persistence | false_reacquisition_rate | 498 | 0.3534 | 0.3012 | 0.4076 | True | 0.4177 |
| occlusion_and_long_gap | top1 | 498 | 0.3534 | 0.3012 | 0.4076 | True | 0.4177 |
| occlusion_and_long_gap | MRR | 498 | 0.5312 | 0.4905 | 0.5734 | True | 0.8434 |
| occlusion_and_long_gap | false_confuser_rate | 498 | 0.3534 | 0.3012 | 0.4076 | True | 0.4177 |
| occlusion_and_long_gap | false_reacquisition_rate | 498 | 0.3534 | 0.3012 | 0.4076 | True | 0.4177 |
| appearance_similar_confuser | top1 | 624 | 0.3830 | 0.3333 | 0.4311 | True | 0.4503 |
| appearance_similar_confuser | MRR | 624 | 0.5744 | 0.5385 | 0.6103 | True | 0.8830 |
| appearance_similar_confuser | false_confuser_rate | 624 | 0.3830 | 0.3333 | 0.4311 | True | 0.4503 |
| appearance_similar_confuser | false_reacquisition_rate | 624 | 0.3830 | 0.3333 | 0.4311 | True | 0.4503 |
| spatially_close_confuser | top1 | 696 | 0.3606 | 0.3175 | 0.4037 | True | 0.4152 |
| spatially_close_confuser | MRR | 696 | 0.5537 | 0.5212 | 0.5878 | True | 0.9152 |
| spatially_close_confuser | false_confuser_rate | 696 | 0.3606 | 0.3175 | 0.4037 | True | 0.4152 |
| spatially_close_confuser | false_reacquisition_rate | 696 | 0.3606 | 0.3175 | 0.4037 | True | 0.4152 |
| crossing_or_overlap_confuser | top1 | 438 | 0.3699 | 0.3151 | 0.4247 | True | 0.4247 |
| crossing_or_overlap_confuser | MRR | 438 | 0.5353 | 0.4896 | 0.5787 | True | 0.9224 |
| crossing_or_overlap_confuser | false_confuser_rate | 438 | 0.3699 | 0.3151 | 0.4247 | True | 0.4247 |
| crossing_or_overlap_confuser | false_reacquisition_rate | 438 | 0.3699 | 0.3151 | 0.4247 | True | 0.4247 |
| OOD_confuser | top1 | 582 | 0.3849 | 0.3351 | 0.4347 | True | 0.4467 |
| OOD_confuser | MRR | 582 | 0.5706 | 0.5325 | 0.6084 | True | 0.8849 |
| OOD_confuser | false_confuser_rate | 582 | 0.3849 | 0.3351 | 0.4347 | True | 0.4467 |
| OOD_confuser | false_reacquisition_rate | 582 | 0.3849 | 0.3351 | 0.4347 | True | 0.4467 |

## full_trace_belief_vs_belief_with_shuffled_gallery

| group | metric | count | mean_delta | ci95_low | ci95_high | zero_excluded | win_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| all_reacquisition_items | top1 | 1038 | 0.3979 | 0.3632 | 0.4345 | True | 0.4605 |
| all_reacquisition_items | MRR | 1038 | 0.5693 | 0.5401 | 0.5995 | True | 0.8719 |
| all_reacquisition_items | false_confuser_rate | 1038 | 0.3979 | 0.3632 | 0.4345 | True | 0.4605 |
| all_reacquisition_items | false_reacquisition_rate | 1038 | 0.3979 | 0.3632 | 0.4345 | True | 0.4605 |
| occlusion_reappearance | top1 | 1038 | 0.3979 | 0.3632 | 0.4345 | True | 0.4605 |
| occlusion_reappearance | MRR | 1038 | 0.5693 | 0.5401 | 0.5995 | True | 0.8719 |
| occlusion_reappearance | false_confuser_rate | 1038 | 0.3979 | 0.3632 | 0.4345 | True | 0.4605 |
| occlusion_reappearance | false_reacquisition_rate | 1038 | 0.3979 | 0.3632 | 0.4345 | True | 0.4605 |
| long_gap_persistence | top1 | 498 | 0.3494 | 0.2972 | 0.4056 | True | 0.4237 |
| long_gap_persistence | MRR | 498 | 0.5272 | 0.4851 | 0.5689 | True | 0.8394 |
| long_gap_persistence | false_confuser_rate | 498 | 0.3494 | 0.2972 | 0.4056 | True | 0.4237 |
| long_gap_persistence | false_reacquisition_rate | 498 | 0.3494 | 0.2972 | 0.4056 | True | 0.4237 |
| occlusion_and_long_gap | top1 | 498 | 0.3494 | 0.2972 | 0.4056 | True | 0.4237 |
| occlusion_and_long_gap | MRR | 498 | 0.5272 | 0.4851 | 0.5689 | True | 0.8394 |
| occlusion_and_long_gap | false_confuser_rate | 498 | 0.3494 | 0.2972 | 0.4056 | True | 0.4237 |
| occlusion_and_long_gap | false_reacquisition_rate | 498 | 0.3494 | 0.2972 | 0.4056 | True | 0.4237 |
| appearance_similar_confuser | top1 | 624 | 0.3686 | 0.3173 | 0.4167 | True | 0.4391 |
| appearance_similar_confuser | MRR | 624 | 0.5599 | 0.5225 | 0.5964 | True | 0.8686 |
| appearance_similar_confuser | false_confuser_rate | 624 | 0.3686 | 0.3173 | 0.4167 | True | 0.4391 |
| appearance_similar_confuser | false_reacquisition_rate | 624 | 0.3686 | 0.3173 | 0.4167 | True | 0.4391 |
| spatially_close_confuser | top1 | 696 | 0.3434 | 0.2989 | 0.3908 | True | 0.4080 |
| spatially_close_confuser | MRR | 696 | 0.5364 | 0.5002 | 0.5703 | True | 0.8980 |
| spatially_close_confuser | false_confuser_rate | 696 | 0.3434 | 0.2989 | 0.3908 | True | 0.4080 |
| spatially_close_confuser | false_reacquisition_rate | 696 | 0.3434 | 0.2989 | 0.3908 | True | 0.4080 |
| crossing_or_overlap_confuser | top1 | 438 | 0.3379 | 0.2785 | 0.3973 | True | 0.4178 |
| crossing_or_overlap_confuser | MRR | 438 | 0.5033 | 0.4540 | 0.5509 | True | 0.8904 |
| crossing_or_overlap_confuser | false_confuser_rate | 438 | 0.3379 | 0.2785 | 0.3973 | True | 0.4178 |
| crossing_or_overlap_confuser | false_reacquisition_rate | 438 | 0.3379 | 0.2785 | 0.3973 | True | 0.4178 |
| OOD_confuser | top1 | 582 | 0.3540 | 0.3007 | 0.4038 | True | 0.4296 |
| OOD_confuser | MRR | 582 | 0.5397 | 0.4998 | 0.5798 | True | 0.8540 |
| OOD_confuser | false_confuser_rate | 582 | 0.3540 | 0.3007 | 0.4038 | True | 0.4296 |
| OOD_confuser | false_reacquisition_rate | 582 | 0.3540 | 0.3007 | 0.4038 | True | 0.4296 |

