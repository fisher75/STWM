# STWM External Candidate Scoring V4 Dev/Heldout Summary

- split_rule: `sha256(item_id)%2`
- no_heldout_tuning: `true`
- selected_weights: `{'distance': 0.0, 'target_appearance': 0.65, 'predicted_semantic': 0.0, 'predicted_identity': 0.1, 'priors': 0.24999999999999997}`
- overfit_warning: `False`

| mode | split | top1 | AP | AUROC | MRR |
|---|---|---:|---:|---:|---:|
| distance_only | dev | 0.1696 | 0.0844 | 0.5327 | 0.3672 |
| distance_only | heldout | 0.1674 | 0.0806 | 0.5351 | 0.3635 |
| weak_posterior_v3 | dev | 0.1579 | 0.0844 | 0.5309 | 0.3585 |
| weak_posterior_v3 | heldout | 0.1721 | 0.0804 | 0.5329 | 0.3673 |
| target_candidate_appearance_only | dev | 0.4035 | 0.2666 | 0.6747 | 0.5842 |
| target_candidate_appearance_only | heldout | 0.3860 | 0.2329 | 0.6589 | 0.5805 |
| predicted_semantic_to_candidate | dev | 0.2047 | 0.0753 | 0.4998 | 0.3886 |
| predicted_semantic_to_candidate | heldout | 0.1907 | 0.0687 | 0.4747 | 0.3978 |
| predicted_identity_to_candidate | dev | 0.1871 | 0.0745 | 0.4724 | 0.3761 |
| predicted_identity_to_candidate | heldout | 0.1767 | 0.0722 | 0.4556 | 0.3827 |
| predicted_semantic_identity_to_candidate | dev | 0.1871 | 0.0758 | 0.4834 | 0.3748 |
| predicted_semantic_identity_to_candidate | heldout | 0.1860 | 0.0708 | 0.4606 | 0.3957 |
| posterior_v4 | dev | 0.1579 | 0.0855 | 0.5402 | 0.3624 |
| posterior_v4 | heldout | 0.1814 | 0.0825 | 0.5474 | 0.3776 |
| posterior_v4_no_distance | dev | 0.2749 | 0.1202 | 0.6063 | 0.4739 |
| posterior_v4_no_distance | heldout | 0.2233 | 0.1074 | 0.5630 | 0.4429 |
| posterior_v4_no_semantic_identity | dev | 0.1579 | 0.0855 | 0.5406 | 0.3613 |
| posterior_v4_no_semantic_identity | heldout | 0.1814 | 0.0829 | 0.5505 | 0.3807 |
| posterior_v4_no_target_candidate_appearance | dev | 0.1579 | 0.0843 | 0.5305 | 0.3593 |
| posterior_v4_no_target_candidate_appearance | heldout | 0.1674 | 0.0803 | 0.5325 | 0.3650 |

## Selected Dev Weights
- dev: `{'metric_items': 171, 'candidate_top1': 0.3742690058479532, 'candidate_MRR': 0.5736406648429352, 'candidate_AP': 0.21211445705468956, 'candidate_AUROC': 0.6769566020313943, 'candidate_positive_rate': 0.06716417910447761}`
- heldout: `{'metric_items': 215, 'candidate_top1': 0.3488372093023256, 'candidate_MRR': 0.5570474260881237, 'candidate_AP': 0.17215546078564506, 'candidate_AUROC': 0.6503726688205752, 'candidate_positive_rate': 0.06884406019852705}`
