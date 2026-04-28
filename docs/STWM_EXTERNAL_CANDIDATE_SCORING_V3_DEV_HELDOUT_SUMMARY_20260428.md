# STWM External Candidate Scoring V3 Dev/Heldout Summary

- split_rule: `sha256(item_id) % 2`
- no_heldout_tuning: `true`
- selected_score_weights: `fixed_posterior_v1_no_tuning`

| mode | split | top1 | AP | AUROC | MRR | metric_items |
|---|---|---:|---:|---:|---:|---:|
| distance_only | dev | 0.1696 | 0.0844 | 0.5327 | 0.3672 | 171 |
| distance_only | heldout | 0.1674 | 0.0806 | 0.5351 | 0.3635 | 215 |
| priors_only | dev | 0.1053 | 0.0797 | 0.5197 | 0.3085 | 171 |
| priors_only | heldout | 0.1535 | 0.0810 | 0.5122 | 0.3456 | 215 |
| semantic_only | dev | 0.2047 | 0.0680 | 0.4900 | 0.4001 | 171 |
| semantic_only | heldout | 0.1860 | 0.0645 | 0.4834 | 0.3894 | 215 |
| identity_only | dev | 0.1754 | 0.0645 | 0.4877 | 0.3674 | 171 |
| identity_only | heldout | 0.2093 | 0.0677 | 0.4881 | 0.4064 | 215 |
| posterior_v1 | dev | 0.1579 | 0.0843 | 0.5287 | 0.3608 | 171 |
| posterior_v1 | heldout | 0.1674 | 0.0801 | 0.5281 | 0.3622 | 215 |
| posterior_no_distance | dev | 0.1930 | 0.0657 | 0.4922 | 0.3915 | 171 |
| posterior_no_distance | heldout | 0.1860 | 0.0645 | 0.4852 | 0.3908 | 215 |
| posterior_no_semantic | dev | 0.1696 | 0.0843 | 0.5294 | 0.3640 | 171 |
| posterior_no_semantic | heldout | 0.1721 | 0.0806 | 0.5337 | 0.3669 | 215 |
