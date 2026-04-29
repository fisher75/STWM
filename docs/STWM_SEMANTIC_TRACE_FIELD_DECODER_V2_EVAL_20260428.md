# STWM Semantic Trace Field Decoder V2 Eval

## Head-Only vs Controlled Semantic Branch

| checkpoint | free proto acc | free proto top5 | free masked CE | freq top5 | trace coord error |
|---|---:|---:|---:|---:|---:|
| head-only V2 | 0.000000 | 0.033854 | 5.012193 | 0.179688 | 0.205717 |
| semantic-branch V2 | 0.000000 | 0.033854 | 5.004487 | 0.179688 | 0.205983 |

## Conclusion

Controlled semantic-branch unfreeze slightly lowers masked CE but does not improve prototype top5 beyond the frequency baseline. Trace rollout regression is `False` and output degeneracy is `false`. The current bottleneck is target/representation quality, not evidence against the STWM world-model direction.
