# STWM Semantic-Only TUSB Unfreeze V1 Eval

| model | C | free proto acc | free proto top5 | freq top5 | top5 gain | masked CE | trace coord error |
|---|---:|---:|---:|---:|---:|---:|---:|
| head-only V2 | 128 | 0.000000 | 0.033854 | 0.179688 | -0.145833 | 5.012193 | 0.205717 |
| head/proj-only V2 | 128 | 0.000000 | 0.033854 | 0.179688 | -0.145833 | 5.004487 | 0.205983 |
| semantic-only TUSB | 64 | 0.010417 | 0.040690 | 0.249023 | -0.208333 | 4.385637 | 0.206147 |
| semantic-only TUSB | 128 | 0.002279 | 0.033854 | 0.179688 | -0.145833 | 4.990012 | 0.207193 |

The true semantic-only TUSB unfreeze did not produce a positive semantic prototype field signal. C=64 is the better of the two runs by free-rollout top5, but it remains well below the frequency baseline.
