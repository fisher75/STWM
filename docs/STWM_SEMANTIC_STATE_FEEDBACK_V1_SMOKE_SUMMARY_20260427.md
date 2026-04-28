# STWM Semantic-State Feedback V1 Smoke Summary

- feedback_smoke_passed: true
- max_items: 128
- mode: readout_only

| variant | event AP | event AUROC | per-horizon AP | per-horizon AUROC | coord error | gate mean | delta norm | degenerate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| disabled | 0.896940 | 0.805556 | 0.157445 | 0.233697 | 0.205128 | 0.000000 | 0.000000 | false |
| alpha=0.05 | 0.896940 | 0.805556 | 0.157445 | 0.233697 | 0.205128 | 0.017986 | 0.000000 | false |
| alpha=0.10 | 0.896940 | 0.805556 | 0.157445 | 0.233697 | 0.205128 | 0.017986 | 0.000000 | false |

Smoke interpretation: alpha variants now instantiate the full-model feedback adapter. The adapter is zero-delta initialized, so metrics match disabled before training; the nonzero gate confirms the feedback path is wired without trace regression.
