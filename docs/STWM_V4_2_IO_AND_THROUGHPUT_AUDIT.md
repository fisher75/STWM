# STWM V4.2 IO And Throughput Audit

## Purpose

This document records phase0/1 throughput and IO audit for lane scale-out decisions.

Audit summarizer:

- `code/stwm/tools/summarize_stwm_v4_2_io_audit.py`

## Scale-Out Gate Criteria

A lane is scale-out ready only if all checks pass:

1. median GPU utilization `>= 85`
2. `step_time_p95 / step_time_p50 <= 1.5`
3. no IO sawtooth (`step_time_spike_share <= 0.10`)
4. `disk_free_gb_min >= 50`
5. checkpoint files `latest.pt` and `best.pt` both exist

Global decision:

- `can_expand_to_4_lanes = all(lane_ok_for_scale_out)`

## Evidence Status

### Non-Compliant Warmup Benchmark (Reference Only)

Run root:

- `outputs/audits/stwm_v4_2_phase01_20260401_155639`

Summary:

- `phase1_audit_summary.json` reported `can_expand_to_4_lanes=true`
- lane0 (1B): util median `93`, step ratio `1.26`
- lane1 (220M): util median `95`, step ratio `1.42`
- resume check passed in both lanes (`start_step=120`, `resolved_steps=122`)

Why reference-only:

- this run used truncated sample coverage (`budget.sample_count=18`), so it cannot be used as main evidence for true-train compliance.

### Compliant Full-Train Warmup (Authoritative)

In-progress run root:

- `outputs/audits/stwm_v4_2_phase01_20260401_161909`

Required completion artifacts:

- `phase1_audit_summary.json`
- `lane0/resume_check.json`
- `lane1/resume_check.json`
- lane summaries with full-train `sample_count` in `mini_val_summary.json`

Final decision must be taken from this compliant run only.

## Data-Wait Observation

Early compliant-run logs (first few steps) indicate high data-wait ratio (about `0.83-0.86`) for both lanes.

Operational interpretation:

1. This is expected to be higher than mini-split warmup due to full-train IO diversity.
2. Lane expansion should still follow formal gate checks above.
3. If gate fails, reduce lane count before any other change.

## Update Procedure

After compliant run finishes:

1. Copy final gate metrics from `phase1_audit_summary.json`.
2. Confirm resume checks are true in both lanes.
3. Confirm checkpoint existence checks are true in both lanes.
4. Record final `can_expand_to_4_lanes` decision and rationale.
