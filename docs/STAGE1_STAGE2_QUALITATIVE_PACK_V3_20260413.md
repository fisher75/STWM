# Stage1 / Stage2 Qualitative Pack V3

- generated_at_utc: 2026-04-13T14:46:23.453217+00:00
- stage1_pack: /home/chen034/workspace/stwm/reports/stage1_qualitative_pack_v3_20260413.json
- stage2_pack: /home/chen034/workspace/stwm/reports/stage2_qualitative_pack_v3_20260413.json
- v7_repaired_overall_best: stage2_semobjv7_alignonly_topk1_seed123_20260413
- v7_repaired_semantic_hard_best: stage2_semobjv7_alignpersist_topk1_seed123_20260413
- v7_repaired_best_effective_persistence: none

## Stage1

| case_id | bucket | dataset | clip_id | render |
|---|---|---|---|---|
| stage1_000 | easy_cases | VSPW | 127_-hIVCYO4C90 | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage1/easy_cases_stage1_000.png |
| stage1_001 | easy_cases | VSPW | 1643_T9npC-YHzuE | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage1/easy_cases_stage1_001.png |
| stage1_002 | easy_cases | VSPW | 112_0OB2IP1enjU | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage1/easy_cases_stage1_002.png |
| stage1_003 | dynamic_change_cases | VSPW | 231_-_w6ZFauJBI | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage1/dynamic_change_cases_stage1_003.png |
| stage1_004 | dynamic_change_cases | VSPW | 2097_HVti7xTm2ow | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage1/dynamic_change_cases_stage1_004.png |
| stage1_005 | dynamic_change_cases | VSPW | 1296_oaQaoEjjG7I | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage1/dynamic_change_cases_stage1_005.png |
| stage1_006 | failure_boundary_cases | VSPW | 1678__qxxSOgqMpc | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage1/failure_boundary_cases_stage1_006.png |
| stage1_007 | failure_boundary_cases | VSPW | 93_qZmq-lc8lAg | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage1/failure_boundary_cases_stage1_007.png |
| stage1_008 | failure_boundary_cases | VSPW | 50_9mZFBNGzmok | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage1/failure_boundary_cases_stage1_008.png |

## Stage2

- 说明: `persistence_declared_probe_cases` 只表示 persistence-declared checkpoint 在个别 hard case 上看起来更好，但 v7 repair audit 判定 persistence telemetry 仍然 inactive，不能当作 persistence 已实际起效的证据。

| case_id | group | dataset | clip_id | why_selected | qualitative_interpretation | render |
|---|---|---|---|---|---|---|
| stage2_000 | persistence_declared_probe_cases | VSPW | 2097_HVti7xTm2ow | persistence-declared checkpoint looked favorable on this semantic-hard clip, but v7 repair audit marked persistence telemetry inactive; keep only as probe case | this clip visually favors the persistence-declared sidecar checkpoint, but it is not valid evidence that the persistence branch became effective | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage2/persistence_active_improved_cases_stage2_000.png |
| stage2_001 | persistence_declared_probe_cases | VSPW | 736_ML-JwZIxno0 | persistence-declared checkpoint looked favorable on this semantic-hard clip, but v7 repair audit marked persistence telemetry inactive; keep only as probe case | this clip visually favors the persistence-declared sidecar checkpoint, but it is not valid evidence that the persistence branch became effective | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage2/persistence_active_improved_cases_stage2_001.png |
| stage2_002 | alignment_only_wins_cases | VSPW | 1010_kI0mOZirPGs | alignment-only branch beats persistence-declared branch and baseline envelope | calibration-only branch appears sufficient on this clip | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage2/alignment_only_wins_cases_stage2_002.png |
| stage2_003 | alignment_only_wins_cases | VSPW | 794_701d4x0AN9E | alignment-only branch beats persistence-declared branch and baseline envelope | calibration-only branch appears sufficient on this clip | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage2/alignment_only_wins_cases_stage2_003.png |
| stage2_004 | persistence_declared_but_inactive_cases | VSPW | 1010_kI0mOZirPGs | persistence was declared but did not show active gain pattern | declared persistence likely inactive or non-contributive | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage2/persistence_declared_but_inactive_cases_stage2_004.png |
| stage2_005 | persistence_declared_but_inactive_cases | VSPW | 794_701d4x0AN9E | persistence was declared but did not show active gain pattern | declared persistence likely inactive or non-contributive | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage2/persistence_declared_but_inactive_cases_stage2_005.png |
| stage2_006 | persistence_declared_but_inactive_cases | VSPW | 1272_kDLzAZhFEVY | persistence was declared but did not show active gain pattern | declared persistence likely inactive or non-contributive | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage2/persistence_declared_but_inactive_cases_stage2_006.png |
| stage2_007 | persistence_declared_but_inactive_cases | VSPW | 1132_ngMJTWUanCA | persistence was declared but did not show active gain pattern | declared persistence likely inactive or non-contributive | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage2/persistence_declared_but_inactive_cases_stage2_007.png |
| stage2_008 | unresolved_or_failure_cases | VSPW | 281__lcLNaySFDc | neither branch clearly dominates under semantic-hard stress | remaining ambiguity/failure case | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage2/unresolved_or_failure_cases_stage2_008.png |
| stage2_009 | unresolved_or_failure_cases | VSPW | 127_-hIVCYO4C90 | neither branch clearly dominates under semantic-hard stress | remaining ambiguity/failure case | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage2/unresolved_or_failure_cases_stage2_009.png |
| stage2_010 | unresolved_or_failure_cases | VSPW | 2097_HVti7xTm2ow | neither branch clearly dominates under semantic-hard stress | remaining ambiguity/failure case | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage2/unresolved_or_failure_cases_stage2_010.png |
| stage2_011 | unresolved_or_failure_cases | VSPW | 1272_kDLzAZhFEVY | neither branch clearly dominates under semantic-hard stress | remaining ambiguity/failure case | /home/chen034/workspace/stwm/outputs/visualizations/stage1_stage2_qualitative_pack_v3_20260413/stage2/unresolved_or_failure_cases_stage2_011.png |
