# Stage2 Bootstrap Data Contract

- generated_at_utc: 2026-04-08T16:33:10.260943+00:00
- bootstrap_interface_ready: True
- source_audit_json: /home/chen034/workspace/data/_manifests/stage2_dataset_audit_20260408.json

## Binding
- core: ['VSPW', 'VIPSeg']
- optional_extension: ['BURST']

## Included Datasets
| dataset | role | status_from_audit | used_in_bootstrap_train | used_in_bootstrap_eval | local_path |
|---|---|---|---|---|---|
| VSPW | Stage2 core data | complete | True | True | /home/chen034/workspace/stwm/data/external/vspw/VSPW |
| VIPSeg | Stage2 core data | complete | True | True | /home/chen034/workspace/stwm/data/external/vipseg/VIPSeg |
| BURST | Stage2 open-world extension | complete | False | True | /home/chen034/workspace/stwm/data/external/burst |

## Excluded Datasets
| dataset | not_in_current_bootstrap | reason |
|---|---|---|
| TAO | True | access_ready |
| VISOR | True | manual_gate |
