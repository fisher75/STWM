# STWM

STWM stands for Semantic Trajectory World Model.

This workspace is organized for:

- reproducible dataset and checkpoint downloads
- centralized caches, logs, and manifests
- lightweight smoke tests before any full preprocessing
- a minimal STWM code skeleton for week-1 integration

All long-running jobs should run inside `tmux` and write logs under `logs/`.

Quick start:

```bash
cd /home/chen034/workspace/stwm
source env/stwm.env
scripts/bootstrap.sh
scripts/start_downloads_tmux.sh
```

Primary data priority:

1. `VSPW`
2. `VISOR`
3. `VIPSeg`
4. `BURST/TAO`

Primary model/backbone priority:

1. `TraceAnything`
2. `OV2VSS`
3. `SAM2`
4. `DEVA`
5. `Cutie`
6. `XMem`
7. `YOLO-World`
