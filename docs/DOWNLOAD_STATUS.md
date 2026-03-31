# Download Status

## Datasets

| Dataset | Priority | Status | Retries | Target path | Notes |
|---|---:|---|---:|---|---|
| VSPW | 1 | downloaded_and_extracted | 0 | `/home/chen034/workspace/stwm/data/raw/vspw` | raw archive present and extracted to `/home/chen034/workspace/stwm/data/external/vspw` |
| VISOR | 2 | downloaded_and_extracted | 0 | `/home/chen034/workspace/stwm/data/raw/visor` | official zip present and extracted to `/home/chen034/workspace/stwm/data/external/visor` |
| VIPSeg | 3 | downloaded_and_extracted | 0 | `/home/chen034/workspace/stwm/data/raw/vipseg` | extractor updated to handle tar payload under `.zip` filename; extracted to `/home/chen034/workspace/stwm/data/external/vipseg` |
| BURST annotations | 4 | downloaded_and_extracted | 0 | `/home/chen034/workspace/stwm/data/raw/burst` | extracted to `/home/chen034/workspace/stwm/data/external/burst/annotations` |
| TAO train | 4 | downloaded_and_extracted | 0 | `/home/chen034/workspace/stwm/data/raw/burst` | extracted to `/home/chen034/workspace/stwm/data/external/burst/images/train` |
| TAO val | 4 | downloaded_and_extracted | 0 | `/home/chen034/workspace/stwm/data/raw/burst` | extracted to `/home/chen034/workspace/stwm/data/external/burst/images/val` |
| TAO test | 4 | downloaded_and_extracted | 0 | `/home/chen034/workspace/stwm/data/raw/burst` | extracted to `/home/chen034/workspace/stwm/data/external/burst/images/test`; verified `982,754` files |

## Models

| Model | Status | Retries | Target path | Notes |
|---|---|---:|---|---|
| TraceAnything pretrained | downloaded | 0 | `/home/chen034/workspace/stwm/models/checkpoints/traceanything` | official Hugging Face checkpoint saved |
| SAM2.1 hiera large | downloaded | 0 | `/home/chen034/workspace/stwm/models/checkpoints/sam2` | official Meta checkpoint saved |
| SAM2.1 hiera base+ | downloaded | 0 | `/home/chen034/workspace/stwm/models/checkpoints/sam2` | official Meta checkpoint saved |
| DEVA public weights | downloaded | 0 | `/home/chen034/workspace/stwm/models/checkpoints/deva` | full DEVA dependency bundle present |
| Cutie pretrained | downloaded | 0 | `/home/chen034/workspace/stwm/models/checkpoints/cutie` | base and interactive checkpoints present |
| YOLO-World weights | downloaded | 0 | `/home/chen034/workspace/stwm/models/checkpoints/yolo_world` | official V2.1 S/M/L checkpoints present |
