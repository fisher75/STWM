#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

ensure_dir "$STWM_ROOT/env" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
conda_setup

ENV_NAME="stwm"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_WHEEL_INDEX="${TORCH_WHEEL_INDEX:-https://download.pytorch.org/whl/cu128}"
EXPORT_PATH="$STWM_ROOT/env/stwm.yml"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists."
else
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --extra-index-url "$TORCH_WHEEL_INDEX" torch torchvision
python -m pip install \
  transformers \
  accelerate \
  opencv-python \
  pillow \
  imageio \
  fvcore \
  matplotlib \
  einops \
  hydra-core \
  iopath \
  omegaconf \
  pyyaml \
  pulp \
  tqdm \
  scipy \
  scikit-image \
  scikit-learn \
  pandas \
  pycocotools \
  jupyter \
  ipykernel \
  gdown \
  aria2p \
  huggingface_hub \
  sentencepiece

if ! python -m pip install --no-build-isolation flash-attn; then
  echo "flash-attn installation skipped or failed; continuing without it."
fi

python -m ipykernel install --user --name "$ENV_NAME" --display-name "STWM"
conda env export -n "$ENV_NAME" > "$EXPORT_PATH"

echo "Environment exported to $EXPORT_PATH"
