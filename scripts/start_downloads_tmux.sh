#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

ensure_dir "$STWM_ROOT/logs"

make_session() {
  local name="$1"
  local cmd="$2"
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "tmux session already exists: $name"
    return 0
  fi
  tmux new-session -d -s "$name" "bash -lc 'cd \"$STWM_ROOT\" && $cmd'"
  echo "started tmux session: $name"
}

SETUP_CMD="scripts/bootstrap.sh 2>&1 | tee -a logs/bootstrap.log && scripts/setup_env.sh 2>&1 | tee -a logs/setup_env.log && scripts/clone_repos.sh 2>&1 | tee -a logs/clone_repos.log"
MODELS_CMD="while [[ ! -d \"$STWM_ROOT/third_party/sam2\" ]]; do sleep 30; done; scripts/download_models.sh --all 2>&1 | tee -a logs/download_models.log"
DATA_CMD="while ! conda run -n stwm python -m gdown --help >/dev/null 2>&1; do sleep 30; done; scripts/download_datasets.sh --all 2>&1 | tee -a logs/download_datasets.log"

make_session "stwm_setup" "$SETUP_CMD"
make_session "stwm_models" "$MODELS_CMD"
make_session "stwm_data" "$DATA_CMD"

tmux ls
