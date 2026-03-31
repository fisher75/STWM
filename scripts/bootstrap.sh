#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

ensure_dir "$STWM_ROOT/docs" "$STWM_ROOT/logs" "$STWM_ROOT/manifests" "$STWM_ROOT/tmp"

REPORT="$STWM_ROOT/docs/BOOTSTRAP_REPORT.md"
TOOLS=(git tmux aria2c wget curl gdown python conda pip nvidia-smi rg)

{
  echo "# Bootstrap Report"
  echo
  echo "- Generated: $(timestamp)"
  echo "- Host: $(hostname)"
  echo "- User: $(whoami)"
  echo "- STWM root: $STWM_ROOT"
  echo
  echo "## Filesystem"
  df -h "$STWM_ROOT" || df -h /home/chen034
  echo
  echo "## GPU"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  else
    echo "nvidia-smi not found"
  fi
  echo
  echo "## Tool Check"
  echo
  echo "| Tool | Path | Status |"
  echo "|---|---|---|"
  for tool in "${TOOLS[@]}"; do
    path="$(command -v "$tool" || true)"
    status="missing"
    if [[ -n "$path" ]]; then
      status="ok"
    fi
    printf "| %s | %s | %s |\n" "$tool" "${path:--}" "$status"
  done
  echo
  echo "## Versions"
  echo
  echo "- Python: $(python --version 2>&1 || true)"
  echo "- Pip: $(pip --version 2>&1 || true)"
  echo "- Conda: $(conda --version 2>&1 || true)"
  echo "- Git: $(git --version 2>&1 || true)"
  echo "- Tmux: $(tmux -V 2>&1 || true)"
  echo "- Aria2c: $(aria2c --version 2>/dev/null | head -n 1 || true)"
} > "$REPORT"

echo "Bootstrap report written to $REPORT"
