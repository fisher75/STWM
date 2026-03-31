#!/usr/bin/env bash
set -euo pipefail

export STWM_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
ENV_FILE="$STWM_ROOT/env/stwm.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

detect_resources() {
  export STWM_CPU_TOTAL="${STWM_CPU_TOTAL:-$(nproc 2>/dev/null || echo 1)}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    export STWM_GPU_TOTAL="${STWM_GPU_TOTAL:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
  else
    export STWM_GPU_TOTAL="${STWM_GPU_TOTAL:-0}"
  fi
  local gpu_divisor=1
  if [[ "${STWM_GPU_TOTAL:-0}" -gt 0 ]]; then
    gpu_divisor="$STWM_GPU_TOTAL"
  fi
  export STWM_CPUS_PER_GPU="${STWM_CPUS_PER_GPU:-$(( STWM_CPU_TOTAL / gpu_divisor ))}"
  export STWM_IO_WORKERS="${STWM_IO_WORKERS:-$(( STWM_CPU_TOTAL / 7 ))}"
  export STWM_PREPROCESS_WORKERS="${STWM_PREPROCESS_WORKERS:-$(( STWM_CPU_TOTAL / 2 ))}"
  export STWM_DOWNLOAD_CONNECTIONS="${STWM_DOWNLOAD_CONNECTIONS:-16}"
  export STWM_DOWNLOAD_SPLITS="${STWM_DOWNLOAD_SPLITS:-16}"
  export STWM_TRAIN_GPUS="${STWM_TRAIN_GPUS:-$STWM_GPU_TOTAL}"
}

detect_resources

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

slug_now() {
  date +"%Y%m%d_%H%M%S"
}

ensure_dir() {
  mkdir -p "$@"
}

retry() {
  local attempts="$1"
  shift
  local n=1
  until "$@"; do
    if (( n >= attempts )); then
      return 1
    fi
    sleep $(( n * 5 ))
    n=$(( n + 1 ))
  done
}

init_csv() {
  local path="$1"
  local header="$2"
  if [[ ! -f "$path" ]]; then
    printf "%s\n" "$header" > "$path"
  fi
}

conda_setup() {
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1090
  source "$conda_base/etc/profile.d/conda.sh"
}

note_blocker() {
  local title="$1"
  local detail="$2"
  local blocker_file="$STWM_ROOT/docs/BLOCKERS.md"
  if [[ ! -f "$blocker_file" ]]; then
    {
      echo "# Blockers"
      echo
    } > "$blocker_file"
  fi
  {
    echo "## $(timestamp) - $title"
    echo
    echo "$detail"
    echo
  } >> "$blocker_file"
}
