#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export MODEL_ARGS_FILE="${SCRIPT_DIR}/../models/qwen3-4B.sh"
export MODEL_PATH="/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/models/Qwen3-4B"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
export ROLLOUT_GPU_SET_CSV="${ROLLOUT_GPU_SET_CSV:-0,1;2,3;4,5;6,7}"

exec "${SCRIPT_DIR}/run-no-fast-restart-weight-update-smoke-32gpu.sh" "$@"
