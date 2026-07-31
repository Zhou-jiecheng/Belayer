#!/bin/bash

# Thin wrapper matching the adaptive 16-GPU qwen3-32B launch style,
# but forcing the static checkpoint path:
# - no shadow worker
# - no slime router
# - env checkpoint enabled
# - no fault injection

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../../../slime" &>/dev/null && pwd)"

export MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${SLIME_DIR}/scripts/models/qwen3-32B.sh}"
export HF_CKPT="${HF_CKPT:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/models/Qwen3-32B}"
export REF_LOAD="${REF_LOAD:-${HF_CKPT}}"

# 16 GPU default topology:
# - 1 training node x 8 GPUs
# - 1 rollout node x 8 GPUs
# - training TP = 8
# - rollout TP = 4 (two engines per rollout node)
export TOTAL_NODE_COUNT="${TOTAL_NODE_COUNT:-4}"
export ACTOR_NODE_COUNT="${ACTOR_NODE_COUNT:-1}"
export ROLLOUT_NODE_COUNT="${ROLLOUT_NODE_COUNT:-3}"
export ACTOR_GPU_SET="${ACTOR_GPU_SET:-0,1,2,3,4,5,6,7}"
export ROLLOUT_GPU_SET_CSV="${ROLLOUT_GPU_SET_CSV:-0,1,2,3,4,5,6,7}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"

export STATIC_MAX_CONCURRENCY="${STATIC_MAX_CONCURRENCY:-256}"
export SWE_CHECKPOINT_POLICY="${SWE_CHECKPOINT_POLICY:-never}"
export SWE_FAULT_INJECTION_ENABLE="${SWE_FAULT_INJECTION_ENABLE:-0}"

exec "${SCRIPT_DIR}/run_swe_rl_static_checkpoint_multinode.sh" "$@"
