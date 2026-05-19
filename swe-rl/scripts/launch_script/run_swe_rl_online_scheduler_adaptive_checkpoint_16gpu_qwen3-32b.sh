#!/bin/bash

# Thin wrapper matching the 32-GPU qwen3-32B launch style:
# keep model/topology defaults in one place, then reuse the generic multi-node
# SWE-RL launcher.

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
export TOTAL_NODE_COUNT="${TOTAL_NODE_COUNT:-2}"
export ACTOR_NODE_COUNT="${ACTOR_NODE_COUNT:-1}"
export ROLLOUT_NODE_COUNT="${ROLLOUT_NODE_COUNT:-1}"
export ACTOR_GPU_SET="${ACTOR_GPU_SET:-0,1,2,3,4,5,6,7}"
export ROLLOUT_GPU_SET_CSV="${ROLLOUT_GPU_SET_CSV:-0,1,2,3;4,5,6,7}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
export SWE_CHECKPOINT_POLICY="${SWE_CHECKPOINT_POLICY:-never}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
export SWE_MAX_CONCURRENT_EVAL=16 # same as the CPU num
export SWE_MAX_CONCURRENT_DIFF=16 # same as the CPU num
export SHADOW_WORKER_READY_TIMEOUT_SEC="${SHADOW_WORKER_READY_TIMEOUT_SEC:-600}"
export SHADOW_WORKER_STABILIZATION_SEC="${SHADOW_WORKER_STABILIZATION_SEC:-20}"

exec "${SCRIPT_DIR}/run_swe_rl_online_scheduler_adaptive_checkpoint_multinode.sh" "$@"
