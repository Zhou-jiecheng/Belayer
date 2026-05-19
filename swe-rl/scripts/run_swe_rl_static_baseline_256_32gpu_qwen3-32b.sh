#!/bin/bash

# SWE-RL static-concurrency baseline (256), checkpoint disabled, on 4 nodes / 32 GPUs.
# Scheduler should inject:
#   MLP_ROLE_INDEX=0/1/2/3
#   MLP_WORKER_0_HOST=<head_ip>
#   MLP_WORKER_1_HOST=<worker1_ip>
#   MLP_WORKER_2_HOST=<worker2_ip>
#   MLP_WORKER_3_HOST=<worker3_ip>
#
# All nodes run this same script. The first node is treated as MASTER_ADDR / Ray head.

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LAUNCH_SCRIPT_DIR="${SCRIPT_DIR}/launch_script"

unset SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK:-1}"

export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
export MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}}"

export TOTAL_NODE_COUNT="${TOTAL_NODE_COUNT:-4}"
export ACTOR_NODE_COUNT="${ACTOR_NODE_COUNT:-1}"
export ROLLOUT_NODE_COUNT="${ROLLOUT_NODE_COUNT:-3}"
export ACTOR_GPU_SET="${ACTOR_GPU_SET:-0,1,2,3,4,5,6,7}"
export ROLLOUT_GPU_SET_CSV="${ROLLOUT_GPU_SET_CSV:-0,1,2,3,4,5,6,7}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
export STATIC_MAX_CONCURRENCY="${STATIC_MAX_CONCURRENCY:-256}"

echo "MLP_ROLE_INDEX=${MLP_ROLE_INDEX}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "TOTAL_NODE_COUNT=${TOTAL_NODE_COUNT}"
echo "ACTOR_NODE_COUNT=${ACTOR_NODE_COUNT}"
echo "ROLLOUT_NODE_COUNT=${ROLLOUT_NODE_COUNT}"
echo "ACTOR_GPU_SET=${ACTOR_GPU_SET}"
echo "ROLLOUT_GPU_SET_CSV=${ROLLOUT_GPU_SET_CSV}"
echo "TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE}"
echo "ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "STATIC_MAX_CONCURRENCY=${STATIC_MAX_CONCURRENCY}"
echo "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=${SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK}"

exec "${LAUNCH_SCRIPT_DIR}/run_swe_rl_static_baseline_256_32gpu_qwen3-32b.sh" "$@"
