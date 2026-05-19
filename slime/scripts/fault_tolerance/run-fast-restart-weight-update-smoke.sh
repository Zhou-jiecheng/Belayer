#!/bin/bash

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
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SLIME_ROOT}/.." &>/dev/null && pwd)"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
export PYTHONUNBUFFERED=1

MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${SCRIPT_DIR}/../models/qwen3-4B.sh}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/models/Qwen3-4B}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/data/dapo-math-17k.jsonl}"

ACTOR_GPU_SET="${ACTOR_GPU_SET:-0,1,2,3}"
ROLLOUT_GPU_SET_CSV="${ROLLOUT_GPU_SET_CSV:-4,5;6,7}"
FIRST_GPU_ID=4
WEIGHT_SERVER_BASE_PORT="${WEIGHT_SERVER_BASE_PORT:-5556}"
RAY_HEAD_IP="${RAY_HEAD_IP:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
KV_CACHE_MEM_FRACTION_STATIC="${KV_CACHE_MEM_FRACTION_STATIC:-0.75}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16400}"

ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-60}"
ROLLOUT_HEALTH_CHECK_INTERVAL="${ROLLOUT_HEALTH_CHECK_INTERVAL:-60}"
ROLLOUT_HEALTH_CHECK_TIMEOUT="${ROLLOUT_HEALTH_CHECK_TIMEOUT:-60}"
ROLLOUT_HEALTH_CHECK_ENABLED="${ROLLOUT_HEALTH_CHECK_ENABLED:-1}"
ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD="${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD:-5}"
ROUTER_HEALTH_CHECK_ENABLED="${ROUTER_HEALTH_CHECK_ENABLED:-0}"
ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD="${ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD:-20}"
ROUTER_HEALTH_CHECK_INTERVAL_SEC="${ROUTER_HEALTH_CHECK_INTERVAL_SEC:-15}"
ROUTER_GENERATE_CHUNK_DEBUG="${ROUTER_GENERATE_CHUNK_DEBUG:-0}"

CI_FAULT_INJECTION_ENABLE="${CI_FAULT_INJECTION_ENABLE:-0}"
CI_FAULT_INJECTION_MODE="${CI_FAULT_INJECTION_MODE:-mid_generate}"
CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD="${CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD:-0}"
CI_FAULT_INJECTION_ENGINE_INDEX="${CI_FAULT_INJECTION_ENGINE_INDEX:-0}"
CI_FAULT_INJECTION_DELAY_SEC="${CI_FAULT_INJECTION_DELAY_SEC:-60}"
CI_FAULT_INJECTION_MID_DELAY_SEC="${CI_FAULT_INJECTION_MID_DELAY_SEC:-60}"
CI_FAULT_INJECTION_PROGRESS_FRACTION="${CI_FAULT_INJECTION_PROGRESS_FRACTION:-0.5}"
CI_FAULT_INJECTION_MID_FALLBACK_SEC="${CI_FAULT_INJECTION_MID_FALLBACK_SEC:-10}"

SLIME_ROUTER_GENERATE_CHECKPOINT_TOKENS="${SLIME_ROUTER_GENERATE_CHECKPOINT_TOKENS:-1024}"
SLIME_ROUTER_GENERATE_COALESCE_CHUNKS="${SLIME_ROUTER_GENERATE_COALESCE_CHUNKS:-1024}"

SHADOW_WORKER_READY_TIMEOUT_SEC="${SHADOW_WORKER_READY_TIMEOUT_SEC:-600}"
SHADOW_WORKER_STABILIZATION_SEC="${SHADOW_WORKER_STABILIZATION_SEC:-30}"

ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-0}"
CI_FAULT_INJECTION_DELAY_SEC="${CI_FAULT_INJECTION_DELAY_SEC:-20}"
SHADOW_WORKER_READY_TIMEOUT_SEC="${SHADOW_WORKER_READY_TIMEOUT_SEC:-600}"
SHADOW_WORKER_STABILIZATION_SEC="${SHADOW_WORKER_STABILIZATION_SEC:-20}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WORKDIR="${WORKDIR:-${SLIME_ROOT}/workdir/slime_fast_restart_weight_update_smoke_${RUN_ID}}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_SEED="${ROLLOUT_SEED:-1234}"


mkdir -p "${WORKDIR}"
LOG_FILE="${WORKDIR}/smoke.log"

PIDS=()
KV_SOCKET_PATHS=()

cleanup() {
    set +e
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "${pid}" >/dev/null 2>&1; then
            kill "${pid}" >/dev/null 2>&1 || true
            wait "${pid}" >/dev/null 2>&1 || true
        fi
    done
    for socket_path in "${KV_SOCKET_PATHS[@]:-}"; do
        rm -f "${socket_path}"
    done
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

for required in python3 ray; do
    if ! command -v "${required}" >/dev/null 2>&1; then
        echo "Missing required command: ${required}" >&2
        exit 1
    fi
done

if [ ! -d "${MODEL_PATH}" ]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

if [ ! -d "${MEGATRON_ROOT}" ]; then
    echo "MEGATRON_ROOT does not exist: ${MEGATRON_ROOT}" >&2
    exit 1
fi

if [ ! -f "${MODEL_ARGS_FILE}" ]; then
    echo "MODEL_ARGS_FILE does not exist: ${MODEL_ARGS_FILE}" >&2
    exit 1
fi

source "${MODEL_ARGS_FILE}"

IFS=',' read -r -a ACTOR_GPU_IDS <<< "${ACTOR_GPU_SET}"
ACTOR_GPU_COUNT="${#ACTOR_GPU_IDS[@]}"
if [ "${ACTOR_GPU_COUNT}" -ne 4 ]; then
    echo "ACTOR_GPU_SET must contain exactly 4 GPU ids for this smoke test: ${ACTOR_GPU_SET}" >&2
    exit 1
fi

IFS=';' read -r -a ROLLOUT_GPU_SETS <<< "${ROLLOUT_GPU_SET_CSV}"
ROLLOUT_ENGINE_COUNT="${#ROLLOUT_GPU_SETS[@]}"
if [ "${ROLLOUT_ENGINE_COUNT}" -ne 2 ]; then
    echo "ROLLOUT_GPU_SET_CSV must contain exactly 2 rollout GPU sets: ${ROLLOUT_GPU_SET_CSV}" >&2
    exit 1
fi

TOTAL_GPU_COUNT="${ACTOR_GPU_COUNT}"
FIRST_ROLLOUT_GPU_ID=""
KV_SOCKET_CSV_PARTS=()
ROLLOUT_TP_SIZE=""
for idx in "${!ROLLOUT_GPU_SETS[@]}"; do
    IFS=',' read -r -a ENGINE_GPU_IDS <<< "${ROLLOUT_GPU_SETS[$idx]}"
    if [ -z "${ROLLOUT_TP_SIZE}" ]; then
        ROLLOUT_TP_SIZE="${#ENGINE_GPU_IDS[@]}"
    fi
    if [ "${#ENGINE_GPU_IDS[@]}" -ne "${ROLLOUT_TP_SIZE}" ]; then
        echo "Each rollout GPU set must contain the same number of GPU ids: ${ROLLOUT_GPU_SET_CSV}" >&2
        exit 1
    fi
    # if [ "${ROLLOUT_TP_SIZE}" -ne 2 ]; then
    #     echo "This smoke test currently expects rollout TP=2, got rollout GPU set size ${ROLLOUT_TP_SIZE}" >&2
    #     exit 1
    # fi
    TOTAL_GPU_COUNT=$((TOTAL_GPU_COUNT + ${#ENGINE_GPU_IDS[@]}))
    for gpu_id in "${ENGINE_GPU_IDS[@]}"; do
        if [ -z "${FIRST_ROLLOUT_GPU_ID}" ] || [ "${gpu_id}" -lt "${FIRST_ROLLOUT_GPU_ID}" ]; then
            FIRST_ROLLOUT_GPU_ID="${gpu_id}"
        fi
    done
    KV_SOCKET_PATHS+=("/tmp/kv_cache_weight_update_smoke_${idx}.sock")
    KV_SOCKET_CSV_PARTS+=("${KV_SOCKET_PATHS[$idx]}")
done

KV_CACHE_SOCKETS="$(IFS=','; echo "${KV_SOCKET_CSV_PARTS[*]}")"

echo "Workdir: ${WORKDIR}"
echo "Model: ${MODEL_PATH}"
echo "Actor GPUs: ${ACTOR_GPU_SET}"
echo "Rollout GPU sets: ${ROLLOUT_GPU_SET_CSV}"
echo "Topology: actor 1x4GPU + rollout ${ROLLOUT_ENGINE_COUNT}x${ROLLOUT_TP_SIZE}GPU (TP=${ROLLOUT_TP_SIZE})"
echo "KV cache mem fraction: ${KV_CACHE_MEM_FRACTION_STATIC}"
echo "Max tokens per GPU: ${MAX_TOKENS_PER_GPU}"
echo "Num rollouts: ${NUM_ROLLOUTS}"
echo "CI fault injection delay: ${CI_FAULT_INJECTION_DELAY_SEC}"
echo "Shadow worker ready timeout: ${SHADOW_WORKER_READY_TIMEOUT_SEC}"
echo "Shadow worker stabilization: ${SHADOW_WORKER_STABILIZATION_SEC}"
echo "Log: ${LOG_FILE}"

ray stop --force >/dev/null 2>&1 || true

launch_parameter_server() {
    local gpu_set="$1"
    local tp_size="$2"
    local port="$3"
    local log_path="$4"
    echo "Starting parameter server on GPUs ${gpu_set}, port ${port}"
    CUDA_VISIBLE_DEVICES="${gpu_set}" \
    PYTHONPATH="${REPO_ROOT}/checkpoint-engine" \
    stdbuf -oL -eL python3 "${REPO_ROOT}/checkpoint-engine/examples/persistent_ps_example.py" \
        --server-ckpts "${MODEL_PATH}" "${tp_size}" "${port}" >"${log_path}" 2>&1 &
    PIDS+=("$!")
}

launch_kv_server() {
    local gpu_set="$1"
    local socket_path="$2"
    local log_path="$3"
    echo "Starting KV cache server on GPUs ${gpu_set}, socket ${socket_path}"
    PYTHONPATH="${REPO_ROOT}/sglang/python" \
    stdbuf -oL -eL python3 -m sglang.srt.mem_cache.kv_cache_server \
        --socket-path "${socket_path}" \
        --gpu-id "${gpu_set}" \
        --model-path "${MODEL_PATH}" \
        --mem-fraction-static "${KV_CACHE_MEM_FRACTION_STATIC}" \
        --page-size 1 \
        --dtype bfloat16 >"${log_path}" 2>&1 &
    PIDS+=("$!")
}

for idx in "${!ROLLOUT_GPU_SETS[@]}"; do
    launch_parameter_server \
        "${ROLLOUT_GPU_SETS[$idx]}" \
        "${ROLLOUT_TP_SIZE}" \
        "$((WEIGHT_SERVER_BASE_PORT + idx))" \
        "${WORKDIR}/parameter_server_${idx}.log"
    launch_kv_server \
        "${ROLLOUT_GPU_SETS[$idx]}" \
        "${KV_SOCKET_PATHS[$idx]}" \
        "${WORKDIR}/kv_cache_${idx}.log"
done

sleep 20

echo "Starting Ray head"
ray start --head \
    --node-ip-address "${RAY_HEAD_IP}" \
    --num-gpus "${TOTAL_GPU_COUNT}" \
    --disable-usage-stats \
    --dashboard-port "${RAY_DASHBOARD_PORT}" >/dev/null

cd "${SLIME_ROOT}"

RUNTIME_ENV_JSON="$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${SLIME_ROOT}/examples/fully_async:${SLIME_ROOT}:${MEGATRON_ROOT}:${REPO_ROOT}/sglang/python",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "SGLANG_KV_CACHE_SOCKET_PATH": "${KV_CACHE_SOCKETS}",
    "WEIGHT_SERVER_BASE_PORT": "${WEIGHT_SERVER_BASE_PORT}",
    "SGLANG_MIN_GPU_ID": "${FIRST_GPU_ID}",
    "SLIME_CI_FAULT_INJECTION_DELAY_SEC": "${CI_FAULT_INJECTION_DELAY_SEC}",
    "SLIME_CI_FAULT_INJECTION_MODE": "${CI_FAULT_INJECTION_MODE}",
    "SLIME_CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD": "${CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD}",
    "SLIME_CI_FAULT_INJECTION_PROGRESS_FRACTION": "${CI_FAULT_INJECTION_PROGRESS_FRACTION}",
    "SLIME_CI_FAULT_INJECTION_MID_DELAY_SEC": "${CI_FAULT_INJECTION_MID_DELAY_SEC}",
    "SLIME_CI_FAULT_INJECTION_ENGINE_INDEX": "${CI_FAULT_INJECTION_ENGINE_INDEX}",
    "SLIME_CI_FAULT_INJECTION_MID_FALLBACK_SEC": "${CI_FAULT_INJECTION_MID_FALLBACK_SEC}",
    "SLIME_CI_FAULT_INJECTION_LOCK_PATH": "/tmp/slime_ci_fault_injection_once_${RUN_ID}.lock",
    "SLIME_ROLLOUT_ENABLE_HEALTH_CHECK": "${ROLLOUT_HEALTH_CHECK_ENABLED}",
    "SLIME_ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD": "${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD}",
    "SLIME_ROUTER_ENABLE_HEALTH_CHECK": "${ROUTER_HEALTH_CHECK_ENABLED}",
    "SLIME_ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD": "${ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD}",
    "SLIME_ROUTER_HEALTH_CHECK_INTERVAL_SEC": "${ROUTER_HEALTH_CHECK_INTERVAL_SEC}",
    "SLIME_ROUTER_GENERATE_CHUNK_DEBUG": "${ROUTER_GENERATE_CHUNK_DEBUG}",
    "PYTHONUNBUFFERED": "1"
  }
}
EOF
)"
# --ci-test inject fault
echo "Submitting async train smoke job"
pwd

#    --verify-rollout-weight-update \
# --verify-rollout-weight-update-num-params 32 \
# --verify-rollout-weight-update-num-values 256 \
# --verify-rollout-weight-update-truncate-size 256 \

set +e
stdbuf -oL -eL ray job submit --address="http://${RAY_HEAD_IP}:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 ${SLIME_ROOT}/train_async.py \
    --hf-checkpoint "${MODEL_PATH}" \
    --ref-load "${MODEL_PATH}" \
    --prompt-data "${PROMPT_DATA}" \
    --input-key prompt \
    --label-key label \
    --apply-chat-template \
    --rm-type deepscaler \
    --reward-key score \
    --num-rollout "${NUM_ROLLOUTS}" \
    --rollout-function-path fully_async_rollout.generate_rollout_fully_async \
    --rollout-shuffle \
    --rollout-seed "${ROLLOUT_SEED}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
    --rollout-temperature "${ROLLOUT_TEMPERATURE}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${ACTOR_GPU_COUNT}" \
    --rollout-num-gpus "$((ROLLOUT_ENGINE_COUNT * ROLLOUT_TP_SIZE))" \
    --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE}" \
    --tensor-model-parallel-size 4 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --megatron-to-hf-mode bridge \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --use-slime-router \
    --use-fault-tolerance \
    --rollout-health-check-interval "${ROLLOUT_HEALTH_CHECK_INTERVAL}" \
    --rollout-health-check-timeout "${ROLLOUT_HEALTH_CHECK_TIMEOUT}" \
    --rollout-health-check-first-wait "${ROLLOUT_HEALTH_CHECK_FIRST_WAIT}" \
    --sglang-enable-fast-restart \
    --sglang-shadow-worker-kv-cache-socket-path "${KV_CACHE_SOCKETS}" \
    --sglang-shadow-worker-weight-server-base-port "${WEIGHT_SERVER_BASE_PORT}" \
    --sglang-shadow-worker-min-gpu-id "${FIRST_ROLLOUT_GPU_ID}" \
    --sglang-shadow-worker-ready-timeout-seconds "${SHADOW_WORKER_READY_TIMEOUT_SEC}" \
    --sglang-shadow-worker-stabilization-seconds "${SHADOW_WORKER_STABILIZATION_SEC}" \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash \
    "${MODEL_ARGS[@]}" 2>&1 | stdbuf -oL -eL tee "${LOG_FILE}"
JOB_RC="${PIPESTATUS[0]}"
set -e

if [ "${JOB_RC}" -ne 0 ]; then
    echo "Smoke job failed with exit code ${JOB_RC}. See ${LOG_FILE}" >&2
    exit "${JOB_RC}"
fi

if ! grep -q "Shadow-worker handover succeeded" "${LOG_FILE}"; then
    echo "Did not observe shadow-worker handover in ${LOG_FILE}" >&2
    exit 2
fi

if ! grep -q "Promoted shadow worker" "${LOG_FILE}"; then
    echo "Did not observe shadow-worker promotion in ${LOG_FILE}" >&2
    exit 3
fi

if ! grep -q "weight-update reconnection" "${LOG_FILE}"; then
    echo "Did not observe a post-handover weight-update reconnection marker in ${LOG_FILE}" >&2
    exit 4
fi

if ! grep -q "Verified rollout engine" "${LOG_FILE}"; then
    echo "Did not observe rollout weight verification logs in ${LOG_FILE}" >&2
    exit 5
fi

if ! grep -q "Rebuilding weight-update connections" "${LOG_FILE}"; then
    echo "Did not observe weight-update connection rebuild in ${LOG_FILE}" >&2
    exit 5
fi

echo "FAST RESTART + WEIGHT UPDATE SMOKE TEST PASSED"
echo "Observed shadow-worker handover and subsequent weight-update connection rebuild."
exit 0
