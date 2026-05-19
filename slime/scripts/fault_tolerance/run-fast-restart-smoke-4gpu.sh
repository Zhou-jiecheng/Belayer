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

# Default to the same 4-GPU Qwen3-8B shape used by swe-rl rollout-only runs.
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${SCRIPT_DIR}/../models/qwen3-8B.sh}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/models/Qwen3-8B}"
GPU_SET_CSV="${GPU_SET_CSV:-0,1;2,3}"
WEIGHT_SERVER_BASE_PORT="${WEIGHT_SERVER_BASE_PORT:-5556}"
RAY_HEAD_IP="${RAY_HEAD_IP:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
KV_CACHE_MEM_FRACTION_STATIC="${KV_CACHE_MEM_FRACTION_STATIC:-0.75}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-60}"
ROLLOUT_HEALTH_CHECK_INTERVAL="${ROLLOUT_HEALTH_CHECK_INTERVAL:-60}"
ROLLOUT_HEALTH_CHECK_TIMEOUT="${ROLLOUT_HEALTH_CHECK_TIMEOUT:-60}"
ROLLOUT_HEALTH_CHECK_ENABLED="${ROLLOUT_HEALTH_CHECK_ENABLED:-1}"
ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD="${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD:-5}"
# ROUTER_HEALTH_CHECK_ENABLED="${ROUTER_HEALTH_CHECK_ENABLED:-0}"
# ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD="${ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD:-20}"
# ROUTER_HEALTH_CHECK_INTERVAL_SEC="${ROUTER_HEALTH_CHECK_INTERVAL_SEC:-15}"
ROUTER_GENERATE_CHUNK_DEBUG="${ROUTER_GENERATE_CHUNK_DEBUG:-1}"
CI_FAULT_INJECTION_DELAY_SEC="${CI_FAULT_INJECTION_DELAY_SEC:-60}"
CI_FAULT_INJECTION_MODE="${CI_FAULT_INJECTION_MODE:-mid_generate}"
CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD="${CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD:-0}"
CI_FAULT_INJECTION_PROGRESS_FRACTION="${CI_FAULT_INJECTION_PROGRESS_FRACTION:-0.5}"
CI_FAULT_INJECTION_MID_DELAY_SEC="${CI_FAULT_INJECTION_MID_DELAY_SEC:-0}"
CI_FAULT_INJECTION_ENGINE_INDEX="${CI_FAULT_INJECTION_ENGINE_INDEX:-0}"
CI_FAULT_INJECTION_MID_FALLBACK_SEC="${CI_FAULT_INJECTION_MID_FALLBACK_SEC:-120}"
SHADOW_WORKER_READY_TIMEOUT_SEC="${SHADOW_WORKER_READY_TIMEOUT_SEC:-600}"
SHADOW_WORKER_STABILIZATION_SEC="${SHADOW_WORKER_STABILIZATION_SEC:-30}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WORKDIR="${WORKDIR:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/slime/workdir/slime_fast_restart_smoke_${RUN_ID}}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/aliyun_data/verl/data/dapo-math-17k.parquet}"
NUM_ROLLOUT="${NUM_ROLLOUT:-5}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.9}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
GPU_MONITOR_INTERVAL_SEC="${GPU_MONITOR_INTERVAL_SEC:-10}"

mkdir -p "${WORKDIR}"
LOG_FILE="${WORKDIR}/smoke.log"
GPU_MONITOR_SCRIPT="${SLIME_ROOT}/scripts/monitor_gpu_sm_activity.py"

PIDS=()
GPU_MONITOR_PID=""

cleanup() {
    set +e
    if [ -n "${GPU_MONITOR_PID}" ] && kill -0 "${GPU_MONITOR_PID}" >/dev/null 2>&1; then
        kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
        wait "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    fi
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

if [ ! -f "${PROMPT_DATA}" ]; then
    echo "PROMPT_DATA does not exist: ${PROMPT_DATA}" >&2
    exit 1
fi

if [ ! -f "${GPU_MONITOR_SCRIPT}" ]; then
    echo "GPU monitor script does not exist: ${GPU_MONITOR_SCRIPT}" >&2
    exit 1
fi

source "${MODEL_ARGS_FILE}"

IFS=';' read -r -a GPU_SETS <<< "${GPU_SET_CSV}"
ENGINE_COUNT="${#GPU_SETS[@]}"
if [ "${ENGINE_COUNT}" -ne 2 ]; then
    echo "GPU_SET_CSV must contain exactly 2 GPU sets for this smoke test: ${GPU_SET_CSV}" >&2
    exit 1
fi

GPU_COUNT=0
FIRST_GPU_ID=""
ALL_GPU_IDS=()
KV_SOCKET_PATHS=()
KV_SOCKET_CSV_PARTS=()
for idx in "${!GPU_SETS[@]}"; do
    IFS=',' read -r -a ENGINE_GPU_IDS <<< "${GPU_SETS[$idx]}"
    # if [ "${#ENGINE_GPU_IDS[@]}" -ne 2 ]; then
    #     echo "Each GPU set must contain exactly 2 GPU ids: ${GPU_SETS[$idx]}" >&2
    #     exit 1
    # fi
    GPU_COUNT=$((GPU_COUNT + ${#ENGINE_GPU_IDS[@]}))
    for gpu_id in "${ENGINE_GPU_IDS[@]}"; do
        ALL_GPU_IDS+=("${gpu_id}")
        if [ -z "${FIRST_GPU_ID}" ] || [ "${gpu_id}" -lt "${FIRST_GPU_ID}" ]; then
            FIRST_GPU_ID="${gpu_id}"
        fi
    done
    KV_SOCKET_PATHS+=("/tmp/kv_cache_smoke_${idx}.sock")
    KV_SOCKET_CSV_PARTS+=("${KV_SOCKET_PATHS[$idx]}")
done

KV_CACHE_SOCKETS="$(IFS=','; echo "${KV_SOCKET_CSV_PARTS[*]}")"
MONITOR_GPU_IDS="$(printf '%s\n' "${ALL_GPU_IDS[@]}" | sort -n | uniq | paste -sd, -)"

echo "Workdir: ${WORKDIR}"
echo "Model: ${MODEL_PATH}"
echo "GPU sets: ${GPU_SET_CSV}"
echo "Rollout topology: ${ENGINE_COUNT} engines, total ${GPU_COUNT} GPUs"
echo "KV cache mem fraction: ${KV_CACHE_MEM_FRACTION_STATIC}"
echo "Max tokens per GPU: ${MAX_TOKENS_PER_GPU}"
echo "CI fault injection delay: ${CI_FAULT_INJECTION_DELAY_SEC}"
echo "CI fault injection mode: ${CI_FAULT_INJECTION_MODE}"
echo "CI fault injection rollout threshold: ${CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD}"
echo "CI fault injection progress fraction: ${CI_FAULT_INJECTION_PROGRESS_FRACTION}"
echo "CI fault injection mid delay: ${CI_FAULT_INJECTION_MID_DELAY_SEC}"
echo "CI fault injection engine index: ${CI_FAULT_INJECTION_ENGINE_INDEX}"
echo "CI mid-generate fallback delay: ${CI_FAULT_INJECTION_MID_FALLBACK_SEC}"
echo "Shadow worker ready timeout: ${SHADOW_WORKER_READY_TIMEOUT_SEC}"
echo "Shadow worker stabilization: ${SHADOW_WORKER_STABILIZATION_SEC}"
echo "Rollout health check enabled: ${ROLLOUT_HEALTH_CHECK_ENABLED}"
echo "Rollout health check interval/timeout: ${ROLLOUT_HEALTH_CHECK_INTERVAL}s / ${ROLLOUT_HEALTH_CHECK_TIMEOUT}s"
echo "Rollout health check failure threshold: ${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD}"
echo "Router generate chunk debug: ${ROUTER_GENERATE_CHUNK_DEBUG}"
echo "GPU monitor interval: ${GPU_MONITOR_INTERVAL_SEC}s"
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

for idx in "${!GPU_SETS[@]}"; do
    IFS=',' read -r -a ENGINE_GPU_IDS <<< "${GPU_SETS[$idx]}"
    launch_parameter_server \
        "${GPU_SETS[$idx]}" \
        "${#ENGINE_GPU_IDS[@]}" \
        "$((WEIGHT_SERVER_BASE_PORT + idx))" \
        "${WORKDIR}/parameter_server_${idx}.log"
    launch_kv_server \
        "${GPU_SETS[$idx]}" \
        "${KV_SOCKET_PATHS[$idx]}" \
        "${WORKDIR}/kv_cache_${idx}.log"
done

sleep 20

echo "Starting Ray head"
ray start --head \
    --node-ip-address "${RAY_HEAD_IP}" \
    --num-gpus "${GPU_COUNT}" \
    --disable-usage-stats \
    --dashboard-host 0.0.0.0 \
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
    "SLIME_ROUTER_GENERATE_CHUNK_DEBUG": "${ROUTER_GENERATE_CHUNK_DEBUG}",
    "PYTHONUNBUFFERED": "1"
  }
}
EOF
)"

echo "Submitting rollout-only smoke job"
echo "Starting GPU monitor for rollout GPUs: ${MONITOR_GPU_IDS}"
stdbuf -oL -eL python3 "${GPU_MONITOR_SCRIPT}" \
    --gpus "${MONITOR_GPU_IDS}" \
    --interval "${GPU_MONITOR_INTERVAL_SEC}" \
    --sm-only-one-line &
GPU_MONITOR_PID="$!"

set +e
stdbuf -oL -eL ray job submit --address="http://${RAY_HEAD_IP}:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train_async_rollout_only.py \
    --debug-rollout-only \
    --hf-checkpoint "${MODEL_PATH}" \
    --ref-load "${MODEL_PATH}" \
    --rollout-function-path fully_async_rollout.generate_rollout_fully_async \
    --prompt-data "${PROMPT_DATA}" \
    --input-key prompt \
    --label-key label \
    --apply-chat-template \
    --rollout-shuffle \
    --rollout-seed "${ROLLOUT_SEED}" \
    --rm-type deepscaler \
    --reward-key score \
    --num-rollout "${NUM_ROLLOUT}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
    --rollout-temperature "${ROLLOUT_TEMPERATURE}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${GPU_COUNT}" \
    --rollout-num-gpus "${GPU_COUNT}" \
    --rollout-num-gpus-per-engine 2 \
    --tensor-model-parallel-size 1 \
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
    --ci-test \
    --rollout-health-check-interval "${ROLLOUT_HEALTH_CHECK_INTERVAL}" \
    --rollout-health-check-timeout "${ROLLOUT_HEALTH_CHECK_TIMEOUT}" \
    --rollout-health-check-first-wait "${ROLLOUT_HEALTH_CHECK_FIRST_WAIT}" \
    --sglang-enable-fast-restart \
    --sglang-shadow-worker-kv-cache-socket-path "${KV_CACHE_SOCKETS}" \
    --sglang-shadow-worker-weight-server-base-port "${WEIGHT_SERVER_BASE_PORT}" \
    --sglang-shadow-worker-min-gpu-id "${FIRST_GPU_ID}" \
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

if [ -n "${GPU_MONITOR_PID}" ] && kill -0 "${GPU_MONITOR_PID}" >/dev/null 2>&1; then
    kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
fi
GPU_MONITOR_PID=""

if [ "${JOB_RC}" -ne 0 ]; then
    echo "Smoke job failed with exit code ${JOB_RC}. See ${LOG_FILE}" >&2
    exit "${JOB_RC}"
fi

SHADOW_LAUNCH_COUNT="$(grep -c "Launch shadow worker" "${LOG_FILE}" || true)"
if [ "${SHADOW_LAUNCH_COUNT}" -lt "${ENGINE_COUNT}" ]; then
    echo "Expected at least ${ENGINE_COUNT} shadow-worker launch markers, found ${SHADOW_LAUNCH_COUNT} in ${LOG_FILE}" >&2
    exit 1
fi

if grep -q "Shadow-worker handover succeeded" "${LOG_FILE}" && grep -q "Promoted shadow worker" "${LOG_FILE}"; then
    echo "FAST RESTART SMOKE TEST PASSED"
    echo "Shadow workers launched for both instances and at least one handover completed successfully."
    exit 0
fi

if grep -q "Recovered .* dead rollout engines" "${LOG_FILE}"; then
    echo "Smoke job finished, but it fell back to full engine restart instead of shadow handover." >&2
    echo "See ${LOG_FILE}" >&2
    exit 2
fi

echo "Smoke job finished, but no successful shadow handover markers were found." >&2
echo "See ${LOG_FILE}" >&2
exit 1
