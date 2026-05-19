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
SGL_ROUTER_ROOT="${SGL_ROUTER_ROOT:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/slime/sglang/sgl-router}"
export PYTHONUNBUFFERED=1

# Keep the same 4-GPU Qwen3-8B rollout-only smoke shape, but without fast-restart or CI fault injection.
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${SCRIPT_DIR}/../models/qwen3-8B.sh}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/models/Qwen3-8B}"
GPU_SET_CSV="${GPU_SET_CSV:-0,1;2,3}"
RAY_HEAD_IP="${RAY_HEAD_IP:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-15}"
ROLLOUT_HEALTH_CHECK_INTERVAL="${ROLLOUT_HEALTH_CHECK_INTERVAL:-15}"
ROLLOUT_HEALTH_CHECK_TIMEOUT="${ROLLOUT_HEALTH_CHECK_TIMEOUT:-20}"
ROLLOUT_HEALTH_CHECK_ENABLED="${ROLLOUT_HEALTH_CHECK_ENABLED:-1}"
ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD="${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD:-4}"
SLIME_ROUTER_GENERATE_CHECKPOINT_TOKENS="${SLIME_ROUTER_GENERATE_CHECKPOINT_TOKENS:-2048}"
SLIME_ROUTER_GENERATE_COALESCE_CHUNKS="${SLIME_ROUTER_GENERATE_COALESCE_CHUNKS:-2048}"
# ROUTER_HEALTH_CHECK_ENABLED="${ROUTER_HEALTH_CHECK_ENABLED:-0}"
# ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD="${ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD:-10}"
# ROUTER_HEALTH_CHECK_INTERVAL_SEC="${ROUTER_HEALTH_CHECK_INTERVAL_SEC:-15}"
ROUTER_GENERATE_CHUNK_DEBUG="${ROUTER_GENERATE_CHUNK_DEBUG:-0}"
ROUTER_GENERATE_PATH="${ROUTER_GENERATE_PATH:-/generate}"
CI_FAULT_INJECTION_ENABLE="${CI_FAULT_INJECTION_ENABLE:-1}"
CI_FAULT_INJECTION_MODE="${CI_FAULT_INJECTION_MODE:-mid_generate}"
CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD="${CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD:-0}"
CI_FAULT_INJECTION_ENGINE_INDEX="${CI_FAULT_INJECTION_ENGINE_INDEX:-0}"
CI_FAULT_INJECTION_DELAY_SEC="${CI_FAULT_INJECTION_DELAY_SEC:-60}"
CI_FAULT_INJECTION_MID_DELAY_SEC="${CI_FAULT_INJECTION_MID_DELAY_SEC:-60}"
CI_FAULT_INJECTION_PROGRESS_FRACTION="${CI_FAULT_INJECTION_PROGRESS_FRACTION:-0.5}"
CI_FAULT_INJECTION_MID_FALLBACK_SEC="${CI_FAULT_INJECTION_MID_FALLBACK_SEC:-10}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WORKDIR="${WORKDIR:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/slime/workdir/slime_no_fast_restart_smoke_${RUN_ID}}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/data/dapo-math-17k.jsonl}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
GPU_MONITOR_INTERVAL_SEC="${GPU_MONITOR_INTERVAL_SEC:-10}"

mkdir -p "${WORKDIR}"
LOG_FILE="${WORKDIR}/smoke.log"
GPU_MONITOR_SCRIPT="${SLIME_ROOT}/scripts/monitor_gpu_sm_activity.py"
GPU_MONITOR_PID=""

cleanup() {
    set +e
    if [ -n "${GPU_MONITOR_PID}" ] && kill -0 "${GPU_MONITOR_PID}" >/dev/null 2>&1; then
        kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
        wait "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    fi
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

if [ ! -d "${SGL_ROUTER_ROOT}/py_src" ]; then
    echo "SGL_ROUTER_ROOT/py_src does not exist: ${SGL_ROUTER_ROOT}/py_src" >&2
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
ALL_GPU_IDS=()
for idx in "${!GPU_SETS[@]}"; do
    IFS=',' read -r -a ENGINE_GPU_IDS <<< "${GPU_SETS[$idx]}"
    GPU_COUNT=$((GPU_COUNT + ${#ENGINE_GPU_IDS[@]}))
    for gpu_id in "${ENGINE_GPU_IDS[@]}"; do
        ALL_GPU_IDS+=("${gpu_id}")
    done
done

MONITOR_GPU_IDS="$(printf '%s\n' "${ALL_GPU_IDS[@]}" | sort -n | uniq | paste -sd, -)"

echo "Workdir: ${WORKDIR}"
echo "Model: ${MODEL_PATH}"
echo "GPU sets: ${GPU_SET_CSV}"
echo "SGL router root: ${SGL_ROUTER_ROOT}"
echo "Rollout topology: ${ENGINE_COUNT} engines, total ${GPU_COUNT} GPUs"
echo "Max tokens per GPU: ${MAX_TOKENS_PER_GPU}"
echo "Rollout health check enabled: ${ROLLOUT_HEALTH_CHECK_ENABLED}"
echo "Rollout health check interval/timeout: ${ROLLOUT_HEALTH_CHECK_INTERVAL}s / ${ROLLOUT_HEALTH_CHECK_TIMEOUT}s"
echo "Rollout health check failure threshold: ${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD}"
# echo "Router health check enabled: ${ROUTER_HEALTH_CHECK_ENABLED}"
# echo "Router health check interval/threshold: ${ROUTER_HEALTH_CHECK_INTERVAL_SEC}s / ${ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD}"
echo "Router generate chunk debug: ${ROUTER_GENERATE_CHUNK_DEBUG}"
echo "Router generate path: ${ROUTER_GENERATE_PATH}"
echo "Rollout max response len: ${ROLLOUT_MAX_RESPONSE_LEN}"
echo "CI fault injection enabled: ${CI_FAULT_INJECTION_ENABLE}"
echo "CI fault injection mode: ${CI_FAULT_INJECTION_MODE}"
echo "CI fault injection rollout threshold: ${CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD}"
echo "CI fault injection engine index: ${CI_FAULT_INJECTION_ENGINE_INDEX}"
echo "CI fault injection pre/mid delay: ${CI_FAULT_INJECTION_DELAY_SEC}s / ${CI_FAULT_INJECTION_MID_DELAY_SEC}s"
echo "GPU monitor interval: ${GPU_MONITOR_INTERVAL_SEC}s"
echo "Log: ${LOG_FILE}"

ray stop --force >/dev/null 2>&1 || true

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
    "SLIME_ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD": "${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD}",
    "SLIME_ROUTER_GENERATE_CHUNK_DEBUG": "${ROUTER_GENERATE_CHUNK_DEBUG}",
    "SLIME_ROUTER_GENERATE_PATH": "${ROUTER_GENERATE_PATH}",
    "SLIME_CI_FAULT_INJECTION_MODE": "${CI_FAULT_INJECTION_MODE}",
    "SLIME_CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD": "${CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD}",
    "SLIME_CI_FAULT_INJECTION_ENGINE_INDEX": "${CI_FAULT_INJECTION_ENGINE_INDEX}",
    "SLIME_CI_FAULT_INJECTION_DELAY_SEC": "${CI_FAULT_INJECTION_DELAY_SEC}",
    "SLIME_CI_FAULT_INJECTION_MID_DELAY_SEC": "${CI_FAULT_INJECTION_MID_DELAY_SEC}",
    "SLIME_CI_FAULT_INJECTION_PROGRESS_FRACTION": "${CI_FAULT_INJECTION_PROGRESS_FRACTION}",
    "SLIME_CI_FAULT_INJECTION_MID_FALLBACK_SEC": "${CI_FAULT_INJECTION_MID_FALLBACK_SEC}",
    "SLIME_CI_FAULT_INJECTION_LOCK_PATH": "/tmp/slime_ci_fault_injection_once_${RUN_ID}.lock",
    "PYTHONUNBUFFERED": "1"
  }
}
EOF
)"

CI_TEST_ARGS=()
if [ "${CI_FAULT_INJECTION_ENABLE}" = "1" ]; then
    CI_TEST_ARGS+=(--ci-test)
fi

echo "Submitting rollout-only smoke job (no fast restart, no fault injection)"
echo "Starting GPU monitor for rollout GPUs: ${MONITOR_GPU_IDS}"
stdbuf -oL -eL python3 "${GPU_MONITOR_SCRIPT}" \
    --gpus "${MONITOR_GPU_IDS}" \
    --interval "${GPU_MONITOR_INTERVAL_SEC}" \
    --sm-only-one-line &
GPU_MONITOR_PID="$!"
# --use-slime-router \
set +e
stdbuf -oL -eL ray job submit --address="http://${RAY_HEAD_IP}:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train_async.py \
    --debug-rollout-only \
    --hf-checkpoint "${MODEL_PATH}" \
    --ref-load "${MODEL_PATH}" \
    --prompt-data "${PROMPT_DATA}" \
    --input-key prompt \
    --rollout-function-path fully_async_rollout.generate_rollout_fully_async \
    --label-key label \
    --apply-chat-template \
    --rollout-shuffle \
    --rollout-seed "${ROLLOUT_SEED}" \
    --rm-type dapo \
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
    --use-slime-router \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --use-fault-tolerance \
    "${CI_TEST_ARGS[@]}" \
    --rollout-health-check-interval "${ROLLOUT_HEALTH_CHECK_INTERVAL}" \
    --rollout-health-check-timeout "${ROLLOUT_HEALTH_CHECK_TIMEOUT}" \
    --rollout-health-check-first-wait "${ROLLOUT_HEALTH_CHECK_FIRST_WAIT}" \
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

if [ "${CI_FAULT_INJECTION_ENABLE}" = "1" ]; then
    if ! grep -q "CI Fault Injection: Simulating crash" "${LOG_FILE}"; then
        echo "Expected CI fault injection crash marker was not found: ${LOG_FILE}" >&2
        exit 2
    fi
    if ! grep -q "Recovered .* dead rollout engines" "${LOG_FILE}"; then
        echo "Expected normal-restart recovery marker was not found: ${LOG_FILE}" >&2
        exit 3
    fi
    if ! grep -q "Immediate non-fast-restart recovery completed" "${LOG_FILE}"; then
        echo "Expected immediate non-fast-restart recovery marker was not found: ${LOG_FILE}" >&2
        exit 4
    fi
else
    if grep -q "CI Fault Injection: Simulating crash" "${LOG_FILE}"; then
        echo "Unexpected CI fault injection detected while disabled: ${LOG_FILE}" >&2
        exit 5
    fi
fi

echo "NO FAST RESTART SMOKE TEST PASSED"
echo "Rollout-only smoke completed successfully with ordinary-restart validation (ci_fault_injection_enable=${CI_FAULT_INJECTION_ENABLE})."
exit 0
