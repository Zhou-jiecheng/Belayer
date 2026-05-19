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
RAY_HEAD_IP="${RAY_HEAD_IP:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16400}"

# Enable normal fault tolerance only: kill one rollout worker mid-generation,
# recover it without shadow workers, then rebuild weight-update links before
# the next update_weights.

ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-60}"
ROLLOUT_HEALTH_CHECK_INTERVAL="${ROLLOUT_HEALTH_CHECK_INTERVAL:-60}"
ROLLOUT_HEALTH_CHECK_TIMEOUT="${ROLLOUT_HEALTH_CHECK_TIMEOUT:-60}"
ROLLOUT_HEALTH_CHECK_ENABLED="${ROLLOUT_HEALTH_CHECK_ENABLED:-1}"
ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD="${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD:-5}"
ROUTER_HEALTH_CHECK_ENABLED="${ROUTER_HEALTH_CHECK_ENABLED:-0}"
ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD="${ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD:-20}"
ROUTER_HEALTH_CHECK_INTERVAL_SEC="${ROUTER_HEALTH_CHECK_INTERVAL_SEC:-15}"
ROUTER_GENERATE_CHUNK_DEBUG="${ROUTER_GENERATE_CHUNK_DEBUG:-0}"
SLIME_ROUTER_GENERATE_PATH="generate_nonstream"

SLIME_ROUTER_GENERATE_CHECKPOINT_TOKENS="${SLIME_ROUTER_GENERATE_CHECKPOINT_TOKENS:-256}"
SLIME_ROUTER_GENERATE_COALESCE_CHUNKS="${SLIME_ROUTER_GENERATE_COALESCE_CHUNKS:-256}"

CI_FAULT_INJECTION_ENABLE="${CI_FAULT_INJECTION_ENABLE:-0}"
CI_FAULT_INJECTION_MODE="${CI_FAULT_INJECTION_MODE:-mid_generate}"
CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD="${CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD:-0}"
CI_FAULT_INJECTION_ENGINE_INDEX="${CI_FAULT_INJECTION_ENGINE_INDEX:-0}"
CI_FAULT_INJECTION_DELAY_SEC="${CI_FAULT_INJECTION_DELAY_SEC:-60}"
CI_FAULT_INJECTION_MID_DELAY_SEC="${CI_FAULT_INJECTION_MID_DELAY_SEC:-60}"
CI_FAULT_INJECTION_PROGRESS_FRACTION="${CI_FAULT_INJECTION_PROGRESS_FRACTION:-0.5}"
CI_FAULT_INJECTION_MID_FALLBACK_SEC="${CI_FAULT_INJECTION_MID_FALLBACK_SEC:-10}"

NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_SEED="${ROLLOUT_SEED:-1234}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WORKDIR="${WORKDIR:-${SLIME_ROOT}/workdir/slime_no_fast_restart_weight_update_smoke_${RUN_ID}}"

mkdir -p "${WORKDIR}"
LOG_FILE="${WORKDIR}/smoke.log"

cleanup() {
    set +e
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

source "${MODEL_ARGS_FILE}"

IFS=',' read -r -a ACTOR_GPU_IDS <<< "${ACTOR_GPU_SET}"
ACTOR_GPU_COUNT="${#ACTOR_GPU_IDS[@]}"
if [ "${ACTOR_GPU_COUNT}" -ne 4 ]; then
    echo "ACTOR_GPU_SET must contain exactly 4 GPU ids for this smoke test: ${ACTOR_GPU_SET}" >&2
    exit 1
fi

IFS=';' read -r -a ROLLOUT_GPU_SETS <<< "${ROLLOUT_GPU_SET_CSV}"
ROLLOUT_ENGINE_COUNT="${#ROLLOUT_GPU_SETS[@]}"


TOTAL_GPU_COUNT="${ACTOR_GPU_COUNT}"
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
    TOTAL_GPU_COUNT=$((TOTAL_GPU_COUNT + ${#ENGINE_GPU_IDS[@]}))
done

echo "Workdir: ${WORKDIR}"
echo "Model: ${MODEL_PATH}"
echo "Actor GPUs: ${ACTOR_GPU_SET}"
echo "Rollout GPU sets: ${ROLLOUT_GPU_SET_CSV}"
echo "Topology: actor 1x4GPU + rollout ${ROLLOUT_ENGINE_COUNT}x${ROLLOUT_TP_SIZE}GPU (TP=${ROLLOUT_TP_SIZE})"
echo "Max tokens per GPU: ${MAX_TOKENS_PER_GPU}"
echo "Num rollouts: ${NUM_ROLLOUTS}"
echo "Rollout max response len: ${ROLLOUT_MAX_RESPONSE_LEN}"
echo "Rollout health check enabled: ${ROLLOUT_HEALTH_CHECK_ENABLED}"
echo "Rollout health check interval/timeout: ${ROLLOUT_HEALTH_CHECK_INTERVAL}s / ${ROLLOUT_HEALTH_CHECK_TIMEOUT}s"
echo "Rollout health check failure threshold: ${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD}"
echo "CI fault injection enabled: ${CI_FAULT_INJECTION_ENABLE}"
echo "CI fault injection mode: ${CI_FAULT_INJECTION_MODE}"
echo "CI fault injection rollout threshold: ${CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD}"
echo "CI fault injection engine index: ${CI_FAULT_INJECTION_ENGINE_INDEX}"
echo "CI fault injection pre/mid delay: ${CI_FAULT_INJECTION_DELAY_SEC}s / ${CI_FAULT_INJECTION_MID_DELAY_SEC}s"
echo "Log: ${LOG_FILE}"

ray stop --force >/dev/null 2>&1 || true

echo "Starting Ray head"
ray start --head \
    --node-ip-address "${RAY_HEAD_IP}" \
    --num-gpus "${TOTAL_GPU_COUNT}" \
    --disable-usage-stats \
    --dashboard-host 0.0.0.0 \
    --dashboard-port "${RAY_DASHBOARD_PORT}" >/dev/null

cd "${SLIME_ROOT}"

RUNTIME_ENV_JSON="$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${SLIME_ROOT}/examples/fully_async:${SLIME_ROOT}:${MEGATRON_ROOT}:${REPO_ROOT}/sglang/python",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "SLIME_CI_FAULT_INJECTION_MODE": "${CI_FAULT_INJECTION_MODE}",
    "SLIME_CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD": "${CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD}",
    "SLIME_CI_FAULT_INJECTION_ENGINE_INDEX": "${CI_FAULT_INJECTION_ENGINE_INDEX}",
    "SLIME_CI_FAULT_INJECTION_DELAY_SEC": "${CI_FAULT_INJECTION_DELAY_SEC}",
    "SLIME_CI_FAULT_INJECTION_MID_DELAY_SEC": "${CI_FAULT_INJECTION_MID_DELAY_SEC}",
    "SLIME_CI_FAULT_INJECTION_PROGRESS_FRACTION": "${CI_FAULT_INJECTION_PROGRESS_FRACTION}",
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

CI_TEST_ARGS=()
if [ "${CI_FAULT_INJECTION_ENABLE}" = "1" ]; then
    CI_TEST_ARGS+=(--ci-test)
fi

echo "Submitting async train smoke job (no fast restart, with weight-update reconnect)"
set +e
stdbuf -oL -eL ray job submit --address="http://${RAY_HEAD_IP}:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 ${SLIME_ROOT}/train_async.py \
    --hf-checkpoint "${MODEL_PATH}" \
    --ref-load "${MODEL_PATH}" \
    --prompt-data "${PROMPT_DATA}" \
    --input-key prompt \
    --label-key label \
    --rollout-function-path fully_async_rollout.generate_rollout_fully_async \
    --apply-chat-template \
    --rm-type deepscaler \
    --reward-key score \
    --num-rollout "${NUM_ROLLOUTS}" \
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

if [ "${JOB_RC}" -ne 0 ]; then
    echo "Smoke job failed with exit code ${JOB_RC}. See ${LOG_FILE}" >&2
    exit "${JOB_RC}"
fi

if [ "${CI_FAULT_INJECTION_ENABLE}" = "1" ]; then
    if ! grep -q "CI Fault Injection: Simulating crash" "${LOG_FILE}"; then
        echo "Expected CI fault injection crash marker was not found: ${LOG_FILE}" >&2
        exit 2
    fi
    if ! grep -q "Immediate non-fast-restart recovery completed" "${LOG_FILE}"; then
        echo "Expected non-fast-restart recovery marker was not found: ${LOG_FILE}" >&2
        exit 3
    fi
    if ! grep -q "Rebuilding weight-update connections" "${LOG_FILE}"; then
        echo "Expected weight-update connection rebuild marker was not found: ${LOG_FILE}" >&2
        exit 4
    fi
    if ! grep -q "Weight-update reconnect finished" "${LOG_FILE}"; then
        echo "Expected weight-update reconnect timing marker was not found: ${LOG_FILE}" >&2
        exit 5
    fi
fi

if ! grep -q "Timer train start" "${LOG_FILE}"; then
    echo "Expected training phase marker was not found: ${LOG_FILE}" >&2
    exit 6
fi

echo "NO FAST RESTART + WEIGHT UPDATE SMOKE TEST PASSED"
echo "Observed rollout crash, immediate normal recovery, and subsequent weight-update connection rebuild."
exit 0
