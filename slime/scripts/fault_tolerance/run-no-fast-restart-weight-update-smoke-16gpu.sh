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

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SLIME_ROOT}/.." &>/dev/null && pwd)"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
export PYTHONUNBUFFERED=1

if [ -z "${MASTER_ADDR:-}" ]; then
    echo "MASTER_ADDR is required for multi-node startup." >&2
    exit 1
fi

CURRENT_NODE_IP="$(hostname --ip-address | awk '{print $1}')"
IS_HEAD_NODE=0
if [ "${CURRENT_NODE_IP}" = "${MASTER_ADDR}" ]; then
    IS_HEAD_NODE=1
fi

MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${SCRIPT_DIR}/../models/qwen3-8B.sh}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/models/Qwen3-8B}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/data/dapo-math-17k.jsonl}"

ACTOR_GPU_SET="${ACTOR_GPU_SET:-0,1,2,3,4,5,6,7}"
ROLLOUT_GPU_SET_CSV="${ROLLOUT_GPU_SET_CSV:-0,1;2,3;4,5;6,7}"
ACTOR_NODE_COUNT="${ACTOR_NODE_COUNT:-1}"
ROLLOUT_NODE_COUNT="${ROLLOUT_NODE_COUNT:-1}"
TOTAL_NODE_COUNT="$((ACTOR_NODE_COUNT + ROLLOUT_NODE_COUNT))"

RAY_HEAD_IP="${RAY_HEAD_IP:-${MASTER_ADDR}}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-9991}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_DASHBOARD_URL="${RAY_DASHBOARD_URL:-http://${RAY_HEAD_IP}:${RAY_DASHBOARD_PORT}}"
RAY_WORKER_JOIN_WAIT_SEC="${RAY_WORKER_JOIN_WAIT_SEC:-20}"

MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16400}"

SLIME_ROUTER_GENERATE_CHECKPOINT_TOKENS="${SLIME_ROUTER_GENERATE_CHECKPOINT_TOKENS:-1024}"
SLIME_ROUTER_GENERATE_COALESCE_CHUNKS="${SLIME_ROUTER_GENERATE_COALESCE_CHUNKS:-1024}"

ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-0}"
ROLLOUT_HEALTH_CHECK_INTERVAL="${ROLLOUT_HEALTH_CHECK_INTERVAL:-5}"
ROLLOUT_HEALTH_CHECK_TIMEOUT="${ROLLOUT_HEALTH_CHECK_TIMEOUT:-10}"
ROLLOUT_HEALTH_CHECK_ENABLED="${ROLLOUT_HEALTH_CHECK_ENABLED:-1}"
ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD="${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD:-3}"

ROUTER_HEALTH_CHECK_ENABLED="${ROUTER_HEALTH_CHECK_ENABLED:-0}"
ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD="${ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD:-20}"
ROUTER_HEALTH_CHECK_INTERVAL_SEC="${ROUTER_HEALTH_CHECK_INTERVAL_SEC:-15}"
ROUTER_GENERATE_CHUNK_DEBUG="${ROUTER_GENERATE_CHUNK_DEBUG:-0}"

CI_FAULT_INJECTION_ENABLE="${CI_FAULT_INJECTION_ENABLE:-1}"
CI_FAULT_INJECTION_DELAY_SEC="${CI_FAULT_INJECTION_DELAY_SEC:-20}"
CI_FAULT_INJECTION_MODE="${CI_FAULT_INJECTION_MODE:-mid_generate}"
CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD="${CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD:-0}"
CI_FAULT_INJECTION_PROGRESS_FRACTION="${CI_FAULT_INJECTION_PROGRESS_FRACTION:-0.3}"
CI_FAULT_INJECTION_MID_DELAY_SEC="${CI_FAULT_INJECTION_MID_DELAY_SEC:-0}"
CI_FAULT_INJECTION_ENGINE_INDEX="${CI_FAULT_INJECTION_ENGINE_INDEX:-0,1}"
CI_FAULT_INJECTION_MID_FALLBACK_SEC="${CI_FAULT_INJECTION_MID_FALLBACK_SEC:-80}"

NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
COORD_JOB_KEY="${JOB_ID:-manual}"
COORD_ROOT="${SLIME_ROOT}/workdir/.fault_tolerance_coord_${MASTER_ADDR//./_}_${COORD_JOB_KEY}"
RUN_ID="${RUN_ID:-}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_SEED="${ROLLOUT_SEED:-1234}"

cleanup() {
    local exit_code="$1"
    trap - EXIT
    set +e
    if [ "${IS_HEAD_NODE}" -eq 1 ] && [ -n "${DONE_FILE:-}" ]; then
        printf '%s\n' "${exit_code}" > "${DONE_FILE}"
    fi
    ray stop --force >/dev/null 2>&1 || true
    exit "${exit_code}"
}
trap 'cleanup $?' EXIT

resolve_run_id() {
    mkdir -p "${COORD_ROOT}"
    local run_id_file="${COORD_ROOT}/run_id"
    if [ "${IS_HEAD_NODE}" -eq 1 ]; then
        if [ -z "${RUN_ID}" ]; then
            RUN_ID="$(date +%Y%m%d_%H%M%S)"
        fi
        printf '%s\n' "${RUN_ID}" > "${run_id_file}"
    elif [ -z "${RUN_ID}" ]; then
        while true; do
            if [ -f "${run_id_file}" ]; then
                RUN_ID="$(cat "${run_id_file}")"
                if [ -n "${RUN_ID}" ]; then
                    break
                fi
            fi
            sleep 1
        done
    fi
}

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

resolve_run_id

WORKDIR="${WORKDIR:-${SLIME_ROOT}/workdir/slime_no_fast_restart_weight_update_smoke_16gpu_${RUN_ID}}"
LOG_FILE="${WORKDIR}/smoke.log"
DONE_FILE="${WORKDIR}/head_done.exitcode"

mkdir -p "${WORKDIR}"

IFS=',' read -r -a ACTOR_GPU_IDS <<< "${ACTOR_GPU_SET}"
ACTOR_GPU_COUNT="${#ACTOR_GPU_IDS[@]}"
if [ "${ACTOR_GPU_COUNT}" -ne 8 ]; then
    echo "ACTOR_GPU_SET must contain exactly 8 GPU ids for this smoke test: ${ACTOR_GPU_SET}" >&2
    exit 1
fi

IFS=';' read -r -a ROLLOUT_GPU_SETS <<< "${ROLLOUT_GPU_SET_CSV}"
ROLLOUT_ENGINE_COUNT_PER_NODE="${#ROLLOUT_GPU_SETS[@]}"
# if [ "${ROLLOUT_ENGINE_COUNT_PER_NODE}" -ne 2 ]; then
#     echo "ROLLOUT_GPU_SET_CSV must contain exactly 2 rollout GPU sets per rollout node: ${ROLLOUT_GPU_SET_CSV}" >&2
#     exit 1
# fi

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
done

ROLLOUT_GPU_COUNT_PER_NODE="$((ROLLOUT_ENGINE_COUNT_PER_NODE * ROLLOUT_TP_SIZE))"
ROLLOUT_TOTAL_GPU_COUNT="$((ROLLOUT_NODE_COUNT * ROLLOUT_GPU_COUNT_PER_NODE))"
ROLLOUT_TOTAL_ENGINE_COUNT="$((ROLLOUT_NODE_COUNT * ROLLOUT_ENGINE_COUNT_PER_NODE))"

wait_for_cluster_nodes() {
    while true; do
        local alive_count
        alive_count="$(
            RAY_HEAD_IP="${RAY_HEAD_IP}" \
            RAY_HEAD_PORT="${RAY_HEAD_PORT}" \
            python3 - <<'PY'
import os
import ray

head = os.environ["RAY_HEAD_IP"]
port = os.environ["RAY_HEAD_PORT"]
ray.init(address=f"{head}:{port}", ignore_reinit_error=True, logging_level=40)
alive = sorted({n["NodeManagerAddress"] for n in ray.nodes() if n.get("Alive")}, key=lambda ip: tuple(map(int, ip.split("."))))
print(len(alive))
PY
        )"
        if [ "${alive_count}" -eq "${TOTAL_NODE_COUNT}" ]; then
            break
        fi
        if [ "${alive_count}" -gt "${TOTAL_NODE_COUNT}" ]; then
            echo "Expected ${TOTAL_NODE_COUNT} Ray nodes, but found ${alive_count}. Refusing to continue." >&2
            exit 1
        fi
        sleep 2
    done
}

wait_for_dashboard_ready() {
    while true; do
        if python3 - "${RAY_DASHBOARD_URL}/api/version" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=2) as resp:
        if resp.status == 200:
            raise SystemExit(0)
except Exception:
    pass
raise SystemExit(1)
PY
        then
            break
        fi
        sleep 2
    done
}

wait_for_head_completion() {
    echo "Waiting for head completion marker: ${DONE_FILE}"
    while [ ! -f "${DONE_FILE}" ]; do
        sleep 2
    done
    local head_exit_code
    head_exit_code="$(tr -d '[:space:]' < "${DONE_FILE}")"
    if [ -z "${head_exit_code}" ]; then
        head_exit_code=0
    fi
    echo "Head node finished with exit code ${head_exit_code}"
    exit "${head_exit_code}"
}

start_head_and_submit_job() {
    echo "Head node: ${CURRENT_NODE_IP}"
    echo "Ray head: ${RAY_HEAD_IP}:${RAY_HEAD_PORT}"
    echo "Workdir: ${WORKDIR}"
    echo "Model: ${MODEL_PATH}"
    echo "Actor GPUs per node: ${ACTOR_GPU_SET}"
    echo "Rollout GPU sets per rollout node: ${ROLLOUT_GPU_SET_CSV}"
    echo "Topology: actor ${ACTOR_NODE_COUNT}x${ACTOR_GPU_COUNT}GPU + rollout ${ROLLOUT_TOTAL_ENGINE_COUNT}x${ROLLOUT_TP_SIZE}GPU across ${ROLLOUT_NODE_COUNT} rollout node(s)"
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
        --port "${RAY_HEAD_PORT}" \
        --num-gpus "${ACTOR_GPU_COUNT}" \
        --disable-usage-stats \
        --dashboard-host 0.0.0.0 \
        --dashboard-port "${RAY_DASHBOARD_PORT}" >/dev/null

    echo "Waiting ${RAY_WORKER_JOIN_WAIT_SEC}s for worker nodes to join Ray..."
    sleep "${RAY_WORKER_JOIN_WAIT_SEC}"
    wait_for_cluster_nodes
    wait_for_dashboard_ready

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
    "SLIME_ROUTER_ENABLE_HEALTH_CHECK": "${ROUTER_HEALTH_CHECK_ENABLED}",
    "SLIME_ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD": "${ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD}",
    "SLIME_ROUTER_HEALTH_CHECK_INTERVAL_SEC": "${ROUTER_HEALTH_CHECK_INTERVAL_SEC}",
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
        -- python3 "${SLIME_ROOT}/train_async.py" \
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
        --actor-num-nodes "${ACTOR_NODE_COUNT}" \
        --actor-num-gpus-per-node "${ACTOR_GPU_COUNT}" \
        --rollout-num-gpus "${ROLLOUT_TOTAL_GPU_COUNT}" \
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
        --sglang-mem-fraction-static 0.7 \
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

    echo "NO FAST RESTART + WEIGHT UPDATE 16-GPU SMOKE TEST PASSED"
    echo "Observed rollout crash, immediate normal recovery, and subsequent weight-update connection rebuild."
}

start_worker_node() {
    echo "Non-head node: ${CURRENT_NODE_IP}"
    echo "MASTER_ADDR: ${MASTER_ADDR}"

    ray stop --force >/dev/null 2>&1 || true
    echo "Starting Ray worker"
    until ray start \
        --address="${MASTER_ADDR}:${RAY_HEAD_PORT}" \
        --node-ip-address "${CURRENT_NODE_IP}" \
        --num-gpus "${ACTOR_GPU_COUNT}" \
        --disable-usage-stats >/dev/null; do
        echo "Ray head ${MASTER_ADDR}:${RAY_HEAD_PORT} not ready yet, retrying worker join..."
        ray stop --force >/dev/null 2>&1 || true
        sleep 2
    done

    wait_for_head_completion
}

if [ "${IS_HEAD_NODE}" -eq 1 ]; then
    start_head_and_submit_job
else
    start_worker_node
fi
