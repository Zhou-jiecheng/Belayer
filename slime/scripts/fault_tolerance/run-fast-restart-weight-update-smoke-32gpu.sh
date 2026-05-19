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
START_TS="$(date +%s)"

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
ROLLOUT_GPU_SET_CSV="${ROLLOUT_GPU_SET_CSV:-0,1,2,3;4,5,6,7}"
ACTOR_NODE_COUNT="${ACTOR_NODE_COUNT:-1}"
ROLLOUT_NODE_COUNT="${ROLLOUT_NODE_COUNT:-3}"
TOTAL_NODE_COUNT="$((ACTOR_NODE_COUNT + ROLLOUT_NODE_COUNT))"
FIRST_GPU_ID=0

WEIGHT_SERVER_BASE_PORT="${WEIGHT_SERVER_BASE_PORT:-5556}"
RAY_HEAD_IP="${RAY_HEAD_IP:-${MASTER_ADDR}}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-9991}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_DASHBOARD_URL="${RAY_DASHBOARD_URL:-http://${RAY_HEAD_IP}:${RAY_DASHBOARD_PORT}}"
RAY_WORKER_JOIN_WAIT_SEC="${RAY_WORKER_JOIN_WAIT_SEC:-45}"

KV_CACHE_MEM_FRACTION_STATIC="${KV_CACHE_MEM_FRACTION_STATIC:-0.7}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16400}"

SLIME_ROUTER_GENERATE_CHECKPOINT_TOKENS="${SLIME_ROUTER_GENERATE_CHECKPOINT_TOKENS:-256}"
SLIME_ROUTER_GENERATE_COALESCE_CHUNKS="${SLIME_ROUTER_GENERATE_COALESCE_CHUNKS:-256}"
SLIME_ROUTER_REROUTE_FAILED_REQUESTS_TO_HEALTHY_WORKERS="${SLIME_ROUTER_REROUTE_FAILED_REQUESTS_TO_HEALTHY_WORKERS:-0}"

ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-0}"
ROLLOUT_HEALTH_CHECK_INTERVAL="${ROLLOUT_HEALTH_CHECK_INTERVAL:-20}"
ROLLOUT_HEALTH_CHECK_TIMEOUT="${ROLLOUT_HEALTH_CHECK_TIMEOUT:-20}"
ROLLOUT_HEALTH_CHECK_ENABLED="${ROLLOUT_HEALTH_CHECK_ENABLED:-1}"
ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD="${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD:-3}"
ROUTER_HEALTH_CHECK_ENABLED="${ROUTER_HEALTH_CHECK_ENABLED:-0}"
ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD="${ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD:-20}"
ROUTER_HEALTH_CHECK_INTERVAL_SEC="${ROUTER_HEALTH_CHECK_INTERVAL_SEC:-15}"
ROUTER_GENERATE_CHUNK_DEBUG="${ROUTER_GENERATE_CHUNK_DEBUG:-0}"

CI_FAULT_INJECTION_ENABLE="${CI_FAULT_INJECTION_ENABLE:-0}"
CI_FAULT_INJECTION_DELAY_SEC="${CI_FAULT_INJECTION_DELAY_SEC:-30}" # work for per generate inject
CI_FAULT_INJECTION_MODE="${CI_FAULT_INJECTION_MODE:-mid_generate}"
CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD="${CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD:-0}"
CI_FAULT_INJECTION_PROGRESS_FRACTION="${CI_FAULT_INJECTION_PROGRESS_FRACTION:-0.8}"
CI_FAULT_INJECTION_MID_DELAY_SEC="${CI_FAULT_INJECTION_MID_DELAY_SEC:-0}"
CI_FAULT_INJECTION_ENGINE_INDEX="${CI_FAULT_INJECTION_ENGINE_INDEX:-0}"
CI_FAULT_INJECTION_MID_FALLBACK_SEC="${CI_FAULT_INJECTION_MID_FALLBACK_SEC:-10}"

SHADOW_WORKER_READY_TIMEOUT_SEC="${SHADOW_WORKER_READY_TIMEOUT_SEC:-600}"
SHADOW_WORKER_STABILIZATION_SEC="${SHADOW_WORKER_STABILIZATION_SEC:-20}"

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

TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"

PIDS=()
KV_SOCKET_PATHS=()

cleanup() {
    local exit_code="$1"
    trap - EXIT
    set +e
    if [ "${IS_HEAD_NODE}" -eq 1 ] && [ -n "${DONE_FILE:-}" ]; then
        printf '%s\n' "${exit_code}" > "${DONE_FILE}"
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

WORKDIR="${WORKDIR:-${SLIME_ROOT}/workdir/slime_fast_restart_weight_update_smoke_32gpu_${RUN_ID}}"
LOG_FILE="${WORKDIR}/smoke.log"
ROLE_MAP_FILE="${WORKDIR}/node_roles.txt"
READY_DIR="${WORKDIR}/rollout_ready"
SELF_READY_FILE="${READY_DIR}/${CURRENT_NODE_IP}.ready"
DONE_FILE="${WORKDIR}/head_done.exitcode"

mkdir -p "${WORKDIR}" "${READY_DIR}"

IFS=',' read -r -a ACTOR_GPU_IDS <<< "${ACTOR_GPU_SET}"
ACTOR_GPU_COUNT="${#ACTOR_GPU_IDS[@]}"
if [ "${ACTOR_GPU_COUNT}" -ne 8 ]; then
    echo "ACTOR_GPU_SET must contain exactly 8 GPU ids for this smoke test: ${ACTOR_GPU_SET}" >&2
    exit 1
fi

IFS=';' read -r -a ROLLOUT_GPU_SETS <<< "${ROLLOUT_GPU_SET_CSV}"
ROLLOUT_ENGINE_COUNT_PER_NODE="${#ROLLOUT_GPU_SETS[@]}"
if [ "${ROLLOUT_ENGINE_COUNT_PER_NODE}" -lt 1 ]; then
    echo "ROLLOUT_GPU_SET_CSV must contain at least 1 rollout GPU set per rollout node: ${ROLLOUT_GPU_SET_CSV}" >&2
    exit 1
fi

ROLLOUT_TP_SIZE=""
KV_SOCKET_CSV_PARTS=()
for idx in "${!ROLLOUT_GPU_SETS[@]}"; do
    IFS=',' read -r -a ENGINE_GPU_IDS <<< "${ROLLOUT_GPU_SETS[$idx]}"
    if [ -z "${ROLLOUT_TP_SIZE}" ]; then
        ROLLOUT_TP_SIZE="${#ENGINE_GPU_IDS[@]}"
    fi
    if [ "${#ENGINE_GPU_IDS[@]}" -ne "${ROLLOUT_TP_SIZE}" ]; then
        echo "Each rollout GPU set must contain the same number of GPU ids: ${ROLLOUT_GPU_SET_CSV}" >&2
        exit 1
    fi
    KV_SOCKET_PATHS+=("/tmp/kv_cache_weight_update_smoke_32gpu_${idx}.sock")
    KV_SOCKET_CSV_PARTS+=("${KV_SOCKET_PATHS[$idx]}")
done

KV_CACHE_SOCKETS="$(IFS=','; echo "${KV_SOCKET_CSV_PARTS[*]}")"
ROLLOUT_GPU_COUNT_PER_NODE="$((ROLLOUT_ENGINE_COUNT_PER_NODE * ROLLOUT_TP_SIZE))"
if [ "${ROLLOUT_GPU_COUNT_PER_NODE}" -ne 8 ]; then
    echo "Rollout GPU sets must use exactly 8 GPUs per rollout node for this 32 GPU smoke test: ${ROLLOUT_GPU_SET_CSV}" >&2
    exit 1
fi
ROLLOUT_TOTAL_GPU_COUNT="$((ROLLOUT_NODE_COUNT * ROLLOUT_GPU_COUNT_PER_NODE))"
ROLLOUT_TOTAL_ENGINE_COUNT="$((ROLLOUT_NODE_COUNT * ROLLOUT_ENGINE_COUNT_PER_NODE))"

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

start_rollout_side_services() {
    for idx in "${!ROLLOUT_GPU_SETS[@]}"; do
        launch_parameter_server \
            "${ROLLOUT_GPU_SETS[$idx]}" \
            "${ROLLOUT_TP_SIZE}" \
            "$((WEIGHT_SERVER_BASE_PORT + idx))" \
            "${WORKDIR}/parameter_server_${CURRENT_NODE_IP//./_}_${idx}.log"
        launch_kv_server \
            "${ROLLOUT_GPU_SETS[$idx]}" \
            "${KV_SOCKET_PATHS[$idx]}" \
            "${WORKDIR}/kv_cache_${CURRENT_NODE_IP//./_}_${idx}.log"
    done
    sleep 20
}

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

write_node_roles() {
    ROLE_MAP_FILE="${ROLE_MAP_FILE}" \
    RAY_HEAD_IP="${RAY_HEAD_IP}" \
    RAY_HEAD_PORT="${RAY_HEAD_PORT}" \
    TOTAL_NODE_COUNT="${TOTAL_NODE_COUNT}" \
    ACTOR_NODE_COUNT="${ACTOR_NODE_COUNT}" \
    python3 - <<'PY'
import os
import ray

role_map_file = os.environ["ROLE_MAP_FILE"]
head = os.environ["RAY_HEAD_IP"]
port = os.environ["RAY_HEAD_PORT"]
total = int(os.environ["TOTAL_NODE_COUNT"])
actor_nodes = int(os.environ["ACTOR_NODE_COUNT"])

ray.init(address=f"{head}:{port}", ignore_reinit_error=True, logging_level=40)
alive = sorted({n["NodeManagerAddress"] for n in ray.nodes() if n.get("Alive")}, key=lambda ip: tuple(map(int, ip.split("."))))
if len(alive) != total:
    raise SystemExit(f"Expected {total} alive nodes, found {len(alive)}: {alive}")

with open(role_map_file, "w", encoding="utf-8") as f:
    for idx, ip in enumerate(alive):
        role = "training" if idx < actor_nodes else "rollout"
        f.write(f"{ip} {role}\n")

print("Resolved node roles:")
for idx, ip in enumerate(alive):
    role = "training" if idx < actor_nodes else "rollout"
    print(f"  {ip} -> {role}")
PY
}

determine_self_role() {
    while [ ! -f "${ROLE_MAP_FILE}" ]; do
        sleep 1
    done
    local role
    role="$(awk -v ip="${CURRENT_NODE_IP}" '$1 == ip {print $2}' "${ROLE_MAP_FILE}")"
    if [ -z "${role}" ]; then
        echo "Current node ${CURRENT_NODE_IP} was not found in ${ROLE_MAP_FILE}" >&2
        exit 1
    fi
    printf '%s\n' "${role}"
}

mark_rollout_ready() {
    printf '%s\n' "${CURRENT_NODE_IP}" > "${SELF_READY_FILE}"
}

wait_for_rollout_readiness() {
    while true; do
        local ready_count
        ready_count="$(find "${READY_DIR}" -maxdepth 1 -type f -name '*.ready' | wc -l | tr -d ' ')"
        if [ "${ready_count}" -eq "${ROLLOUT_NODE_COUNT}" ]; then
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
    echo "Actor GPUs: ${ACTOR_GPU_SET}"
    echo "Rollout GPU sets per rollout node: ${ROLLOUT_GPU_SET_CSV}"
    echo "Topology: actor ${ACTOR_NODE_COUNT}x${ACTOR_GPU_COUNT}GPU + rollout ${ROLLOUT_TOTAL_ENGINE_COUNT}x${ROLLOUT_TP_SIZE}GPU across ${ROLLOUT_NODE_COUNT} rollout nodes"
    echo "KV cache mem fraction: ${KV_CACHE_MEM_FRACTION_STATIC}"
    echo "Max tokens per GPU: ${MAX_TOKENS_PER_GPU}"
    echo "Num rollouts: ${NUM_ROLLOUTS}"
    echo "CI fault injection delay: ${CI_FAULT_INJECTION_DELAY_SEC}"
    echo "Shadow worker ready timeout: ${SHADOW_WORKER_READY_TIMEOUT_SEC}"
    echo "Shadow worker stabilization: ${SHADOW_WORKER_STABILIZATION_SEC}"
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
    write_node_roles

    local self_role
    self_role="$(determine_self_role)"
    echo "Current node role: ${self_role}"
    if [ "${self_role}" = "rollout" ]; then
        echo "Head node participates in rollout slice; starting local rollout-side services."
        start_rollout_side_services
        mark_rollout_ready
    fi

    echo "Waiting for all rollout nodes to report ready..."
    wait_for_rollout_readiness

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
    "SLIME_ROUTER_GENERATE_CHECKPOINT_TOKENS": "${SLIME_ROUTER_GENERATE_CHECKPOINT_TOKENS}",
    "SLIME_ROUTER_GENERATE_COALESCE_CHUNKS": "${SLIME_ROUTER_GENERATE_COALESCE_CHUNKS}",
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

    echo "Submitting async train smoke job"
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
        --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
        --sequence-parallel \
        --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 \
        --expert-model-parallel-size 1 \
        --expert-tensor-parallel-size 1 \
        --recompute-granularity full \
        --recompute-method uniform \
        --recompute-num-layers 1 \
        --ci-test \
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

    if ! grep -q "Rebuilding weight-update connections" "${LOG_FILE}"; then
        echo "Did not observe weight-update connection rebuild in ${LOG_FILE}" >&2
        exit 5
    fi

    echo "FAST RESTART + WEIGHT UPDATE 32-GPU SMOKE TEST PASSED"
    echo "Observed shadow-worker handover and subsequent weight-update connection rebuild."
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

    local self_role
    self_role="$(determine_self_role)"
    echo "Current node role: ${self_role}"
    if [ "${self_role}" = "rollout" ]; then
        echo "Starting rollout-side parameter/KV services on local GPUs ${ROLLOUT_GPU_SET_CSV}"
        start_rollout_side_services
        mark_rollout_ready
    fi

    wait_for_head_completion
}

if [ "${IS_HEAD_NODE}" -eq 1 ]; then
    start_head_and_submit_job
else
    start_worker_node
fi
