#!/bin/bash

# Multi-node SWE-RL launcher that combines:
# 1. online scheduler admission for the remote SWE env pool,
# 2. adaptive-risk checkpointing for env-side recovery, and
# 3. SGLang shadow-worker fast restart.
#
# Topology is controlled by node-role slicing, aligned with
# slime/scripts/fault_tolerance/run-fast-restart-weight-update-smoke-32gpu.sh:
# - the first ACTOR_NODE_COUNT nodes (sorted by IP) are the training slice
# - the remaining ROLLOUT_NODE_COUNT nodes are the rollout slice
#
# All nodes run this same script. MASTER_ADDR is only used for the Ray head.

set -euo pipefail

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SWE_RL_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SWE_RL_DIR}/../slime" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SWE_RL_DIR}/.." &>/dev/null && pwd)"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${SLIME_DIR}/scripts/models/qwen3-32B.sh}"
CURRENT_NODE_IP="$(hostname --ip-address | awk '{print $1}')"

source "${SCRIPT_DIR}/swe_rl_online_scheduler_adaptive_checkpoint_common.sh"

if [[ -f "${SWE_RL_DIR}/.env.swe" ]]; then
    source "${SWE_RL_DIR}/.env.swe"
fi

if [[ -z "${MASTER_ADDR:-}" ]]; then
    echo "MASTER_ADDR is required for multi-node launch." >&2
    exit 1
fi

if [[ ! -d "${MEGATRON_LM_PATH}" ]]; then
    echo "MEGATRON_LM_PATH does not exist: ${MEGATRON_LM_PATH}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_ARGS_FILE}" ]]; then
    echo "MODEL_ARGS_FILE does not exist: ${MODEL_ARGS_FILE}" >&2
    exit 1
fi

source "${MODEL_ARGS_FILE}"

IS_HEAD_NODE=0
if [[ "${CURRENT_NODE_IP}" == "${MASTER_ADDR}" ]]; then
    IS_HEAD_NODE=1
fi

# Model / data paths.
export HF_CKPT="${HF_CKPT:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/models/Qwen3-32B}"
export REF_LOAD="${REF_LOAD:-${HF_CKPT}}"
MODEL_PATH="${MODEL_PATH:-${HF_CKPT}}"
export PROMPT_DATA="${PROMPT_DATA:-${SWE_RL_DIR}/data/train.jsonl}"
export SAVE_CKPT="${SAVE_CKPT:-${SWE_RL_DIR}/../export/ckpt/swe-rl-online-scheduler-adaptive-fast-restart}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${PROMPT_DATA}" ]]; then
    echo "PROMPT_DATA does not exist: ${PROMPT_DATA}" >&2
    exit 1
fi

# Node/GPU topology.
# Default 32-GPU layout:
# - 1 training node x 8 GPUs
# - 3 rollout nodes x 8 GPUs each
# - training TP = 8
# - rollout TP = 8 (one engine per rollout node)
ACTOR_GPU_SET="${ACTOR_GPU_SET:-0,1,2,3,4,5,6,7}"
ROLLOUT_GPU_SET_CSV="${ROLLOUT_GPU_SET_CSV:-0,1,2,3,4,5,6,7}"
ACTOR_NODE_COUNT="${ACTOR_NODE_COUNT:-1}"
ROLLOUT_NODE_COUNT="${ROLLOUT_NODE_COUNT:-3}"
TOTAL_NODE_COUNT="$((ACTOR_NODE_COUNT + ROLLOUT_NODE_COUNT))"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-${PROC_PER_NODE:-8}}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
WEIGHT_SERVER_BASE_PORT="${WEIGHT_SERVER_BASE_PORT:-5556}"
FIRST_GPU_ID="${FIRST_GPU_ID:-0}"

IFS=',' read -r -a ACTOR_GPU_IDS <<< "${ACTOR_GPU_SET}"
ACTOR_GPU_COUNT="${#ACTOR_GPU_IDS[@]}"
if [[ "${ACTOR_GPU_COUNT}" -ne 8 ]]; then
    echo "ACTOR_GPU_SET must contain exactly 8 GPU ids: ${ACTOR_GPU_SET}" >&2
    exit 1
fi

IFS=';' read -r -a ROLLOUT_GPU_SETS <<< "${ROLLOUT_GPU_SET_CSV}"
ROLLOUT_ENGINE_COUNT_PER_NODE="${#ROLLOUT_GPU_SETS[@]}"
if [[ "${ROLLOUT_ENGINE_COUNT_PER_NODE}" -lt 1 ]]; then
    echo "ROLLOUT_GPU_SET_CSV must contain at least one rollout GPU set: ${ROLLOUT_GPU_SET_CSV}" >&2
    exit 1
fi

ROLLOUT_TP_SIZE=""
KV_SOCKET_PATHS=()
KV_SOCKET_CSV_PARTS=()
for idx in "${!ROLLOUT_GPU_SETS[@]}"; do
    IFS=',' read -r -a ENGINE_GPU_IDS <<< "${ROLLOUT_GPU_SETS[$idx]}"
    if [[ -z "${ROLLOUT_TP_SIZE}" ]]; then
        ROLLOUT_TP_SIZE="${#ENGINE_GPU_IDS[@]}"
    fi
    if [[ "${#ENGINE_GPU_IDS[@]}" -ne "${ROLLOUT_TP_SIZE}" ]]; then
        echo "Each rollout GPU set must have the same size: ${ROLLOUT_GPU_SET_CSV}" >&2
        exit 1
    fi
    KV_SOCKET_PATHS+=("/tmp/swe_rl_fast_restart_kv_${idx}.sock")
    KV_SOCKET_CSV_PARTS+=("${KV_SOCKET_PATHS[$idx]}")
done
KV_CACHE_SOCKETS="$(IFS=','; echo "${KV_SOCKET_CSV_PARTS[*]}")"
ROLLOUT_GPU_COUNT_PER_NODE="$((ROLLOUT_ENGINE_COUNT_PER_NODE * ROLLOUT_TP_SIZE))"
if [[ "${ROLLOUT_GPU_COUNT_PER_NODE}" -ne 8 ]]; then
    echo "Rollout GPU sets must use exactly 8 GPUs per rollout node: ${ROLLOUT_GPU_SET_CSV}" >&2
    exit 1
fi
ROLLOUT_TOTAL_GPU_COUNT="$((ROLLOUT_NODE_COUNT * ROLLOUT_GPU_COUNT_PER_NODE))"
ROLLOUT_TOTAL_ENGINE_COUNT="$((ROLLOUT_NODE_COUNT * ROLLOUT_ENGINE_COUNT_PER_NODE))"

# Fast-restart / health-check knobs.
KV_CACHE_MEM_FRACTION_STATIC="${KV_CACHE_MEM_FRACTION_STATIC:-0.7}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-0}"
ROLLOUT_HEALTH_CHECK_INTERVAL="${ROLLOUT_HEALTH_CHECK_INTERVAL:-60}"
ROLLOUT_HEALTH_CHECK_TIMEOUT="${ROLLOUT_HEALTH_CHECK_TIMEOUT:-60}"
ROLLOUT_HEALTH_CHECK_ENABLED="${ROLLOUT_HEALTH_CHECK_ENABLED:-1}"
ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD="${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD:-5}"
ROUTER_HEALTH_CHECK_ENABLED="${ROUTER_HEALTH_CHECK_ENABLED:-0}"
ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD="${ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD:-20}"
ROUTER_HEALTH_CHECK_INTERVAL_SEC="${ROUTER_HEALTH_CHECK_INTERVAL_SEC:-15}"
ROUTER_GENERATE_CHUNK_DEBUG="${ROUTER_GENERATE_CHUNK_DEBUG:-1}"
SHADOW_WORKER_READY_TIMEOUT_SEC="${SHADOW_WORKER_READY_TIMEOUT_SEC:-600}"
SHADOW_WORKER_STABILIZATION_SEC="${SHADOW_WORKER_STABILIZATION_SEC:-20}"

CI_FAULT_INJECTION_ENABLE="${CI_FAULT_INJECTION_ENABLE:-0}"
CI_FAULT_INJECTION_DELAY_SEC="${CI_FAULT_INJECTION_DELAY_SEC:-20}"
CI_FAULT_INJECTION_MODE="${CI_FAULT_INJECTION_MODE:-mid_generate}"
CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD="${CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD:-0}"
CI_FAULT_INJECTION_PROGRESS_FRACTION="${CI_FAULT_INJECTION_PROGRESS_FRACTION:-0.5}"
CI_FAULT_INJECTION_MID_DELAY_SEC="${CI_FAULT_INJECTION_MID_DELAY_SEC:-0}"
CI_FAULT_INJECTION_ENGINE_INDEX="${CI_FAULT_INJECTION_ENGINE_INDEX:-0}"
CI_FAULT_INJECTION_MID_FALLBACK_SEC="${CI_FAULT_INJECTION_MID_FALLBACK_SEC:-20}"
SWE_SCHED_PRESERVE_PROMPT_ORDER=0

# Runtime defaults shared with the single-node rollout-only debug script.
export DEBUG_MODE="${DEBUG_MODE:-0}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
export TARGET_TOTAL_SAMPLES="${TARGET_TOTAL_SAMPLES:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
export GLOBAL_BATCH_SIZE=512
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-auto}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK="0"

export SWE_LITELLM_MODEL_NAME="${SWE_LITELLM_MODEL_NAME:-openai/Qwen/Qwen3-32B}"
export SWE_ENV_SERVER_BIND_HOST="${SWE_ENV_SERVER_BIND_HOST:-0.0.0.0}"
export SWE_ENV_SERVER_PORT="${SWE_ENV_SERVER_PORT:-18090}"
export SWE_ENV_SERVER_HOST="${SWE_ENV_SERVER_HOST:-${MASTER_ADDR}}"
export SWE_ENV_SERVER_URL="${SWE_ENV_SERVER_URL:-http://${SWE_ENV_SERVER_HOST}:${SWE_ENV_SERVER_PORT}}"
export SWE_EXEC_SERVER_URLS="${SWE_EXEC_SERVER_URLS:-http://100.103.147.252:5000}"
export SWE_MAX_CONCURRENT="${SWE_MAX_CONCURRENT:-4096}"
export SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL="${SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL:-0}"

swe_rl_apply_online_scheduler_adaptive_checkpoint_defaults

# Match the working fault-tolerance smoke launchers for shadow-worker startup:
# TorchMemorySaver used by fast restart is incompatible with expandable_segments.
# Clear both allocator env vars after sourcing common defaults so they are not
# reintroduced into the Ray job / child SGLang processes.
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

ALL_EXEC_HOSTS="$(echo "${SWE_EXEC_SERVER_URLS}" | tr ',' '\n' | sed -E 's#https?://([^:/]+).*#\1#' | tr '\n' ',' | sed 's/,$//')"
export NO_PROXY="localhost,127.0.0.1,${MASTER_ADDR},${CURRENT_NODE_IP},${SWE_ENV_SERVER_HOST},${ALL_EXEC_HOSTS}"
export no_proxy="${NO_PROXY}"

COORD_JOB_KEY="${JOB_ID:-manual}"
COORD_ROOT="${SWE_RL_DIR}/../export/.swe_rl_fast_restart_coord_${MASTER_ADDR//./_}_${COORD_JOB_KEY}"
RUN_ID="${RUN_ID:-}"
RAY_HEAD_IP="${RAY_HEAD_IP:-${MASTER_ADDR}}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-9991}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_DASHBOARD_URL="${RAY_DASHBOARD_URL:-http://${RAY_HEAD_IP}:${RAY_DASHBOARD_PORT}}"
RAY_WORKER_JOIN_WAIT_SEC="${RAY_WORKER_JOIN_WAIT_SEC:-45}"

resolve_run_id() {
    mkdir -p "${COORD_ROOT}"
    local run_id_file="${COORD_ROOT}/run_id"
    if [[ "${IS_HEAD_NODE}" -eq 1 ]]; then
        if [[ -z "${RUN_ID}" ]]; then
            RUN_ID="$(date +%Y%m%d_%H%M%S)"
        fi
        printf '%s\n' "${RUN_ID}" > "${run_id_file}"
    elif [[ -z "${RUN_ID}" ]]; then
        while true; do
            if [[ -f "${run_id_file}" ]]; then
                RUN_ID="$(cat "${run_id_file}")"
                if [[ -n "${RUN_ID}" ]]; then
                    break
                fi
            fi
            sleep 1
        done
    fi
}

resolve_run_id

WORKDIR="${WORKDIR:-${SWE_RL_DIR}/../export/swe_rl_online_scheduler_adaptive_fast_restart_${RUN_ID}}"
LOG_FILE="${WORKDIR}/main_logs.log"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
ROLE_MAP_FILE="${WORKDIR}/node_roles.txt"
READY_DIR="${WORKDIR}/rollout_ready"
SELF_READY_FILE="${READY_DIR}/${CURRENT_NODE_IP}.ready"
DONE_FILE="${WORKDIR}/head_done.exitcode"
export SWE_SAVE_TRAJ_DIR="${SWE_SAVE_TRAJ_DIR:-${WORKDIR}/swe_rollouts}"
mkdir -p "${WORKDIR}" "${LOG_DIR}" "${READY_DIR}" "${SWE_SAVE_TRAJ_DIR}" "$(dirname "${SAVE_CKPT}")"

PIDS=()
SWE_POOL_PID=""

cleanup() {
    local exit_code="$1"
    trap - EXIT
    set +e
    if [[ "${IS_HEAD_NODE}" -eq 1 ]]; then
        printf '%s\n' "${exit_code}" > "${DONE_FILE}"
    fi
    if [[ -n "${SWE_POOL_PID}" ]] && kill -0 "${SWE_POOL_PID}" 2>/dev/null; then
        kill "${SWE_POOL_PID}" >/dev/null 2>&1 || true
        wait "${SWE_POOL_PID}" >/dev/null 2>&1 || true
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

launch_parameter_server() {
    local gpu_set="$1"
    local tp_size="$2"
    local port="$3"
    local log_path="$4"
    echo "[swe-fast-restart] Starting parameter server on GPUs ${gpu_set}, port ${port}"
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
    echo "[swe-fast-restart] Starting KV cache server on GPUs ${gpu_set}, socket ${socket_path}"
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

ray.init(address=f"{os.environ['RAY_HEAD_IP']}:{os.environ['RAY_HEAD_PORT']}", ignore_reinit_error=True, logging_level=40)
alive = sorted({n["NodeManagerAddress"] for n in ray.nodes() if n.get("Alive")}, key=lambda ip: tuple(map(int, ip.split("."))))
print(len(alive))
PY
        )"
        if [[ "${alive_count}" -eq "${TOTAL_NODE_COUNT}" ]]; then
            break
        fi
        if [[ "${alive_count}" -gt "${TOTAL_NODE_COUNT}" ]]; then
            echo "Expected ${TOTAL_NODE_COUNT} Ray nodes, found ${alive_count}" >&2
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
        raise SystemExit(0 if resp.status == 200 else 1)
except Exception:
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
PY
}

determine_self_role() {
    while [[ ! -f "${ROLE_MAP_FILE}" ]]; do
        sleep 1
    done
    local role
    role="$(awk -v ip="${CURRENT_NODE_IP}" '$1 == ip {print $2}' "${ROLE_MAP_FILE}")"
    if [[ -z "${role}" ]]; then
        echo "Current node ${CURRENT_NODE_IP} not found in ${ROLE_MAP_FILE}" >&2
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
        if [[ "${ready_count}" -eq "${ROLLOUT_NODE_COUNT}" ]]; then
            break
        fi
        sleep 2
    done
}

wait_for_head_completion() {
    echo "[swe-fast-restart] Waiting for head completion marker: ${DONE_FILE}"
    while [[ ! -f "${DONE_FILE}" ]]; do
        sleep 2
    done
    local head_exit_code
    head_exit_code="$(tr -d '[:space:]' < "${DONE_FILE}")"
    if [[ -z "${head_exit_code}" ]]; then
        head_exit_code=0
    fi
    echo "[swe-fast-restart] Head finished with exit code ${head_exit_code}"
    exit "${head_exit_code}"
}

start_env_pool_server() {

    MINISWE_DIR="${SWE_RL_DIR}/mini-swe-agent"
    MINISWE_VERSION="v1.12.0"

    # if ! python3 -c "import minisweagent" 2>/dev/null; then
    #     export http_proxy=http://100.100.63.247:7890
    #     export https_proxy=http://100.100.63.247:7890
    #     if [ ! -d "${MINISWE_DIR}" ]; then
    #         echo "Cloning mini-swe-agent ${MINISWE_VERSION}..."
    #         git clone --branch "${MINISWE_VERSION}" --depth 1 \
    #             https://github.com/SWE-agent/mini-swe-agent.git "${MINISWE_DIR}"
    #     fi
    #     echo "Installing mini-swe-agent from local source (editable mode)..."
    #     pip install -e "${MINISWE_DIR}"
    #     pip install itsdangerous
    #     pip install --no-deps flask
    #     pip install litellm
    #     unset http_proxy
    #     unset https_proxy
    # fi

    local pool_log="${LOG_DIR}/swe_env_pool_server.log"
    PYTHONPATH="${SLIME_DIR}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${PYTHONPATH:-}" \
    python3 -m swe_env_pool_server \
        --host "${SWE_ENV_SERVER_BIND_HOST}" \
        --port "${SWE_ENV_SERVER_PORT}" \
        --exec-server-urls "${SWE_EXEC_SERVER_URLS}" \
        --max-containers-per-node "${SWE_MAX_CONTAINERS_PER_NODE}" \
        --max-total-leases "${SWE_POOL_MAX_TOTAL_LEASES}" \
        --max-concurrent-allocates "${SWE_POOL_MAX_CONCURRENT_ALLOCATES}" \
        --allocate-min-interval-sec "${SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC}" \
        --create-timeout-sec "${SWE_POOL_CREATE_TIMEOUT_SEC}" \
        > "${pool_log}" 2>&1 &
    SWE_POOL_PID="$!"
    echo "[swe-fast-restart] SWE env pool server PID=${SWE_POOL_PID}, log=${pool_log}"

    for _ in {1..60}; do
        if curl -fsS "${SWE_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    if ! curl -fsS "${SWE_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
        echo "SWE env pool server failed to start: ${SWE_ENV_SERVER_URL}/healthz" >&2
        exit 1
    fi
}

check_exec_servers() {
    IFS=',' read -r -a _exec_urls <<< "${SWE_EXEC_SERVER_URLS}"
    for exec_url in "${_exec_urls[@]}"; do
        if ! curl -fsS --max-time 60 "${exec_url}/healthz" >/dev/null; then
            echo "SWE exec server is not healthy: ${exec_url}/healthz" >&2
            exit 1
        fi
    done
}

build_runtime_env_json() {
    cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${MEGATRON_LM_PATH}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${SLIME_DIR}:${REPO_ROOT}/sglang/python",
    "PYTHONUNBUFFERED": "${PYTHONUNBUFFERED}",
    "PYTHONFAULTHANDLER": "${PYTHONFAULTHANDLER}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTORCH_CUDA_ALLOC_CONF": "",
    "PYTORCH_ALLOC_CONF": "",
    "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
    "OPENAI_API_KEY": "${OPENAI_API_KEY}",
    "LITELLM_MODEL_REGISTRY_PATH": "${LITELLM_MODEL_REGISTRY_PATH}",
    "SWE_LITELLM_MODEL_NAME": "${SWE_LITELLM_MODEL_NAME}",
    "SWE_SAVE_TRAJ_DIR": "${SWE_SAVE_TRAJ_DIR}",
    "SWE_CONFIG_PATH": "${SWE_RL_DIR}/swebench.yaml",
    "SWE_ENV_SERVER_URL": "${SWE_ENV_SERVER_URL}",
    "SWE_ENABLE_ONLINE_ENV_DOCKER_SCHEDULER": "${SWE_ENABLE_ONLINE_ENV_DOCKER_SCHEDULER}",
    "SWE_CHECKPOINT_POLICY": "${SWE_CHECKPOINT_POLICY}",
    "SWE_ADAPTIVE_TAIL_ROOT": "${SWE_ADAPTIVE_TAIL_ROOT}",
    "SWE_ADAPTIVE_CHECKPOINT_BUDGET_SEC": "${SWE_ADAPTIVE_CHECKPOINT_BUDGET_SEC}",
    "SWE_ADAPTIVE_DECISION_INTERVAL_SEC": "${SWE_ADAPTIVE_DECISION_INTERVAL_SEC}",
    "SWE_ADAPTIVE_FAILURE_PROB": "${SWE_ADAPTIVE_FAILURE_PROB}",
    "SWE_ADAPTIVE_MIN_DELTA_ENV_COST_SEC": "${SWE_ADAPTIVE_MIN_DELTA_ENV_COST_SEC}",
    "SWE_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS": "${SWE_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS}",
    "SWE_FAULT_INJECTION_ENABLE": "${SWE_FAULT_INJECTION_ENABLE}",
    "SWE_FAULT_INJECTION_PROB": "${SWE_FAULT_INJECTION_PROB}",
    "SWE_MAX_CONCURRENT": "${SWE_MAX_CONCURRENT}",
    "SWE_MAX_CONCURRENT_DOCKER_CREATE": "${SWE_MAX_CONCURRENT_DOCKER_CREATE}",
    "SWE_DOCKER_CREATE_MIN_INTERVAL_SEC": "${SWE_DOCKER_CREATE_MIN_INTERVAL_SEC}",
    "SWE_SCHED_DOCKER_CREATE_MAX_CONCURRENT": "${SWE_SCHED_DOCKER_CREATE_MAX_CONCURRENT}",
    "SWE_SCHED_DOCKER_CREATE_MIN_INTERVAL_SEC": "${SWE_SCHED_DOCKER_CREATE_MIN_INTERVAL_SEC}",
    "SWE_POOL_CREATE_TIMEOUT_SEC": "${SWE_POOL_CREATE_TIMEOUT_SEC}",
    "SWE_ENV_HTTP_MAX_RETRIES": "${SWE_ENV_HTTP_MAX_RETRIES}",
    "SWE_ALLOCATE_HTTP_MAX_RETRIES": "${SWE_ALLOCATE_HTTP_MAX_RETRIES}",
    "SWE_ENV_APP_MAX_RETRIES": "${SWE_ENV_APP_MAX_RETRIES}",
    "SWE_ALLOCATE_APP_MAX_RETRIES": "${SWE_ALLOCATE_APP_MAX_RETRIES}",
    "SWE_ENV_APP_RETRY_DELAY_SEC": "${SWE_ENV_APP_RETRY_DELAY_SEC}",
    "SWE_ENV_APP_RETRY_JITTER_SEC": "${SWE_ENV_APP_RETRY_JITTER_SEC}",
    "SWE_ENV_APP_RETRY_MAX_DELAY_SEC": "${SWE_ENV_APP_RETRY_MAX_DELAY_SEC}",
    "SWE_SCHED_SAMPLING_INTERVAL_SEC": "${SWE_SCHED_SAMPLING_INTERVAL_SEC}",
    "SWE_SCHED_SAFETY_MARGIN": "${SWE_SCHED_SAFETY_MARGIN}",
    "SWE_SCHED_MEMORY_OVERSELL_RATIO": "${SWE_SCHED_MEMORY_OVERSELL_RATIO}",
    "SWE_SCHED_MEMORY_PEAK_SCALE": "${SWE_SCHED_MEMORY_PEAK_SCALE}",
    "SWE_SCHED_CPU_OVERSELL_RATIO": "${SWE_SCHED_CPU_OVERSELL_RATIO}",
    "SWE_SCHED_DISK_READ_OVERSELL_RATIO": "${SWE_SCHED_DISK_READ_OVERSELL_RATIO}",
    "SWE_SCHED_DISK_WRITE_OVERSELL_RATIO": "${SWE_SCHED_DISK_WRITE_OVERSELL_RATIO}",
    "SWE_SCHED_MAX_UNKNOWN_REPO_CONCURRENCY": "${SWE_SCHED_MAX_UNKNOWN_REPO_CONCURRENCY}",
    "SWE_SCHED_STARTUP_MAX_ACTIVE_PROMPTS": "${SWE_SCHED_STARTUP_MAX_ACTIVE_PROMPTS}",
    "SWE_SCHED_STARTUP_CAP_DURATION_SEC": "${SWE_SCHED_STARTUP_CAP_DURATION_SEC}",
    "SWE_SCHED_DISABLE_LIVE_STATS_POLLING": "${SWE_SCHED_DISABLE_LIVE_STATS_POLLING}",
    "SWE_SCHED_USE_RESOURCE_STATS_DIR": "${SWE_SCHED_USE_RESOURCE_STATS_DIR}",
    "SWE_SCHED_USE_REALTIME_SERVER_MEMORY": "${SWE_SCHED_USE_REALTIME_SERVER_MEMORY}",
    "SWE_SCHED_USE_REALTIME_SERVER_CPU": "${SWE_SCHED_USE_REALTIME_SERVER_CPU}",
    "SWE_SCHED_USE_REALTIME_SERVER_DISK": "${SWE_SCHED_USE_REALTIME_SERVER_DISK}",
    "SWE_SCHED_SERVER_MEMORY_REFRESH_SEC": "${SWE_SCHED_SERVER_MEMORY_REFRESH_SEC}",
    "SWE_SCHED_VERBOSE_LOGGING": "${SWE_SCHED_VERBOSE_LOGGING}",
    "SWE_SCHED_BLOCKED_LOG_INTERVAL_SEC": "${SWE_SCHED_BLOCKED_LOG_INTERVAL_SEC}",
    "SWE_SCHED_VERBOSE_LOG_INTERVAL_SEC": "${SWE_SCHED_VERBOSE_LOG_INTERVAL_SEC}",
    "SWE_SCHED_HEAD_BLOCK_REQUEUE_THRESHOLD": "${SWE_SCHED_HEAD_BLOCK_REQUEUE_THRESHOLD}",
    "SWE_SCHED_HEAD_BLOCK_REQUEUE_OFFSET": "${SWE_SCHED_HEAD_BLOCK_REQUEUE_OFFSET}",
    "SWE_SCHED_DURATION_PRIORITY_WEIGHT": "${SWE_SCHED_DURATION_PRIORITY_WEIGHT}",
    "SWE_SCHED_DURATION_PRIORITY_REF_SEC": "${SWE_SCHED_DURATION_PRIORITY_REF_SEC}",
    "SWE_SCHED_REALTIME_LOCAL_ACTIVE_DISCOUNT": "${SWE_SCHED_REALTIME_LOCAL_ACTIVE_DISCOUNT}",
    "SWE_SCHED_INTERNAL_MAX_INFLIGHT": "${SWE_SCHED_INTERNAL_MAX_INFLIGHT}",
    "SWE_SCHED_PENDING_GROUP_WINDOW_MULTIPLIER": "${SWE_SCHED_PENDING_GROUP_WINDOW_MULTIPLIER}",
    "SWE_SCHED_PENDING_GROUP_WINDOW_EXTRA": "${SWE_SCHED_PENDING_GROUP_WINDOW_EXTRA}",
    "SWE_SCHED_PENDING_GROUP_WINDOW_CAP": "${SWE_SCHED_PENDING_GROUP_WINDOW_CAP}",
    "SWE_SCHED_PRESERVE_PROMPT_ORDER": "${SWE_SCHED_PRESERVE_PROMPT_ORDER}",
    "SWE_RESOURCE_STATS_DIR": "${SWE_RESOURCE_STATS_DIR}",
    "SWE_REPO_RESOURCE_STATS_PATH": "${SWE_REPO_RESOURCE_STATS_PATH}",
    "SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL": "${SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL}",
    "SGLANG_KV_CACHE_SOCKET_PATH": "${KV_CACHE_SOCKETS}",
    "WEIGHT_SERVER_BASE_PORT": "${WEIGHT_SERVER_BASE_PORT}",
    "SGLANG_MIN_GPU_ID": "${FIRST_GPU_ID}",
    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "${SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK}",
    "SLIME_ROLLOUT_ENABLE_HEALTH_CHECK": "${ROLLOUT_HEALTH_CHECK_ENABLED}",
    "SLIME_ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD": "${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD}",
    "SLIME_ROUTER_ENABLE_HEALTH_CHECK": "${ROUTER_HEALTH_CHECK_ENABLED}",
    "SLIME_ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD": "${ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD}",
    "SLIME_ROUTER_HEALTH_CHECK_INTERVAL_SEC": "${ROUTER_HEALTH_CHECK_INTERVAL_SEC}",
    "SLIME_ROUTER_GENERATE_CHUNK_DEBUG": "${ROUTER_GENERATE_CHUNK_DEBUG}",
    "MSWEA_DOCKER_EXEC_MODE": "${MSWEA_DOCKER_EXEC_MODE}",
    "NO_PROXY": "${NO_PROXY}",
    "no_proxy": "${no_proxy}"
  }
}
EOF
}

echo "[swe-fast-restart] MASTER_ADDR=${MASTER_ADDR}"
echo "[swe-fast-restart] CURRENT_NODE_IP=${CURRENT_NODE_IP}"
echo "[swe-fast-restart] MODEL_PATH=${MODEL_PATH}"
echo "[swe-fast-restart] PROMPT_DATA=${PROMPT_DATA}"
echo "[swe-fast-restart] ACTOR_NODE_COUNT=${ACTOR_NODE_COUNT}"
echo "[swe-fast-restart] ROLLOUT_NODE_COUNT=${ROLLOUT_NODE_COUNT}"
echo "[swe-fast-restart] ACTOR_GPU_SET=${ACTOR_GPU_SET}"
echo "[swe-fast-restart] ROLLOUT_GPU_SET_CSV=${ROLLOUT_GPU_SET_CSV}"
echo "[swe-fast-restart] TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE}"
echo "[swe-fast-restart] ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE}"
echo "[swe-fast-restart] ROLLOUT_TOTAL_GPU_COUNT=${ROLLOUT_TOTAL_GPU_COUNT}"
echo "[swe-fast-restart] WEIGHT_SERVER_BASE_PORT=${WEIGHT_SERVER_BASE_PORT}"
echo "[swe-fast-restart] KV_CACHE_SOCKETS=${KV_CACHE_SOCKETS}"
swe_rl_print_online_scheduler_adaptive_checkpoint_summary "[swe-fast-restart]"
# --use-slime-router \
start_head_and_submit_job() {
    echo "[swe-fast-restart] Head node: ${CURRENT_NODE_IP}"
    echo "[swe-fast-restart] Ray head: ${RAY_HEAD_IP}:${RAY_HEAD_PORT}"
    echo "[swe-fast-restart] Workdir: ${WORKDIR}"
    echo "[swe-fast-restart] Topology: actor ${ACTOR_NODE_COUNT}x${ACTOR_GPU_COUNT}GPU + rollout ${ROLLOUT_TOTAL_ENGINE_COUNT}x${ROLLOUT_TP_SIZE}GPU across ${ROLLOUT_NODE_COUNT} rollout node(s)"

    ray stop --force >/dev/null 2>&1 || true
    echo "[swe-fast-restart] Starting Ray head"
    ray start --head \
        --node-ip-address "${RAY_HEAD_IP}" \
        --port "${RAY_HEAD_PORT}" \
        --num-gpus "${ACTOR_GPU_COUNT}" \
        --disable-usage-stats \
        --dashboard-host 0.0.0.0 \
        --dashboard-port "${RAY_DASHBOARD_PORT}" >/dev/null

    echo "[swe-fast-restart] Waiting ${RAY_WORKER_JOIN_WAIT_SEC}s for worker nodes to join Ray..."
    sleep "${RAY_WORKER_JOIN_WAIT_SEC}"
    wait_for_cluster_nodes
    wait_for_dashboard_ready
    write_node_roles

    local self_role
    self_role="$(determine_self_role)"
    echo "[swe-fast-restart] Current node role: ${self_role}"
    if [[ "${self_role}" == "rollout" ]]; then
        echo "[swe-fast-restart] Head participates in rollout slice; starting local rollout sidecars."
        start_rollout_side_services
        mark_rollout_ready
    fi

    echo "[swe-fast-restart] Waiting for all rollout nodes to report ready..."
    wait_for_rollout_readiness
    start_env_pool_server
    check_exec_servers
    unset http_proxy
    unset https_proxy

    local custom_func_file="${SWE_RL_DIR}/generate_with_swe_remote.py"
    local runtime_env_json
    runtime_env_json="$(build_runtime_env_json)"
    local ray_job_submission_id
    ray_job_submission_id="${RAY_JOB_SUBMISSION_ID:-swe_rl_online_sched_adaptive_fast_restart_${RUN_ID}}"
    local ray_job_log_file="${WORKDIR}/ray_job_full.log"

    ray job submit --address="${RAY_DASHBOARD_URL}" \
        --submission-id "${ray_job_submission_id}" \
        --no-wait \
        --runtime-env-json="${runtime_env_json}" \
        -- python3 -u "${SLIME_DIR}/train.py" \
        --actor-num-nodes "${ACTOR_NODE_COUNT}" \
        --actor-num-gpus-per-node "${ACTOR_GPU_COUNT}" \
        --rollout-num-gpus "${ROLLOUT_TOTAL_GPU_COUNT}" \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint "${HF_CKPT}" \
        --ref-load "${REF_LOAD}" \
        --save "${SAVE_CKPT}" \
        --save-interval 20 \
        --prompt-data "${PROMPT_DATA}" \
        --input-key text \
        --metadata-key metadata \
        --rollout-shuffle \
        --reward-key score \
        --num-rollout "${NUM_ROLLOUT}" \
        --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
        --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        --rollout-max-response-len 4096 \
        --rollout-max-context-len 32768 \
        --rollout-temperature 1 \
        --num-steps-per-rollout 1 \
        --advantage-estimator grpo \
        --use-kl-loss \
        --kl-loss-coef 0.00 \
        --kl-loss-type low_var_kl \
        --entropy-coef 0.00 \
        --eps-clip 0.2 \
        --eps-clip-high 0.28 \
        --optimizer adam \
        --lr 1e-6 \
        --lr-decay-style constant \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.98 \
        --optimizer-cpu-offload \
        --overlap-cpu-optimizer-d2h-h2d \
        --use-precision-aware-optimizer \
        --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
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
        --log-probs-chunk-size 1024 \
        --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE}" \
        --sglang-mem-fraction-static 0.7 \
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
        --custom-generate-function-path "${custom_func_file}:generate" \
        --custom-rm-path "${custom_func_file}:reward_func" \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --accumulate-allreduce-grads-in-fp32 \
        --attention-softmax-in-fp32 \
        --attention-backend flash 2>&1 | stdbuf -oL -eL tee "${LOG_FILE}"

    set +e
    {
        echo
        echo "[swe-fast-restart] ===== ray job logs: ${ray_job_submission_id} ====="
    } | tee -a "${LOG_FILE}" "${ray_job_log_file}"
    ray job logs --address="${RAY_DASHBOARD_URL}" "${ray_job_submission_id}" -f --log-style=record 2>&1 | \
        stdbuf -oL -eL tee -a "${LOG_FILE}" "${ray_job_log_file}"
    local ray_log_exit=${PIPESTATUS[0]}
    local ray_status_output
    ray_status_output="$(ray job status --address="${RAY_DASHBOARD_URL}" "${ray_job_submission_id}" --log-style=record 2>&1)"
    echo "${ray_status_output}" | tee -a "${LOG_FILE}" "${ray_job_log_file}"
    set -e
    if [[ "${ray_status_output}" == *"SUCCEEDED"* ]]; then
        exit 0
    fi
    echo "Ray job failed (submission id: ${ray_job_submission_id}, logs exit: ${ray_log_exit})" >&2
    exit 1
}

start_worker_node() {
    echo "[swe-fast-restart] Non-head node: ${CURRENT_NODE_IP}"
    echo "[swe-fast-restart] MASTER_ADDR: ${MASTER_ADDR}"

    ray stop --force >/dev/null 2>&1 || true
    echo "[swe-fast-restart] Starting Ray worker"
    until ray start \
        --address="${MASTER_ADDR}:${RAY_HEAD_PORT}" \
        --node-ip-address "${CURRENT_NODE_IP}" \
        --num-gpus "${NUM_GPUS_PER_NODE}" \
        --disable-usage-stats >/dev/null; do
        echo "[swe-fast-restart] Ray head ${MASTER_ADDR}:${RAY_HEAD_PORT} not ready yet, retrying worker join..."
        ray stop --force >/dev/null 2>&1 || true
        sleep 2
    done

    local self_role
    self_role="$(determine_self_role)"
    echo "[swe-fast-restart] Current node role: ${self_role}"
    if [[ "${self_role}" == "rollout" ]]; then
        echo "[swe-fast-restart] Starting rollout-side parameter/KV services on local GPUs ${ROLLOUT_GPU_SET_CSV}"
        start_rollout_side_services
        mark_rollout_ready
    fi

    wait_for_head_completion
}

if [[ "${IS_HEAD_NODE}" -eq 1 ]]; then
    start_head_and_submit_job
else
    start_worker_node
fi
