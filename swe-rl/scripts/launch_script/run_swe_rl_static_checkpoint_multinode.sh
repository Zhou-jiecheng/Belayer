#!/bin/bash

# Multi-node SWE-RL launcher with:
# 1. static env-pool concurrency cap,
# 2. checkpoint policy enabled, and
# 3. otherwise aligned training/topology defaults with the static baseline launcher.

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

source "${SCRIPT_DIR}/swe_rl_runtime_defaults.sh"

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
export SAVE_CKPT="${SAVE_CKPT:-${SWE_RL_DIR}/../export/ckpt/swe-rl-static-checkpoint-256}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${PROMPT_DATA}" ]]; then
    echo "PROMPT_DATA does not exist: ${PROMPT_DATA}" >&2
    exit 1
fi

# Node/GPU topology.
ACTOR_GPU_SET="${ACTOR_GPU_SET:-0,1,2,3,4,5,6,7}"
ROLLOUT_GPU_SET_CSV="${ROLLOUT_GPU_SET_CSV:-0,1,2,3,4,5,6,7}"
ACTOR_NODE_COUNT="${ACTOR_NODE_COUNT:-1}"
ROLLOUT_NODE_COUNT="${ROLLOUT_NODE_COUNT:-3}"
TOTAL_NODE_COUNT="$((ACTOR_NODE_COUNT + ROLLOUT_NODE_COUNT))"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-${PROC_PER_NODE:-8}}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"

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
for rollout_gpu_set in "${ROLLOUT_GPU_SETS[@]}"; do
    IFS=',' read -r -a ENGINE_GPU_IDS <<< "${rollout_gpu_set}"
    if [[ -z "${ROLLOUT_TP_SIZE}" ]]; then
        ROLLOUT_TP_SIZE="${#ENGINE_GPU_IDS[@]}"
    fi
    if [[ "${#ENGINE_GPU_IDS[@]}" -ne "${ROLLOUT_TP_SIZE}" ]]; then
        echo "Each rollout GPU set must have the same size: ${ROLLOUT_GPU_SET_CSV}" >&2
        exit 1
    fi
done

ROLLOUT_GPU_COUNT_PER_NODE="$((ROLLOUT_ENGINE_COUNT_PER_NODE * ROLLOUT_TP_SIZE))"
if [[ "${ROLLOUT_GPU_COUNT_PER_NODE}" -ne 8 ]]; then
    echo "Rollout GPU sets must use exactly 8 GPUs per rollout node: ${ROLLOUT_GPU_SET_CSV}" >&2
    exit 1
fi
ROLLOUT_TOTAL_GPU_COUNT="$((ROLLOUT_NODE_COUNT * ROLLOUT_GPU_COUNT_PER_NODE))"
ROLLOUT_TOTAL_ENGINE_COUNT="$((ROLLOUT_NODE_COUNT * ROLLOUT_ENGINE_COUNT_PER_NODE))"

# Rollout health-check knobs.
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-0}"
ROLLOUT_HEALTH_CHECK_INTERVAL="${ROLLOUT_HEALTH_CHECK_INTERVAL:-60}"
ROLLOUT_HEALTH_CHECK_TIMEOUT="${ROLLOUT_HEALTH_CHECK_TIMEOUT:-60}"
ROLLOUT_HEALTH_CHECK_ENABLED="${ROLLOUT_HEALTH_CHECK_ENABLED:-1}"
ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD="${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD:-5}"

# Baseline-specific overrides.
export STATIC_MAX_CONCURRENCY="${STATIC_MAX_CONCURRENCY:-256}"
export CONTROL_PLANE_CONCURRENCY="${CONTROL_PLANE_CONCURRENCY:-${STATIC_MAX_CONCURRENCY}}"
export SWE_STATIC_CAPACITY_HEADROOM="${SWE_STATIC_CAPACITY_HEADROOM:-8}"
export SWE_CHECKPOINT_POLICY="${SWE_CHECKPOINT_POLICY:-adaptive-risk}"

if ! [[ "${STATIC_MAX_CONCURRENCY}" =~ ^[0-9]+$ ]] || [[ "${STATIC_MAX_CONCURRENCY}" -le 0 ]]; then
    echo "STATIC_MAX_CONCURRENCY must be a positive integer, got '${STATIC_MAX_CONCURRENCY}'" >&2
    exit 1
fi

if ! [[ "${CONTROL_PLANE_CONCURRENCY}" =~ ^[0-9]+$ ]] || [[ "${CONTROL_PLANE_CONCURRENCY}" -le 0 ]]; then
    echo "CONTROL_PLANE_CONCURRENCY must be a positive integer, got '${CONTROL_PLANE_CONCURRENCY}'" >&2
    exit 1
fi

if [[ "${CONTROL_PLANE_CONCURRENCY}" -gt "${STATIC_MAX_CONCURRENCY}" ]]; then
    CONTROL_PLANE_CONCURRENCY="${STATIC_MAX_CONCURRENCY}"
fi

STATIC_CAPACITY_LIMIT="$((STATIC_MAX_CONCURRENCY + SWE_STATIC_CAPACITY_HEADROOM))"
CONTROL_PLANE_CAPACITY_LIMIT="$((CONTROL_PLANE_CONCURRENCY + SWE_STATIC_CAPACITY_HEADROOM))"

export SWE_POOL_MAX_TOTAL_LEASES="${SWE_POOL_MAX_TOTAL_LEASES:-${STATIC_CAPACITY_LIMIT}}"
export SWE_MAX_CONTAINERS_PER_NODE="${SWE_MAX_CONTAINERS_PER_NODE:-${STATIC_CAPACITY_LIMIT}}"
export SWE_POOL_MAX_CONCURRENT_ALLOCATES="${SWE_POOL_MAX_CONCURRENT_ALLOCATES:-${CONTROL_PLANE_CAPACITY_LIMIT}}"
export SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC="${SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC:-0.05}"
export SWE_MAX_CONCURRENT_DOCKER_CREATE="${SWE_MAX_CONCURRENT_DOCKER_CREATE:-${CONTROL_PLANE_CAPACITY_LIMIT}}"
export SWE_DOCKER_CREATE_MIN_INTERVAL_SEC="${SWE_DOCKER_CREATE_MIN_INTERVAL_SEC:-0.05}"

# Runtime defaults shared with the adaptive launcher.
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK="0"
export DEBUG_MODE="${DEBUG_MODE:-0}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
export SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL="${SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL:-0}"
export TARGET_TOTAL_SAMPLES="${TARGET_TOTAL_SAMPLES:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
export GLOBAL_BATCH_SIZE=512
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-auto}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

export SWE_LITELLM_MODEL_NAME="${SWE_LITELLM_MODEL_NAME:-openai/Qwen/Qwen3-32B}"
export SWE_ENV_SERVER_BIND_HOST="${SWE_ENV_SERVER_BIND_HOST:-0.0.0.0}"
export SWE_ENV_SERVER_PORT="${SWE_ENV_SERVER_PORT:-18090}"
export SWE_ENV_SERVER_HOST="${SWE_ENV_SERVER_HOST:-${MASTER_ADDR}}"
export SWE_ENV_SERVER_URL="${SWE_ENV_SERVER_URL:-http://${SWE_ENV_SERVER_HOST}:${SWE_ENV_SERVER_PORT}}"
export SWE_EXEC_SERVER_URLS="${SWE_EXEC_SERVER_URLS:-http://100.103.147.252:5000}"
export SWE_MAX_CONCURRENT="${SWE_MAX_CONCURRENT:-${STATIC_MAX_CONCURRENCY}}"

swe_rl_apply_runtime_defaults

# Keep no-fault-injection semantics even if the caller inherited those env vars elsewhere.
export SWE_FAULT_INJECTION_ENABLE=0
export SWE_FAULT_INJECTION_PROB=0

ALL_EXEC_HOSTS="$(echo "${SWE_EXEC_SERVER_URLS}" | tr ',' '\n' | sed -E 's#https?://([^:/]+).*#\1#' | tr '\n' ',' | sed 's/,$//')"
export NO_PROXY="localhost,127.0.0.1,${MASTER_ADDR},${CURRENT_NODE_IP},${SWE_ENV_SERVER_HOST},${ALL_EXEC_HOSTS}"
export no_proxy="${NO_PROXY}"

COORD_JOB_KEY="${JOB_ID:-manual}"
COORD_ROOT="${SWE_RL_DIR}/../export/.swe_rl_static_checkpoint_coord_${MASTER_ADDR//./_}_${COORD_JOB_KEY}"
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

WORKDIR="${WORKDIR:-${SWE_RL_DIR}/../export/swe_rl_static_checkpoint_256_${RUN_ID}}"
LOG_FILE="${WORKDIR}/main_logs.log"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
ROLE_MAP_FILE="${WORKDIR}/node_roles.txt"
DONE_FILE="${WORKDIR}/head_done.exitcode"
export SWE_SAVE_TRAJ_DIR="${SWE_SAVE_TRAJ_DIR:-${WORKDIR}/swe_rollouts}"
mkdir -p "${WORKDIR}" "${LOG_DIR}" "${SWE_SAVE_TRAJ_DIR}" "$(dirname "${SAVE_CKPT}")"

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
    ray stop --force >/dev/null 2>&1 || true
    exit "${exit_code}"
}
trap 'cleanup $?' EXIT

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

wait_for_head_completion() {
    echo "[swe-static-checkpoint] Waiting for head completion marker: ${DONE_FILE}"
    while [[ ! -f "${DONE_FILE}" ]]; do
        sleep 2
    done
    local head_exit_code
    head_exit_code="$(tr -d '[:space:]' < "${DONE_FILE}")"
    if [[ -z "${head_exit_code}" ]]; then
        head_exit_code=0
    fi
    echo "[swe-static-checkpoint] Head finished with exit code ${head_exit_code}"
    exit "${head_exit_code}"
}

start_env_pool_server() {
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
    echo "[swe-static-checkpoint] SWE env pool server PID=${SWE_POOL_PID}, log=${pool_log}"

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
    "PYTORCH_CUDA_ALLOC_CONF": "${PYTORCH_CUDA_ALLOC_CONF}",
    "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
    "OPENAI_API_KEY": "${OPENAI_API_KEY}",
    "LITELLM_MODEL_REGISTRY_PATH": "${LITELLM_MODEL_REGISTRY_PATH}",
    "SWE_LITELLM_MODEL_NAME": "${SWE_LITELLM_MODEL_NAME}",
    "SWE_SAVE_TRAJ_DIR": "${SWE_SAVE_TRAJ_DIR}",
    "SWE_CONFIG_PATH": "${SWE_RL_DIR}/swebench.yaml",
    "SWE_ENV_SERVER_URL": "${SWE_ENV_SERVER_URL}",
    "SWE_CHECKPOINT_POLICY": "${SWE_CHECKPOINT_POLICY}",
    "SWE_FAULT_INJECTION_ENABLE": "${SWE_FAULT_INJECTION_ENABLE}",
    "SWE_MAX_CONCURRENT": "${SWE_MAX_CONCURRENT}",
    "SWE_MAX_CONCURRENT_DOCKER_CREATE": "${SWE_MAX_CONCURRENT_DOCKER_CREATE}",
    "SWE_DOCKER_CREATE_MIN_INTERVAL_SEC": "${SWE_DOCKER_CREATE_MIN_INTERVAL_SEC}",
    "SWE_POOL_CREATE_TIMEOUT_SEC": "${SWE_POOL_CREATE_TIMEOUT_SEC}",
    "SWE_ENV_HTTP_MAX_RETRIES": "${SWE_ENV_HTTP_MAX_RETRIES}",
    "SWE_ALLOCATE_HTTP_MAX_RETRIES": "${SWE_ALLOCATE_HTTP_MAX_RETRIES}",
    "SWE_ENV_APP_MAX_RETRIES": "${SWE_ENV_APP_MAX_RETRIES}",
    "SWE_ALLOCATE_APP_MAX_RETRIES": "${SWE_ALLOCATE_APP_MAX_RETRIES}",
    "SWE_ENV_APP_RETRY_DELAY_SEC": "${SWE_ENV_APP_RETRY_DELAY_SEC}",
    "SWE_ENV_APP_RETRY_JITTER_SEC": "${SWE_ENV_APP_RETRY_JITTER_SEC}",
    "SWE_ENV_APP_RETRY_MAX_DELAY_SEC": "${SWE_ENV_APP_RETRY_MAX_DELAY_SEC}",
    "SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL": "${SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL}",
    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "${SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK}",
    "SLIME_ROLLOUT_ENABLE_HEALTH_CHECK": "${ROLLOUT_HEALTH_CHECK_ENABLED}",
    "SLIME_ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD": "${ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD}",
    "MSWEA_DOCKER_EXEC_MODE": "${MSWEA_DOCKER_EXEC_MODE}",
    "NO_PROXY": "${NO_PROXY}",
    "no_proxy": "${no_proxy}"
  }
}
EOF
}

echo "[swe-static-checkpoint] MASTER_ADDR=${MASTER_ADDR}"
echo "[swe-static-checkpoint] CURRENT_NODE_IP=${CURRENT_NODE_IP}"
echo "[swe-static-checkpoint] MODEL_PATH=${MODEL_PATH}"
echo "[swe-static-checkpoint] PROMPT_DATA=${PROMPT_DATA}"
echo "[swe-static-checkpoint] ACTOR_NODE_COUNT=${ACTOR_NODE_COUNT}"
echo "[swe-static-checkpoint] ROLLOUT_NODE_COUNT=${ROLLOUT_NODE_COUNT}"
echo "[swe-static-checkpoint] ACTOR_GPU_SET=${ACTOR_GPU_SET}"
echo "[swe-static-checkpoint] ROLLOUT_GPU_SET_CSV=${ROLLOUT_GPU_SET_CSV}"
echo "[swe-static-checkpoint] TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE}"
echo "[swe-static-checkpoint] ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE}"
echo "[swe-static-checkpoint] ROLLOUT_TOTAL_GPU_COUNT=${ROLLOUT_TOTAL_GPU_COUNT}"
echo "[swe-static-checkpoint] STATIC_MAX_CONCURRENCY=${STATIC_MAX_CONCURRENCY}"
echo "[swe-static-checkpoint] CONTROL_PLANE_CONCURRENCY=${CONTROL_PLANE_CONCURRENCY}"
echo "[swe-static-checkpoint] SWE_STATIC_CAPACITY_HEADROOM=${SWE_STATIC_CAPACITY_HEADROOM}"
echo "[swe-static-checkpoint] SWE_POOL_MAX_TOTAL_LEASES=${SWE_POOL_MAX_TOTAL_LEASES}"
echo "[swe-static-checkpoint] SWE_MAX_CONTAINERS_PER_NODE=${SWE_MAX_CONTAINERS_PER_NODE}"
echo "[swe-static-checkpoint] SWE_MAX_CONCURRENT=${SWE_MAX_CONCURRENT}"
echo "[swe-static-checkpoint] SWE_POOL_MAX_CONCURRENT_ALLOCATES=${SWE_POOL_MAX_CONCURRENT_ALLOCATES}"
echo "[swe-static-checkpoint] SWE_MAX_CONCURRENT_DOCKER_CREATE=${SWE_MAX_CONCURRENT_DOCKER_CREATE}"
echo "[swe-static-checkpoint] SWE_CHECKPOINT_POLICY=${SWE_CHECKPOINT_POLICY}"
echo "[swe-static-checkpoint] SWE_FAULT_INJECTION_ENABLE=${SWE_FAULT_INJECTION_ENABLE}"
echo "[swe-static-checkpoint] NUM_ROLLOUT=${NUM_ROLLOUT}"
echo "[swe-static-checkpoint] ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}"
echo "[swe-static-checkpoint] N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}"
echo "[swe-static-checkpoint] OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE}"
echo "[swe-static-checkpoint] TARGET_TOTAL_SAMPLES=${TARGET_TOTAL_SAMPLES}"
echo "[swe-static-checkpoint] SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL=${SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL}"

start_head_and_submit_job() {
    echo "[swe-static-checkpoint] Head node: ${CURRENT_NODE_IP}"
    echo "[swe-static-checkpoint] Ray head: ${RAY_HEAD_IP}:${RAY_HEAD_PORT}"
    echo "[swe-static-checkpoint] Workdir: ${WORKDIR}"
    echo "[swe-static-checkpoint] Topology: actor ${ACTOR_NODE_COUNT}x${ACTOR_GPU_COUNT}GPU + rollout ${ROLLOUT_TOTAL_ENGINE_COUNT}x${ROLLOUT_TP_SIZE}GPU across ${ROLLOUT_NODE_COUNT} rollout node(s)"

    ray stop --force >/dev/null 2>&1 || true
    echo "[swe-static-checkpoint] Starting Ray head"
    ray start --head \
        --node-ip-address "${RAY_HEAD_IP}" \
        --port "${RAY_HEAD_PORT}" \
        --num-gpus "${ACTOR_GPU_COUNT}" \
        --disable-usage-stats \
        --dashboard-host 0.0.0.0 \
        --dashboard-port "${RAY_DASHBOARD_PORT}" >/dev/null

    echo "[swe-static-checkpoint] Waiting ${RAY_WORKER_JOIN_WAIT_SEC}s for worker nodes to join Ray..."
    sleep "${RAY_WORKER_JOIN_WAIT_SEC}"
    wait_for_cluster_nodes
    wait_for_dashboard_ready
    write_node_roles
    start_env_pool_server
    check_exec_servers
    unset http_proxy
    unset https_proxy

    local custom_func_file="${SWE_RL_DIR}/generate_with_swe_remote.py"
    local runtime_env_json
    runtime_env_json="$(build_runtime_env_json)"
    local ray_job_submission_id
    ray_job_submission_id="${RAY_JOB_SUBMISSION_ID:-swe_rl_static_checkpoint_256_${RUN_ID}}"
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
        echo "[swe-static-checkpoint] ===== ray job logs: ${ray_job_submission_id} ====="
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
    echo "[swe-static-checkpoint] Non-head node: ${CURRENT_NODE_IP}"
    echo "[swe-static-checkpoint] MASTER_ADDR: ${MASTER_ADDR}"

    ray stop --force >/dev/null 2>&1 || true
    echo "[swe-static-checkpoint] Starting Ray worker"
    until ray start \
        --address="${MASTER_ADDR}:${RAY_HEAD_PORT}" \
        --node-ip-address "${CURRENT_NODE_IP}" \
        --num-gpus "${NUM_GPUS_PER_NODE}" \
        --disable-usage-stats >/dev/null; do
        echo "[swe-static-checkpoint] Ray head ${MASTER_ADDR}:${RAY_HEAD_PORT} not ready yet, retrying worker join..."
        ray stop --force >/dev/null 2>&1 || true
        sleep 2
    done

    local self_role
    self_role="$(determine_self_role)"
    echo "[swe-static-checkpoint] Current node role: ${self_role}"

    wait_for_head_completion
}

if [[ "${IS_HEAD_NODE}" -eq 1 ]]; then
    start_head_and_submit_job
else
    start_worker_node
fi
