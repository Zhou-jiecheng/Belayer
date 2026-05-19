#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWE_RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SWE_RL_DIR}/.." && pwd)"
SLIME_DIR="${REPO_ROOT}/slime"

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/miniconda/envs/osworld/bin/python}"
TRAJ_ROOT="${TRAJ_ROOT:-${1:-${REPO_ROOT}/export/swe_rl_online_scheduler_adaptive_fast_restart_20260501_152520/swe_rollouts}}"
if [[ $# -gt 0 ]]; then
    shift
fi

export SWE_ENV_SERVER_BIND_HOST="${SWE_ENV_SERVER_BIND_HOST:-127.0.0.1}"
export SWE_ENV_SERVER_PORT="${SWE_ENV_SERVER_PORT:-18090}"
export SWE_ENV_SERVER_HOST="${SWE_ENV_SERVER_HOST:-127.0.0.1}"
export SWE_ENV_SERVER_URL="${SWE_ENV_SERVER_URL:-http://${SWE_ENV_SERVER_HOST}:${SWE_ENV_SERVER_PORT}}"
export SWE_EXEC_SERVER_URLS="${SWE_EXEC_SERVER_URLS:-http://100.103.147.254:5000}"

export SWE_MAX_CONTAINERS_PER_NODE="${SWE_MAX_CONTAINERS_PER_NODE:-64}"
export SWE_POOL_MAX_TOTAL_LEASES="${SWE_POOL_MAX_TOTAL_LEASES:-64}"
export SWE_POOL_MAX_CONCURRENT_ALLOCATES="${SWE_POOL_MAX_CONCURRENT_ALLOCATES:-8}"
export SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC="${SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC:-0.05}"
export SWE_POOL_CREATE_TIMEOUT_SEC="${SWE_POOL_CREATE_TIMEOUT_SEC:-180}"

PROFILE_MAX_CONCURRENCY="${PROFILE_MAX_CONCURRENCY:-64}"
PROFILE_LIMIT="${PROFILE_LIMIT:-}"
PROFILE_SAMPLE_INTERVAL_SEC="${PROFILE_SAMPLE_INTERVAL_SEC:-1.0}"
PROFILE_NEIGHBOR_COUNT="${PROFILE_NEIGHBOR_COUNT:-5}"
SIMULATE_LLM_DELAY="${SIMULATE_LLM_DELAY:-1}"
PROFILE_OUTPUT_ROOT="${PROFILE_OUTPUT_ROOT:-${REPO_ROOT}/export/replay_image_resource_profile_$(date +%Y%m%d_%H%M%S)}"
PROFILE_OUTPUT_JSON="${PROFILE_OUTPUT_JSON:-${PROFILE_OUTPUT_ROOT}/summary.json}"

mkdir -p "${PROFILE_OUTPUT_ROOT}"
POOL_LOG="${POOL_LOG:-${PROFILE_OUTPUT_ROOT}/swe_env_pool_server.log}"
PROFILE_LOG="${PROFILE_LOG:-${PROFILE_OUTPUT_ROOT}/replay_image_resource_profile.log}"

ALL_EXEC_HOSTS="$(echo "${SWE_EXEC_SERVER_URLS}" | tr ',' '\n' | sed -E 's#https?://([^:/]+).*#\1#' | tr '\n' ',' | sed 's/,$//')"
export NO_PROXY="localhost,127.0.0.1,${SWE_ENV_SERVER_HOST},${ALL_EXEC_HOSTS}"
export no_proxy="${NO_PROXY}"

SWE_POOL_PID=""
cleanup() {
    set +e
    if [[ -n "${SWE_POOL_PID}" ]] && kill -0 "${SWE_POOL_PID}" 2>/dev/null; then
        kill "${SWE_POOL_PID}" || true
    fi
}
trap cleanup EXIT INT TERM

echo "[replay-profile] PYTHON_BIN=${PYTHON_BIN}"
echo "[replay-profile] TRAJ_ROOT=${TRAJ_ROOT}"
echo "[replay-profile] SWE_ENV_SERVER_URL=${SWE_ENV_SERVER_URL}"
echo "[replay-profile] SWE_EXEC_SERVER_URLS=${SWE_EXEC_SERVER_URLS}"
echo "[replay-profile] PROFILE_OUTPUT_ROOT=${PROFILE_OUTPUT_ROOT}"

PYTHONPATH="${SLIME_DIR}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${PYTHONPATH:-}" \
"${PYTHON_BIN}" -m swe_env_pool_server \
    --host "${SWE_ENV_SERVER_BIND_HOST}" \
    --port "${SWE_ENV_SERVER_PORT}" \
    --exec-server-urls "${SWE_EXEC_SERVER_URLS}" \
    --max-containers-per-node "${SWE_MAX_CONTAINERS_PER_NODE}" \
    --max-total-leases "${SWE_POOL_MAX_TOTAL_LEASES}" \
    --max-concurrent-allocates "${SWE_POOL_MAX_CONCURRENT_ALLOCATES}" \
    --allocate-min-interval-sec "${SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC}" \
    --create-timeout-sec "${SWE_POOL_CREATE_TIMEOUT_SEC}" \
    > "${POOL_LOG}" 2>&1 &
SWE_POOL_PID=$!
echo "[replay-profile] started swe_env_pool_server pid=${SWE_POOL_PID}, log=${POOL_LOG}"

for _ in {1..60}; do
    if curl -fsS "${SWE_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
        echo "[replay-profile] SWE env pool server is ready: ${SWE_ENV_SERVER_URL}"
        break
    fi
    sleep 1
done

if ! curl -fsS "${SWE_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
    echo "ERROR: swe_env_pool_server failed to start. See ${POOL_LOG}" >&2
    exit 1
fi

IFS=',' read -r -a _exec_urls <<< "${SWE_EXEC_SERVER_URLS}"
for exec_url in "${_exec_urls[@]}"; do
    if ! curl -fsS --max-time 8 "${exec_url}/healthz" >/dev/null; then
        echo "ERROR: SWE exec server is not healthy: ${exec_url}/healthz" >&2
        exit 1
    fi
done

profile_args=(
    "${TRAJ_ROOT}"
    "--base-url" "${SWE_ENV_SERVER_URL}"
    "--max-concurrency" "${PROFILE_MAX_CONCURRENCY}"
    "--sample-interval-sec" "${PROFILE_SAMPLE_INTERVAL_SEC}"
    "--neighbor-count" "${PROFILE_NEIGHBOR_COUNT}"
    "--output-root" "${PROFILE_OUTPUT_ROOT}"
    "--output-json" "${PROFILE_OUTPUT_JSON}"
)

if [[ -n "${PROFILE_LIMIT}" ]]; then
    profile_args+=("--limit" "${PROFILE_LIMIT}")
fi
if [[ "${SIMULATE_LLM_DELAY}" == "1" ]]; then
    profile_args+=("--simulate-llm-delay")
else
    profile_args+=("--no-simulate-llm-delay")
fi

if [[ $# -gt 0 ]]; then
    profile_args+=("$@")
fi

echo "[replay-profile] launch command:"
printf '  %q' "${PYTHON_BIN}" "${SWE_RL_DIR}/tools/replay_swe_image_resource_profile.py" "${profile_args[@]}"
printf '\n'

PYTHONPATH="${SLIME_DIR}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${PYTHONPATH:-}" \
"${PYTHON_BIN}" "${SWE_RL_DIR}/tools/replay_swe_image_resource_profile.py" \
    "${profile_args[@]}" \
    2>&1 | tee "${PROFILE_LOG}"

echo "[replay-profile] done"
echo "[replay-profile] pool_log=${POOL_LOG}"
echo "[replay-profile] profile_log=${PROFILE_LOG}"
echo "[replay-profile] summary_json=${PROFILE_OUTPUT_JSON}"
echo "[replay-profile] output_root=${PROFILE_OUTPUT_ROOT}"
