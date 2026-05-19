#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWE_RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SWE_RL_DIR}/.." && pwd)"
SLIME_DIR="${REPO_ROOT}/slime"

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/miniconda/envs/osworld/bin/python}"
TRAJ_ROOT="${TRAJ_ROOT:-${REPO_ROOT}/export/swe_rollouts_profile_20260325_093408}"

export SWE_ENV_SERVER_BIND_HOST="${SWE_ENV_SERVER_BIND_HOST:-0.0.0.0}"
export SWE_ENV_SERVER_PORT="${SWE_ENV_SERVER_PORT:-18090}"
export SWE_ENV_SERVER_HOST="${SWE_ENV_SERVER_HOST:-127.0.0.1}"
export SWE_ENV_SERVER_URL="${SWE_ENV_SERVER_URL:-http://${SWE_ENV_SERVER_HOST}:${SWE_ENV_SERVER_PORT}}"
export SWE_EXEC_SERVER_URLS="${SWE_EXEC_SERVER_URLS:-http://100.101.233.34:5000}"
export SWE_MAX_CONTAINERS_PER_NODE="${SWE_MAX_CONTAINERS_PER_NODE:-128}"
export SWE_POOL_MAX_TOTAL_LEASES="${SWE_POOL_MAX_TOTAL_LEASES:-0}"
export SWE_POOL_MAX_CONCURRENT_ALLOCATES="${SWE_POOL_MAX_CONCURRENT_ALLOCATES:-8}"
export SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC="${SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC:-0.05}"
export SWE_POOL_CREATE_TIMEOUT_SEC="${SWE_POOL_CREATE_TIMEOUT_SEC:-180}"

REPLAY_MAX_CONCURRENCY="${REPLAY_MAX_CONCURRENCY:-32}"
REPLAY_LIMIT="${REPLAY_LIMIT:-}"
SIMULATE_LLM_DELAY="${SIMULATE_LLM_DELAY:-1}"
REPLAY_OUTPUT_ROOT="${REPLAY_OUTPUT_ROOT:-${REPO_ROOT}/export/checkpoint_replay_debug_$(date +%Y%m%d_%H%M%S)}"
REPLAY_OUTPUT_JSON="${REPLAY_OUTPUT_JSON:-${REPLAY_OUTPUT_ROOT}/summary.json}"
REPLAY_OUTPUT_DIR="${REPLAY_OUTPUT_DIR:-${REPLAY_OUTPUT_ROOT}/per_traj}"

CHECKPOINT_AFTER_STEP="${CHECKPOINT_AFTER_STEP:-}"
WAIT_CHECKPOINT_READY="${WAIT_CHECKPOINT_READY:-0}"
WAIT_CHECKPOINT_READY_TIMEOUT="${WAIT_CHECKPOINT_READY_TIMEOUT:-300}"
CHECKPOINT_POLL_INTERVAL="${CHECKPOINT_POLL_INTERVAL:-1.0}"
CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-manual-replay}"
CHECKPOINT_REASON="${CHECKPOINT_REASON:-traj_replay}"
RERUN_AFTER_STEP="${RERUN_AFTER_STEP:-}"
RERUN_TIMEOUT="${RERUN_TIMEOUT:-120}"
GC_KEEP_LATEST="${GC_KEEP_LATEST:-}"
GC_DRY_RUN="${GC_DRY_RUN:-0}"
KEEP_LEASE_OPEN="${KEEP_LEASE_OPEN:-0}"

mkdir -p "${REPLAY_OUTPUT_ROOT}" "${REPLAY_OUTPUT_DIR}"
POOL_LOG="${POOL_LOG:-${REPLAY_OUTPUT_ROOT}/swe_env_pool_server.log}"
REPLAY_LOG="${REPLAY_LOG:-${REPLAY_OUTPUT_ROOT}/replay.log}"

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

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "TRAJ_ROOT=${TRAJ_ROOT}"
echo "SWE_ENV_SERVER_URL=${SWE_ENV_SERVER_URL}"
echo "SWE_EXEC_SERVER_URLS=${SWE_EXEC_SERVER_URLS}"
echo "REPLAY_OUTPUT_ROOT=${REPLAY_OUTPUT_ROOT}"

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
echo "Started swe_env_pool_server pid=${SWE_POOL_PID}, log=${POOL_LOG}"

for i in {1..60}; do
    if curl -fsS "${SWE_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
        echo "SWE env pool server is ready: ${SWE_ENV_SERVER_URL}"
        break
    fi
    sleep 2
done

if ! curl -fsS "${SWE_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
    echo "ERROR: SWE env pool server failed to start: ${SWE_ENV_SERVER_URL}/healthz"
    echo "Check log: ${POOL_LOG}"
    exit 1
fi

IFS=',' read -r -a _exec_urls <<< "${SWE_EXEC_SERVER_URLS}"
for exec_url in "${_exec_urls[@]}"; do
    if ! curl -fsS --max-time 8 "${exec_url}/healthz" >/dev/null; then
        echo "ERROR: SWE exec server is not healthy: ${exec_url}/healthz"
        exit 1
    fi
done

replay_args=(
    "${TRAJ_ROOT}"
    "--base-url" "${SWE_ENV_SERVER_URL}"
    "--max-concurrency" "${REPLAY_MAX_CONCURRENCY}"
    "--output-json" "${REPLAY_OUTPUT_JSON}"
    "--output-dir" "${REPLAY_OUTPUT_DIR}"
)

if [[ -n "${REPLAY_LIMIT}" ]]; then
    replay_args+=("--limit" "${REPLAY_LIMIT}")
fi
if [[ "${SIMULATE_LLM_DELAY}" == "1" ]]; then
    replay_args+=("--simulate-llm-delay")
fi
if [[ -n "${CHECKPOINT_AFTER_STEP}" ]]; then
    IFS=',' read -r -a _ckpt_steps <<< "${CHECKPOINT_AFTER_STEP}"
    for step in "${_ckpt_steps[@]}"; do
        replay_args+=("--checkpoint-after-step" "${step}")
    done
fi
if [[ "${WAIT_CHECKPOINT_READY}" == "1" ]]; then
    replay_args+=("--wait-checkpoint-ready")
    replay_args+=("--wait-checkpoint-ready-timeout" "${WAIT_CHECKPOINT_READY_TIMEOUT}")
    replay_args+=("--checkpoint-poll-interval" "${CHECKPOINT_POLL_INTERVAL}")
fi
if [[ -n "${RERUN_AFTER_STEP}" ]]; then
    replay_args+=("--rerun-after-step" "${RERUN_AFTER_STEP}")
    replay_args+=("--rerun-timeout" "${RERUN_TIMEOUT}")
fi
if [[ -n "${GC_KEEP_LATEST}" ]]; then
    replay_args+=("--gc-keep-latest" "${GC_KEEP_LATEST}")
fi
if [[ "${GC_DRY_RUN}" == "1" ]]; then
    replay_args+=("--gc-dry-run")
fi
if [[ "${KEEP_LEASE_OPEN}" == "1" ]]; then
    replay_args+=("--keep-lease-open")
fi
replay_args+=("--checkpoint-policy" "${CHECKPOINT_POLICY}")
replay_args+=("--checkpoint-reason" "${CHECKPOINT_REASON}")

if [[ $# -gt 0 ]]; then
    replay_args+=("$@")
fi

echo "Replay command:"
printf '  %q' "${PYTHON_BIN}" "${SWE_RL_DIR}/tools/replay_swe_traj_checkpoint.py" "${replay_args[@]}"
printf '\n'

PYTHONPATH="${SLIME_DIR}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${PYTHONPATH:-}" \
"${PYTHON_BIN}" "${SWE_RL_DIR}/tools/replay_swe_traj_checkpoint.py" \
    "${replay_args[@]}" \
    2>&1 | tee "${REPLAY_LOG}"

echo "Replay finished. Summary: ${REPLAY_OUTPUT_JSON}"
echo "Per-trajectory reports: ${REPLAY_OUTPUT_DIR}"
