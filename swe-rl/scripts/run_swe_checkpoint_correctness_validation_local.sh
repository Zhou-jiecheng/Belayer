#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWE_RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SWE_RL_DIR}/.." && pwd)"
SLIME_DIR="${REPO_ROOT}/slime"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAJ_ROOT="${TRAJ_ROOT:-${1:-${REPO_ROOT}/export/swe_rl_static_baseline_128_no_ckpt/swe_rollouts}}"
if [[ $# -gt 0 ]]; then
    shift
fi

export SWE_ENV_SERVER_BIND_HOST="${SWE_ENV_SERVER_BIND_HOST:-127.0.0.1}"
export SWE_ENV_SERVER_PORT="${SWE_ENV_SERVER_PORT:-18090}"
export SWE_ENV_SERVER_HOST="${SWE_ENV_SERVER_HOST:-127.0.0.1}"
export SWE_ENV_SERVER_URL="${SWE_ENV_SERVER_URL:-http://${SWE_ENV_SERVER_HOST}:${SWE_ENV_SERVER_PORT}}"
export SWE_EXEC_SERVER_URLS="${SWE_EXEC_SERVER_URLS:-http://100.103.28.35:5000}"

export SWE_MAX_CONTAINERS_PER_NODE="${SWE_MAX_CONTAINERS_PER_NODE:-128}"
export SWE_POOL_MAX_TOTAL_LEASES="${SWE_POOL_MAX_TOTAL_LEASES:-128}"
export SWE_POOL_MAX_CONCURRENT_ALLOCATES="${SWE_POOL_MAX_CONCURRENT_ALLOCATES:-4}"
export SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC="${SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC:-0.05}"
export SWE_POOL_CREATE_TIMEOUT_SEC="${SWE_POOL_CREATE_TIMEOUT_SEC:-180}"

VALIDATION_OUTPUT_ROOT="${VALIDATION_OUTPUT_ROOT:-${REPO_ROOT}/export/checkpoint_correctness_validation_$(date +%Y%m%d_%H%M%S)}"
VALIDATION_ACTION_SLEEP_SEC="${VALIDATION_ACTION_SLEEP_SEC:-1.5}"
VALIDATION_EXEC_TIMEOUT="${VALIDATION_EXEC_TIMEOUT:-180}"
VALIDATION_RERUN_TIMEOUT="${VALIDATION_RERUN_TIMEOUT:-180}"
VALIDATION_PHASES="${VALIDATION_PHASES:-before_action mid_action after_action_before_observation after_observation_before_checkpoint before_commit after_commit_before_ready after_checkpoint_ready}"
VALIDATION_RANDOM_TRIALS="${VALIDATION_RANDOM_TRIALS:-0}"
VALIDATION_RANDOM_SEED="${VALIDATION_RANDOM_SEED:-20260521}"
VALIDATION_RANDOM_MIN_DELAY_SEC="${VALIDATION_RANDOM_MIN_DELAY_SEC:-0.0}"
VALIDATION_RANDOM_MAX_DELAY_SEC="${VALIDATION_RANDOM_MAX_DELAY_SEC:-8.0}"
VALIDATION_RANDOM_CHECKPOINT_INTERRUPT_PROBABILITY="${VALIDATION_RANDOM_CHECKPOINT_INTERRUPT_PROBABILITY:-0.25}"
VALIDATION_RANDOM_CHECKPOINT_INTERRUPT_PHASES="${VALIDATION_RANDOM_CHECKPOINT_INTERRUPT_PHASES:-before_commit after_commit_before_ready}"
VALIDATION_RANDOM_CHECKPOINT_INTERRUPT_DELAY_SEC="${VALIDATION_RANDOM_CHECKPOINT_INTERRUPT_DELAY_SEC:-0.0}"

mkdir -p "${VALIDATION_OUTPUT_ROOT}"
POOL_LOG="${POOL_LOG:-${VALIDATION_OUTPUT_ROOT}/swe_env_pool_server.log}"
VALIDATION_LOG="${VALIDATION_LOG:-${VALIDATION_OUTPUT_ROOT}/validation.log}"

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

echo "[correctness] output_root=${VALIDATION_OUTPUT_ROOT}"
echo "[correctness] trajectory_root=${TRAJ_ROOT}"
echo "[correctness] exec_server_urls=${SWE_EXEC_SERVER_URLS}"
echo "[correctness] env_pool_url=${SWE_ENV_SERVER_URL}"

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

for _ in {1..60}; do
    if curl -fsS "${SWE_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! curl -fsS "${SWE_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
    echo "ERROR: swe_env_pool_server failed to start. See ${POOL_LOG}" >&2
    exit 1
fi

ARGS=(
    "${TRAJ_ROOT}"
    "--base-url" "${SWE_ENV_SERVER_URL}"
    "--output-root" "${VALIDATION_OUTPUT_ROOT}"
    "--exec-timeout" "${VALIDATION_EXEC_TIMEOUT}"
    "--rerun-timeout" "${VALIDATION_RERUN_TIMEOUT}"
    "--action-sleep-sec" "${VALIDATION_ACTION_SLEEP_SEC}"
    "--random-trials" "${VALIDATION_RANDOM_TRIALS}"
    "--random-seed" "${VALIDATION_RANDOM_SEED}"
    "--random-min-delay-sec" "${VALIDATION_RANDOM_MIN_DELAY_SEC}"
    "--random-max-delay-sec" "${VALIDATION_RANDOM_MAX_DELAY_SEC}"
    "--random-checkpoint-interrupt-probability" "${VALIDATION_RANDOM_CHECKPOINT_INTERRUPT_PROBABILITY}"
    "--random-checkpoint-interrupt-delay-sec" "${VALIDATION_RANDOM_CHECKPOINT_INTERRUPT_DELAY_SEC}"
    "--random-checkpoint-interrupt-phases"
)

for phase in ${VALIDATION_RANDOM_CHECKPOINT_INTERRUPT_PHASES}; do
    ARGS+=("${phase}")
done

ARGS+=(
    "--phases"
)

for phase in ${VALIDATION_PHASES}; do
    ARGS+=("${phase}")
done

if [[ -n "${VALIDATION_IMAGE_NAME:-}" ]]; then
    ARGS+=("--image-name" "${VALIDATION_IMAGE_NAME}")
fi

if [[ -n "${VALIDATION_INSTANCE_ID:-}" ]]; then
    ARGS+=("--instance-id" "${VALIDATION_INSTANCE_ID}")
fi

if [[ "${VALIDATION_RANDOM_ONLY:-0}" == "1" ]]; then
    ARGS+=("--random-only")
fi

if [[ $# -gt 0 ]]; then
    ARGS+=("$@")
fi

PYTHONPATH="${SLIME_DIR}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${PYTHONPATH:-}" \
"${PYTHON_BIN}" "${SWE_RL_DIR}/tools/validate_swe_checkpoint_correctness.py" \
    "${ARGS[@]}" \
    2>&1 | tee "${VALIDATION_LOG}"

echo "[correctness] done"
echo "[correctness] pool_log=${POOL_LOG}"
echo "[correctness] validation_log=${VALIDATION_LOG}"
echo "[correctness] output_root=${VALIDATION_OUTPUT_ROOT}"
