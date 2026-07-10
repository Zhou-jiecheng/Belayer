#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWE_RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SWE_RL_DIR}/.." && pwd)"
SLIME_DIR="${REPO_ROOT}/slime"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAJ_ROOT="${TRAJ_ROOT:-${1:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/export/swe_rl_online_scheduler_adaptive_fast_restart_no_ckpt/swe_rollouts}"
if [[ -z "${TRAJ_ROOT}" ]]; then
    echo "Usage: $0 <trajectory_root> [extra replay args...]" >&2
    echo "Example: $0 ${REPO_ROOT}/export/swe_rollouts_profile_20260325_093408 --limit 32" >&2
    exit 2
fi
if [[ $# -gt 0 ]]; then
    shift
fi

export SWE_ENV_SERVER_BIND_HOST="${SWE_ENV_SERVER_BIND_HOST:-127.0.0.1}"
export SWE_ENV_SERVER_PORT="${SWE_ENV_SERVER_PORT:-18090}"
export SWE_ENV_SERVER_HOST="${SWE_ENV_SERVER_HOST:-127.0.0.1}"
export SWE_ENV_SERVER_URL="${SWE_ENV_SERVER_URL:-http://${SWE_ENV_SERVER_HOST}:${SWE_ENV_SERVER_PORT}}"
export SWE_EXEC_SERVER_URLS="${SWE_EXEC_SERVER_URLS:-http://100.103.28.35:5000}"

export SWE_MAX_CONTAINERS_PER_NODE="${SWE_MAX_CONTAINERS_PER_NODE:-40}"
export SWE_POOL_MAX_TOTAL_LEASES="${SWE_POOL_MAX_TOTAL_LEASES:-40}"
export SWE_POOL_MAX_CONCURRENT_ALLOCATES="${SWE_POOL_MAX_CONCURRENT_ALLOCATES:-8}"
export SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC="${SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC:-0.05}"
export SWE_POOL_CREATE_TIMEOUT_SEC="${SWE_POOL_CREATE_TIMEOUT_SEC:-180}"

EXPERIMENT_LIMIT="${EXPERIMENT_LIMIT:-128}"
EXPERIMENT_MAX_CONCURRENCY="${EXPERIMENT_MAX_CONCURRENCY:-128}"
EXPERIMENT_INJECTION_PROBABILITY="${EXPERIMENT_INJECTION_PROBABILITY:-0.01}"
EXPERIMENT_INJECTION_SEED="${EXPERIMENT_INJECTION_SEED:-20260407}"
SIMULATE_LLM_DELAY="${SIMULATE_LLM_DELAY:-1}"
POLICIES="${POLICIES:-adaptive-risk oracle-no-fault-no-checkpoint never always every-3}" #oracle-no-fault-no-checkpoint never always
EXPERIMENT_OUTPUT_ROOT="${EXPERIMENT_OUTPUT_ROOT:-${REPO_ROOT}/export/checkpoint_policy_fault_experiment_$(date +%Y%m%d_%H%M%S)}"
ADAPTIVE_RISK_CHECKPOINT_OVERHEAD=13
ADAPTIVE_FAILURE_PROB=0.01

mkdir -p "${EXPERIMENT_OUTPUT_ROOT}"
POOL_LOG="${POOL_LOG:-${EXPERIMENT_OUTPUT_ROOT}/swe_env_pool_server.log}"
EXPERIMENT_LOG="${EXPERIMENT_LOG:-${EXPERIMENT_OUTPUT_ROOT}/experiment.log}"

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

echo "[replay-exp] output_root=${EXPERIMENT_OUTPUT_ROOT}"
echo "[replay-exp] trajectory_root=${TRAJ_ROOT}"
echo "[replay-exp] exec_server_urls=${SWE_EXEC_SERVER_URLS}"
echo "[replay-exp] env_pool_url=${SWE_ENV_SERVER_URL}"

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
    "--limit" "${EXPERIMENT_LIMIT}"
    "--max-concurrency" "${EXPERIMENT_MAX_CONCURRENCY}"
    "--injection-probability" "${EXPERIMENT_INJECTION_PROBABILITY}"
    "--injection-seed" "${EXPERIMENT_INJECTION_SEED}"
    "--output-root" "${EXPERIMENT_OUTPUT_ROOT}"
    "--adaptive-checkpoint-budget-sec" "${ADAPTIVE_RISK_CHECKPOINT_OVERHEAD}"
    "--policies"
)

for policy in ${POLICIES}; do
    ARGS+=("${policy}")
done

if [[ "${SIMULATE_LLM_DELAY}" == "1" ]]; then
    ARGS+=("--simulate-llm-delay")
fi

if [[ $# -gt 0 ]]; then
    ARGS+=("$@")
fi

echo "[replay-exp] launching replay_swe_checkpoint_fault_experiment.py"
PYTHONPATH="${SLIME_DIR}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${PYTHONPATH:-}" \
"${PYTHON_BIN}" "${SWE_RL_DIR}/tools/replay_swe_checkpoint_fault_experiment.py" \
    "${ARGS[@]}" \
    2>&1 | tee "${EXPERIMENT_LOG}"

echo "[replay-exp] done"
echo "[replay-exp] pool_log=${POOL_LOG}"
echo "[replay-exp] experiment_log=${EXPERIMENT_LOG}"
echo "[replay-exp] output_root=${EXPERIMENT_OUTPUT_ROOT}"
