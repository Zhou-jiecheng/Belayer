#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SWE_RL_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SWE_RL_DIR}/.." &>/dev/null && pwd)"

TS="$(date +%Y%m%d_%H%M%S)"
BATCH_ROOT_DEFAULT="${REPO_ROOT}/export/swe_rollout_checkpoint_batch_4policies_${TS}"
STDOUT_LOG_ROOT_DEFAULT="${SWE_RL_DIR}/logs"

export BATCH_EXPORT_ROOT="${BATCH_EXPORT_ROOT:-${BATCH_ROOT_DEFAULT}}"
export BATCH_LOG_DIR="${BATCH_LOG_DIR:-${STDOUT_LOG_ROOT_DEFAULT}}"

mkdir -p "${BATCH_EXPORT_ROOT}" "${BATCH_LOG_DIR}"

BATCH_LOG="${BATCH_LOG_DIR}/batch.log"
BATCH_ENV_LOG="${BATCH_LOG_DIR}/batch_env.log"
BATCH_STATUS_TSV="${BATCH_EXPORT_ROOT}/batch_status.tsv"

exec > >(tee -a "${BATCH_LOG}") 2>&1

echo "[checkpoint-batch] BATCH_EXPORT_ROOT=${BATCH_EXPORT_ROOT}"
echo "[checkpoint-batch] BATCH_LOG_DIR=${BATCH_LOG_DIR}"

# Keep batch-wide rollout shape fixed unless explicitly overridden.
export DEBUG_MODE="${DEBUG_MODE:-1}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
TARGET_TOTAL_SAMPLES="${TARGET_TOTAL_SAMPLES:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"

# Shared checkpoint / fault defaults. Per-policy overrides happen in the loop.
export SWE_ADAPTIVE_TAIL_ROOT="${SWE_ADAPTIVE_TAIL_ROOT:-${REPO_ROOT}/export/swe_rollouts_profile_20260325_083236}"
export SWE_ADAPTIVE_CHECKPOINT_BUDGET_SEC="${SWE_ADAPTIVE_CHECKPOINT_BUDGET_SEC:-6}"
export SWE_ADAPTIVE_DECISION_INTERVAL_SEC="${SWE_ADAPTIVE_DECISION_INTERVAL_SEC:-1.0}"
export SWE_ADAPTIVE_FAILURE_PROB="${SWE_ADAPTIVE_FAILURE_PROB:-0.01}"
export SWE_ADAPTIVE_MIN_DELTA_ENV_COST_SEC="${SWE_ADAPTIVE_MIN_DELTA_ENV_COST_SEC:-1}"
export SWE_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS="${SWE_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS:-4}"

export SWE_FAULT_INJECTION_PROB="${SWE_FAULT_INJECTION_PROB:-0.01}"

export STATIC_MAX_CONCURRENCY="${STATIC_MAX_CONCURRENCY:-256}"
export SWE_STATIC_CAPACITY_HEADROOM="${SWE_STATIC_CAPACITY_HEADROOM:-8}"
STATIC_CAPACITY_LIMIT=$((STATIC_MAX_CONCURRENCY + SWE_STATIC_CAPACITY_HEADROOM))

export SWE_MAX_CONCURRENT="${SWE_MAX_CONCURRENT:-${STATIC_CAPACITY_LIMIT}}"
export SWE_POOL_MAX_TOTAL_LEASES="${SWE_POOL_MAX_TOTAL_LEASES:-${STATIC_CAPACITY_LIMIT}}"
export SWE_MAX_CONTAINERS_PER_NODE="${SWE_MAX_CONTAINERS_PER_NODE:-${STATIC_CAPACITY_LIMIT}}"
export SWE_POOL_MAX_CONCURRENT_ALLOCATES="${SWE_POOL_MAX_CONCURRENT_ALLOCATES:-${STATIC_CAPACITY_LIMIT}}"
export SWE_MAX_CONCURRENT_DOCKER_CREATE="${SWE_MAX_CONCURRENT_DOCKER_CREATE:-${STATIC_CAPACITY_LIMIT}}"

export SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC="${SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC:-0.05}"
export SWE_DOCKER_CREATE_MIN_INTERVAL_SEC="${SWE_DOCKER_CREATE_MIN_INTERVAL_SEC:-0.05}"

export SWE_POOL_HEALTH_CHECK_FAILURE_THRESHOLD="${SWE_POOL_HEALTH_CHECK_FAILURE_THRESHOLD:-3}"

export SWE_ROLLOUT_TIMEOUT="${SWE_ROLLOUT_TIMEOUT:-1800}"
export SWE_AGENT_RUNTIME_TIMEOUT="${SWE_AGENT_RUNTIME_TIMEOUT:-${SWE_ROLLOUT_TIMEOUT}}"
export SWE_EVAL_TIMEOUT="${SWE_EVAL_TIMEOUT:-${SWE_ROLLOUT_TIMEOUT}}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

export SWE_BATCH_INTER_TASK_GC_ENABLE="${SWE_BATCH_INTER_TASK_GC_ENABLE:-1}"
export SWE_BATCH_INTER_TASK_GC_KEEP_LATEST="${SWE_BATCH_INTER_TASK_GC_KEEP_LATEST:-0}"
export SWE_BATCH_INTER_TASK_GC_DRAIN_TIMEOUT_SEC="${SWE_BATCH_INTER_TASK_GC_DRAIN_TIMEOUT_SEC:-180}"
export SWE_BATCH_INTER_TASK_GC_DRAIN_POLL_INTERVAL_SEC="${SWE_BATCH_INTER_TASK_GC_DRAIN_POLL_INTERVAL_SEC:-0.5}"

{
    echo "TS=${TS}"
    echo "BATCH_EXPORT_ROOT=${BATCH_EXPORT_ROOT}"
    echo "BATCH_LOG_DIR=${BATCH_LOG_DIR}"
    echo "DEBUG_MODE=${DEBUG_MODE}"
    echo "NUM_ROLLOUT=${NUM_ROLLOUT}"
    echo "ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}"
    echo "N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}"
    echo "OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE}"
    echo "TARGET_TOTAL_SAMPLES=${TARGET_TOTAL_SAMPLES}"
    echo "SWE_ADAPTIVE_TAIL_ROOT=${SWE_ADAPTIVE_TAIL_ROOT}"
    echo "SWE_ADAPTIVE_CHECKPOINT_BUDGET_SEC=${SWE_ADAPTIVE_CHECKPOINT_BUDGET_SEC}"
    echo "SWE_ADAPTIVE_DECISION_INTERVAL_SEC=${SWE_ADAPTIVE_DECISION_INTERVAL_SEC}"
    echo "SWE_ADAPTIVE_FAILURE_PROB=${SWE_ADAPTIVE_FAILURE_PROB}"
    echo "SWE_ADAPTIVE_MIN_DELTA_ENV_COST_SEC=${SWE_ADAPTIVE_MIN_DELTA_ENV_COST_SEC}"
    echo "SWE_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS=${SWE_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS}"
    echo "SWE_FAULT_INJECTION_PROB=${SWE_FAULT_INJECTION_PROB}"
    echo "STATIC_MAX_CONCURRENCY=${STATIC_MAX_CONCURRENCY}"
    echo "SWE_STATIC_CAPACITY_HEADROOM=${SWE_STATIC_CAPACITY_HEADROOM}"
    echo "STATIC_CAPACITY_LIMIT=${STATIC_CAPACITY_LIMIT}"
    echo "SWE_MAX_CONCURRENT=${SWE_MAX_CONCURRENT}"
    echo "SWE_POOL_MAX_TOTAL_LEASES=${SWE_POOL_MAX_TOTAL_LEASES}"
    echo "SWE_MAX_CONTAINERS_PER_NODE=${SWE_MAX_CONTAINERS_PER_NODE}"
    echo "SWE_POOL_MAX_CONCURRENT_ALLOCATES=${SWE_POOL_MAX_CONCURRENT_ALLOCATES}"
    echo "SWE_MAX_CONCURRENT_DOCKER_CREATE=${SWE_MAX_CONCURRENT_DOCKER_CREATE}"
    echo "SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC=${SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC}"
    echo "SWE_DOCKER_CREATE_MIN_INTERVAL_SEC=${SWE_DOCKER_CREATE_MIN_INTERVAL_SEC}"
    echo "SWE_POOL_HEALTH_CHECK_FAILURE_THRESHOLD=${SWE_POOL_HEALTH_CHECK_FAILURE_THRESHOLD}"
    echo "SWE_ROLLOUT_TIMEOUT=${SWE_ROLLOUT_TIMEOUT}"
    echo "SWE_AGENT_RUNTIME_TIMEOUT=${SWE_AGENT_RUNTIME_TIMEOUT}"
    echo "SWE_EVAL_TIMEOUT=${SWE_EVAL_TIMEOUT}"
    echo "SWE_BATCH_INTER_TASK_GC_ENABLE=${SWE_BATCH_INTER_TASK_GC_ENABLE}"
    echo "SWE_BATCH_INTER_TASK_GC_KEEP_LATEST=${SWE_BATCH_INTER_TASK_GC_KEEP_LATEST}"
    echo "SWE_BATCH_INTER_TASK_GC_DRAIN_TIMEOUT_SEC=${SWE_BATCH_INTER_TASK_GC_DRAIN_TIMEOUT_SEC}"
    echo "SWE_BATCH_INTER_TASK_GC_DRAIN_POLL_INTERVAL_SEC=${SWE_BATCH_INTER_TASK_GC_DRAIN_POLL_INTERVAL_SEC}"
    echo "SWE_ENV_SERVER_URL=${SWE_ENV_SERVER_URL:-}"
    echo "SWE_EXEC_SERVER_URLS=${SWE_EXEC_SERVER_URLS:-}"
} | tee "${BATCH_ENV_LOG}"

echo -e "policy\tfault_injection\texport_root\tstatus\tend_ts" > "${BATCH_STATUS_TSV}"

run_inter_task_checkpoint_gc() {
    local completed_policy="$1"
    local next_policy="$2"

    if [[ "${SWE_BATCH_INTER_TASK_GC_ENABLE}" == "0" ]]; then
        echo "[checkpoint-batch] inter-task checkpoint GC disabled after policy=${completed_policy}"
        return 0
    fi

    local exec_urls="${SWE_EXEC_SERVER_URLS:-}"
    if [[ -z "${exec_urls}" ]]; then
        echo "[checkpoint-batch] skipping inter-task checkpoint GC after policy=${completed_policy}: SWE_EXEC_SERVER_URLS is empty"
        return 0
    fi

    local gc_failed=0
    IFS=',' read -r -a exec_url_items <<< "${exec_urls}"
    for raw_url in "${exec_url_items[@]}"; do
        local exec_url="${raw_url//[[:space:]]/}"
        if [[ -z "${exec_url}" ]]; then
            continue
        fi

        echo "[checkpoint-batch] inter-task checkpoint GC start node=${exec_url} after policy=${completed_policy} before policy=${next_policy}"
        if ! curl -fsS -X POST \
            -H 'Content-Type: application/json' \
            -d "{\"keep_latest\": ${SWE_BATCH_INTER_TASK_GC_KEEP_LATEST}, \"dry_run\": false}" \
            "${exec_url%/}/container/checkpoint/gc"; then
            echo "[checkpoint-batch] inter-task checkpoint GC request failed node=${exec_url}"
            gc_failed=1
            continue
        fi

        if ! curl -fsS -X POST \
            -H 'Content-Type: application/json' \
            -d "{\"timeout_sec\": ${SWE_BATCH_INTER_TASK_GC_DRAIN_TIMEOUT_SEC}, \"poll_interval_sec\": ${SWE_BATCH_INTER_TASK_GC_DRAIN_POLL_INTERVAL_SEC}}" \
            "${exec_url%/}/container/checkpoint/gc/drain"; then
            echo "[checkpoint-batch] inter-task checkpoint GC drain failed node=${exec_url}"
            gc_failed=1
            continue
        fi

        echo "[checkpoint-batch] inter-task checkpoint GC done node=${exec_url}"
    done

    if [[ "${gc_failed}" != "0" ]]; then
        echo "[checkpoint-batch] inter-task checkpoint GC completed with failures after policy=${completed_policy}"
        return 1
    fi
    return 0
}

run_policy() {
    local policy="$1"
    local fault_injection_enable="$2"
    shift 2
    local policy_root="${BATCH_EXPORT_ROOT}/${policy}"
    local policy_stdout_log="${BATCH_LOG_DIR}/${policy}_concurrency_128_total_32_fault_inject_prob_001.log"

    mkdir -p "${policy_root}"

    echo "[checkpoint-batch] starting policy=${policy} fault_injection=${fault_injection_enable} export_root=${policy_root}"

    if env \
        EXPORT_ROOT="${policy_root}" \
        LOG_DIR="${policy_root}/logs" \
        SWE_SAVE_TRAJ_DIR="${policy_root}/rollouts" \
        SWE_CHECKPOINT_POLICY="${policy}" \
        SWE_FAULT_INJECTION_ENABLE="${fault_injection_enable}" \
        SWE_FAULT_INJECTION_PROB="${SWE_FAULT_INJECTION_PROB}" \
        DEBUG_MODE="${DEBUG_MODE}" \
        NUM_ROLLOUT="${NUM_ROLLOUT}" \
        ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" \
        N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT}" \
        OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE}" \
        TARGET_TOTAL_SAMPLES="${TARGET_TOTAL_SAMPLES}" \
        STATIC_MAX_CONCURRENCY="${STATIC_MAX_CONCURRENCY}" \
        SWE_MAX_CONCURRENT="${SWE_MAX_CONCURRENT}" \
        SWE_POOL_MAX_TOTAL_LEASES="${SWE_POOL_MAX_TOTAL_LEASES}" \
        SWE_MAX_CONTAINERS_PER_NODE="${SWE_MAX_CONTAINERS_PER_NODE}" \
        SWE_POOL_MAX_CONCURRENT_ALLOCATES="${SWE_POOL_MAX_CONCURRENT_ALLOCATES}" \
        SWE_MAX_CONCURRENT_DOCKER_CREATE="${SWE_MAX_CONCURRENT_DOCKER_CREATE}" \
        SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC="${SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC}" \
        SWE_DOCKER_CREATE_MIN_INTERVAL_SEC="${SWE_DOCKER_CREATE_MIN_INTERVAL_SEC}" \
        SWE_POOL_HEALTH_CHECK_FAILURE_THRESHOLD="${SWE_POOL_HEALTH_CHECK_FAILURE_THRESHOLD}" \
        SWE_ROLLOUT_TIMEOUT="${SWE_ROLLOUT_TIMEOUT}" \
        SWE_AGENT_RUNTIME_TIMEOUT="${SWE_AGENT_RUNTIME_TIMEOUT}" \
        SWE_EVAL_TIMEOUT="${SWE_EVAL_TIMEOUT}" \
        SWE_ADAPTIVE_TAIL_ROOT="${SWE_ADAPTIVE_TAIL_ROOT}" \
        SWE_ADAPTIVE_CHECKPOINT_BUDGET_SEC="${SWE_ADAPTIVE_CHECKPOINT_BUDGET_SEC}" \
        SWE_ADAPTIVE_DECISION_INTERVAL_SEC="${SWE_ADAPTIVE_DECISION_INTERVAL_SEC}" \
        SWE_ADAPTIVE_FAILURE_PROB="${SWE_ADAPTIVE_FAILURE_PROB}" \
        SWE_ADAPTIVE_MIN_DELTA_ENV_COST_SEC="${SWE_ADAPTIVE_MIN_DELTA_ENV_COST_SEC}" \
        SWE_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS="${SWE_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS}" \
        PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
        PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER}" \
        bash "${SCRIPT_DIR}/run_swe_rl_static_rollout_only_checkpoint_debug.sh" "$@" \
        2>&1 | tee "${policy_stdout_log}"; then
        echo -e "${policy}\t${fault_injection_enable}\t${policy_root}\tsuccess\t$(date +%Y%m%d_%H%M%S)" >> "${BATCH_STATUS_TSV}"
        echo "[checkpoint-batch] finished policy=${policy} status=success"
    else
        echo -e "${policy}\t${fault_injection_enable}\t${policy_root}\tfailed\t$(date +%Y%m%d_%H%M%S)" >> "${BATCH_STATUS_TSV}"
        echo "[checkpoint-batch] finished policy=${policy} status=failed"
        return 1
    fi
}

run_policy "adaptive-risk" "1" "$@"
run_policy "never" "1" "$@"
# run_policy "always" "1" "$@"
run_policy "oracle-no-fault-no-checkpoint" "1" "$@"

echo "[checkpoint-batch] all policies completed"
echo "[checkpoint-batch] status_tsv=${BATCH_STATUS_TSV}"
