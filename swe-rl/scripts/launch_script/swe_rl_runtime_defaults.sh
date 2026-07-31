#!/bin/bash

# Shared runtime defaults for the static SWE-RL launchers.
#
# This file is sourced by concrete launch scripts. It only exports rollout,
# checkpoint, control-plane, retry, and process runtime settings.

swe_rl_apply_runtime_defaults() {
    if [[ -z "${SWE_RL_DIR:-}" ]]; then
        echo "SWE_RL_DIR must be set before sourcing common launcher defaults." >&2
        return 1
    fi

    # Synchronous rollout defaults shared across launchers.
    export DEBUG_MODE="${DEBUG_MODE:-0}"
    export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
    export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
    export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
    export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
    export SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL="${SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL:-0}"
    export TARGET_TOTAL_SAMPLES="${TARGET_TOTAL_SAMPLES:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"

    if [[ "${OVER_SAMPLING_BATCH_SIZE}" -ne "${ROLLOUT_BATCH_SIZE}" ]]; then
        echo "Synchronous rollout requires OVER_SAMPLING_BATCH_SIZE == ROLLOUT_BATCH_SIZE, got ${OVER_SAMPLING_BATCH_SIZE} vs ${ROLLOUT_BATCH_SIZE}" >&2
        return 1
    fi

    if [[ "${SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL}" != "0" ]]; then
        echo "Synchronous rollout requires SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL=0, got ${SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL}" >&2
        return 1
    fi

    # Checkpoint policy and optional server-side fault injection.
    export SWE_CHECKPOINT_POLICY="${SWE_CHECKPOINT_POLICY:-adaptive-risk}"
    export SWE_ADAPTIVE_TAIL_ROOT="${SWE_ADAPTIVE_TAIL_ROOT:-${SWE_RL_DIR}/../export/swe_rollouts_profile_20260325_083236}"
    export SWE_ADAPTIVE_CHECKPOINT_BUDGET_SEC="${SWE_ADAPTIVE_CHECKPOINT_BUDGET_SEC:-7.0}"
    export SWE_ADAPTIVE_DECISION_INTERVAL_SEC="${SWE_ADAPTIVE_DECISION_INTERVAL_SEC:-2.0}"
    export SWE_ADAPTIVE_FAILURE_PROB="${SWE_ADAPTIVE_FAILURE_PROB:-0.01}"
    export SWE_ADAPTIVE_MIN_DELTA_ENV_COST_SEC="${SWE_ADAPTIVE_MIN_DELTA_ENV_COST_SEC:-30}"
    export SWE_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS="${SWE_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS:-4}"
    export SWE_FAULT_INJECTION_ENABLE="${SWE_FAULT_INJECTION_ENABLE:-0}"
    export SWE_FAULT_INJECTION_PROB="${SWE_FAULT_INJECTION_PROB:-0.003}"

    # Control-plane throughput and timeout guardrails.
    export SWE_POOL_MAX_CONCURRENT_ALLOCATES="${SWE_POOL_MAX_CONCURRENT_ALLOCATES:-8}"
    export SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC="${SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC:-0.05}"
    export SWE_POOL_CREATE_TIMEOUT_SEC="${SWE_POOL_CREATE_TIMEOUT_SEC:-1800}"
    export SWE_POOL_HEALTH_CHECK_FAILURE_THRESHOLD="${SWE_POOL_HEALTH_CHECK_FAILURE_THRESHOLD:-3}"
    export SWE_ROLLOUT_TIMEOUT="${SWE_ROLLOUT_TIMEOUT:-2400}"
    export SWE_AGENT_RUNTIME_TIMEOUT="${SWE_AGENT_RUNTIME_TIMEOUT:-${SWE_ROLLOUT_TIMEOUT}}"
    export SWE_EVAL_TIMEOUT="${SWE_EVAL_TIMEOUT:-${SWE_ROLLOUT_TIMEOUT}}"
    export SWE_ENV_HTTP_MAX_RETRIES="${SWE_ENV_HTTP_MAX_RETRIES:-10}"
    export SWE_ALLOCATE_HTTP_MAX_RETRIES="${SWE_ALLOCATE_HTTP_MAX_RETRIES:-1}"
    export SWE_ENV_APP_MAX_RETRIES="${SWE_ENV_APP_MAX_RETRIES:-3}"
    export SWE_ALLOCATE_APP_MAX_RETRIES="${SWE_ALLOCATE_APP_MAX_RETRIES:-360}"
    export SWE_ENV_APP_RETRY_DELAY_SEC="${SWE_ENV_APP_RETRY_DELAY_SEC:-1.0}"
    export SWE_ENV_APP_RETRY_JITTER_SEC="${SWE_ENV_APP_RETRY_JITTER_SEC:-0.2}"
    export SWE_ENV_APP_RETRY_MAX_DELAY_SEC="${SWE_ENV_APP_RETRY_MAX_DELAY_SEC:-5.0}"

    # Process runtime defaults.
    export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-1024}"
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
    export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
    export LITELLM_MODEL_REGISTRY_PATH="${LITELLM_MODEL_REGISTRY_PATH:-${SWE_RL_DIR}/litellm.json}"
    export MSWEA_DOCKER_EXEC_MODE="${MSWEA_DOCKER_EXEC_MODE:-api}"
    export RAY_health_check_failure_threshold="${RAY_health_check_failure_threshold:-20}"
    export RAY_health_check_period_ms="${RAY_health_check_period_ms:-5000}"
    export RAY_health_check_timeout_ms="${RAY_health_check_timeout_ms:-30000}"
    export RAY_num_heartbeats_timeout="${RAY_num_heartbeats_timeout:-60}"
    export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
}
