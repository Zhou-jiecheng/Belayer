#!/bin/bash

# Single-node rollout-only debug launcher that combines:
# 1. the SWE online env docker scheduler, and
# 2. the adaptive-risk checkpoint policy.
#
# This stays close to the existing rollout-only debug scripts, but centralizes the
# shared scheduler/checkpoint settings so the same defaults can be reused by the
# multi-node training launcher.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SWE_RL_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SWE_RL_DIR}/../slime" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SWE_RL_DIR}/.." &>/dev/null && pwd)"

source "${SCRIPT_DIR}/swe_rl_online_scheduler_adaptive_checkpoint_common.sh"

# Debug profile: keep rollout-only execution with synchronous rollout settings.
export DEBUG_MODE="${DEBUG_MODE:-1}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
export SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL="${SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL:-0}"
export TRAIN_ASYNC_ENTRY="${TRAIN_ASYNC_ENTRY:-${SLIME_DIR}/train_rollout_only_sync.py}"
export SLIME_EXTRA_ARGS="${SLIME_EXTRA_ARGS:---debug-rollout-only --sglang-server-concurrency ${SGLANG_SERVER_CONCURRENCY:-1024}}"

swe_rl_apply_online_scheduler_adaptive_checkpoint_defaults
swe_rl_print_online_scheduler_adaptive_checkpoint_summary "[swe-rollout-only]"

TS="$(date +%Y%m%d_%H%M%S)"
DEBUG_ROOT_DEFAULT="${REPO_ROOT}/export/swe_online_sched_adaptive_ckpt_debug_${TS}"
export EXPORT_ROOT="${EXPORT_ROOT:-${DEBUG_ROOT_DEFAULT}}"
export LOG_DIR="${LOG_DIR:-${EXPORT_ROOT}/logs}"
export SWE_SAVE_TRAJ_DIR="${SWE_SAVE_TRAJ_DIR:-${EXPORT_ROOT}/rollouts}"
mkdir -p "${EXPORT_ROOT}" "${LOG_DIR}" "${SWE_SAVE_TRAJ_DIR}"

echo "[swe-rollout-only] EXPORT_ROOT=${EXPORT_ROOT}"
echo "[swe-rollout-only] LOG_DIR=${LOG_DIR}"
echo "[swe-rollout-only] SWE_SAVE_TRAJ_DIR=${SWE_SAVE_TRAJ_DIR}"
echo "[swe-rollout-only] TRAIN_ASYNC_ENTRY=${TRAIN_ASYNC_ENTRY}"
echo "[swe-rollout-only] SLIME_EXTRA_ARGS=${SLIME_EXTRA_ARGS}"

exec bash "${SWE_RL_DIR}/scripts/run_swe_rl.sh" "$@"
