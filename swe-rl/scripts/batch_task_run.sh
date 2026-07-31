#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Fixed rollout target for all batch jobs:
# - stop after one rollout
# - test 64 prompt groups x 4 samples/group = 256 total samples
# - disable oversampling beyond the target group count
# - disable rolling refill
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
export SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL="${SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL:-0}"

TARGET_TOTAL_SAMPLES=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))

if [[ "${OVER_SAMPLING_BATCH_SIZE}" -ne "${ROLLOUT_BATCH_SIZE}" ]]; then
    echo "ERROR: this batch benchmark requires OVER_SAMPLING_BATCH_SIZE == ROLLOUT_BATCH_SIZE"
    echo "       got OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE}, ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}"
    exit 1
fi

if [[ "${SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL}" != "0" ]]; then
    echo "ERROR: this batch benchmark requires SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL=0"
    exit 1
fi

run_and_log() {
    local log_file="$1"
    shift
    "$@" 2>&1 | tee "${log_file}"
}

# echo "[batch-task-run] NUM_ROLLOUT=${NUM_ROLLOUT}"
# echo "[batch-task-run] ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}"
# echo "[batch-task-run] N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}"
# echo "[batch-task-run] OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE}"
# echo "[batch-task-run] TARGET_TOTAL_SAMPLES_PER_ROLLOUT=${TARGET_TOTAL_SAMPLES}"
# echo "[batch-task-run] SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL=${SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL}"

# run_and_log static_sync_64_concurrency_64x4_no_oversampling_no_refill.log \
#     env STATIC_MAX_CONCURRENCY="${STATIC_MAX_CONCURRENCY_64:-64}" \
#         CONTROL_PLANE_CONCURRENCY="${CONTROL_PLANE_CONCURRENCY_64:-64}" \
#         NUM_ROLLOUT="${NUM_ROLLOUT}" \
#         ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" \
#         N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT}" \
#         OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE}" \
#         SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL="${SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL}" \
#     bash "${SCRIPT_DIR}/run_swe_rl_static_rollout_only_sync.sh"

# sleep "${SLEEP_BETWEEN_RUNS_SEC:-10}"

run_and_log static_sync_128_concurrency_64x4_no_oversampling_no_refill.log \
    env STATIC_MAX_CONCURRENCY="${STATIC_MAX_CONCURRENCY_128:-128}" \
        CONTROL_PLANE_CONCURRENCY="${CONTROL_PLANE_CONCURRENCY_128:-128}" \
        NUM_ROLLOUT="${NUM_ROLLOUT}" \
        ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" \
        N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT}" \
        OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE}" \
        SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL="${SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL}" \
    bash "${SCRIPT_DIR}/run_swe_rl_static_rollout_only_sync.sh"
