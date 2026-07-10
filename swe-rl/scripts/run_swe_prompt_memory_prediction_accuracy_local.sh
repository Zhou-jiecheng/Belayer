#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWE_RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SWE_RL_DIR}/.." && pwd)"

DEFAULT_PYTHON_BIN="/mnt/shared-storage-user/ailab-sys/zhoujiecheng/miniconda/envs/osworld/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

TRAJ_ROOT="${TRAJ_ROOT:-${1:-${REPO_ROOT}/export/E2E/swe_rl_online_scheduler_adaptive_fast_restart_no_ckpt/swe_rollouts}}"
if [[ $# -gt 0 ]]; then
    shift
fi

SOURCE_N_SAMPLES_PER_PROMPT="${SOURCE_N_SAMPLES_PER_PROMPT:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
TRAIN_SAMPLE_SLOT_START="${TRAIN_SAMPLE_SLOT_START:-0}"
TRAIN_SAMPLE_SLOT_END="${TRAIN_SAMPLE_SLOT_END:-4}"
VALID_SAMPLE_SLOT_START="${VALID_SAMPLE_SLOT_START:-${TRAIN_SAMPLE_SLOT_END}}"
VALID_SAMPLE_SLOT_END="${VALID_SAMPLE_SLOT_END:-${SOURCE_N_SAMPLES_PER_PROMPT}}"

EXPERIMENT_ROOT="${EXPERIMENT_OUTPUT_ROOT:-${REPO_ROOT}/export/prompt_memory_prediction_accuracy_$(date +%Y%m%d_%H%M%S)}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${EXPERIMENT_ROOT}/train_slots_${TRAIN_SAMPLE_SLOT_START}_${TRAIN_SAMPLE_SLOT_END}}"
VALID_OUTPUT_ROOT="${VALID_OUTPUT_ROOT:-${EXPERIMENT_ROOT}/valid_slots_${VALID_SAMPLE_SLOT_START}_${VALID_SAMPLE_SLOT_END}}"
PROFILE_JSON="${PROFILE_JSON:-${EXPERIMENT_ROOT}/repo_resource_stats.json}"
ANALYSIS_JSON="${ANALYSIS_JSON:-${EXPERIMENT_ROOT}/memory_prediction_accuracy.json}"
ANALYSIS_CSV="${ANALYSIS_CSV:-${EXPERIMENT_ROOT}/memory_prediction_rows.csv}"
SCATTER_PREFIX="${SCATTER_PREFIX:-${EXPERIMENT_ROOT}/prompt_resource_prediction}"

mkdir -p "${EXPERIMENT_ROOT}"

export PYTHON_BIN
export SWE_REPLAY_SEED="${SWE_REPLAY_SEED:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SWE_REPLAY_SEED}}"
export SWE_SCHED_RANDOM_SEED="${SWE_SCHED_RANDOM_SEED:-${SWE_REPLAY_SEED}}"
export SWE_REPLAY_SOURCE_N_SAMPLES_PER_PROMPT="${SWE_REPLAY_SOURCE_N_SAMPLES_PER_PROMPT:-${SOURCE_N_SAMPLES_PER_PROMPT}}"
export REORDER_MODE="${REORDER_MODE:-breadth-first}"

export SWE_ENABLE_ONLINE_ENV_DOCKER_SCHEDULER=1
export SWE_SCHED_USE_RESOURCE_STATS_DIR=0
export SWE_REPO_RESOURCE_STATS_PATH="${PROFILE_JSON}"
export SWE_SCHED_MEMORY_PEAK_SCALE="${SWE_SCHED_MEMORY_PEAK_SCALE:-1}"
export SWE_SCHED_MIN_PROFILE_MEMORY_BYTES="${SWE_SCHED_MIN_PROFILE_MEMORY_BYTES:-0}"
export SWE_SCHED_MIN_LIVE_PROFILE_SAMPLES="${SWE_SCHED_MIN_LIVE_PROFILE_SAMPLES:-1}"
export SWE_SCHED_DISABLE_LIVE_STATS_POLLING="${SWE_SCHED_DISABLE_LIVE_STATS_POLLING:-0}"
export SWE_REPLAY_DISABLE_BATCH_SHAPE_FILTER="${SWE_REPLAY_DISABLE_BATCH_SHAPE_FILTER:-0}"

cat > "${EXPERIMENT_ROOT}/experiment_config.env" <<EOF
TRAJ_ROOT=${TRAJ_ROOT}
SOURCE_N_SAMPLES_PER_PROMPT=${SOURCE_N_SAMPLES_PER_PROMPT}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}
TRAIN_SAMPLE_SLOT_START=${TRAIN_SAMPLE_SLOT_START}
TRAIN_SAMPLE_SLOT_END=${TRAIN_SAMPLE_SLOT_END}
VALID_SAMPLE_SLOT_START=${VALID_SAMPLE_SLOT_START}
VALID_SAMPLE_SLOT_END=${VALID_SAMPLE_SLOT_END}
SWE_REPLAY_SEED=${SWE_REPLAY_SEED}
PROFILE_JSON=${PROFILE_JSON}
EOF

echo "[memory-prediction] root=${EXPERIMENT_ROOT}"
echo "[memory-prediction] profile=${PROFILE_JSON}"
echo "[memory-prediction] train slots=[${TRAIN_SAMPLE_SLOT_START}, ${TRAIN_SAMPLE_SLOT_END})"
echo "[memory-prediction] valid slots=[${VALID_SAMPLE_SLOT_START}, ${VALID_SAMPLE_SLOT_END})"

echo "[memory-prediction] running train replay"
SWE_SCHED_ENABLE_LIVE_PROFILE_UPDATES=1 \
N_SAMPLES_PER_PROMPT="${TRAIN_SAMPLE_SLOT_END}" \
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" \
EXPERIMENT_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT}" \
bash "${SCRIPT_DIR}/run_swe_replay_online_scheduler_experiment_local.sh" \
    "${TRAJ_ROOT}" \
    --sample-slot-start "${TRAIN_SAMPLE_SLOT_START}" \
    --sample-slot-end "${TRAIN_SAMPLE_SLOT_END}" \
    "$@"

if [[ ! -s "${PROFILE_JSON}" ]]; then
    echo "ERROR: profile json was not created: ${PROFILE_JSON}" >&2
    exit 1
fi

echo "[memory-prediction] running frozen validation replay"
SWE_SCHED_ENABLE_LIVE_PROFILE_UPDATES=0 \
N_SAMPLES_PER_PROMPT="${SOURCE_N_SAMPLES_PER_PROMPT}" \
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" \
EXPERIMENT_OUTPUT_ROOT="${VALID_OUTPUT_ROOT}" \
bash "${SCRIPT_DIR}/run_swe_replay_online_scheduler_experiment_local.sh" \
    "${TRAJ_ROOT}" \
    --sample-slot-start "${VALID_SAMPLE_SLOT_START}" \
    --sample-slot-end "${VALID_SAMPLE_SLOT_END}" \
    "$@"

echo "[memory-prediction] analyzing validation log"
"${PYTHON_BIN}" "${SWE_RL_DIR}/tools/analyze_prompt_memory_prediction_accuracy.py" \
    "${VALID_OUTPUT_ROOT}/experiment.log" \
    --output-json "${ANALYSIS_JSON}" \
    --output-csv "${ANALYSIS_CSV}" \
    --exclude-cold-start

echo "[memory-prediction] plotting prediction scatter"
"${PYTHON_BIN}" "${SWE_RL_DIR}/tools/plot_prompt_resource_prediction_scatter.py" \
    "${VALID_OUTPUT_ROOT}/experiment.log" \
    --output-prefix "${SCATTER_PREFIX}" \
    --exclude-cold-start \
    --min-valid-leases 1

echo "[memory-prediction] done"
echo "[memory-prediction] train=${TRAIN_OUTPUT_ROOT}"
echo "[memory-prediction] validation=${VALID_OUTPUT_ROOT}"
echo "[memory-prediction] analysis_json=${ANALYSIS_JSON}"
echo "[memory-prediction] analysis_csv=${ANALYSIS_CSV}"
echo "[memory-prediction] scatter_png=${SCATTER_PREFIX}.prediction_scatter.png"
echo "[memory-prediction] scatter_pdf=${SCATTER_PREFIX}.prediction_scatter.pdf"
