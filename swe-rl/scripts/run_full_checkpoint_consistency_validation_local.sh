#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWE_RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SWE_RL_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${FULL_CONSISTENCY_OUTPUT_ROOT:-${WORKSPACE_ROOT}/export/full_checkpoint_consistency_$(date +%Y%m%d_%H%M%S)}"
IMAGE="${FULL_CONSISTENCY_IMAGE:-python:3.12-slim}"
HEAP_MB="${FULL_CONSISTENCY_HEAP_MB:-16}"
PAYLOAD_MB="${FULL_CONSISTENCY_PAYLOAD_MB:-16}"
WALL_CLOCK_TRIALS="${FULL_CONSISTENCY_WALL_CLOCK_TRIALS:-60}"
WALL_CLOCK_SEED="${FULL_CONSISTENCY_WALL_CLOCK_SEED:-20260721}"

mkdir -p "${OUTPUT_ROOT}"

ARGS=(
    "--output-root" "${OUTPUT_ROOT}"
    "--image" "${IMAGE}"
    "--heap-mb" "${HEAP_MB}"
    "--payload-mb" "${PAYLOAD_MB}"
    "--wall-clock-trials" "${WALL_CLOCK_TRIALS}"
    "--wall-clock-seed" "${WALL_CLOCK_SEED}"
)

if [[ "${FULL_CONSISTENCY_DIRECTED_DIAGNOSTICS:-0}" != "1" ]]; then
    ARGS+=("--wall-clock-only")
fi

if [[ "${FULL_CONSISTENCY_NO_PULL:-1}" == "1" ]]; then
    ARGS+=("--no-pull")
fi
if [[ "${FULL_CONSISTENCY_KEEP_ARTIFACTS:-0}" == "1" ]]; then
    ARGS+=("--keep-artifacts")
fi
if [[ "${FULL_CONSISTENCY_STRICT:-0}" == "1" ]]; then
    ARGS+=("--strict")
fi
if [[ $# -gt 0 ]]; then
    ARGS+=("$@")
fi

echo "[full-consistency] output_root=${OUTPUT_ROOT}"
echo "[full-consistency] image=${IMAGE} heap_mb=${HEAP_MB} payload_mb=${PAYLOAD_MB} wall_clock_trials=${WALL_CLOCK_TRIALS}"

if [[ "${EUID}" -eq 0 ]]; then
    env PYTHONDONTWRITEBYTECODE=1 \
        "${PYTHON_BIN}" "${SWE_RL_DIR}/tools/validate_full_checkpoint_consistency.py" "${ARGS[@]}"
else
    sudo -n env PYTHONDONTWRITEBYTECODE=1 \
        "${PYTHON_BIN}" "${SWE_RL_DIR}/tools/validate_full_checkpoint_consistency.py" "${ARGS[@]}"
fi

echo "[full-consistency] result=${OUTPUT_ROOT}/result.json"
