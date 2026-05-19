#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export STATIC_MAX_CONCURRENCY=${STATIC_MAX_CONCURRENCY:-128}
exec bash "${SCRIPT_DIR}/run_swe_rl_static_rollout_only_baseline.sh" "$@"

