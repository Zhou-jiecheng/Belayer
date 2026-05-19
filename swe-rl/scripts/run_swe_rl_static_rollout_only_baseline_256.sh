#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export STATIC_MAX_CONCURRENCY=${STATIC_MAX_CONCURRENCY:-256}
export SWE_MAX_CONCURRENT_EVAL=512
export SWE_MAX_CONCURRENT_DIFF=512
exec bash "${SCRIPT_DIR}/run_swe_rl_static_rollout_only_baseline.sh" "$@"

