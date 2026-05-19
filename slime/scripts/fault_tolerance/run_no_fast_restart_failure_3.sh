#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export CI_FAULT_INJECTION_ENABLE=1
export SLIME_ROUTER_REROUTE_FAILED_REQUESTS_TO_HEALTHY_WORKERS=0

# fault inject
export CI_FAULT_INJECTION_DELAY_SEC=0
export CI_FAULT_INJECTION_MODE="mid_generate"
export CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD=0
export CI_FAULT_INJECTION_PROGRESS_FRACTION=0.3
export CI_FAULT_INJECTION_MID_DELAY_SEC=0
export CI_FAULT_INJECTION_ENGINE_INDEX="0,1,2"
export CI_FAULT_INJECTION_MID_FALLBACK_SEC=80

# /mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/slime/scripts/fault_tolerance/run-no-fast-restart-weight-update-smoke-16gpu.sh
exec "${SCRIPT_DIR}/run-no-fast-restart-weight-update-smoke-16gpu.sh" "$@"
