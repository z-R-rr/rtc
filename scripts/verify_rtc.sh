#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/rtc_env.sh"
cd "$RTC_ROOT"
RTC_PYTHON="${RTC_PYTHON:-python}"
mkdir -p outputs/verification
printf 'RTC verification started: %s\n' "$(date --iso-8601=seconds)"
export JAX_PLATFORMS=cpu
"$RTC_PYTHON" -m pytest tests/rtc -q
"$RTC_PYTHON" -m compileall -q src/openpi scripts packages/openpi-client/src/openpi_client
printf 'RTC verification completed: %s\n' "$(date --iso-8601=seconds)"
