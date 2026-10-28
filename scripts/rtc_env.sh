#!/usr/bin/env bash
# Source from any directory. Does not modify another checkout or its environment.
RTC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$RTC_ROOT/src:$RTC_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export RTC_ROOT
