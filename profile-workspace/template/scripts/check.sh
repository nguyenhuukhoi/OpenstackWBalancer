#!/usr/bin/env bash
set -euo pipefail
# Recovery only: inspect actual state after an interrupted/failed attempt.
# 0 = desired state verified; 10 = verified safe to execute apply again.
# Any other exit code = UNKNOWN; never use 10 as a generic fallback.
echo 'PROFILE_NOT_IMPLEMENTED: outcome cannot yet be reconciled' >&2
exit 20
