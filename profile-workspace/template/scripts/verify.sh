#!/usr/bin/env bash
set -euo pipefail
# Read-only acceptance before restore; this stage also runs if reboot is skipped.
# Compare actual state with the expected result, not just process existence.
echo 'PROFILE_NOT_IMPLEMENTED: implement acceptance checks' >&2
exit 78
