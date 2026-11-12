#!/bin/sh
set -eu
# This remains read-only. Refresh apt metadata separately before starting the run.
stamp=/var/lib/apt/periodic/update-success-stamp
test -f "$stamp" || { echo 'apt metadata freshness unknown; run apt-get update first' >&2; exit 1; }
age=$(( $(date +%s) - $(stat -c %Y "$stamp") ))
test "$age" -ge 0 && test "$age" -le 86400 || { echo 'apt metadata older than 24h; refresh first' >&2; exit 1; }
simulation=$(apt-get -s -o Debug::NoLocking=1 dist-upgrade) || exit 1
if printf '%s\n' "$simulation" | grep -q '^Remv '; then echo 'package removals require a separate reviewed plan' >&2; exit 1; fi
printf '%s\n' "$simulation" | grep -q '^Inst ' || exit 3
printf '%s\n' "$simulation" | grep '^Inst '
