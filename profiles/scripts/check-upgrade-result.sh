#!/bin/sh
set -eu
# rc 0  = the approved upgrade is complete (nothing left to install).
# rc 10 = provably safe to run dist-upgrade again: dpkg is consistent and every
#         remaining package was in the approved plan (partial upgrades and
#         version bumps of approved packages included).
# rc 1  = UNKNOWN for operator review (dpkg inconsistent, removals, or
#         packages outside the approved plan).
audit=$(dpkg --audit) || exit 1
test -z "$audit" || exit 1
simulation=$(apt-get -s -o Debug::NoLocking=1 dist-upgrade) || exit 1
if printf '%s\n' "$simulation" | grep -q '^Remv '; then exit 1; fi
current=$(printf '%s\n' "$simulation" | awk '$1 == "Inst" {print $2}' | sort -u)
if [ -z "$current" ]; then
    exit 0
fi
: "${WLB_PACKAGE_PLAN:?approved package plan missing}"
approved=$(printf '%s\n' "$WLB_PACKAGE_PLAN" | awk '$1 == "Inst" {print $2}' | sort -u)
test -n "$approved" || exit 1
extra=$(printf '%s\n' "$current" | grep -vxF "$approved" || true)
test -z "$extra" || exit 1
exit 10
