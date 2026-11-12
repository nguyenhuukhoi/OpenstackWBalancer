#!/bin/sh
set -eu
# Upgrade only packages that were approved at pre_drain.
#
# The plan is captured before the drain, the upgrade runs after it. apt-daily
# and unattended-upgrades may meanwhile refresh lists or install part of the
# plan, which changes versions in the "Inst" lines. Accepted: the remaining
# work is a subset (by package name) of the approved plan, no removals.
# Rejected: any package name that was not approved, or any removal.
: "${WLB_PACKAGE_PLAN:?approved package plan missing}"
approved=$(printf '%s\n' "$WLB_PACKAGE_PLAN" | awk '$1 == "Inst" {print $2}' | sort -u)
test -n "$approved" || { echo 'approved plan lists no package' >&2; exit 1; }
simulation=$(apt-get -s -o Debug::NoLocking=1 dist-upgrade) || exit 1
if printf '%s\n' "$simulation" | grep -q '^Remv '; then echo 'plan now removes packages' >&2; exit 1; fi
current=$(printf '%s\n' "$simulation" | awk '$1 == "Inst" {print $2}' | sort -u)
if [ -z "$current" ]; then
    echo 'nothing left to upgrade (approved plan already installed)'
    exit 0
fi
extra=$(printf '%s\n' "$current" | grep -vxF "$approved" || true)
if [ -n "$extra" ]; then
    echo "package plan gained packages that were not approved: $(echo $extra)" >&2
    exit 1
fi
DEBIAN_FRONTEND=noninteractive apt-get -y -q -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold dist-upgrade
