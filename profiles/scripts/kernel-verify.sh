#!/bin/sh
set -eu
: "${WLB_EXPECTED_KERNEL:?expected kernel was not captured before reboot}"
running=$(uname -r)
printf 'running=%s expected=%s\n' "$running" "$WLB_EXPECTED_KERNEL"
test "$running" = "$WLB_EXPECTED_KERNEL"
