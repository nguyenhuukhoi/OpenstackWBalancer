#!/bin/sh
set -eu
running=$(uname -r)
newest=$(find /boot -maxdepth 1 -name 'vmlinuz-*' -printf '%f\n' | sed 's/^vmlinuz-//' | sort -V | tail -1)
test -n "$newest" || exit 1
top=$(printf '%s\n%s\n' "$running" "$newest" | sort -V | tail -1)
if [ "$top" = "$running" ]; then echo 'already on newest kernel'; exit 3; fi
