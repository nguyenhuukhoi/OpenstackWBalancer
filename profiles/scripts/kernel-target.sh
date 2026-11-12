#!/bin/sh
set -eu
newest=$(find /boot -maxdepth 1 -name 'vmlinuz-*' -printf '%f\n' | sed 's/^vmlinuz-//' | sort -V | tail -1)
test -n "$newest" || { echo 'no installed kernel' >&2; exit 1; }
test -s "/boot/initrd.img-$newest"
test -d "/lib/modules/$newest/kernel"
audit=$(dpkg --audit) || { echo 'dpkg --audit failed' >&2; exit 1; }
test -z "$audit" || { printf '%s\n' "$audit" >&2; exit 1; }
def=$(bash -c '. /etc/default/grub; for file in /etc/default/grub.d/*.cfg; do test ! -f "$file" || . "$file"; done; printf "%s" "${GRUB_DEFAULT:-0}"')
test "$def" = 0 || { echo 'GRUB default is not numeric 0; unsupported boot policy' >&2; exit 1; }
env=$(grub-editenv /boot/grub/grubenv list) || exit 1
if printf '%s\n' "$env" | grep -Eq '^(next_entry|saved_entry)=.+'; then
    echo 'GRUB has saved/one-shot entry; cannot prove target kernel' >&2; exit 1
fi
first=$(grep -m1 -oP '^\s*linux\s+\S*vmlinuz-\K\S+' /boot/grub/grub.cfg)
test "$first" = "$newest" || { echo 'GRUB entry does not match target kernel' >&2; exit 1; }
printf '%s\n' "$newest"
