#!/usr/bin/env bash
# Exercise the production prepush reader on test/operations/dune with one
# exact selected alias. The dedicated probe still runs if that alias
# leaves scans. The probe starts no build and reads no run logs.
set -eu

runner=$(dirname "$0")/../../tools/test-run.sh
member=test/operations/runtest-slot_kind_live
selected=$("$BASH" "$runner" _prepush-scan-members "@$member")
if [ "$selected" != "$member" ]; then
  printf 'prepush scan membership: expected %s, got <%s>\n' "$member" "$selected" >&2
  exit 1
fi
