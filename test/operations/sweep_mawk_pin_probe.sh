#!/usr/bin/env bash
# The staging#2244 in-inside-assignment probe uses bare awk so the driver's PATH
# decides the interpreter. Exit 0 is the exact BWK result, 1 is the exact
# mawk defect, and 2 covers any other output or a failed awk invocation.
probe() {
  local out
  out=$(awk 'BEGIN { a["k"] = ("k" in a) ? a["k"] "; x" : "x"; print "got=[" a["k"] "]" }') || return 2
  printf '%s\n' "$out"
  case $out in
    'got=[x]') return 0 ;;
    'got=[; x]') return 1 ;;
    *) return 2 ;;
  esac
}

system_status=0
(
  export PATH=/usr/bin:/bin
  probe
) || system_status=$?
driver_status=0
probe || driver_status=$?
if [ "$system_status:$driver_status" != 0:1 ]; then
  printf 'staging#2244 probe expected system awk exit 0 and driver awk exit 1, got %s and %s\n' \
    "$system_status" "$driver_status" >&2
  exit 1
fi
