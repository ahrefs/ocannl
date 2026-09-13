#!/usr/bin/env bash
# Shared shell test contracts. Sourcing only defines functions; Dune actions
# can use rejection checks without parsing arguments, installing traps or Dune.

harness_args() {
  KEEP=0 failures=0 skipped=0
  for arg in "$@"; do
    case $arg in
      --keep) KEEP=1 ;;
      -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
      *) echo "${0##*/}: unknown argument '$arg'" >&2; exit 2 ;;
    esac
  done
}
report() { # RC LABEL [DETAIL]
  if [ "$1" -eq 0 ]; then
    printf 'PASS  %s\n' "$2"
  else
    failures=$((failures + 1))
    printf 'FAIL  %s\n' "$2"
    [ $# -lt 3 ] || printf '      %s\n' "$3"
  fi
  return 0
}
skip() { # LABEL REASON
  skipped=$((skipped + 1))
  printf 'SKIP  %s\n      %s\n' "$1" "$2"
}
finish() {
  echo
  if [ "$failures" -eq 0 ]; then
    echo "all legs passed ($skipped skipped)"
  else
    echo "$failures leg(s) failed ($skipped skipped)"
  fi
  exit $((failures > 0 ? 1 : 0))
}
harness_cleanup() {
  # Callers release their owned children even with --keep.
  if declare -F cleanup_fixture >/dev/null; then cleanup_fixture; fi
  if [ "${KEEP:-0}" = 1 ]; then
    printf 'kept %s\n' "$TMP"
  elif [ -n "${TMP:-}" ] && [ -d "$TMP" ] && [ "$TMP" != / ]; then
    rm -rf "$TMP"
  fi
  return 0
}
harness_scratch() { # scratch prefix
  TMP=$(mktemp -d "${TMPDIR:-/tmp}/$1.XXXXXX" 2>/dev/null) || TMP=
  if [ -z "$TMP" ] || [ ! -d "$TMP" ]; then
    echo "could not create a temporary directory under ${TMPDIR:-/tmp}" >&2
    exit 2
  fi
  trap harness_cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
}
harness_require() { # executable prerequisites; missing host tools are coverage skips
  local tool missing=0
  for tool in "$@"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      skip "every leg requiring $tool" "no $tool on PATH"
      missing=1
    fi
  done
  [ "$missing" = 0 ] || finish
}
mutant() { # NAME AWK_PROGRAM [AWK_OPTION...], uses caller's SRC and TMP
  local program=$2 out="$TMP/$1.sh"
  shift 2
  awk "$@" "$program" "$SRC" >"$out" || return 1
  bash -n "$out" || return 1
  printf '%s' "$out"
}
# In-process re-entrant rejection primitive: require the precise status AND
# diagnostic, never treat an unavailable command or syntax error as a verdict.
harness_rejected() { # EXPECTED_RC PATTERN LOG COMMAND [ARG...]
  local expected=$1 pattern=$2 log=$3 rc=0
  shift 3
  "$@" >"$log" 2>&1 || rc=$?
  [ "$rc" = "$expected" ] && grep -qE -- "$pattern" "$log"
}
expect_rejected() { # LABEL SUBJECT ORACLE [GREP_PATTERN]
  local label=$1 subject=$2 oracle=$3 pattern=${4-}
  local run="mutant-$(printf '%s' "$label" | tr ' ' '-')"
  if "$oracle" "$subject" "$run"; then
    report 1 "negative control: $label" "the shipping oracle accepted the mutant"
  elif [ -n "$pattern" ] && ! grep -qE -- "$pattern" "$TMP/runs/$run/stdout"; then
    report 1 "negative control: $label" "rejected without printing /$pattern/; see $TMP/runs/$run"
  else
    report 0 "negative control: $label"
  fi
}
# Call only for an unreaped direct child. DEAD is an independent host state
# reader; the caller probes its availability before running process legs.
harness_wait_child() { # PID DEAD LABEL [poll count at 50ms]
  local pid=$1 dead=$2 label=$3 polls=${4:-300} i
  for ((i=0; i<polls; i++)); do "$dead" "$pid" && break; sleep .05; done
  if [ "$i" -eq "$polls" ]; then
    report 1 "$label"
    kill -KILL "$pid" 2>/dev/null
  fi
  harness_child_rc=0
  wait "$pid" 2>/dev/null || harness_child_rc=$?
}
harness_kill_recorded() { # PID_FILE TOKEN_FILE [group]; caller supplies identity reader
  local pid
  if proc_identity_matches "$1" "$2"; then
    pid=$(cat "$1")
    if [ "${3:-}" = group ]; then kill -KILL -- "-$pid" 2>/dev/null
    else kill -KILL "$pid" 2>/dev/null; fi
  fi
  return 0
}
