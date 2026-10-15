#!/usr/bin/env bash
# Shared shell test contracts. Sourcing only defines functions; Dune actions
# can use rejection checks without parsing arguments, installing traps or Dune.

harness_args() {
  KEEP=0 failures=0 skipped=0 harness_finished=0
  harness_only= harness_only_set=0 legs_selected=0 legs_filtered=0
  harness_ran=() harness_skips=()
  while [ $# -gt 0 ]; do
    case $1 in
      --keep) KEEP=1 ;;
      --only=*) harness_only=${1#--only=} harness_only_set=1 ;;
      --only)
        [ $# -ge 2 ] || { echo "${0##*/}: --only needs a regular expression" >&2; exit 2; }
        harness_only=$2 harness_only_set=1
        shift ;;
      -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
      *) echo "${0##*/}: unknown argument '$1'" >&2; exit 2 ;;
    esac
    shift
  done
  if [ "$harness_only_set" = 1 ]; then
    # An empty pattern matches every leg: a "filter" that filters nothing.
    [ -n "$harness_only" ] || { echo "${0##*/}: --only needs a non-empty regular expression" >&2; exit 2; }
    local rc=0
    [[ '' =~ $harness_only ]] || rc=$?
    [ "$rc" != 2 ] || { echo "${0##*/}: --only '$harness_only' is not a valid extended regular expression" >&2; exit 2; }
  fi
}
# Gate one leg's body: `if harness_leg 'NAME'; then ... fi`. Without --only
# every leg runs; with it, a leg whose NAME does not match the extended
# regular expression is reported FILTERED and its body is not run. Keep a
# leg's shared setup (helpers, fixtures later legs read) outside the gate.
# A NAME may end in a `#group` tag, which `--only '#group'` selects whole.
harness_leg() { # NAME
  if [ "${harness_only_set:-0}" != 1 ] || [[ $1 =~ $harness_only ]]; then
    legs_selected=$((legs_selected + 1))
    harness_ran+=("$1")
    return 0
  fi
  legs_filtered=$((legs_filtered + 1))
  printf 'FILTERED  %s\n' "$1"
  return 1
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
  # A host expected to decide every leg (a CI step that installed the
  # toolchain) sets OCANNL_TOOL_HARNESS_STRICT=1: an undecided leg is red,
  # unless the run's list of expected platform skips names it (below).
  local expected=
  if [ "${OCANNL_TOOL_HARNESS_STRICT:-0}" = 1 ]; then
    if ! harness_skip_expected "$1"; then
      report 1 "$1" "skipped under OCANNL_TOOL_HARNESS_STRICT=1: $2"
      return 0
    fi
    expected=" (expected: listed in ${OCANNL_TOOL_HARNESS_EXPECTED_SKIPS##*/})"
    printf '%s\t%s\n' "$OCANNL_TOOL_HARNESS_NAME" "$1" >>"$OCANNL_TOOL_HARNESS_SKIP_RECORD"
  fi
  skipped=$((skipped + 1))
  harness_skips+=("$1 -- $2")
  printf 'SKIP  %s\n      %s%s\n' "$1" "$2" "$expected"
}
# Whether a strict run's list of expected platform skips names this skip.
# tools/test-harnesses.sh --expected-skips FILE sets all three variables and
# validates FILE before any member runs; here only an exact
# `HARNESS<TAB>LABEL` line matches, so anything else in FILE allows nothing.
harness_skip_expected() { # LABEL
  [ -n "${OCANNL_TOOL_HARNESS_EXPECTED_SKIPS:-}" ] && [ -n "${OCANNL_TOOL_HARNESS_NAME:-}" ] &&
    [ -n "${OCANNL_TOOL_HARNESS_SKIP_RECORD:-}" ] &&
    grep -qxF -- "$OCANNL_TOOL_HARNESS_NAME"$'\t'"$1" "$OCANNL_TOOL_HARNESS_EXPECTED_SKIPS"
}
finish() {
  harness_finished=1
  echo
  if [ "${harness_only_set:-0}" = 1 ]; then
    # A filtered run never prints the full-pass line, and one that selected
    # nothing tested nothing: that is a usage error, not a green run.
    if [ "$legs_selected" -eq 0 ] && [ "$legs_filtered" -eq 0 ]; then
      echo "--only '$harness_only': this run reached no leg gate (the harness declares none, or stopped before its first): nothing was tested" >&2
      exit 2
    elif [ "$legs_selected" -eq 0 ]; then
      echo "--only '$harness_only' selected no leg ($legs_filtered filtered out): nothing was tested" >&2
      exit 2
    fi
    # The coverage account: which legs ran, and every check a ran leg
    # skipped, with its reason. The FILTERED lines above name the rest.
    local leg note
    echo "ran $legs_selected leg(s):"
    for leg in "${harness_ran[@]}"; do printf '  RAN   %s\n' "$leg"; done
    if [ "$skipped" -gt 0 ]; then
      echo "skipped $skipped check(s):"
      for note in "${harness_skips[@]}"; do printf '  SKIP  %s\n' "$note"; done
    fi
    if [ "$failures" -eq 0 ]; then
      echo "FILTERED RUN, not a full pass: $legs_selected leg(s) selected, $legs_filtered filtered out by --only '$harness_only'; no failure ($skipped skipped)"
    else
      echo "$failures leg(s) failed ($skipped skipped); FILTERED RUN: $legs_selected leg(s) selected, $legs_filtered filtered out by --only '$harness_only'"
    fi
  elif [ "$failures" -eq 0 ]; then
    echo "all legs passed ($skipped skipped)"
  else
    echo "$failures leg(s) failed ($skipped skipped)"
  fi
  exit $((failures > 0 ? 1 : 0))
}
harness_cleanup() {
  local rc=$?
  if [ "${harness_finished:-0}" != 1 ]; then
    printf 'aborted before finish (rc=%s)\n' "$rc" >&2
  fi
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
expect_rejected() { # LABEL SUBJECT ORACLE GREP_PATTERN [REASON_CALLBACK]
  local label=$1 subject=$2 oracle=$3 pattern=${4-} reason=${5-}
  local run="mutant-$(printf '%s' "$label" | tr ' ' '-')"
  if "$oracle" "$subject" "$run"; then
    report 1 "negative control: $label" "the shipping oracle accepted the mutant"
  elif [ -n "$pattern" ] && ! grep -qE -- "$pattern" "$TMP/runs/$run/stdout"; then
    report 1 "negative control: $label" "rejected without printing /$pattern/; see $TMP/runs/$run"
  elif [ -n "$reason" ] && ! "$reason" "$run"; then
    report 1 "negative control: $label" "rejected for an unrelated reason; see $TMP/runs/$run"
  elif [ -z "$pattern$reason" ]; then
    report 1 "negative control: $label" "a diagnostic pattern or reason callback is required"
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
