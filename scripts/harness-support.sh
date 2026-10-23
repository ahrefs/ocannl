#!/usr/bin/env bash
# Shared shell test contracts. Sourcing only defines functions; Dune actions
# can use rejection checks without parsing arguments, installing traps or Dune.

harness_args() {
  KEEP=0 failures=0 skipped=0 harness_finished=0
  harness_only= harness_only_set=0 legs_selected=0 legs_filtered=0
  harness_ran=() harness_skips=() harness_leg_running= harness_leg_current=
  harness_list=0 harness_listed=0 harness_leg_names=$'\n' harness_group_seen=$'\n'
  harness_number_keys=() harness_number_legs=()
  while [ $# -gt 0 ]; do
    case $1 in
      --keep) KEEP=1 ;;
      --list-legs) harness_list=1 ;;
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
    [ "$harness_list" = 0 ] || { echo "${0##*/}: --list-legs lists every leg; it takes no --only" >&2; exit 2; }
  fi
  harness_groups=$(harness_header_groups)
}
# A harness that groups its legs declares each group in its header (the
# comment block --help prints) as `#NAME (LEGS)`: LEGS a comma-separated list
# of leg numbers and ranges, a piece of prose after a colon allowed
# (`#lifecycle (40-47, 65, 74: the run/start lifecycle)`). Prints one line per
# group: NAME, then each leg number it lists.
harness_header_groups() {
  sed -n '2,${/^#/!q;p;}' "$0" 2>/dev/null | sed 's/^# \{0,1\}//' | tr '\n' ' ' | awk '{
    s = $0
    while (match(s, /#[A-Za-z0-9_-]+ \([^)]*\)/)) {
      d = substr(s, RSTART + 1, RLENGTH - 2); s = substr(s, RSTART + RLENGTH)
      out = d; sub(/ .*/, "", out); sub(/^[^(]*\(/, "", d)
      n = split(d, piece, ",")
      for (i = 1; i <= n; i++) {
        p = piece[i]; sub(/^ +/, "", p)
        if (!match(p, /^[0-9]+(-[0-9]+)?/)) continue
        p = substr(p, 1, RLENGTH); lo = p; hi = p
        if (index(p, "-")) { lo = substr(p, 1, index(p, "-") - 1); hi = substr(p, index(p, "-") + 1) }
        for (k = lo + 0; k <= hi + 0; k++) out = out " " k
      }
      print out
    }
  }'
}
# The leg numbers a leg name's LABEL (the text before its first colon) claims,
# one `N SUFFIX` line each: `2-3` claims 2 and 3, `59-63, 66` six numbers,
# `55a` 55 with suffix a. A label that is not of that form claims none.
harness_label_numbers() { # LABEL
  awk -v l="$1" 'BEGIN {
    if (l !~ /^[0-9]+[a-z]?(-[0-9]+[a-z]?)?(, ?[0-9]+[a-z]?(-[0-9]+[a-z]?)?)*$/) exit
    n = split(l, piece, ",")
    for (i = 1; i <= n; i++) {
      p = piece[i]; sub(/^ /, "", p); sfx = p; gsub(/[0-9-]/, "", sfx); gsub(/[a-z]/, "", p)
      lo = p; hi = p
      if (index(p, "-")) { lo = substr(p, 1, index(p, "-") - 1); hi = substr(p, index(p, "-") + 1) }
      for (k = lo + 0; k <= hi + 0; k++) print k, sfx
    }
  }'
}
# Leg naming, checked at every gate whether or not the leg is selected
# (gh-ocannl-1367): a name selects one leg, so `--only` and
# tools/harness-legs-alone.sh can run it alone; a leg number (with its
# suffix letter: 80 and 80b are two legs) numbers one leg; and in a harness
# whose header declares groups, every leg closes with ` #GROUP` of a declared
# group that lists its numbers. finish checks the other direction: every
# number a group lists is carried by a leg of that group.
harness_leg_lint() { # NAME
  local name=$1 tag= nums= label pairs n sfx i line clash=0
  case $harness_leg_names in
    *$'\n'"$name"$'\n'*)
      report 1 "leg naming: two legs are named '$name'" "a leg's name is what selects it alone; give each leg its own"
      return 0 ;;
  esac
  harness_leg_names=$harness_leg_names$name$'\n'
  case $name in *'#'*) tag=${name##*#} ;; esac
  [[ $tag =~ ^[A-Za-z0-9_-]+$ ]] || tag=
  if [ -n "$harness_groups" ]; then
    line=
    if [ -n "$tag" ]; then
      while IFS= read -r line; do
        case $line in "$tag"|"$tag "*) break ;; esac
        line=
      done <<<"$harness_groups"
    fi
    if [ -z "$line" ]; then
      report 1 "leg naming: '$name' does not close with a group its header declares" \
        "declared: $(printf '%s\n' "$harness_groups" | sed 's/ .*//; s/^/#/' | tr '\n' ' ')"
      tag=
    else
      nums=" ${line#"$tag"} "
    fi
  else
    tag= # tags are free where the header declares no group
  fi
  label=${name%%:*}
  [ "$label" != "$name" ] || return 0
  pairs=$(harness_label_numbers "$label")
  [ -n "$pairs" ] || return 0
  while read -r n sfx; do
    for ((i = 0; i < ${#harness_number_keys[@]}; i++)); do
      [ "$clash" = 0 ] && [ "${harness_number_keys[i]}" = "$n$sfx" ] || continue
      report 1 "leg naming: '$name' numbers leg $n$sfx, as '${harness_number_legs[i]}' does" \
        "a leg number names one leg; a further leg beside it takes a suffix letter (${label}b)"
      clash=1
    done
    harness_number_keys+=("$n$sfx") harness_number_legs+=("$name")
    [ -n "$tag" ] || continue
    harness_group_seen=$harness_group_seen$tag:$n$'\n'
    case $nums in
      '  ') ;;
      *" $n "*) ;;
      *) report 1 "leg naming: '$name' numbers leg $n, which the header does not list under #$tag" ;;
    esac
  done <<<"$pairs"
  return 0
}
harness_header_unclaimed() { # finish: a number a declared group lists that no leg of it carries
  local line tag n
  [ -n "$harness_groups" ] && [ "$harness_leg_names" != $'\n' ] || return 0
  while IFS= read -r line; do
    tag=${line%% *}
    for n in ${line#"$tag"}; do
      case $harness_group_seen in
        *$'\n'"$tag:$n"$'\n'*) ;;
        *) report 1 "leg naming: the header lists leg $n under #$tag, and no leg of #$tag numbers it" ;;
      esac
    done
  done <<<"$harness_groups"
  return 0
}
# Gate one leg's body: `if harness_leg 'NAME'; then ... fi`. Without --only
# every leg runs; with it, a leg whose NAME does not match the extended
# regular expression is reported FILTERED and its body is not run. Keep a
# leg's shared setup (helpers, fixtures later legs read) outside the gate:
# a leg must pass run alone, which tools/harness-legs-alone.sh checks.
# A NAME may end in a `#group` tag, which `--only '#group'` selects whole.
# Under --list-legs every gate prints `LEG  NAME` and runs no body.
harness_leg() { # NAME
  # The leg a later skip is charged to (OCANNL_TOOL_HARNESS_SKIP_LEGS): the
  # last gate reached, selected or not, so shared code between two gates is
  # charged alike in a full run and a filtered one.
  harness_leg_current=$1
  harness_leg_lint "$1"
  if [ "$harness_list" = 1 ]; then
    harness_listed=$((harness_listed + 1))
    printf 'LEG  %s\n' "$1"
    return 1
  fi
  if [ "${harness_only_set:-0}" != 1 ] || [[ $1 =~ $harness_only ]]; then
    legs_selected=$((legs_selected + 1))
    harness_ran+=("$1")
    return 0
  fi
  legs_filtered=$((legs_filtered + 1))
  printf 'FILTERED  %s\n' "$1"
  return 1
}
# Run one leg whose body is a FUNCTION: `harness_leg_run 'NAME' leg_fn`, gated
# as harness_leg gates. A body that cannot go on -- a fixture that is not what
# its checks assume, an extraction that matched nothing -- ends with
# `harness_abandon DETAIL; return`, which fails that leg alone; every later leg
# still runs and is reported (gh-ocannl-1374). An `exit` there would end the
# whole harness and hide them. The body's own status is not a verdict: its
# last command's status is incidental, and only report/skip/harness_abandon
# say anything.
harness_leg_run() { # NAME FUNCTION
  harness_leg "$1" || return 0
  harness_leg_running=$1
  "$2"
  harness_leg_running=
  return 0
}
harness_abandon() { # DETAIL -- inside a harness_leg_run body, then `return`
  report 1 "${harness_leg_running:-a leg}: abandoned, its later checks did not run" "$1"
}
# Run one leg whose body checks under errexit: `harness_leg_run_errexit 'NAME'
# leg_fn`, gated as harness_leg gates (gh-ocannl-1539). The body runs in a
# subshell of its own with errexit on and the ERR trap inherited by its
# functions, so a check that fails -- a command errexit weighs, an `exit`, an
# unbound variable under `set -u` -- ends the subshell and with it the leg,
# never the harness: the leg is abandoned (harness_abandon), naming the
# command that failed or the status the body ended with, and every later leg
# still runs. A body passes only by reaching its end. Call it as a statement
# of its own: inside an `if`, `&&`, `||` or `!` bash ignores errexit in the
# whole body, so no check there could fail, and the leg is abandoned instead.
# Being a subshell, the body's variables, exports, traps and `cd` stay its
# own, and so would its counts: `report 0` prints as usual, while `report 1`,
# `skip` and `harness_abandon` end the body as an abandoned leg, with their
# text. What the body changes on disk does outlive it: a harness whose legs
# leave shared state behind defines harness_leg_restore, which runs, contained
# the same way, after every abandoned leg that ran (a FAIL of its own if it
# fails).
harness_leg_run_errexit() { # NAME FUNCTION
  harness_leg "$1" || return 0
  harness_leg_running=$1
  harness_errexit_body "$2"
  if [ -n "$harness_errexit_detail" ]; then
    harness_abandon "$harness_errexit_detail"
    if [ "$harness_errexit_ran" = 1 ] && declare -F harness_leg_restore >/dev/null; then
      harness_errexit_body harness_leg_restore
      [ -z "$harness_errexit_detail" ] ||
        report 1 "$1: the shared state it left was not restored; a later leg may fail for it" "$harness_errexit_detail"
    fi
  fi
  harness_leg_running=
  return 0
}
# harness_leg_run_errexit's containment: run FUNCTION in an errexit subshell,
# then set harness_errexit_detail to why it did not reach its end (empty when
# it did), and harness_errexit_ran to 0 when it was not run at all. It returns
# 0 whatever the body did: the subshell exits 0 through its own EXIT trap, so
# the caller's errexit never sees the body's status, and the call never sits
# in a condition that would turn errexit off inside it.
harness_errexit_body() { # FUNCTION
  local mark rc
  mark=$TMP/.harness-errexit
  rm -f "$mark.rc" "$mark.done" "$mark.failed" "$mark.ignored"
  harness_errexit_subshell "$1" "$mark"
  harness_errexit_detail= harness_errexit_ran=1
  [ ! -e "$mark.done" ] || return 0
  if [ -e "$mark.ignored" ]; then
    harness_errexit_ran=0
    harness_errexit_detail="it was not run: called inside a condition (if, &&, || or !), where bash ignores errexit in the whole body, no check of it could fail"
  elif [ -s "$mark.failed" ]; then
    harness_errexit_detail=$(cat "$mark.failed")
  else
    rc=$(cat "$mark.rc" 2>/dev/null) || rc='unknown (it was killed)'
    harness_errexit_detail="it ended with status $rc before its end: an exit, or an unbound variable under set -u (the lines above say which)"
  fi
  return 0
}
# The subshell itself, a function whose body is one, so its `set -e` is its
# own: MARK.done when FUNCTION reached its end, MARK.failed naming why it did
# not, MARK.rc the status it ended with, MARK.ignored when errexit could not
# take effect at all.
harness_errexit_subshell() ( # FUNCTION MARK
  harness_errexit_mark=$2
  trap 'printf "%s\n" "$?" >"$harness_errexit_mark.rc"; exit 0' EXIT
  # Whether errexit can take effect here at all: under a condition bash
  # ignores it, even when set again inside, so the probe would reach exit 3.
  set +e
  (set -e; false; exit 3)
  [ "$?" != 3 ] || { : >"$harness_errexit_mark.ignored"; exit 1; }
  report() { # RC LABEL [DETAIL]
    [ "$1" -ne 0 ] || { printf 'PASS  %s\n' "$2"; return 0; }
    harness_abandon "report $1 inside the body: $2${3:+ -- $3}"
  }
  skip() { harness_abandon "skip inside the body: $1 -- $2"; }
  harness_abandon() { printf '%s\n' "$1" >"$harness_errexit_mark.failed"; exit 1; }
  set -eE
  # Commands inside a command substitution report at a deeper level; only the
  # body's own level names the check that ended it.
  trap '[ "$BASH_SUBSHELL" != '"$BASH_SUBSHELL"' ] || printf "a check failed in %s: %s\n" "${FUNCNAME[0]:-the body}" "$BASH_COMMAND" >"$harness_errexit_mark.failed"' ERR
  "$1"
  : >"$harness_errexit_mark.done"
)
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
  # tools/harness-legs-alone.sh compares each leg's skips alone with the
  # same leg's in the full run, so it asks which leg each skip came from.
  [ -z "${OCANNL_TOOL_HARNESS_SKIP_LEGS:-}" ] ||
    printf '%s\t%s\n' "${harness_leg_current:-(before any gate)}" "$1" >>"$OCANNL_TOOL_HARNESS_SKIP_LEGS"
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
  harness_header_unclaimed
  echo
  if [ "$harness_list" = 1 ]; then
    # A listing ran no leg body: it reports only the gates it reached, and
    # fails only on a naming defect or a shared fixture's own FAIL.
    echo "listed $harness_listed leg(s); no leg ran ($failures failure(s) outside the legs, $skipped skipped)"
  elif [ "${harness_only_set:-0}" = 1 ]; then
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
    if [ -n "${harness_leg_running:-}" ]; then
      printf 'aborted before finish (rc=%s) inside leg: %s\n' "$rc" "$harness_leg_running" >&2
    else
      printf 'aborted before finish (rc=%s)\n' "$rc" >&2
    fi
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
