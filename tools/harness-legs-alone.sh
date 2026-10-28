#!/usr/bin/env bash
# Run every gated leg of the shell harnesses alone, and every group of legs,
# and report each one that fails or skips alone (gh-ocannl-1367).
#
#   tools/harness-legs-alone.sh                     # every gated harness
#   tools/harness-legs-alone.sh tools/test-promote.sh ...
#   tools/harness-legs-alone.sh --logs DIR ...      # keep every run's output
#
# A leg gated by harness_leg/harness_leg_run (scripts/harness-support.sh) must
# pass when `--only` selects it alone: the shared fixtures outside the gates
# run as in a full run, so a leg that needs anything an EARLIER leg's body
# made -- a variable, a file, a process -- passes in every full run and fails
# only when someone iterates on it with --only (two such legs surfaced when
# tools/test-test-run.sh was first gated). Per HARNESS (a path relative to the
# repository root, where every run starts) this runs:
#   1. HARNESS --list-legs: the gated legs, and the naming checks every gate
#      makes (one leg per name and per leg number, declared groups);
#   2. HARNESS once in full, for its verdict and the checks it skipped;
#   3. HARNESS --only '#GROUP$' for each group tag the leg names close with;
#   4. HARNESS --only '^NAME$' for each leg, NAME escaped so that the run
#      selects exactly that leg.
# Every run starts from the same checkout: one that changes its tracked or
# untracked files is a finding (CHECKOUT CHANGED), since a later run could
# pass on what it left. State a leg keeps outside the checkout and its own
# scratch directory is beyond this check, as it is beyond harness_scratch's
# contract.
# A selected run is a finding when it fails (ALONE FAIL), when its coverage
# account does not say it ran exactly the legs meant -- or the selector, read
# by harness_leg's own matcher, would not select exactly them (ALONE MISS) --
# or when it skips a check the same leg decided in the full run (ALONE SKIP:
# each skip is charged to the leg it came from, OCANNL_TOOL_HARNESS_SKIP_LEGS). Each finding says
# whether the full run passed: "passes in the full run" names an
# order-dependent leg; a red full run is reported first and on its own.
#
# With no HARNESS, every member of tools/test-harnesses.sh whose gates list a
# leg on this host. Each member runs in the caller's environment (strict
# skips included), except that no list of expected skips reaches it: that
# list belongs to the aggregate. CI runs this on its daily schedule.
#
# Exit: 0 every run passed; 1 a finding, a red full run or a failed listing;
# 2 a usage error, or a HARNESS named that lists no leg. HUP, INT and TERM
# stop the check and keep their status (129, 130, 143).
set -u
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P) || exit 2
logs= harnesses=()
while [ $# -gt 0 ]; do
  case $1 in
    --logs)
      [ $# -ge 2 ] && [ -n "$2" ] || { echo "harness-legs-alone.sh: --logs needs a directory" >&2; exit 2; }
      logs=$2
      shift ;;
    -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "harness-legs-alone.sh: unknown argument '$1'" >&2; exit 2 ;;
    *) harnesses+=("$1") ;;
  esac
  shift
done
named=${#harnesses[@]}
if [ "$named" -eq 0 ]; then
  while IFS= read -r script; do
    # A pre-filter only: a gate call outside a comment, its arguments on the
    # same line or a continued one. The listing below is the authority, and a
    # candidate that lists no leg is printed as not checked.
    grep -qE '^[^#]*(^|[^A-Za-z0-9_])harness_leg(_run)?([[:space:]]|\\|$)' "$root/$script" 2>/dev/null &&
      harnesses+=("$script")
  done < <("$root/tools/test-harnesses.sh" --list)
  [ ${#harnesses[@]} -gt 0 ] || { echo "harness-legs-alone.sh: no member of tools/test-harnesses.sh gates a leg" >&2; exit 2; }
fi
if [ -n "$logs" ]; then
  mkdir -p "$logs" && logs=$(CDPATH= cd -- "$logs" && pwd -P) ||
    { echo "harness-legs-alone.sh: cannot use $logs" >&2; exit 2; }
  keep=1
else
  logs=$(mktemp -d "${TMPDIR:-/tmp}/harness-legs-alone.XXXXXX") && logs=$(CDPATH= cd -- "$logs" && pwd -P) ||
    { echo "harness-legs-alone.sh: cannot create a log directory" >&2; exit 2; }
  keep=0
fi
unset OCANNL_TOOL_HARNESS_EXPECTED_SKIPS OCANNL_TOOL_HARNESS_SKIP_RECORD OCANNL_TOOL_HARNESS_SKIP_LEGS \
  OCANNL_TOOL_HARNESS_DUNE_SKIP_RECORD
# The runs' own output may not land in the checkout they are checked against
# (both paths resolved physically, so a symbolic link cannot hide it).
case "$logs/" in
  "$root/"*)
    echo "harness-legs-alone.sh: the log directory $logs is inside the checkout, whose files every run must leave unchanged" >&2
    [ "$keep" = 1 ] || rm -rf "$logs"
    exit 2 ;;
esac
cd "$root" || exit 2

# NAME as an extended regular expression matching exactly NAME.
ere_escape() {
  local s=$1 out= c i
  for ((i = 0; i < ${#s}; i++)); do
    c=${s:i:1}
    case $c in
      '\'|'.'|'*'|'^'|'$'|'('|')'|'+'|'?'|'{'|'}'|'|'|'['|']') out+="\\$c" ;;
      *) out+=$c ;;
    esac
  done
  printf '%s' "$out"
}
# A signal to this check stops the member running too, and waits for its
# cleanup, before the check exits with the signal's status. The member is
# sent TERM whatever the signal: a background child of a non-interactive
# shell starts with INT ignored, and every harness exits on TERM through
# its cleanup (harness_scratch).
child=
stop() { # STATUS
  trap '' HUP INT TERM
  if [ -n "$child" ]; then
    kill -TERM "$child" 2>/dev/null
    wait "$child" 2>/dev/null
  fi
  printf 'harness-legs-alone: interrupted (exit %s)%s\n' "$1" "${current_log:+ in $current_log}" >&2
  exit "$1"
}
trap 'stop 129' HUP
trap 'stop 130' INT
trap 'stop 143' TERM
# What the runs may not change: the checkout's tracked and untracked files
# (ignored ones such as _build are out of reach). A run that leaves state
# there could hand it to a later run of the same leg, which would then pass
# alone on what an earlier run made, so it is a finding of its own.
# Contents too, not only each path's status: a run that rewrites a file the
# checkout already had modified, or an untracked one, changes no status line.
# And untracked directories, an empty one included, which name no file.
checkout_state() {
  git -C "$root" status --porcelain=v1 --untracked-files=all 2>/dev/null || return 0
  git -C "$root" ls-files --others --directory --exclude-standard 2>/dev/null
  git -C "$root" diff HEAD --binary 2>/dev/null | git hash-object --stdin
  # Only a regular file is opened to hash it: a FIFO would block the read.
  # Anything else is recorded by its kind (and a link by its target).
  git -C "$root" ls-files -z --others --exclude-standard 2>/dev/null |
    (cd "$root" && while IFS= read -r -d '' path; do
      if [ -L "$path" ]; then printf 'link %s -> %s\n' "$path" "$(readlink "$path")"
      elif [ -f "$path" ]; then printf '%s %s\n' "$(git hash-object -- "$path" 2>/dev/null)" "$path"
      else printf 'special %s\n' "$path"
      fi
    done)
}
member() { # LOG SCRIPT [ARG...] -- one run; rc in $rc
  local log=$1 script=$2 after
  shift 2
  rc=0 current_log=$log
  : >"$log.skips"
  OCANNL_TOOL_HARNESS_NAME=$script OCANNL_TOOL_HARNESS_SKIP_LEGS=$log.skips \
    "$root/$script" "$@" >"$log" 2>&1 </dev/null &
  child=$!
  # A signal to this check runs stop(); the member's own status, 143
  # included, is the member's verdict.
  wait "$child" || rc=$?
  child= current_log=
  after=$(checkout_state)
  if [ "$after" != "$baseline" ]; then
    findings=$((findings + 1))
    printf '  CHECKOUT CHANGED  by %s %s: %s\n' "$script" "$*" "$log"
    diff <(printf '%s\n' "$baseline") <(printf '%s\n' "$after") | sed -n 's/^[<>] /        /p' | head -10
    baseline=$after
  fi
}
excerpt() { # LOG -- the lines a finding is read from
  grep -E '^(FAIL  |      |SKIP  |aborted |--only )' "$1" | head -20 | sed 's/^/        /'
}
findings=0 red=0 checked=0 bad_listing=0
baseline=$(checkout_state)
for script in "${harnesses[@]}"; do
  [ -f "$root/$script" ] || { echo "harness-legs-alone.sh: no harness $script" >&2; exit 2; }
  slug=$(printf '%s' "$script" | tr '/ ' '__')
  mkdir -p "$logs/$slug"
  member "$logs/$slug/list" "$script" --list-legs
  legs=$(sed -n 's/^LEG  //p' "$logs/$slug/list")
  if [ -z "$legs" ]; then
    if [ "$named" -gt 0 ]; then
      echo "harness-legs-alone.sh: $script lists no leg (exit $rc; see $logs/$slug/list)" >&2
      exit 2
    fi
    # Only a listing that finished cleanly says the host gates no leg: one
    # that died before its first gate (a shared fixture's own failure) hid
    # every leg, and is a failed listing.
    if [ "$rc" != 0 ]; then
      bad_listing=$((bad_listing + 1))
      printf '\nHARNESS %s: LISTING FAILED (exit %s) before any gate: %s\n' "$script" "$rc" "$logs/$slug/list"
      excerpt "$logs/$slug/list"
      continue
    fi
    printf '\nHARNESS %s: gates no leg on this host; not checked\n' "$script"
    continue
  fi
  checked=$((checked + 1))
  groups=$(printf '%s\n' "$legs" | sed -n 's/.*#\([A-Za-z0-9_-][A-Za-z0-9_-]*\)$/\1/p' | LC_ALL=C sort -u)
  printf '\nHARNESS %s: %s leg(s), %s group(s)\n' "$script" "$(printf '%s\n' "$legs" | wc -l | tr -d ' ')" \
    "$(printf '%s' "$groups" | grep -c . | tr -d ' ')"
  if [ "$rc" != 0 ]; then
    bad_listing=$((bad_listing + 1))
    printf '  LISTING FAILED (exit %s): %s\n' "$rc" "$logs/$slug/list"
    excerpt "$logs/$slug/list"
  fi
  member "$logs/$slug/full" "$script"
  full=$rc
  # Each skip as LEG<TAB>LABEL: a label one leg skips in the full run says
  # nothing about another leg skipping it alone.
  LC_ALL=C sort -u "$logs/$slug/full.skips" -o "$logs/$slug/full.skips"
  if [ "$full" = 0 ]; then
    printf '  full run: PASS (%s check(s) skipped)\n' "$(wc -l <"$logs/$slug/full.skips" | tr -d ' ')"
    verdict='passes in the full run'
  else
    red=$((red + 1))
    printf '  full run: FAIL (exit %s): %s\n' "$full" "$logs/$slug/full"
    excerpt "$logs/$slug/full"
    verdict="the full run is red too (exit $full)"
  fi
  n=0
  while IFS= read -r selection; do
    [ -n "$selection" ] || continue
    # The legs the selection must run: a group's by their tag, a leg itself.
    case $selection in
      '#'*)
        regex="$(ere_escape "$selection")\$" expected=
        while IFS= read -r leg; do
          [ "${leg##*#}" != "${selection#\#}" ] || expected=$expected$leg$'\n'
        done <<<"$legs" ;;
      *) regex="^$(ere_escape "$selection")\$" expected=$selection$'\n' ;;
    esac
    # What --only REGEX selects, by the matcher harness_leg uses: anything
    # else is this checker's defect, reported without a run.
    matched= want=0
    while IFS= read -r leg; do
      [[ $leg =~ $regex ]] || continue
      matched=$matched$leg$'\n' want=$((want + 1))
    done <<<"$legs"
    n=$((n + 1))
    if [ "$matched" != "$expected" ]; then
      findings=$((findings + 1))
      printf '  ALONE MISS  %s: --only %s selects %s leg(s), not exactly the ones meant; not run\n' \
        "$selection" "$regex" "$want"
      continue
    fi
    log="$logs/$slug/alone-$n"
    member "$log" "$script" --only "$regex"
    ran=$(sed -n 's/^  RAN   //p' "$log")
    new_skips=$(LC_ALL=C sort -u "$log.skips" | LC_ALL=C comm -23 - "$logs/$slug/full.skips")
    if [ "$rc" != 0 ]; then
      findings=$((findings + 1))
      printf '  ALONE FAIL  %s (exit %s; %s): %s\n' "$selection" "$rc" "$verdict" "$log"
      excerpt "$log"
    elif ! grep -qxF "ran $want leg(s):" "$log" || [ "$ran"$'\n' != "$expected" ]; then
      findings=$((findings + 1))
      printf '  ALONE MISS  %s: --only %s did not run exactly the %s leg(s) selected: %s\n' \
        "$selection" "$regex" "$want" "$log"
    elif [ -n "$new_skips" ]; then
      findings=$((findings + 1))
      printf '  ALONE SKIP  %s: skips alone what the full run decided: %s\n' "$selection" "$log"
      printf '%s\n' "$new_skips" | awk -F '\t' '{ printf "        SKIP  %s  (in leg: %s)\n", $2, $1 }'
    else
      printf '  ALONE PASS  %s\n' "$selection"
    fi
  done < <(printf '%s\n' "$groups" | sed '/./s/^/#/'; printf '%s\n' "$legs")
done
printf '\nharness-legs-alone: %s harness(es) checked; %s red full run(s), %s failed listing(s), %s finding(s) run alone\n' \
  "$checked" "$red" "$bad_listing" "$findings"
status=$((findings + red + bad_listing > 0 ? 1 : 0))
if [ "$keep" = 1 ] || [ "$status" != 0 ]; then
  printf 'logs: %s\n' "$logs"
else
  rm -rf "$logs"
fi
exit "$status"
