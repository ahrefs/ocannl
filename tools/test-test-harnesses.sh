#!/usr/bin/env bash
# Opposing controls for the aggregate: run all selected members, preserve
# failures, select the CI groups, hold a strict run to its list of expected
# skips, run the CI shell job's own check (--ci-shell) with that job's
# absences, hold its hidden-dune list to the skips the hidden dune caused,
# run the real-dune job's check (--ci-real-dune) on the selected legs alone,
# pass a leg declared not applicable with no list naming it while an unlisted
# skip beside it is still refused,
# refuse arguments (or a malformed list) before executing work, take the
# fleet's run-time slot through the shared probe (gh-ocannl-1576), and run
# exactly what a change reaches (--affected, with the real `affected
# --harnesses` on a fixture change reaching two members).
# --jobs runs members beside one another and prints each one's output whole,
# in the manifest's order (gh-ocannl-2002).
# Usage: tools/test-test-harnesses.sh [--keep|--help]
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_require python3 perl git
harness_scratch test-harnesses
# No run reaches this host's fleet: the subject's slot (gh-ocannl-1576) is
# driven by the fake fleet-worker.sh below, in the slot legs alone.
export OCANNL_TOOL_FLEET_WORKER=none
fixture="$TMP/repository with spaces"
mkdir -p "$fixture/tools"
fixture=$(CDPATH= cd -- "$fixture" && pwd)
cp "$here/test-harnesses.sh" "$here/fleet-worker-candidates.sh" "$fixture/tools/"
subject="$fixture/tools/test-harnesses.sh"
# Derive fixture membership through the public interface, rather than keeping
# another list that would go stale as CI gains a harness.
"$subject" --list >"$TMP/all" || exit 2
"$subject" --shell --list >"$TMP/shell" || exit 2
"$subject" --toolchain --list >"$TMP/toolchain" || exit 2
"$subject" --promotion --list >"$TMP/promotion" || exit 2
[ -s "$TMP/shell" ] && [ -s "$TMP/toolchain" ] && [ -s "$TMP/promotion" ] || exit 2
cat "$TMP/shell" "$TMP/toolchain" "$TMP/promotion" | sort >"$TMP/groups"
sort "$TMP/all" >"$TMP/sorted"
if cmp -s "$TMP/groups" "$TMP/sorted" \
   && [ "$(sort -u "$TMP/all" | wc -l)" -eq "$(wc -l <"$TMP/all")" ]; then
  report 0 'groups partition the complete list without duplicates'
else report 1 'groups partition the complete list without duplicates'; fi
while IFS= read -r script; do
  mkdir -p "$fixture/$(dirname "$script")"
  cat >"$fixture/$script" <<'STUB'
#!/usr/bin/env bash
name=${0#"$FIXTURE/"}
printf '%s\n' "$name" >>"$RECORD"
[ "$PWD" = "$FIXTURE" ] || exit 90
if [ -n "${BARRIER:-}" ]; then
  # Passed only beside another member: two of them here at once, ever.
  : >"$BARRIER/$$"
  i=0
  while [ "$(ls "$BARRIER" | wc -l)" -lt 2 ]; do
    i=$((i + 1))
    [ "$i" -lt 60 ] || exit 9
    sleep 0.05
  done
fi
[ -z "${ECHO_NAME:-}" ] || echo "ran $name"
# Held until stopped: a TERM ends it as a harness's cleanup trap would.
if [ "$name" = "${HOLD_MEMBER:-}" ]; then trap 'exit 143' TERM; while :; do sleep 0.1; done; fi
[ -z "${SLOT_RECORD:-}" ] || printf '%s slot=%s\n' "$name" "${FAKE_SLOT_HELD-none}" >>"$SLOT_RECORD"
if [ -n "${ENV_RECORD:-}" ]; then
  wake=unset
  [ -z "${OCANNL_TOOL_SWEEP_WAKE_LAB+x}" ] || { wake=present; [ -e "$OCANNL_TOOL_SWEEP_WAKE_LAB" ] || wake=absent; }
  printf '%s no-dune=%s fleet=%s wake-lab=%s\n' "$name" "${OCANNL_TOOL_HARNESS_NO_DUNE-unset}" \
    "${FLEET_BOXES-unset}" "$wake" >>"$ENV_RECORD"
fi
[ -z "${ARGS_RECORD:-}" ] ||
  printf '%s strict=%s args=%s\n' "$name" "${OCANNL_TOOL_HARNESS_STRICT-unset}" "$*" >>"$ARGS_RECORD"
[ -z "${ARGV_RECORD:-}" ] || { printf '%s' "$name"; printf ' [%s]' "$@"; echo; } >>"$ARGV_RECORD"
[ "$name" != "${PASS_MEMBER:-}" ] || printf 'PASS  %s\n' "$PASS_LABEL"
if [ "$name" = "${FAIL_MEMBER:-}" ]; then
  if [ -n "${FAIL_SIGNAL:-}" ]; then
    # Background launchers can inherit ignored INT; reset the member itself
    # before the real signal so the oracle is independent of that parent.
    exec perl -e '$SIG{$ARGV[0]} = "DEFAULT"; kill $ARGV[0], $$;' "$FAIL_SIGNAL"
  fi
  exit 7
fi
if [ "$name" = "${SKIP_MEMBER:-}" ]; then
  . "$SUPPORT"
  harness_args
  if [ -n "${SKIP_DUNE:-}" ]; then
    harness_real_dune || skip "$SKIP_LABEL" "$harness_dune_absent"
  else
    skip "$SKIP_LABEL" 'not on the fixture platform'
  fi
  finish
fi
if [ "$name" = "${NA_MEMBER:-}" ]; then
  . "$SUPPORT"
  harness_args
  na_leg() { harness_not_applicable 'its subject is absent from the fixture platform'; return; }
  harness_leg_run "$NA_LEG" na_leg
  [ -z "${NA_SKIP:-}" ] || skip "$SKIP_LABEL" 'not on the fixture platform'
  finish
fi
exit 0
STUB
  chmod +x "$fixture/$script"
done <"$TMP/all"
export FIXTURE="$fixture" RECORD="$TMP/record"
run() {
  : >"$RECORD"
  got=0
  (cd "$TMP" && "$subject" "$@") >"$TMP/out" 2>&1 || got=$?
}
for group in all shell toolchain promotion; do
  case $group in all) run ;; *) run "--$group" ;; esac
  if [ "$got" = 0 ] && cmp -s "$TMP/$group" "$RECORD"; then
    report 0 "$group executes every selected member from the repository root"
  else report 1 "$group executes every selected member from the repository root"; fi
done
export FAIL_MEMBER
FAIL_MEMBER=$(head -n 1 "$TMP/all")
run
if [ "$got" = 1 ] && cmp -s "$TMP/all" "$RECORD" \
   && grep -q 'exit: 7' "$TMP/out" && grep -q '1 failed$' "$TMP/out"; then
  report 0 'a failing first harness stays red while every later harness runs'
else report 1 'a failing first harness stays red while every later harness runs'; fi
# A fail-fast twin must lose the continuation property, proving the oracle
# above sees the defect instead of merely observing a failing command.
python3 - "$subject" "$TMP/fail-fast.sh" <<'PY_CONTROL'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text()
anchor = '"$root/$script" ${args[@]+"${args[@]}"} || rc=$?'
assert anchor in text
Path(sys.argv[2]).write_text(text.replace(anchor, '"$root/$script" || exit 1'))
PY_CONTROL
# Keep the twin at the same tools/ depth so root resolution stays identical.
cp "$TMP/fail-fast.sh" "$fixture/tools/fail-fast.sh"
chmod +x "$fixture/tools/fail-fast.sh"
: >"$RECORD"
twin_rc=0
"$fixture/tools/fail-fast.sh" >"$TMP/twin-out" 2>&1 || twin_rc=$?
if [ "$twin_rc" = 1 ] && ! cmp -s "$TMP/all" "$RECORD"; then
  report 0 'the fail-fast twin loses the all-members oracle'
else report 1 'the fail-fast twin loses the all-members oracle'; fi
# Actual signals on the member preserve cancellation and suppress later work.
export FAIL_SIGNAL FAIL_MEMBER
FAIL_MEMBER=$(head -n 1 "$TMP/all")
head -n 1 "$TMP/all" >"$TMP/first"
python3 - "$subject" "$fixture/tools/continue.sh" <<'PY_INTERRUPT'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text()
assert '129|130|143)' in text
Path(sys.argv[2]).write_text(text.replace('129|130|143)', '250)'))
PY_INTERRUPT
chmod +x "$fixture/tools/continue.sh"
for signal in HUP INT TERM; do
  case $signal in HUP) want=129 ;; INT) want=130 ;; TERM) want=143 ;; esac
  FAIL_SIGNAL=$signal
  run
  if [ "$got" = "$want" ] && cmp -s "$TMP/first" "$RECORD" \
     && grep -q "^harnesses: interrupted (exit $want)$" "$TMP/out"; then
    report 0 "$signal stops after the interrupted member and preserves exit $want"
  else report 1 "$signal stops after the interrupted member and preserves exit $want"; fi
  : >"$RECORD"
  continuation_rc=0
  "$fixture/tools/continue.sh" >"$TMP/continuation-out" 2>&1 || continuation_rc=$?
  if [ "$continuation_rc" = 1 ] && cmp -s "$TMP/all" "$RECORD"; then
    report 0 "$signal continuation twin launches the later members and loses the interrupt oracle"
  else report 1 "$signal continuation twin launches the later members and loses the interrupt oracle"; fi
done
unset FAIL_MEMBER FAIL_SIGNAL
# --expected-skips (gh-ocannl-1236): one shell member skips through the real
# harness-support.sh. Listed, the run is green and the skip counted; unlisted,
# the member is red; listed but not skipped, the tier is red; a list the
# aggregator inherited rather than was given relaxes nothing.
export SUPPORT="$here/../scripts/harness-support.sh" SKIP_MEMBER SKIP_LABEL
SKIP_MEMBER=$(head -n 1 "$TMP/shell")
SKIP_LABEL='queries: a fixture platform check'
printf '# expected\n\n%s\t%s\n' "$SKIP_MEMBER" "$SKIP_LABEL" >"$TMP/expected-listed"
printf '# nothing expected\n' >"$TMP/expected-none"
run --shell --expected-skips expected-listed
if [ "$got" = 0 ] && cmp -s "$TMP/shell" "$RECORD" \
   && grep -qxF "SKIP  $SKIP_LABEL" "$TMP/out" \
   && grep -qxF '      not on the fixture platform (expected: listed in expected-listed)' "$TMP/out" \
   && grep -q '^harnesses: [0-9]* run, 0 failed; 0 expected skip(s) did not occur$' "$TMP/out"; then
  report 0 'a listed skip is counted, from a list named relative to the caller'
else report 1 'a listed skip is counted, from a list named relative to the caller' "exit $got; $(cat "$TMP/out")"; fi
run --shell --expected-skips "$TMP/expected-none"
if [ "$got" = 1 ] && cmp -s "$TMP/shell" "$RECORD" && grep -qxF "FAIL  $SKIP_LABEL" "$TMP/out" \
   && grep -qxF "HARNESS $SKIP_MEMBER exit: 1" "$TMP/out"; then
  report 0 'an unlisted skip fails its member and the tier, every member still run'
else report 1 'an unlisted skip fails its member and the tier, every member still run' "exit $got; $(cat "$TMP/out")"; fi
(OCANNL_TOOL_HARNESS_STRICT=1 OCANNL_TOOL_HARNESS_EXPECTED_SKIPS=$TMP/expected-listed \
  OCANNL_TOOL_HARNESS_SKIP_RECORD=$TMP/ambient-record; export OCANNL_TOOL_HARNESS_STRICT \
  OCANNL_TOOL_HARNESS_EXPECTED_SKIPS OCANNL_TOOL_HARNESS_SKIP_RECORD; run --shell; exit "$got") || got=$?
if [ "$got" = 1 ] && grep -qxF "FAIL  $SKIP_LABEL" "$TMP/out"; then
  report 0 'an inherited list relaxes no strict run'
else report 1 'an inherited list relaxes no strict run' "exit $got; $(cat "$TMP/out")"; fi
SKIP_MEMBER=
run --shell --expected-skips "$TMP/expected-listed"
if [ "$got" = 1 ] && cmp -s "$TMP/shell" "$RECORD" \
   && grep -qxF "harnesses: expected skip did not occur: $(head -n 1 "$TMP/shell"): $SKIP_LABEL" "$TMP/out" \
   && grep -q '^harnesses: [0-9]* run, 0 failed; 1 expected skip(s) did not occur$' "$TMP/out"; then
  report 0 'a listed skip that no longer occurs fails the tier'
else report 1 'a listed skip that no longer occurs fails the tier' "exit $got; $(cat "$TMP/out")"; fi
# The list's three line forms are an allowlist: each other shape refuses the
# run before a member starts, naming the line.
first=$(head -n 1 "$TMP/shell")
for bad in space cr tabs empty unknown group twice; do
  case $bad in
    space) printf '%s %s\n' "$first" "$SKIP_LABEL" ;;
    cr) printf '%s\t%s\r\n' "$first" "$SKIP_LABEL" ;;
    tabs) printf '%s\t%s\tmore\n' "$first" "$SKIP_LABEL" ;;
    empty) printf '%s\t\n' "$first" ;;
    unknown) printf 'tools/no-such-harness.sh\t%s\n' "$SKIP_LABEL" ;;
    group) printf '%s\t%s\n' "$(head -n 1 "$TMP/toolchain")" "$SKIP_LABEL" ;;
    twice) printf '%s\t%s\n%s\t%s\n' "$first" "$SKIP_LABEL" "$first" "$SKIP_LABEL" ;;
  esac >"$TMP/expected-bad"
  case $bad in twice) want_line=2 ;; *) want_line=1 ;; esac
  run --shell --expected-skips "$TMP/expected-bad"
  if [ "$got" = 2 ] && [ ! -s "$RECORD" ] && grep -q "expected-bad:$want_line: " "$TMP/out"; then
    report 0 "a list line with $bad is refused before any harness executes"
  else report 1 "a list line with $bad is refused before any harness executes" "exit $got; $(cat "$TMP/out")"; fi
done
for missing in '' "$TMP/no-such-list"; do
  run --shell --expected-skips "$missing"
  if [ "$got" = 2 ] && [ ! -s "$RECORD" ]; then
    report 0 "--expected-skips '${missing##*/}' is refused before any harness executes"
  else report 1 "--expected-skips '${missing##*/}' is refused before any harness executes" "exit $got; $(cat "$TMP/out")"; fi
done
# --ci-shell (gh-ocannl-1593): the job's list from the repository, on macOS
# with the Darwin list joined to it, and every member run with dune hidden,
# FLEET_BOXES unset and the wake-lab script pointed at nothing. A stand-in
# `uname` says which platform the run is on.
mkdir -p "$TMP/uname-Linux" "$TMP/uname-Darwin"
for os in Linux Darwin; do
  printf '#!/bin/sh\necho %s\n' "$os" >"$TMP/uname-$os/uname"
  chmod +x "$TMP/uname-$os/uname"
done
SKIP_MEMBER=$first
ci_list=$fixture/tools/harness-skips-ci-shell.txt darwin_list=$fixture/tools/harness-skips-ci-shell-darwin.txt
dune_list=$fixture/tools/harness-skips-ci-shell-dune.txt
printf '# dune\n' >"$dune_list"
ci_run() { # OS ARGS...: --ci-shell run from TMP on that platform, the fleet's facts ambient
  local os=$1
  shift
  : >"$TMP/env-record"
  got=0
  (PATH=$TMP/uname-$os:$PATH FLEET_BOXES='box-a box-b' OCANNL_TOOL_SWEEP_WAKE_LAB=$TMP/out \
    ENV_RECORD=$TMP/env-record; export PATH FLEET_BOXES OCANNL_TOOL_SWEEP_WAKE_LAB ENV_RECORD; run "$@"
   exit "$got") || got=$?
}
env_all() { # WANT: every member ran, each with exactly these facts
  [ "$(sed 's/ .*//' "$TMP/env-record")" = "$(cat "$TMP/shell")" ] &&
    [ "$(sed 's/^[^ ]* //' "$TMP/env-record" | sort -u)" = "$1" ]
}
printf '# job\n%s\t%s\n' "$SKIP_MEMBER" "$SKIP_LABEL" >"$ci_list"
printf '# darwin\n' >"$darwin_list"
for os in Linux Darwin; do
  ci_run "$os" --ci-shell
  if [ "$got" = 0 ] && cmp -s "$TMP/shell" "$RECORD" && env_all 'no-dune=1 fleet=unset wake-lab=absent' \
     && grep -qxF "SKIP  $SKIP_LABEL" "$TMP/out" \
     && grep -q '^harnesses: [0-9]* run, 0 failed; 0 expected skip(s) did not occur; 0 hidden-dune skip(s) misfiled$' "$TMP/out"; then
    report 0 "--ci-shell on $os runs the shell tier against the job's list with dune and the fleet tables hidden"
  else report 1 "--ci-shell on $os runs the shell tier against the job's list with dune and the fleet tables hidden" \
    "exit $got; $(cat "$TMP/out" "$TMP/env-record")"; fi
done
# The Darwin list is read on macOS alone: a skip only it lists passes there
# and fails its member on Linux.
printf '# job\n' >"$ci_list"
printf '# darwin\n%s\t%s\n' "$SKIP_MEMBER" "$SKIP_LABEL" >"$darwin_list"
ci_run Darwin --ci-shell
darwin_rc=$got
darwin_out=$(cat "$TMP/out")
ci_run Linux --ci-shell
if [ "$darwin_rc" = 0 ] && grep -qxF "SKIP  $SKIP_LABEL" <<<"$darwin_out" \
   && [ "$got" = 1 ] && grep -qxF "FAIL  $SKIP_LABEL" "$TMP/out"; then
  report 0 '--ci-shell expects the Darwin list on macOS alone'
else report 1 '--ci-shell expects the Darwin list on macOS alone' "Darwin exit $darwin_rc; Linux exit $got; $darwin_out; $(cat "$TMP/out")"; fi
# Its entries are held like the job's: one that no longer occurs on macOS
# fails the tier there.
SKIP_MEMBER=
ci_run Darwin --ci-shell
if [ "$got" = 1 ] && grep -qxF "harnesses: expected skip did not occur: $first: $SKIP_LABEL" "$TMP/out"; then
  report 0 '--ci-shell on macOS fails on a Darwin-listed skip that no longer occurs'
else report 1 '--ci-shell on macOS fails on a Darwin-listed skip that no longer occurs' "exit $got; $(cat "$TMP/out")"; fi
SKIP_MEMBER=$first
# Without --ci-shell nothing is hidden: an ambient switch is cleared, and the
# fleet's facts pass through.
got=0
(OCANNL_TOOL_HARNESS_NO_DUNE=1; export OCANNL_TOOL_HARNESS_NO_DUNE; SKIP_MEMBER=; ci_run Darwin --shell; exit "$got") || got=$?
if [ "$got" = 0 ] && env_all "no-dune=unset fleet=box-a box-b wake-lab=present"; then
  report 0 'without --ci-shell an ambient OCANNL_TOOL_HARNESS_NO_DUNE is cleared and the fleet facts kept'
else report 1 'without --ci-shell an ambient OCANNL_TOOL_HARNESS_NO_DUNE is cleared and the fleet facts kept' \
  "exit $got; $(cat "$TMP/out" "$TMP/env-record")"; fi
# The hidden-dune list (gh-ocannl-1650) is read on every platform, and holds
# exactly the skips the hidden dune caused, told by their reason: listed
# there, such a skip passes; listed in the job's list, it fails the tier, as
# does a dune-list skip whose reason is another.
export SKIP_DUNE
printf '# job\n' >"$ci_list"
printf '# darwin\n' >"$darwin_list"
printf '# dune\n%s\t%s\n' "$SKIP_MEMBER" "$SKIP_LABEL" >"$dune_list"
for os in Linux Darwin; do
  SKIP_DUNE=1 ci_run "$os" --ci-shell
  if [ "$got" = 0 ] && grep -qxF "SKIP  $SKIP_LABEL" "$TMP/out" \
     && grep -qF 'dune hidden by OCANNL_TOOL_HARNESS_NO_DUNE=1' "$TMP/out" \
     && grep -q '; 0 hidden-dune skip(s) misfiled$' "$TMP/out"; then
    report 0 "--ci-shell on $os expects a hidden dune's skip from the dune list"
  else report 1 "--ci-shell on $os expects a hidden dune's skip from the dune list" "exit $got; $(cat "$TMP/out")"; fi
done
printf '# job\n%s\t%s\n' "$SKIP_MEMBER" "$SKIP_LABEL" >"$ci_list"
printf '# dune\n' >"$dune_list"
SKIP_DUNE=1 ci_run Linux --ci-shell
if [ "$got" = 1 ] && grep -qxF "harnesses: a skip of the hidden dune, listed outside harness-skips-ci-shell-dune.txt: $SKIP_MEMBER: $SKIP_LABEL" "$TMP/out" \
   && grep -q '^harnesses: [0-9]* run, 0 failed; 0 expected skip(s) did not occur; 1 hidden-dune skip(s) misfiled$' "$TMP/out"; then
  report 0 "--ci-shell fails a hidden dune's skip listed in the job's list"
else report 1 "--ci-shell fails a hidden dune's skip listed in the job's list" "exit $got; $(cat "$TMP/out")"; fi
printf '# job\n' >"$ci_list"
printf '# dune\n%s\t%s\n' "$SKIP_MEMBER" "$SKIP_LABEL" >"$dune_list"
SKIP_DUNE= ci_run Linux --ci-shell
if [ "$got" = 1 ] && grep -qxF "harnesses: listed in harness-skips-ci-shell-dune.txt, and skipped for another reason than the hidden dune: $SKIP_MEMBER: $SKIP_LABEL" "$TMP/out" \
   && grep -q '; 1 hidden-dune skip(s) misfiled$' "$TMP/out"; then
  report 0 '--ci-shell fails a dune-list skip whose reason is not the hidden dune'
else report 1 '--ci-shell fails a dune-list skip whose reason is not the hidden dune' "exit $got; $(cat "$TMP/out")"; fi
SKIP_DUNE=
# A leg declared not applicable (gh-ocannl-1839) is no skip: --ci-shell passes
# it with no list naming it, on either platform, and the same member's
# undeclared skip beside it is still refused -- the N/A relaxes nothing.
export NA_MEMBER NA_LEG='1: a leg whose subject the fixture platform lacks' NA_SKIP
NA_MEMBER=$first SKIP_MEMBER=
printf '# job\n' >"$ci_list"
printf '# dune\n' >"$dune_list"
for os in Linux Darwin; do
  NA_SKIP= ci_run "$os" --ci-shell
  if [ "$got" = 0 ] && grep -qxF "N/A   $NA_LEG" "$TMP/out" \
     && grep -qxF 'all legs passed (0 skipped, 1 leg(s) not applicable)' "$TMP/out" \
     && grep -q '^harnesses: [0-9]* run, 0 failed; 0 expected skip(s) did not occur; 0 hidden-dune skip(s) misfiled$' "$TMP/out"; then
    report 0 "--ci-shell on $os passes a leg declared not applicable, with no list naming it"
  else report 1 "--ci-shell on $os passes a leg declared not applicable, with no list naming it" "exit $got; $(cat "$TMP/out")"; fi
done
NA_SKIP=1 ci_run Linux --ci-shell
if [ "$got" = 1 ] && grep -qxF "N/A   $NA_LEG" "$TMP/out" && grep -qxF "FAIL  $SKIP_LABEL" "$TMP/out" \
   && grep -qxF "HARNESS $first exit: 1" "$TMP/out"; then
  report 0 '--ci-shell still refuses an undeclared skip beside a leg declared not applicable (negative control)'
else report 1 '--ci-shell still refuses an undeclared skip beside a leg declared not applicable (negative control)' "exit $got; $(cat "$TMP/out")"; fi
NA_MEMBER= NA_SKIP= SKIP_MEMBER=$first
printf '# job\n%s\t%s\n' "$SKIP_MEMBER" "$SKIP_LABEL" >"$ci_list"
printf '# dune\n' >"$dune_list"
for args in group expected missing missing-dune; do
  case $args in
    group) ci_run Linux --ci-shell --shell ;;
    expected) ci_run Linux --expected-skips "$ci_list" --ci-shell ;;
    missing) rm -f "$darwin_list"; ci_run Darwin --ci-shell ;;
    missing-dune) printf '# darwin\n' >"$darwin_list"; rm -f "$dune_list"; ci_run Linux --ci-shell ;;
  esac
  if [ "$got" = 2 ] && [ ! -s "$RECORD" ]; then
    report 0 "--ci-shell with $args is refused before any harness executes"
  else report 1 "--ci-shell with $args is refused before any harness executes" "exit $got; $(cat "$TMP/out")"; fi
done
unset SKIP_MEMBER SKIP_LABEL SKIP_DUNE
# --ci-real-dune (gh-ocannl-1650): only the members with a leg selector run,
# each strict and given its selector as --only, and the tier is green only
# when every check the dune list names printed PASS. The members it runs are
# read through --list, the selectors from the subject's own table.
printf '# dune\n' >"$dune_list"
"$subject" --ci-real-dune --list >"$TMP/real-dune" || exit 2
[ -s "$TMP/real-dune" ] || exit 2
rd_member=$(head -n 1 "$TMP/real-dune")
# The selector table is the subject's own function, defined here from its text.
eval "$(sed -n '/^real_dune_legs() {$/,/^}$/p' "$subject")"
rd_selector=$(real_dune_legs | awk -F '\t' -v m="$rd_member" '$1 == m { print $2 }')
[ -n "$rd_selector" ] || report 1 '--ci-real-dune: the fixture reads the selector of its first member' "$rd_member"
rd_other=$(grep -vxF -f "$TMP/real-dune" "$TMP/shell" | head -n 1)
export ARGS_RECORD=$TMP/args-record PASS_MEMBER PASS_LABEL
rd_label='against a real dune: a fixture check'
rd_run() { : >"$ARGS_RECORD"; run "$@"; }
printf '# dune\n%s\t%s\n' "$rd_member" "$rd_label" >"$dune_list"
PASS_MEMBER=$rd_member PASS_LABEL=$rd_label
rd_run --ci-real-dune
if [ "$got" = 0 ] && cmp -s "$TMP/real-dune" "$RECORD" \
   && [ "$(cat "$ARGS_RECORD")" = "$(real_dune_legs | awk -F '\t' '{ print $1 " strict=1 args=--only " $2 }')" ] \
   && grep -q '^harnesses: [0-9]* run, 0 failed; 0 real-dune check(s) did not pass$' "$TMP/out"; then
  report 0 '--ci-real-dune runs only the selected members, strict with their selectors, and passes on every listed PASS'
else report 1 '--ci-real-dune runs only the selected members, strict with their selectors, and passes on every listed PASS' \
  "exit $got; $(cat "$TMP/out" "$ARGS_RECORD")"; fi
# Negative controls: a listed check that printed no PASS (its leg left out
# of the selector, or skipped), and a red member.
PASS_LABEL='a different check'
rd_run --ci-real-dune
if [ "$got" = 1 ] && grep -qxF "harnesses: a real-dune check did not PASS (unreached by its selector, skipped or red): $rd_member: $rd_label" "$TMP/out" \
   && grep -q '; 1 real-dune check(s) did not pass$' "$TMP/out"; then
  report 0 '--ci-real-dune fails a listed check that printed no PASS'
else report 1 '--ci-real-dune fails a listed check that printed no PASS' "exit $got; $(cat "$TMP/out")"; fi
PASS_LABEL=$rd_label
FAIL_MEMBER=$rd_member
export FAIL_MEMBER
rd_run --ci-real-dune
if [ "$got" = 1 ] && grep -qxF "HARNESS $rd_member exit: 7" "$TMP/out"; then
  report 0 '--ci-real-dune fails a red member'
else report 1 '--ci-real-dune fails a red member' "exit $got; $(cat "$TMP/out")"; fi
unset FAIL_MEMBER
for args in group expected both unselected missing; do
  printf '# dune\n%s\t%s\n' "$rd_member" "$rd_label" >"$dune_list"
  case $args in
    group) rd_run --ci-real-dune --shell ;;
    expected) rd_run --ci-real-dune --expected-skips "$dune_list" ;;
    both) rd_run --ci-real-dune --ci-shell ;;
    unselected)
      if [ -z "$rd_other" ]; then report 1 '--ci-real-dune: the fixture finds a member with no selector'; continue; fi
      printf '%s\t%s\n' "$rd_other" "$rd_label" >>"$dune_list"
      rd_run --ci-real-dune ;;
    missing) rm -f "$dune_list"; rd_run --ci-real-dune ;;
  esac
  if [ "$got" = 2 ] && [ ! -s "$RECORD" ]; then
    report 0 "--ci-real-dune with $args is refused before any harness executes"
  else report 1 "--ci-real-dune with $args is refused before any harness executes" "exit $got; $(cat "$TMP/out")"; fi
done
unset ARGS_RECORD PASS_MEMBER PASS_LABEL
# Empty membership is an invocation error, never a vacuous green tier.
python3 - "$subject" "$fixture/tools/empty.sh" <<'PY_EMPTY'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text()
head, tail = text.split("cat <<'HARNESS_LIST'\n", 1)
_, tail = tail.split('HARNESS_LIST\n', 1)
Path(sys.argv[2]).write_text(head + "cat <<'HARNESS_LIST'\nHARNESS_LIST\n" + tail)
PY_EMPTY
chmod +x "$fixture/tools/empty.sh"
empty_rc=0
"$fixture/tools/empty.sh" --list >"$TMP/empty-out" 2>&1 || empty_rc=$?
if [ "$empty_rc" = 2 ] && grep -q '^no harnesses selected$' "$TMP/empty-out"; then
  report 0 'empty membership is refused even for read-only listing'
else report 1 'empty membership is refused even for read-only listing'; fi
for args in unknown conflicting; do
  case $args in unknown) run --unknown ;; conflicting) run --shell --toolchain ;; esac
  if [ "$got" = 2 ] && [ ! -s "$RECORD" ]; then
    report 0 "$args arguments refuse before any harness executes"
  else report 1 "$args arguments refuse before any harness executes"; fi
done
run --list
if [ "$got" = 0 ] && [ ! -s "$RECORD" ] && cmp -s "$TMP/all" "$TMP/out"; then
  report 0 'listing membership executes no harness'
else report 1 'listing membership executes no harness'; fi
# The fleet's run-time slot (gh-ocannl-1576): on a box whose fleet-worker.sh
# answers the probe, the whole run happens inside ONE `execution slot --cpu`,
# taken after every refusal and never for --list; a refused slot is SLOT
# REFUSED (exit 75) with no member run; the run's own status wins over the
# wrapper's. The fake records each call and, holding the slot, marks the
# command's environment (FAKE_SLOT_HELD) for the member stubs to record.
fake_fw=$TMP/fake-fleet-worker.sh
cat >"$fake_fw" <<'FAKE_FW'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$FAKE_FW_CALLS"
if [ "$1 $2 $3" = "execution slot --probe" ]; then
  [ "${FAKE_FW_MODE:-}" = noprobe ] && { echo "fleet-worker.sh: execution slot [--wait <seconds>] ..." >&2; exit 2; }
  echo "EXECUTION SLOT PROBE fakebox 4 1"
  exit 0
fi
while [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done
shift
case ${FAKE_FW_MODE:-} in
  refuse) echo "EXECUTION SLOT REFUSED fakebox: all 4 slots busy after 0s"; exit 1 ;;
  unreachable) echo "EXECUTION SLOT UNREACHABLE anchor: registry unread, no slot taken"; exit 4 ;;
  # The command started, then was killed before it could record its status.
  killed) : >"$OCANNL_TOOL_HARNESSES_IN_SLOT"; exit 137 ;;
esac
rc=0
FLEET_SLOT_HELD=1 FAKE_SLOT_HELD=1 "$@" || rc=$?
[ "${FAKE_FW_MODE:-}" != wrapfail ] || exit 3
exit "$rc"
FAKE_FW
chmod +x "$fake_fw"
export SLOT_RECORD=$TMP/slot-record FAKE_FW_CALLS=$TMP/fw-calls
unset FAIL_MEMBER FAIL_SIGNAL SKIP_MEMBER
slot_run() { # [VAR=VALUE...] -- SUBJECT_ARG...
  local env_args=()
  while [ "$1" != -- ]; do env_args+=("$1"); shift; done
  shift
  : >"$SLOT_RECORD"
  : >"$FAKE_FW_CALLS"
  : >"$RECORD"
  got=0
  (cd "$TMP" && env OCANNL_TOOL_FLEET_WORKER="$fake_fw" ${env_args[@]+"${env_args[@]}"} \
    "$subject" "$@") >"$TMP/out" 2>&1 || got=$?
  slot_calls=$(grep -v -- '--probe' "$FAKE_FW_CALLS" | sed 's/ -- .*//')
  probes=$(grep -c -- '--probe' "$FAKE_FW_CALLS")
}
# Every member of the group ran, each with the slot held (WANT: 1) or not (none).
members_ran_in() { # GROUP WANT
  [ -s "$TMP/$1" ] && cmp -s "$TMP/$1" "$RECORD" &&
    [ "$(sed "s/\$/ slot=$2/" "$TMP/$1")" = "$(cat "$SLOT_RECORD")" ]
}
slot_run -- --shell
if [ "$got" = 0 ] && [ "$probes" = 1 ] && [ "$slot_calls" = "execution slot --wait 600 --cpu" ] &&
   grep -qxF -- "execution slot --wait 600 --cpu -- ${BASH:-bash} $subject --shell" "$FAKE_FW_CALLS" &&
   members_ran_in shell 1 &&
   grep -Fq "harnesses: taking one of fakebox's 4 fleet correctness slots" "$TMP/out"; then
  report 0 'on a fleet box the run takes one --cpu slot, probed once, every member inside it'
else report 1 'on a fleet box the run takes one --cpu slot, probed once, every member inside it' \
  "exit $got; probes $probes; calls: $(tr '\n' '|' <"$FAKE_FW_CALLS"); members: $(tr '\n' '|' <"$SLOT_RECORD")"; fi
slot_run OCANNL_TOOL_SLOT_WAIT=7 -- --promotion
if [ "$got" = 0 ] && [ "$slot_calls" = "execution slot --wait 7 --cpu" ] && members_ran_in promotion 1; then
  report 0 "OCANNL_TOOL_SLOT_WAIT bounds the slot's wait"
else report 1 "OCANNL_TOOL_SLOT_WAIT bounds the slot's wait" "exit $got; calls: $(tr '\n' '|' <"$FAKE_FW_CALLS")"; fi
slot_run FAIL_MEMBER="$(head -n 1 "$TMP/shell")" -- --shell
if [ "$got" = 1 ] && [ -n "$slot_calls" ] && members_ran_in shell 1 && grep -q '1 failed$' "$TMP/out"; then
  report 0 'a failing member in the slot keeps the run red, every member run'
else report 1 'a failing member in the slot keeps the run red, every member run' "exit $got; $(tail -n 3 "$TMP/out")"; fi
for mode in refused unreachable; do
  slot_run FAKE_FW_MODE=${mode%d} -- --shell
  if [ "$got" = 75 ] && [ ! -s "$RECORD" ] && grep -q '^harnesses: FAILED (exit 75): SLOT REFUSED' "$TMP/out"; then
    report 0 "a slot $mode is SLOT REFUSED (exit 75), and no member ran"
  else report 1 "a slot $mode is SLOT REFUSED (exit 75), and no member ran" "exit $got; $(tail -n 3 "$TMP/out")"; fi
done
slot_run FAKE_FW_MODE=killed -- --shell
if [ "$got" = 137 ] && grep -q '^harnesses: FAILED (exit 137): the harnesses under the fleet slot ended without a status' "$TMP/out"; then
  report 0 'a run killed in the slot is told apart from a refused slot'
else report 1 'a run killed in the slot is told apart from a refused slot' "exit $got; $(tail -n 3 "$TMP/out")"; fi
slot_run FAKE_FW_MODE=wrapfail -- --shell
if [ "$got" = 0 ] && members_ran_in shell 1; then
  report 0 "the run's recorded status wins over the wrapper's own failure after it"
else report 1 "the run's recorded status wins over the wrapper's own failure after it" "exit $got"; fi
# The negative control of the slot oracle: a fleet-worker.sh without the
# probe (or the slot turned off) runs the members directly, outside any slot.
slot_run FAKE_FW_MODE=noprobe -- --shell
if [ "$got" = 0 ] && [ "$probes" = 1 ] && [ -z "$slot_calls" ] && members_ran_in shell none &&
   ! grep -qi 'slot' "$TMP/out"; then
  report 0 'a fleet-worker.sh without the probe runs the members directly, silently'
else report 1 'a fleet-worker.sh without the probe runs the members directly, silently' \
  "exit $got; calls: $(tr '\n' '|' <"$FAKE_FW_CALLS"); members: $(tr '\n' '|' <"$SLOT_RECORD")"; fi
for args in --list --unknown expected-skips; do
  case $args in
    expected-skips)
      printf 'not a list line\n' >"$TMP/malformed-list"
      slot_run -- --shell --expected-skips "$TMP/malformed-list" ;;
    *) slot_run -- "$args" ;;
  esac
  want=2
  [ "$args" != --list ] || want=0
  if [ "$got" = "$want" ] && [ ! -s "$FAKE_FW_CALLS" ] && [ ! -s "$RECORD" ]; then
    report 0 "$args asks the fleet nothing and runs no member"
  else report 1 "$args asks the fleet nothing and runs no member" "exit $got; calls: $(tr '\n' '|' <"$FAKE_FW_CALLS")"; fi
done
# --affected --since BASE: the real `affected --harnesses`, this fixture being
# a git checkout whose last commit changes one file two members reach -- the
# first of the shell tier by its text, whole, and the first of the promotion
# tier by a declaration, through an --only whose ERE holds a space, so it is
# seen to reach the member as one word. The rest of the tier stays unrun.
cp "$here/test-run.sh" "$here/affected-tests.sh" "$fixture/tools/" || exit 2
# Whatever test-run.sh sources, derived from it rather than listed here.
af_staged=0
while IFS= read -r rel; do
  mkdir -p "$fixture/$(dirname "$rel")" && cp "$here/../$rel" "$fixture/$rel" || exit 2
  af_staged=$((af_staged + 1))
done < <(sed -n 's/^\. \([A-Za-z0-9_./-]*\)$/\1/p' "$here/test-run.sh")
[ "$af_staged" -ge 3 ] || exit 2
af_whole=$(head -n 1 "$TMP/shell") af_leg=$(head -n 1 "$TMP/promotion") af_path=tools/fixture-reached.tbl
af_ere='^2 b:'
printf ': %s\n' "$af_path" >>"$fixture/$af_whole"
{ head -n 1 "$fixture/$af_leg"; printf "# affected-reads: %s --only '%s'\n" "$af_path" "$af_ere"
  sed 1d "$fixture/$af_leg"; } >"$TMP/af-stub" && cat "$TMP/af-stub" >"$fixture/$af_leg" || exit 2
af_git() { git -C "$fixture" -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }
printf 'before\n' >"$fixture/$af_path"
af_git init -q && af_git add -A && af_git commit -qm base || exit 2
printf 'after\n' >"$fixture/$af_path"
af_git commit -qam change || exit 2
printf '%s\n' "$af_whole" "$af_leg" >"$TMP/af-names"
printf '%s\n' "$af_whole" "$af_leg --only '$af_ere'" >"$TMP/af-commands"
printf '%s []\n%s [--only] [%s]\n' "$af_whole" "$af_leg" "$af_ere" >"$TMP/af-argv"
export ARGV_RECORD=$TMP/argv-record
# Ran exactly the two commands, each once with its own argv.
af_ran() { cmp -s "$TMP/af-names" "$RECORD" && cmp -s "$TMP/af-argv" "$ARGV_RECORD"; }
af_slot() { : >"$ARGV_RECORD"; slot_run "$@"; }
af_detail() { echo "exit $got; calls: $(tr '\n' '|' <"$FAKE_FW_CALLS"); ran: $(tr '\n' '|' <"$RECORD"); argv: $(tr '\n' '|' <"$ARGV_RECORD"); $(tail -n 5 "$TMP/out")"; }
bash "$fixture/tools/test-run.sh" affected --harnesses --since HEAD~1 >"$TMP/af-real" 2>"$TMP/af-real-err" ||
  report 1 '--affected: the fixture change is asked of the real affected --harnesses' "$(cat "$TMP/af-real-err")"
if [ "$(sort "$TMP/af-real")" = "$(sort "$TMP/af-commands")" ]; then
  report 0 '--affected: the fixture change reaches two members, one whole and one by --only'
else report 1 '--affected: the fixture change reaches two members, one whole and one by --only' "$(cat "$TMP/af-real" "$TMP/af-real-err")"; fi
af_slot -- --affected --since HEAD~1 --list
if [ "$got" = 0 ] && [ ! -s "$FAKE_FW_CALLS" ] && [ ! -s "$RECORD" ] &&
   [ "$(grep -v '^affected: ' "$TMP/out" | sort)" = "$(sort "$TMP/af-real")" ]; then
  report 0 '--affected --list prints the commands affected --harnesses prints and takes no slot'
else report 1 '--affected --list prints the commands affected --harnesses prints and takes no slot' "$(af_detail)"; fi
af_slot -- --affected --since HEAD~1
if [ "$got" = 0 ] && af_ran && [ "$probes" = 1 ] && [ "$slot_calls" = "execution slot --wait 600 --cpu" ] &&
   grep -qxF -- "execution slot --wait 600 --cpu -- ${BASH:-bash} $subject --affected --since HEAD~1" "$FAKE_FW_CALLS" &&
   [ "$(sed 's/.* //' "$SLOT_RECORD" | sort -u)" = slot=1 ] &&
   [ "$(grep -c '^harnesses: ' "$TMP/out")" = 2 ] && [ "$(tail -n 1 "$TMP/out")" = 'harnesses: 2 run, 0 failed' ] &&
   [ "$(grep -c '^affected: 2 harness command(s)' "$TMP/out")" = 1 ]; then
  report 0 '--affected runs exactly the two commands, in one --cpu slot under one footer'
else report 1 '--affected runs exactly the two commands, in one --cpu slot under one footer' "$(af_detail)"; fi
af_slot FAIL_MEMBER="$af_whole" -- --affected --since HEAD~1
if [ "$got" = 1 ] && af_ran && [ "$(tail -n 1 "$TMP/out")" = 'harnesses: 2 run, 1 failed' ]; then
  report 0 '--affected: a failing member keeps the run red, the other command still run'
else report 1 '--affected: a failing member keeps the run red, the other command still run' "$(af_detail)"; fi
af_slot FAKE_FW_MODE=noprobe -- --affected --since HEAD~1
if [ "$got" = 0 ] && af_ran && [ -z "$slot_calls" ] && [ "$(sed 's/.* //' "$SLOT_RECORD" | sort -u)" = slot=none ]; then
  report 0 '--affected off the fleet runs the same commands directly'
else report 1 '--affected off the fleet runs the same commands directly' "$(af_detail)"; fi
# The one command prepush closes with: the form test-run.sh prints, run as it
# stands, so the two tools cannot drift apart on its spelling.
af_form=$(sed -n '/test-run: prepush: run them/s/.*: tools\/test-harnesses\.sh \([^%]*\)%s.*/\1/p' "$here/test-run.sh")
# shellcheck disable=SC2086 # the printed form's words, split on purpose
af_slot -- $af_form HEAD~1
if [ -n "$af_form" ] && [ "$got" = 0 ] && af_ran; then
  report 0 "the command prepush closes with runs what the change reaches"
else report 1 "the command prepush closes with runs what the change reaches" "form [$af_form]; $(af_detail)"; fi
# A change reaching no harness is a result: green, no slot, nothing run.
af_slot -- --affected --since HEAD
if [ "$got" = 0 ] && [ ! -s "$FAKE_FW_CALLS" ] && [ ! -s "$RECORD" ] &&
   [ "$(tail -n 1 "$TMP/out")" = 'harnesses: 0 run, 0 failed (this change reaches no harness)' ]; then
  report 0 '--affected on a change reaching no harness is green, with no slot and no member run'
else report 1 '--affected on a change reaching no harness is green, with no slot and no member run' "$(af_detail)"; fi
# --files asks the same of named paths, and a group keeps its own members of
# what the paths reach: one shell member here, one promotion member, no
# toolchain one, which is a result like any change reaching nothing.
af_slot -- --affected --files "$af_path"
if [ "$got" = 0 ] && af_ran; then
  report 0 '--affected --files runs what those paths reach, as --since runs a diff'
else report 1 '--affected --files runs what those paths reach, as --since runs a diff' "$(af_detail)"; fi
af_slot -- --affected --files no/such/path.sh
if [ "$got" = 0 ] && [ ! -s "$RECORD" ] && [ ! -s "$FAKE_FW_CALLS" ]; then
  report 0 '--affected --files on a path no harness reaches runs nothing, with no slot'
else report 1 '--affected --files on a path no harness reaches runs nothing, with no slot' "$(af_detail)"; fi
af_slot -- --shell --affected --list --files "$af_path"
if [ "$got" = 0 ] && [ "$(grep -v '^affected: ' "$TMP/out")" = "$af_whole" ]; then
  report 0 '--shell --affected --list keeps the shell member, leaving out the promotion one'
else report 1 '--shell --affected --list keeps the shell member, leaving out the promotion one' "$(af_detail)"; fi
af_slot -- --promotion --affected --since HEAD~1
if [ "$got" = 0 ] && [ "$(cat "$RECORD")" = "$af_leg" ] && [ "$(cat "$ARGV_RECORD")" = "$af_leg [--only] [$af_ere]" ]; then
  report 0 '--promotion --affected runs the promotion member alone, with its --only'
else report 1 '--promotion --affected runs the promotion member alone, with its --only' "$(af_detail)"; fi
af_slot -- --toolchain --affected --since HEAD~1
if [ "$got" = 0 ] && [ ! -s "$FAKE_FW_CALLS" ] && [ ! -s "$RECORD" ] &&
   [ "$(tail -n 1 "$TMP/out")" = 'harnesses: 0 run, 0 failed (this change reaches no harness of the toolchain group)' ]; then
  report 0 'a group the change does not reach is green, with no slot and no member run'
else report 1 'a group the change does not reach is green, with no slot and no member run' "$(af_detail)"; fi
for args in no-since since-alone files-alone no-files both-asks option-path expected ci-shell ci-real-dune bad-base; do
  case $args in
    no-since) af_slot -- --affected ;;
    since-alone) af_slot -- --since HEAD~1 ;;
    files-alone) af_slot -- --files "$af_path" ;;
    no-files) af_slot -- --affected --files ;;
    both-asks) af_slot -- --affected --since HEAD~1 --files "$af_path" ;;
    option-path) af_slot -- --affected --files "$af_path" --list ;;
    expected) af_slot -- --affected --since HEAD~1 --expected-skips "$TMP/expected-none" ;;
    ci-shell) af_slot -- --affected --since HEAD~1 --ci-shell ;;
    ci-real-dune) af_slot -- --affected --since HEAD~1 --ci-real-dune ;;
    bad-base) af_slot -- --affected --since no-such-revision ;;
  esac
  if [ "$got" = 2 ] && [ ! -s "$FAKE_FW_CALLS" ] && [ ! -s "$RECORD" ] &&
     { [ "$args" != bad-base ] || grep -qF 'not a revision: no-such-revision' "$TMP/out"; }; then
    report 0 "--affected with $args is refused before a slot or any harness"
  else report 1 "--affected with $args is refused before a slot or any harness" "$(af_detail)"; fi
done
# The commands are read through an allowlist, never evaluated: a stand-in for
# the ask prints each shape; only the two forms pass, the --only form as one
# word again. The first case is the control: the stand-in itself runs members.
mv "$fixture/tools/test-run.sh" "$TMP/test-run.aside" || exit 2
printf '#!/usr/bin/env bash\ncat "$AF_COMMANDS"\nexit "${AF_RC:-0}"\n' >"$fixture/tools/test-run.sh"
export AF_COMMANDS=$TMP/af-standin
printf "%s --only '^1 a:'\n" "$af_leg" >"$AF_COMMANDS"
af_slot -- --affected --since HEAD~1
if [ "$got" = 0 ] && [ "$(cat "$RECORD")" = "$af_leg" ] && [ "$(cat "$ARGV_RECORD")" = "$af_leg [--only] [^1 a:]" ]; then
  report 0 '--affected: a stand-in printing the --only form runs that member alone'
else report 1 '--affected: a stand-in printing the --only form runs that member alone' "$(af_detail)"; fi
for bad in unknown evaluated option double-quoted two-selectors twice failed-ask; do
  case $bad in
    unknown) echo tools/no-such-harness.sh ;;
    evaluated) printf '%s; touch %s\n' "$af_whole" "$TMP/af-evaluated" ;;
    option) printf '%s --keep\n' "$af_whole" ;;
    double-quoted) printf '%s --only "^1:"\n' "$af_whole" ;;
    two-selectors) printf "%s --only '^1:' --only '^2:'\n" "$af_whole" ;;
    twice) printf '%s\n%s\n' "$af_whole" "$af_whole" ;;
    failed-ask) echo "$af_whole" ;;
  esac >"$AF_COMMANDS"
  case $bad in failed-ask) rc_var=AF_RC=3 ;; *) rc_var=AF_RC=0 ;; esac
  af_slot "$rc_var" -- --affected --since HEAD~1
  if [ "$got" = 2 ] && [ ! -s "$FAKE_FW_CALLS" ] && [ ! -s "$RECORD" ] && [ ! -e "$TMP/af-evaluated" ] &&
     grep -q '^test-harnesses.sh: --affected: ' "$TMP/out"; then
    report 0 "--affected refuses $bad before a slot or any harness"
  else report 1 "--affected refuses $bad before a slot or any harness" "$(af_detail)"; fi
done
# A group narrows what the allowlist passes, not what it reads: a good line of
# another group is left out, a line naming no member still refuses.
{ echo "$af_leg"; echo tools/no-such-harness.sh; } >"$AF_COMMANDS"
af_slot -- --shell --affected --since HEAD~1
if [ "$got" = 2 ] && [ ! -s "$FAKE_FW_CALLS" ] && [ ! -s "$RECORD" ] && grep -q '^test-harnesses.sh: --affected: ' "$TMP/out"; then
  report 0 '--shell --affected still refuses a line naming no member'
else report 1 '--shell --affected still refuses a line naming no member' "$(af_detail)"; fi
mv "$TMP/test-run.aside" "$fixture/tools/test-run.sh" || exit 2
unset AF_COMMANDS
# Negative controls of the run's oracle: a twin at the same tools/ depth that
# ignores the selection, and one that drops the selector, each lose it.
for mutant in every-member no-selector; do
  case $mutant in
    every-member) af_from='[ -n "$reached" ] || continue' af_to=':' ;;
    no-selector) af_from='[ -z "${run_selectors[$1]}" ] || args=(--only "${run_selectors[$1]}")' af_to=':' ;;
  esac
  af_twin=$fixture/tools/affected-twin.sh
  if ! AF_FROM=$af_from AF_TO=$af_to perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{AF_FROM}\E/$ENV{AF_TO}/g; END { exit($n == 1 ? 0 : 1) }' \
      "$subject" >"$af_twin"; then
    report 1 "negative control: an --affected twin ($mutant)" 'the mutation anchor does not occur exactly once'
    continue
  fi
  chmod +x "$af_twin"
  : >"$RECORD"
  : >"$ARGV_RECORD"
  twin_rc=0
  (cd "$TMP" && "$af_twin" --affected --since HEAD~1) >"$TMP/twin-out" 2>&1 || twin_rc=$?
  if [ "$twin_rc" = 0 ] && ! af_ran; then
    report 0 "negative control: an --affected twin ($mutant) loses the exactly-these-commands oracle"
  else report 1 "negative control: an --affected twin ($mutant) loses the exactly-these-commands oracle" "exit $twin_rc; $(tr '\n' '|' <"$ARGV_RECORD")"; fi
  rm -f "$af_twin"
done
unset ARGV_RECORD
unset SLOT_RECORD FAKE_FW_CALLS
# One probe for every fleet asker, one rerun helper for the two tools that run
# themselves again inside the slot (gh-ocannl-1576, gh-ocannl-1841): the
# slot-takers call fleet_slot_probe (test-run.sh) or fleet_slot_rerun
# (fmt-check.sh, test-harnesses.sh), the sweep's registry reader calls
# fleet_slot_probe, and the GPU-token reading ships fleet_slot_probe to the box;
# outside tools/fleet-worker-candidates.sh no code line of a tool asks
# `execution slot --probe` itself (comments, and the `answered 'execution slot
# --probe'` of a message saying none did, may name it), and neither rerunning
# tool runs `execution slot` itself. The tools' own harnesses (test-*.sh), whose
# fakes answer the verbs, are not askers.
probe_copies=
for asker in test-run.sh:'fleet_slot_probe ' fmt-check.sh:'fleet_slot_rerun ' \
  test-harnesses.sh:'fleet_slot_rerun ' fleet-registry.sh:'fleet_slot_probe ' \
  gpu-tokens.sh:'declare -f fleet_slot_alarm fleet_slot_count fleet_slot_probe'; do
  grep -qF -- "${asker#*:}" "$here/${asker%%:*}" || probe_copies="$probe_copies ${asker%%:*}:no-call"
done
for tool in "$here"/*.sh; do
  case ${tool##*/} in test-*.sh | fleet-worker-candidates.sh) continue ;; esac
  ! grep -v '^[[:space:]]*#' "$tool" | grep -v 'answered [^ ]*execution slot --probe' |
    grep -q 'slot --probe' || probe_copies="$probe_copies ${tool##*/}:own-probe"
done
for taker in fmt-check.sh test-harnesses.sh; do
  ! grep -v '^[[:space:]]*#' "$here/$taker" | grep -q 'execution slot' || probe_copies="$probe_copies $taker:own-rerun"
done
if [ -z "$probe_copies" ] && grep -q 'slot --probe' "$here/fleet-worker-candidates.sh" &&
   grep -q 'execution slot --wait' "$here/fleet-worker-candidates.sh"; then
  report 0 'the fleet askers share one probe and one rerun helper'
else report 1 'the fleet askers share one probe and one rerun helper' "${probe_copies# }"; fi
# --jobs N (gh-ocannl-2002): members beside one another. Each stub passes only
# once two of them have been running at once, so one at a time fails them all:
# the control.
unset FAIL_MEMBER FAIL_SIGNAL
export BARRIER="$TMP/barrier" ECHO_NAME=1
rm -rf "$BARRIER"; mkdir "$BARRIER"
run --shell --jobs 3
# What a run of one at a time prints, member by member, in the manifest's order.
while IFS= read -r script; do
  printf '\nHARNESS %s (shell)\nran %s\nHARNESS %s exit: 0\n' "$script" "$script" "$script"
done <"$TMP/shell" >"$TMP/jobs-want"
printf '\nharnesses: %s run, 0 failed\n' "$(wc -l <"$TMP/shell" | tr -d ' ')" >>"$TMP/jobs-want"
if [ "$got" = 0 ] && cmp -s "$TMP/jobs-want" "$TMP/out" && cmp -s <(sort "$TMP/shell") <(sort "$RECORD"); then
  report 0 '--jobs 3 runs members beside one another, each output whole and in the manifest order'
else
  report 1 '--jobs 3 runs members beside one another, each output whole and in the manifest order' \
    "exit $got: $(diff "$TMP/jobs-want" "$TMP/out" | head -n 8 | tr '\n' '|')"
fi
rm -rf "$BARRIER"; mkdir "$BARRIER"
head -n 2 "$TMP/shell" >"$TMP/jobs-two"
run --shell --jobs 1
if [ "$got" = 1 ] && grep -qxF "HARNESS $(head -n 1 "$TMP/shell") exit: 9" "$TMP/out"; then
  report 0 'negative control: one at a time, no member ever runs beside another'
else report 1 'negative control: one at a time, no member ever runs beside another' "exit $got: $(tail -n 3 "$TMP/out" | tr '\n' '|')"; fi
unset BARRIER
export FAIL_MEMBER
FAIL_MEMBER=$(sed -n 2p "$TMP/shell")
run --shell --jobs 4
if [ "$got" = 1 ] && cmp -s <(sort "$TMP/shell") <(sort "$RECORD") &&
   grep -qxF "HARNESS $FAIL_MEMBER exit: 7" "$TMP/out" && grep -q ' run, 1 failed$' "$TMP/out"; then
  report 0 '--jobs: a failing member stays red while every other member runs'
else report 1 '--jobs: a failing member stays red while every other member runs' "exit $got: $(tail -n 3 "$TMP/out" | tr '\n' '|')"; fi
# A member that ends on a signal ends the run with that status as soon as it
# is seen to have ended: the first member, still running, is stopped, and the
# third, whose turn its slot was, never starts -- though the second's output
# comes after the first's in the log.
export FAIL_SIGNAL=TERM HOLD_MEMBER
HOLD_MEMBER=$(head -n 1 "$TMP/shell")
run --shell --jobs 2
if [ "$got" = 143 ] && grep -q '^harnesses: interrupted (exit 143)$' "$TMP/out" &&
   [ "$(head -n 2 "$TMP/shell" | sort)" = "$(sort "$RECORD")" ] && ! grep -q ' run, ' "$TMP/out" &&
   grep -qxF "HARNESS $HOLD_MEMBER exit: 143" "$TMP/out" && grep -qxF "HARNESS $FAIL_MEMBER exit: 143" "$TMP/out"; then
  report 0 '--jobs: an interrupted member ends the run with its status at once, stopping the member beside it and starting no further one'
else report 1 '--jobs: an interrupted member ends the run with its status at once, stopping the member beside it and starting no further one' "exit $got: ran $(tr '\n' ' ' <"$RECORD"); $(tail -n 4 "$TMP/out" | tr '\n' '|')"; fi
unset HOLD_MEMBER
unset FAIL_MEMBER FAIL_SIGNAL ECHO_NAME
jobs_bad=
for jobs_argv in '--jobs' '--jobs 0' '--jobs x' '--jobs -1' '--jobs 2x'; do
  # shellcheck disable=SC2086
  run --shell $jobs_argv
  { [ "$got" = 2 ] && [ ! -s "$RECORD" ] && grep -q 'needs a positive integer' "$TMP/out"; } ||
    jobs_bad="$jobs_bad [$jobs_argv: exit $got]"
done
if [ -z "$jobs_bad" ]; then report 0 '--jobs refuses anything but a positive integer before a member starts'
else report 1 '--jobs refuses anything but a positive integer before a member starts' "$jobs_bad"; fi

finish
