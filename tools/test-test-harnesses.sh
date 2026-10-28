#!/usr/bin/env bash
# Opposing controls for the aggregate: run all selected members, preserve
# failures, select the CI groups, hold a strict run to its list of expected
# skips, run the CI shell job's own check (--ci-shell) with that job's
# absences, hold its hidden-dune list to the skips the hidden dune caused,
# run the real-dune job's check (--ci-real-dune) on the selected legs alone,
# and refuse arguments (or a malformed list) before executing work.
# Usage: tools/test-test-harnesses.sh [--keep|--help]
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_require python3 perl
harness_scratch test-harnesses
fixture="$TMP/repository with spaces"
mkdir -p "$fixture/tools"
fixture=$(CDPATH= cd -- "$fixture" && pwd)
cp "$here/test-harnesses.sh" "$fixture/tools/test-harnesses.sh"
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
if [ -n "${ENV_RECORD:-}" ]; then
  wake=unset
  [ -z "${OCANNL_TOOL_SWEEP_WAKE_LAB+x}" ] || { wake=present; [ -e "$OCANNL_TOOL_SWEEP_WAKE_LAB" ] || wake=absent; }
  printf '%s no-dune=%s fleet=%s wake-lab=%s\n' "$name" "${OCANNL_TOOL_HARNESS_NO_DUNE-unset}" \
    "${FLEET_BOXES-unset}" "$wake" >>"$ENV_RECORD"
fi
[ -z "${ARGS_RECORD:-}" ] ||
  printf '%s strict=%s args=%s\n' "$name" "${OCANNL_TOOL_HARNESS_STRICT-unset}" "$*" >>"$ARGS_RECORD"
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
assert '"$root/$script" || rc=$?' in text
Path(sys.argv[2]).write_text(text.replace('"$root/$script" || rc=$?',
                                        '"$root/$script" || exit 1'))
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
   && [ "$(cat "$ARGS_RECORD")" = "$rd_member strict=1 args=--only $rd_selector" ] \
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
finish
