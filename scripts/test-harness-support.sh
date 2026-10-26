#!/usr/bin/env bash
# Exercise shared harness lifecycle, counted skips (and a strict run's list of
# expected ones), the shared real-dune probe and its hiding switch, the --only
# leg filter with its coverage account, function leg bodies that abandon
# themselves or are abandoned at a failed errexit check, and reason-specific
# rejection.
# Usage: scripts/test-harness-support.sh [--keep] [--help]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/harness-support.sh"
harness_args "$@"
harness_scratch harness-support
cat >"$TMP/probe.sh" <<'PROBE'
#!/usr/bin/env bash
# Shared support fixture help.
# Last help line.
. "$SUPPORT"
harness_args "$@"
harness_scratch contract
printf '%s\n' "$TMP" >"$RECORD"
case $MODE in
  skip) harness_require definitely_missing_harness_tool ;;
  skips) skip 'first platform check' 'no first platform'; skip 'second platform check' 'no second platform' ;;
  fail) report 1 'injected failure' ;;
  signal) kill -TERM "$$" ;;
  abort) exit 50 ;;
  errexit) set -e; (exit 7) ;;
  unfinished) exit 0 ;;
  dune)
    # Asked twice, as several legs of one harness ask it.
    for asked in 1 2; do
      if harness_real_dune; then echo "FOUND $harness_dune"; else echo "ABSENT $harness_dune_absent"; fi
    done
    echo "PATH $PATH" ;;
  legs)
    # Each body leaves a trace, so a filtered leg is seen not to have run.
    if harness_leg 'alpha leg #odd'; then echo alpha >>"$RECORD.ran"; report 0 'alpha body'; fi
    if harness_leg 'beta leg #even'; then
      echo beta >>"$RECORD.ran"; report 0 'beta body'
      [ -z "${BETA_SKIP:-}" ] || skip 'beta check' 'no beta host'
    fi
    if harness_leg 'gamma leg #odd'; then echo gamma >>"$RECORD.ran"; report "${GAMMA_RC:-0}" 'gamma body'; fi ;;
  abandon)
    # Function bodies (harness_leg_run): delta gives up partway, epsilon's last
    # command fails after its verdict, and zeta, when asked, exits mid-leg.
    delta() { echo delta >>"$RECORD.ran"; harness_abandon 'the fixture is not there'; return; echo late >>"$RECORD.ran"; }
    epsilon() { echo epsilon >>"$RECORD.ran"; report 0 'epsilon body'; false; }
    zeta() { echo zeta >>"$RECORD.ran"; exit 9; }
    harness_leg_run 'delta leg #odd' delta
    harness_leg_run 'epsilon leg #even' epsilon
    [ -z "${ZETA:-}" ] || harness_leg_run 'zeta leg #odd' zeta ;;
  errexit_legs)
    # Bodies checked under errexit (harness_leg_run_errexit), in a harness that
    # itself runs under errexit: a check failing in a nested function, a body
    # that passes and leaves state behind in its own shell, an exit (after a
    # command that failed harmlessly inside a substitution), an unbound
    # variable, a report 1, a skip, a harness_abandon -- and a call inside a
    # condition.
    set -e
    check() { grep -q never-there "$RECORD"; }
    eta() { echo eta >>"$RECORD.ran"; check; echo late >>"$RECORD.ran"; }
    theta() { echo theta >>"$RECORD.ran"; report 0 'theta body'; export THETA_LEAK=1; cd /; }
    iota() {
      echo "iota ${THETA_LEAK:-unset} $([ "$PWD" = "$PROBE_DIR" ] && echo home)" >>"$RECORD.ran"
      harmless=$(false; echo fine)
      exit 9
    }
    kappa() { echo kappa >>"$RECORD.ran"; set -u; : "$kappa_unbound"; echo late >>"$RECORD.ran"; }
    lambda() { echo lambda >>"$RECORD.ran"; report 1 'lambda label' 'lambda detail'; echo late >>"$RECORD.ran"; }
    mu() { echo mu >>"$RECORD.ran"; skip 'mu check' 'no mu host'; echo late >>"$RECORD.ran"; }
    nu() { echo nu >>"$RECORD.ran"; harness_abandon 'nu gave up'; echo late >>"$RECORD.ran"; }
    xi() { echo xi >>"$RECORD.ran"; false; echo late >>"$RECORD.ran"; }
    harness_leg_restore() { echo restore >>"$RECORD.ran"; [ -z "${RESTORE_FAILS:-}" ]; }
    PROBE_DIR=$PWD
    harness_leg_run_errexit 'eta leg #odd' eta
    harness_leg_run_errexit 'theta leg #even' theta
    for leg in iota kappa lambda mu nu; do harness_leg_run_errexit "$leg leg" "$leg"; done
    if harness_leg_run_errexit 'xi leg' xi; then :; fi ;;
esac
finish
PROBE
export SUPPORT=$HERE/harness-support.sh RECORD=$TMP/record MODE=pass
# The probes' skip verdict is the subject; an ambient strict mode, or an
# ambient list of expected skips, is not.
unset OCANNL_TOOL_HARNESS_STRICT OCANNL_TOOL_HARNESS_EXPECTED_SKIPS \
  OCANNL_TOOL_HARNESS_NAME OCANNL_TOOL_HARNESS_SKIP_RECORD OCANNL_TOOL_HARNESS_NO_DUNE
probe() {
  local rc=0
  bash "$TMP/probe.sh" "$@" >"$TMP/out" 2>&1 || rc=$?
  return "$rc"
}
probe; rc=$?
path=$(cat "$RECORD")
if [ "$rc" = 0 ] && [ ! -d "$path" ] && grep -q 'all legs passed (0 skipped)' "$TMP/out" && ! grep -q 'aborted before finish' "$TMP/out"; then
  report 0 'success cleans scratch and counts zero skips'
else report 1 'success cleans scratch and counts zero skips'; fi
MODE=skip probe; rc=$?
if [ "$rc" = 0 ] && grep -q '^SKIP ' "$TMP/out" && grep -q 'all legs passed (1 skipped)' "$TMP/out"; then
  report 0 'missing host capability counts a skip'
else report 1 'missing host capability counts a skip'; fi
OCANNL_TOOL_HARNESS_STRICT=1 MODE=skip probe; rc=$?
if [ "$rc" = 1 ] && grep -q '^FAIL  every leg requiring definitely_missing_harness_tool$' "$TMP/out" \
   && grep -q '1 leg(s) failed (0 skipped)' "$TMP/out"; then
  report 0 'strict mode turns an undecided leg into a failure'
else report 1 'strict mode turns an undecided leg into a failure'; fi
# A strict run's list of expected platform skips (gh-ocannl-1236): the one
# exact HARNESS<TAB>LABEL line it names is a counted, recorded skip; the
# other skip of the same member stays red, and so does the listed one when
# the harness name, or the record the aggregator reads back, is missing.
printf '# comment\nprobe\tfirst platform check\nprobe first platform check\n' >"$TMP/expected"
expected_probe() { # NAME RECORD
  : >"$TMP/skip-record"
  OCANNL_TOOL_HARNESS_STRICT=1 OCANNL_TOOL_HARNESS_EXPECTED_SKIPS=$TMP/expected \
    OCANNL_TOOL_HARNESS_NAME=$1 OCANNL_TOOL_HARNESS_SKIP_RECORD=$2 MODE=skips probe
}
expected_probe probe "$TMP/skip-record"; rc=$?
if [ "$rc" = 1 ] && grep -q '^SKIP  first platform check$' "$TMP/out" \
   && grep -q '^      no first platform (expected: listed in expected)$' "$TMP/out" \
   && grep -q '^FAIL  second platform check$' "$TMP/out" \
   && grep -q '1 leg(s) failed (1 skipped)' "$TMP/out" \
   && [ "$(cat "$TMP/skip-record")" = "probe	first platform check" ]; then
  report 0 'strict mode counts and records only the skip its list names'
else report 1 'strict mode counts and records only the skip its list names' "exit $rc; $(cat "$TMP/out")"; fi
for control in name record; do
  case $control in
    name) expected_probe other "$TMP/skip-record"; rc=$? ;;
    record) expected_probe probe ''; rc=$? ;;
  esac
  if [ "$rc" = 1 ] && grep -q '^FAIL  first platform check$' "$TMP/out" \
     && grep -q '2 leg(s) failed (0 skipped)' "$TMP/out" && [ ! -s "$TMP/skip-record" ]; then
    report 0 "strict mode lists nothing without the matching $control"
  else report 1 "strict mode lists nothing without the matching $control" "exit $rc; $(cat "$TMP/out")"; fi
done
# The shared real-dune probe (gh-ocannl-1593), on a copy of the support file
# whose tools/opam-env.sh puts a stand-in dune on PATH: found through that
# environment, the caller's PATH left its own; hidden by
# OCANNL_TOOL_HARNESS_NO_DUNE=1 with the reason a skip prints; any other value
# one FAIL however often the probe is asked; and a twin that never reads the
# switch finds the dune it should hide, so the hidden check sees the switch.
# The probes' PATH is a directory of links to the few tools a harness runs,
# so no dune the host has in its system directories decides any of this.
dune_repo=$TMP/dune-repo dune_tools=$TMP/dune-tools dune_tools_missing=
mkdir -p "$dune_repo/scripts" "$dune_repo/tools" "$dune_repo/opam-bin" "$TMP/dune-twin/scripts" "$dune_tools"
for tool in mktemp sed awk tr rm dirname cat grep; do
  tool_path=$(command -v "$tool") && [ "${tool_path#/}" != "$tool_path" ] &&
    ln -s "$tool_path" "$dune_tools/$tool" || dune_tools_missing="$dune_tools_missing $tool"
done
cp "$HERE/harness-support.sh" "$dune_repo/scripts/harness-support.sh"
printf '#!/bin/sh\nexit 0\n' >"$dune_repo/opam-bin/dune"
chmod +x "$dune_repo/opam-bin/dune"
printf 'PATH="%s:$PATH"\n' "$dune_repo/opam-bin" >"$dune_repo/tools/opam-env.sh"
cp -R "$dune_repo/tools" "$TMP/dune-twin/" # its environment finds the same stand-in
sed 's/case ${OCANNL_TOOL_HARNESS_NO_DUNE-} in/case "" in/' "$HERE/harness-support.sh" \
  >"$TMP/dune-twin/scripts/harness-support.sh"
dune_probe() { # SUPPORT_ROOT [SWITCH_VALUE]
  local rc=0
  (
    [ $# -lt 2 ] || export OCANNL_TOOL_HARNESS_NO_DUNE="$2"
    SUPPORT=$1/scripts/harness-support.sh PATH=$dune_tools MODE=dune "$BASH" "$TMP/probe.sh"
  ) >"$TMP/out" 2>&1 || rc=$?
  return "$rc"
}
found="FOUND $dune_repo/opam-bin/dune"
hidden='ABSENT dune hidden by OCANNL_TOOL_HARNESS_NO_DUNE=1, as on the toolchain-free CI job'
if [ -n "$dune_tools_missing" ]; then
  report 1 'the real-dune probe: its fixture PATH' "no executable for:$dune_tools_missing"
elif cmp -s "$HERE/harness-support.sh" "$TMP/dune-twin/scripts/harness-support.sh"; then
  report 1 'the real-dune probe: its switch-blind twin' 'the switch could not be cut out of harness-support.sh'
else
  for value in unset 0; do
    case $value in unset) dune_probe "$dune_repo" ;; *) dune_probe "$dune_repo" "$value" ;; esac; rc=$?
    if [ "$rc" = 0 ] && [ "$(grep -cxF -- "$found" "$TMP/out")" = 2 ] \
       && grep -qxF "PATH $dune_tools" "$TMP/out" && grep -q 'all legs passed (0 skipped)' "$TMP/out"; then
      report 0 "the real-dune probe finds a dune through the opam environment, the switch $value"
    else report 1 "the real-dune probe finds a dune through the opam environment, the switch $value" "exit $rc; $(cat "$TMP/out")"; fi
  done
  dune_probe "$dune_repo" 1; rc=$?
  if [ "$rc" = 0 ] && [ "$(grep -cxF -- "$hidden" "$TMP/out")" = 2 ] && ! grep -q '^FOUND' "$TMP/out" \
     && grep -q 'all legs passed (0 skipped)' "$TMP/out"; then
    report 0 'OCANNL_TOOL_HARNESS_NO_DUNE=1 hides the dune, naming itself as the reason'
  else report 1 'OCANNL_TOOL_HARNESS_NO_DUNE=1 hides the dune, naming itself as the reason' "exit $rc; $(cat "$TMP/out")"; fi
  dune_probe "$dune_repo" yes; rc=$?
  if [ "$rc" = 1 ] && [ "$(grep -c '^FAIL  ' "$TMP/out")" = 1 ] \
     && grep -qxF 'FAIL  the real-dune probe: OCANNL_TOOL_HARNESS_NO_DUNE is 0 or 1' "$TMP/out" \
     && [ "$(grep -cxF "ABSENT OCANNL_TOOL_HARNESS_NO_DUNE='yes' is neither 0 nor 1" "$TMP/out")" = 2 ] \
     && grep -q '1 leg(s) failed (0 skipped)' "$TMP/out"; then
    report 0 'a switch neither 0 nor 1 is one FAIL however often the probe is asked, and hides the dune'
  else report 1 'a switch neither 0 nor 1 is one FAIL however often the probe is asked, and hides the dune' "exit $rc; $(cat "$TMP/out")"; fi
  dune_probe "$TMP/dune-twin" 1; rc=$?
  if [ "$rc" = 0 ] && [ "$(grep -cxF -- "$found" "$TMP/out")" = 2 ] && ! grep -qxF -- "$hidden" "$TMP/out"; then
    report 0 'negative control: a probe blind to the switch finds the dune it should hide'
  else report 1 'negative control: a probe blind to the switch finds the dune it should hide' "exit $rc; $(cat "$TMP/out")"; fi
fi
MODE=fail probe; rc=$?
if [ "$rc" = 1 ] && grep -q '1 leg(s) failed (0 skipped)' "$TMP/out" && ! grep -q 'aborted before finish' "$TMP/out"; then
  report 0 'failure controls exit and footer'
else report 1 'failure controls exit and footer'; fi
probe --keep; rc=$?
path=$(cat "$RECORD")
if [ "$rc" = 0 ] && [ -d "$path" ] && grep -q '^kept ' "$TMP/out"; then
  report 0 '--keep retains and identifies scratch'
else report 1 '--keep retains and identifies scratch'; fi
rm -rf "$path"
probe --help; rc=$?
if [ "$rc" = 0 ] && grep -q '^Last help line.$' "$TMP/out"; then
  report 0 '--help reads the full header'
else report 1 '--help reads the full header'; fi
probe --unknown; rc=$?
report "$([ "$rc" = 2 ]; echo $?)" 'unknown arguments exit 2'
MODE=signal probe; rc=$?
path=$(cat "$RECORD")
if [ "$rc" = 143 ] && [ ! -d "$path" ] && grep -q '^aborted before finish (rc=143)$' "$TMP/out"; then report 0 'TERM exits 143 and cleans scratch'
else report 1 'TERM exits 143 and cleans scratch'; fi
for mode in abort errexit unfinished; do
  case $mode in abort) want=50 ;; errexit) want=7 ;; unfinished) want=0 ;; esac
  MODE=$mode probe; rc=$?
  path=$(cat "$RECORD")
  if [ "$rc" = "$want" ] && [ ! -d "$path" ] \
     && grep -q "^aborted before finish (rc=$want)$" "$TMP/out" \
     && ! grep -q 'legs passed\|leg(s) failed' "$TMP/out"; then
    report 0 "$mode reports unfinished harness and preserves exit $want"
  else report 1 "$mode reports unfinished harness and preserves exit $want"; fi
done
MODE=abort probe --keep; rc=$?
path=$(cat "$RECORD")
if [ "$rc" = 50 ] && [ -d "$path" ] && grep -q '^aborted before finish (rc=50)$' "$TMP/out"; then
  report 0 'aborted --keep retains scratch and reports the original status'
else report 1 'aborted --keep retains scratch and reports the original status'; fi
rm -rf "$path"
# The leg filter: unfiltered, every leg runs and the full-pass line is earned;
# filtered, only matching bodies run, the rest say FILTERED, and the footer is
# never the full-pass line; a filter that selects nothing exits 2.
legs_probe() { # expected bodies run (space-separated), then probe arguments
  local want=$1 rc=0
  shift
  rm -f "$RECORD.ran"
  MODE=legs probe "$@" || rc=$?
  ran=
  [ ! -f "$RECORD.ran" ] || ran=$(tr '\n' ' ' <"$RECORD.ran" | sed 's/ $//')
  [ "$ran" = "$want" ] || { printf 'bodies run: "%s", wanted "%s"\n' "$ran" "$want" >>"$TMP/out"; rc=99; }
  return "$rc"
}
legs_probe 'alpha beta gamma'; rc=$?
if [ "$rc" = 0 ] && grep -q 'all legs passed (0 skipped)' "$TMP/out" && ! grep -q '^FILTERED' "$TMP/out" \
   && [ "$(grep -c '^PASS  .* body$' "$TMP/out")" = 3 ]; then
  report 0 'unfiltered: every leg runs and earns the full-pass line'
else report 1 'unfiltered: every leg runs and earns the full-pass line' "exit $rc; $(cat "$TMP/out")"; fi
legs_probe 'alpha' --only alpha; rc=$?
if [ "$rc" = 0 ] && grep -q '^PASS  alpha body$' "$TMP/out" \
   && grep -q '^FILTERED  beta leg #even$' "$TMP/out" && grep -q '^FILTERED  gamma leg #odd$' "$TMP/out" \
   && ! grep -q 'all legs passed' "$TMP/out" \
   && grep -q "^FILTERED RUN, not a full pass: 1 leg(s) selected, 2 filtered out by --only 'alpha'; no failure (0 skipped)$" "$TMP/out"; then
  report 0 '--only runs the matching leg alone and never prints the full-pass line'
else report 1 '--only runs the matching leg alone and never prints the full-pass line' "exit $rc; $(cat "$TMP/out")"; fi
legs_probe 'alpha gamma' '--only=^(alpha|gamma) leg'; rc=$?
if [ "$rc" = 0 ] && grep -q '^FILTERED  beta leg #even$' "$TMP/out" && ! grep -q 'all legs passed' "$TMP/out" \
   && grep -q 'FILTERED RUN, not a full pass: 2 leg(s) selected, 1 filtered out' "$TMP/out"; then
  report 0 '--only=REGEX is an extended regular expression over leg names'
else report 1 '--only=REGEX is an extended regular expression over leg names' "exit $rc; $(cat "$TMP/out")"; fi
GAMMA_RC=1 legs_probe 'gamma' --only gamma; rc=$?
if [ "$rc" = 1 ] && grep -q '^FAIL  gamma body$' "$TMP/out" \
   && grep -q "^1 leg(s) failed (0 skipped); FILTERED RUN: 1 leg(s) selected, 2 filtered out by --only 'gamma'$" "$TMP/out"; then
  report 0 'a failing selected leg exits 1 and its footer says the run was filtered'
else report 1 'a failing selected leg exits 1 and its footer says the run was filtered' "exit $rc; $(cat "$TMP/out")"; fi
# A `#group` tag closing a leg's name selects the group through the same
# filter, and a filtered run accounts for its coverage before the footer:
# each leg that ran, then each skipped check with its reason.
BETA_SKIP=1 legs_probe 'alpha beta gamma' --only '#odd|#even'; rc=$?
account=$(sed -n '/^ran 3 leg(s):$/,$p' "$TMP/out")
want_account="ran 3 leg(s):
  RAN   alpha leg #odd
  RAN   beta leg #even
  RAN   gamma leg #odd
skipped 1 check(s):
  SKIP  beta check -- no beta host
FILTERED RUN, not a full pass: 3 leg(s) selected, 0 filtered out by --only '#odd|#even'; no failure (1 skipped)"
if [ "$rc" = 0 ] && [ "$account" = "$want_account" ]; then
  report 0 'a filtered run accounts for each leg it ran and each check it skipped'
else report 1 'a filtered run accounts for each leg it ran and each check it skipped' "exit $rc; $(cat "$TMP/out")"; fi
legs_probe 'alpha gamma' --only '#odd'; rc=$?
if [ "$rc" = 0 ] && grep -q '^FILTERED  beta leg #even$' "$TMP/out" \
   && grep -q '^  RAN   alpha leg #odd$' "$TMP/out" && grep -q '^  RAN   gamma leg #odd$' "$TMP/out" \
   && ! grep -q '^skipped ' "$TMP/out"; then
  report 0 'a #group tag selects its legs whole'
else report 1 'a #group tag selects its legs whole' "exit $rc; $(cat "$TMP/out")"; fi
BETA_SKIP=1 legs_probe 'alpha beta gamma'; rc=$?
if [ "$rc" = 0 ] && grep -q 'all legs passed (1 skipped)' "$TMP/out" \
   && ! grep -q '^ran \|^  RAN \|^skipped ' "$TMP/out"; then
  report 0 'an unfiltered run keeps its output: no coverage account'
else report 1 'an unfiltered run keeps its output: no coverage account' "exit $rc; $(cat "$TMP/out")"; fi
OCANNL_TOOL_HARNESS_STRICT=1 legs_probe 'beta' --only beta; rc=$?
if [ "$rc" = 0 ] && grep -q 'FILTERED RUN, not a full pass: 1 leg(s) selected, 2 filtered out' "$TMP/out" \
   && ! grep -q '^FAIL' "$TMP/out"; then
  report 0 'a filtered leg is not a capability skip, so strict mode leaves it alone'
else report 1 'a filtered leg is not a capability skip, so strict mode leaves it alone' "exit $rc; $(cat "$TMP/out")"; fi
legs_probe '' --only 'no such leg'; rc=$?
if [ "$rc" = 2 ] && [ "$(grep -c '^FILTERED  ' "$TMP/out")" = 3 ] \
   && grep -q "^--only 'no such leg' selected no leg (3 filtered out): nothing was tested$" "$TMP/out" \
   && ! grep -q 'legs passed\|leg(s) failed\|FILTERED RUN\|aborted before finish' "$TMP/out"; then
  report 0 'a filter matching no leg fails loudly with exit 2'
else report 1 'a filter matching no leg fails loudly with exit 2' "exit $rc; $(cat "$TMP/out")"; fi
probe --only pass; rc=$?
if [ "$rc" = 2 ] && grep -q "^--only 'pass': this run reached no leg gate" "$TMP/out" \
   && ! grep -q 'legs passed\|FILTERED RUN' "$TMP/out"; then
  report 0 'a filter over a harness that declares no legs fails loudly'
else report 1 'a filter over a harness that declares no legs fails loudly' "exit $rc; $(cat "$TMP/out")"; fi
for bad in "--only" "--only=" "--only=("; do
  legs_probe '' $bad; rc=$?
  if [ "$rc" = 2 ] && grep -q -- '--only' "$TMP/out" && ! grep -q '^PASS\|^FILTERED' "$TMP/out"; then
    report 0 "'$bad' is refused before any leg runs"
  else report 1 "'$bad' is refused before any leg runs" "exit $rc; $(cat "$TMP/out")"; fi
done
# A leg whose body is a function (gh-ocannl-1374): giving up through
# harness_abandon fails that leg alone, named, and the next leg still runs; the
# body's own last status is no verdict; a filtered body does not run; and an
# exit inside a body still aborts, naming the leg it was in.
abandon_probe() { # expected bodies run, then probe arguments
  local want=$1 rc=0
  shift
  rm -f "$RECORD.ran"
  MODE=abandon probe "$@" || rc=$?
  ran=
  [ ! -f "$RECORD.ran" ] || ran=$(tr '\n' ' ' <"$RECORD.ran" | sed 's/ $//')
  [ "$ran" = "$want" ] || { printf 'bodies run: "%s", wanted "%s"\n' "$ran" "$want" >>"$TMP/out"; rc=99; }
  return "$rc"
}
abandon_probe 'delta epsilon'; rc=$?
if [ "$rc" = 1 ] && grep -qx 'FAIL  delta leg #odd: abandoned, its later checks did not run' "$TMP/out" \
   && grep -qx '      the fixture is not there' "$TMP/out" && grep -qx 'PASS  epsilon body' "$TMP/out" \
   && grep -qx '1 leg(s) failed (0 skipped)' "$TMP/out" && ! grep -q 'aborted before finish' "$TMP/out"; then
  report 0 'an abandoned leg fails alone, and the leg after it runs; a body'"'"'s last status is no verdict'
else report 1 'an abandoned leg fails alone, and the leg after it runs; a body'"'"'s last status is no verdict' "exit $rc; $(cat "$TMP/out")"; fi
abandon_probe 'epsilon' --only '#even'; rc=$?
if [ "$rc" = 0 ] && grep -qx 'FILTERED  delta leg #odd' "$TMP/out" && grep -qx '  RAN   epsilon leg #even' "$TMP/out"; then
  report 0 'harness_leg_run gates a function body as harness_leg does'
else report 1 'harness_leg_run gates a function body as harness_leg does' "exit $rc; $(cat "$TMP/out")"; fi
ZETA=1 abandon_probe 'delta epsilon zeta'; rc=$?
if [ "$rc" = 9 ] && grep -qx 'aborted before finish (rc=9) inside leg: zeta leg #odd' "$TMP/out" \
   && ! grep -q 'legs passed\|leg(s) failed' "$TMP/out"; then
  report 0 'an exit inside a leg body still aborts, naming the leg'
else report 1 'an exit inside a leg body still aborts, naming the leg' "exit $rc; $(cat "$TMP/out")"; fi
# A body checked under errexit (gh-ocannl-1539): each failing shape abandons
# its leg alone, named with why -- the failed command and the function it ran
# in, the exit status, the report/skip/abandon text -- and every later leg runs;
# no `late` line means no body went on past its failure; restoration runs after
# each abandoned leg that ran, not after the passing one nor the one called
# inside a condition, which never runs; the passing body's export and `cd` stay
# its own. A restoration that fails is a FAIL of its own.
errexit_probe() { # expected bodies run, then probe arguments
  local want=$1 rc=0
  shift
  rm -f "$RECORD.ran"
  MODE=errexit_legs probe "$@" || rc=$?
  ran=
  [ ! -f "$RECORD.ran" ] || ran=$(tr '\n' '|' <"$RECORD.ran" | sed 's/|$//')
  [ "$ran" = "$want" ] || { printf 'bodies run: "%s", wanted "%s"\n' "$ran" "$want" >>"$TMP/out"; rc=99; }
  return "$rc"
}
errexit_probe 'eta|restore|theta|iota unset home|restore|kappa|restore|lambda|restore|mu|restore|nu|restore'; rc=$?
abandoned() { grep -qx "FAIL  $1 leg: abandoned, its later checks did not run" "$TMP/out" && grep -qxF "      $2" "$TMP/out"; }
if [ "$rc" = 1 ] && grep -qx 'FAIL  eta leg #odd: abandoned, its later checks did not run' "$TMP/out" \
   && grep -qxF '      a check failed in check: grep -q never-there "$RECORD"' "$TMP/out" \
   && grep -qx 'PASS  theta body' "$TMP/out" \
   && abandoned iota 'it ended with status 9 before its end: an exit, or an unbound variable under set -u (the lines above say which)' \
   && grep -qx 'FAIL  kappa leg: abandoned, its later checks did not run' "$TMP/out" \
   && grep -q 'kappa_unbound' "$TMP/out" \
   && abandoned lambda 'report 1 inside the body: lambda label -- lambda detail' \
   && abandoned mu 'skip inside the body: mu check -- no mu host' \
   && abandoned nu 'nu gave up' \
   && abandoned xi 'it was not run: called inside a condition (if, &&, || or !), where bash ignores errexit in the whole body, no check of it could fail' \
   && grep -qx '7 leg(s) failed (0 skipped)' "$TMP/out" && ! grep -q 'aborted before finish\|^SKIP' "$TMP/out"; then
  report 0 'an errexit body abandons its leg alone at its first failed check, named, and is restored after'
else report 1 'an errexit body abandons its leg alone at its first failed check, named, and is restored after' "exit $rc; $(cat "$TMP/out")"; fi
RESTORE_FAILS=1 errexit_probe 'eta|restore|theta|iota unset home|restore|kappa|restore|lambda|restore|mu|restore|nu|restore'; rc=$?
if [ "$rc" = 1 ] && grep -q '^FAIL  eta leg #odd: the shared state it left was not restored; a later leg may fail for it$' "$TMP/out" \
   && grep -qxF '      a check failed in harness_leg_restore: [ -z "${RESTORE_FAILS:-}" ]' "$TMP/out" \
   && grep -qx '13 leg(s) failed (0 skipped)' "$TMP/out"; then
  report 0 'a restoration that fails is a FAIL of its own'
else report 1 'a restoration that fails is a FAIL of its own' "exit $rc; $(cat "$TMP/out")"; fi
errexit_probe 'theta' --only '#even'; rc=$?
if [ "$rc" = 0 ] && grep -qx 'FILTERED  eta leg #odd' "$TMP/out" && grep -qx '  RAN   theta leg #even' "$TMP/out"; then
  report 0 'harness_leg_run_errexit gates a body as harness_leg does'
else report 1 'harness_leg_run_errexit gates a body as harness_leg does' "exit $rc; $(cat "$TMP/out")"; fi
# Leg naming (gh-ocannl-1367), on a probe whose header declares its groups:
# #odd lists legs 1, 3 and 5, #even 2 and 4, #free none. Each case gates the
# legs LEGS names, one per line; every gate is checked, selected or not.
cat >"$TMP/grouped.sh" <<'PROBE'
#!/usr/bin/env bash
# Grouped fixture: #odd (1, 3, 5: the odd legs), #even (2,
# 4) and #free (no numbers).
. "$SUPPORT"
harness_args "$@"
harness_scratch grouped
while IFS= read -r name; do
  if harness_leg "$name"; then echo "$name" >>"$RECORD.ran"; report 0 "$name body"; fi
done <<<"$LEGS"
finish
PROBE
good_legs='1: one #odd
2: two #even
3: three #odd
3b: beside three #odd
4: four #even
5: five #odd
free leg #free'
grouped() { # LEGS, then probe arguments
  local legs=$1 rc=0
  shift
  rm -f "$RECORD.ran"
  LEGS=$legs bash "$TMP/grouped.sh" "$@" >"$TMP/out" 2>&1 || rc=$?
  return "$rc"
}
grouped "$good_legs"; rc=$?
if [ "$rc" = 0 ] && grep -q 'all legs passed (0 skipped)' "$TMP/out" && ! grep -q 'leg naming' "$TMP/out"; then
  report 0 'naming: declared groups, listed numbers and a suffixed leg pass'
else report 1 'naming: declared groups, listed numbers and a suffixed leg pass' "exit $rc; $(cat "$TMP/out")"; fi
naming_case() { # LABEL LEGS EXPECTED-FAIL-LINE [probe arguments]
  local label=$1 legs=$2 want=$3 rc=0
  shift 3
  grouped "$legs" "$@" || rc=$?
  if [ "$rc" = 1 ] && grep -qxF -- "FAIL  $want" "$TMP/out" \
     && [ "$(grep -c '^FAIL  leg naming' "$TMP/out")" = 1 ]; then
    report 0 "naming: $label"
  else report 1 "naming: $label" "exit $rc; $(cat "$TMP/out")"; fi
}
naming_case 'a name that gates two legs fails' "$good_legs
2: two #even" "leg naming: two legs are named '2: two #even'"
naming_case 'a leg number that numbers two legs fails' "$good_legs
3: again #odd" "leg naming: '3: again #odd' numbers leg 3, as '3: three #odd' does"
naming_case 'a leg closing with no group fails' "$good_legs
untagged leg" "leg naming: 'untagged leg' does not close with a group its header declares"
naming_case 'a leg closing with an undeclared group fails' "$good_legs
stray leg #nope" "leg naming: 'stray leg #nope' does not close with a group its header declares"
naming_case 'a number its group does not list fails' "$good_legs
6: six #odd" "leg naming: '6: six #odd' numbers leg 6, which the header does not list under #odd"
naming_case 'a listed number no leg carries fails' "$(printf '%s\n' "$good_legs" | grep -v '^5: ')" \
  "leg naming: the header lists leg 5 under #odd, and no leg of #odd numbers it"
naming_case 'the naming checks every gate, a filtered one too' "$good_legs
3: again #odd" "leg naming: '3: again #odd' numbers leg 3, as '3: three #odd' does" --only '^1:'
# --list-legs: every gate printed in order, no body run, the naming checked.
grouped "$good_legs" --list-legs; rc=$?
if [ "$rc" = 0 ] && [ "$(sed -n 's/^LEG  //p' "$TMP/out")" = "$good_legs" ] && [ ! -f "$RECORD.ran" ] \
   && grep -qx 'listed 7 leg(s); no leg ran (0 failure(s) outside the legs, 0 skipped)' "$TMP/out"; then
  report 0 '--list-legs prints every gate in order and runs no leg body'
else report 1 '--list-legs prints every gate in order and runs no leg body' "exit $rc; $(cat "$TMP/out")"; fi
naming_case '--list-legs fails on a naming defect' "$good_legs
2: two #even" "leg naming: two legs are named '2: two #even'" --list-legs
grouped "$good_legs" --list-legs --only one; rc=$?
if [ "$rc" = 2 ] && grep -q -- '--list-legs lists every leg; it takes no --only' "$TMP/out" && [ ! -f "$RECORD.ran" ]; then
  report 0 '--list-legs refuses --only'
else report 1 '--list-legs refuses --only' "exit $rc; $(cat "$TMP/out")"; fi
reason() { echo 'claimed defect'; return 1; }
wrong_reason() { echo 'unrelated defect'; return 1; }
wrong_status() { echo 'claimed defect'; return 127; }
accepted() { echo 'claimed defect'; return 0; }
if harness_rejected 1 '^claimed defect$' "$TMP/rejection" reason; then
  report 0 'in-process twin rejects for its claimed reason'
else report 1 'in-process twin rejects for its claimed reason'; fi
for control in wrong_reason wrong_status accepted; do
  if harness_rejected 1 '^claimed defect$' "$TMP/rejection" "$control"; then
    report 1 "rejection refuses $control"
  else report 0 "rejection refuses $control"; fi
done
# The positive lifetime test owns an unreaped child. The independent timeout
# must kill/reap it and record exactly one failure, rather than hang on wait.
( sleep 30 ) & child=$!
never_dead() { return 1; }
before=$failures
harness_wait_child "$child" never_dead 'injected deadline' 1 >"$TMP/deadline"
if [ "$failures" = "$((before + 1))" ] && [ "$harness_child_rc" = 137 ]; then
  failures=$before
  report 0 'owned-child deadline kills and reaps the timed-out child'
else report 1 'owned-child deadline kills and reaps the timed-out child'; fi
finish
