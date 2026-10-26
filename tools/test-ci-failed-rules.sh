#!/usr/bin/env bash
# Legs for tools/ci-failed-rules.sh (gh-ocannl-1304), the listing a red CI
# build job ends with. Its input is Dune's own output, so most legs build a
# small real Dune project and read what that build printed: a red build lists
# exactly the rule that failed, and the same project built green prints
# nothing. The rest feed that captured log through what GitHub does to it
# (timestamps, colour, CRLF, a byte-order mark, `gh run view --log` prefixes),
# read it from a checkout of another revision, and pin the CI mode's
# annotations and refusals. A mutant with the timestamp handling removed must
# lose the GitHub-served leg, so that leg is seen to discriminate.
# Usage: tools/test-ci-failed-rules.sh [--keep] [--only REGEX]
#
# In the `promotion` harness group, beside the other readers of Dune's output,
# so CI runs it on current Dune and again at the declared floor.
set -u
. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/ci-failed-rules.sh
[ -f "$SRC" ] || { echo "no $SRC" >&2; exit 2; }
harness_scratch test-ci-failed-rules
unset DUNE_BUILD_DIR DUNE_ROOT

subject() { bash "$SRC" "$@"; }

# Logs that need no build: a warning that did not fail the build stays out,
# and the CI mode's refusals and host line.
if harness_leg 'warning and alert blocks are left out #canned'; then
  cat >"$TMP/warn.log" <<'LOG'
File "lib/dune", line 3, characters 1-20:
3 |  (flags (:standard))
     ^^^^^^^^^^^^^^^^^^^
Warning: this field is deprecated.
File "lib/x.ml", line 2, characters 4-5:
2 | let y = 1
        ^
Alert deprecated: Old.f
File "lib/y.ml", line 4, characters 8-9:
4 | let z = w
            ^
Error: Unbound value w
LOG
  out=$(cd "$TMP" && subject warn.log 2>&1); rc=$?
  want='Failed dune rules (1), in the order dune reported them:
  lib/y.ml:4  compile error
      Error: Unbound value w'
  report "$([ "$rc" = 0 ] && [ "$out" = "$want" ]; echo $?)" 'warning and alert blocks are left out' "exit $rc; $out"
fi
if harness_leg 'CI mode on a missing log, and usage errors #canned'; then
  out=$(subject --github "$TMP/absent.log" 2>&1); rc=$?
  report "$([ "$rc" = 0 ] && [ "$out" = "No dune build log at $TMP/absent.log: this job failed before its dune build ran." ]; echo $?)" \
    '--github on a missing log says the build never ran, exit 0' "exit $rc; $out"
  out=$(subject "$TMP/absent.log" 2>&1); rc=$?
  report "$([ "$rc" = 2 ] && [[ $out == *"cannot read"* ]]; echo $?)" 'a missing log is exit 2 outside CI mode' "exit $rc; $out"
  out=$(subject --bogus 2>&1); rc=$?
  report "$([ "$rc" = 2 ] && [[ $out == *"unknown argument"* ]]; echo $?)" 'an unknown option is exit 2' "exit $rc; $out"
  : >"$TMP/empty.log"
  printf 'OCANNL host: cpu="fixture"\n' >"$TMP/host.txt"
  out=$(subject --github --host "$TMP/host.txt" "$TMP/empty.log" 2>&1); rc=$?
  want="OCANNL host: cpu=\"fixture\"
No failed dune rule in this job's build log: the failure lies in another step."
  report "$([ "$rc" = 0 ] && [ "$out" = "$want" ]; echo $?)" '--host prints the host line first; --github says when no rule failed' "exit $rc; $out"
fi

command -v dune >/dev/null 2>&1 || . "$HERE/opam-env.sh" 2>/dev/null
harness_require dune

proj=$TMP/proj
mkdir -p "$proj/t" || exit 2
printf '(lang dune 3.20)\n' >"$proj/dune-project"
# The failing rule's name field sits in the middle of a stanza long enough
# that dune's excerpt elides it, so only the source can name it -- the
# revision leg below depends on that.
write_fixture() { # EXIT-STATUS-OF-THE-FAILING-TEST
  cat >"$proj/t/dune" <<DUNE
(rule
 (alias runtest-good)
 (action
  (progn)))

(rule
 ; The test under scrutiny. Its program exits
 ; with the status the leg chooses, and the diff
 ; rule below reads what it wrote. These lines
 ; are long enough a stanza for dune to elide
 ; its middle when it quotes the stanza, the
 ; target field among it.
 (target bad.actual)
 (action
  (with-stdout-to
   %{target}
   (run sh -c "echo noise from the failing test; exit $1"))))

(rule
 (alias runtest-bad)
 (action
  (diff bad.expected bad.actual)))

(test
 (name passing)
 (modules passing))

(alias
 (name runtest)
 (deps
  (alias runtest-good)
  (alias runtest-bad)))
DUNE
}
printf 'noise from the failing test\n' >"$proj/t/bad.expected"
printf 'let () = print_string "ok\\n"\n' >"$proj/t/passing.ml"
printf 'ok\n' >"$proj/t/passing.expected"
build() { # LOG TARGET...
  local log=$1
  shift
  # CI=true: the display GitHub's runners get, which echoes each failing
  # command.
  (cd "$proj" && CI=true dune build --root . "$@") >"$log" 2>&1
}
first_two='Failed dune rules (1), in the order dune reported them:
  t/dune:6-17  (rule) target bad.actual'
ran_re="^      ran: \(cd _build/default/t && [^ ]*sh -c 'echo noise from the failing test; exit 3'\) > _build/default/t/bad\.actual$"
# The listing of a red build: the header names the rule, its own target field
# names it, and the command is the one dune ran. Nothing else is listed.
listing_ok() { # OUTPUT RC
  [ "$2" = 0 ] && [ "$(printf '%s\n' "$1" | sed -n '1,2p')" = "$first_two" ] &&
    [ "$(printf '%s\n' "$1" | wc -l | tr -d ' ')" = 3 ] &&
    printf '%s\n' "$1" | sed -n '3p' | grep -qE "$ran_re"
}

write_fixture 3
build "$TMP/red.log" @runtest; red_brc=$?
# Dune's locationless error, which two legs read: built here, outside both
# gates, so each passes run alone (tools/harness-legs-alone.sh).
build "$TMP/alias.log" @t/runtest-nonesuch; alias_brc=$?
if harness_leg 'a red build lists exactly the failing rule #dune'; then
  brc=$red_brc
  out=$(cd "$proj" && subject "$TMP/red.log" 2>&1); rc=$?
  if [ "$brc" = 1 ] && listing_ok "$out" "$rc"; then
    report 0 'a red build lists exactly the failing rule'
  else report 1 'a red build lists exactly the failing rule' "build exit $brc, exit $rc; $out"; fi
  out=$(cd "$proj" && subject --github "$TMP/red.log" 2>&1); rc=$?
  ann=$(printf '%s\n' "$out" | grep '^::error ')
  if [ "$rc" = 0 ] && [ "$(printf '%s\n' "$ann" | wc -l | tr -d ' ')" = 1 ] &&
     [[ $ann == '::error file=t/dune,line=6,endLine=17,title=Failed dune rule::t/dune:6-17  (rule) target bad.actual -- ran: (cd _build/default/t && '* ]]; then
    report 0 '--github annotates the failing rule once'
  else report 1 '--github annotates the failing rule once' "exit $rc; $out"; fi
fi
if harness_leg 'a GitHub-served log reads the same #dune'; then
  # A job log as GitHub serves it: a byte-order mark, a timestamp on every
  # line, dune's colour around the header, CRLF line ends; then the same lines
  # behind the JOB<TAB>STEP<TAB> prefix of `gh run view --log-failed`.
  esc=$(printf '\033')
  LC_ALL=C awk -v esc="$esc" 'BEGIN { printf "\357\273\277" }
    /^File "/ { $0 = esc "[0;1m" $0 esc "[0m" }
    { printf "2026-10-19T13:11:44.%07dZ %s\r\n", NR, $0 }' "$TMP/red.log" >"$TMP/served.log"
  LC_ALL=C awk '{ sub(/^\357\273\277/, ""); printf "Build (ubuntu-latest, 5.5.x, main)\tRun opam exec -- dune build\t%s\n", $0 }' \
    "$TMP/served.log" >"$TMP/gh-view.log"
  # Under a UTF-8 locale too, where an awk reads the BOM as one character.
  ok=0
  for locale in C en_US.UTF-8; do
    for form in served gh-view; do
      out=$(cd "$proj" && LC_ALL=$locale subject "$TMP/$form.log" 2>&1); rc=$?
      listing_ok "$out" "$rc" || { ok=1; detail="$form, LC_ALL=$locale: exit $rc; $out"; }
    done
  done
  report "$ok" 'a GitHub-served log reads the same' "${detail:-}"
  # The negative control: without the timestamp handling the header is not
  # at the start of its line, and the served log lists nothing.
  if mut=$(mutant no-timestamps '{ sub(/^  sub\("\^" ts, "", s\)$/, "  # timestamps kept") } 1'); then
    out=$(cd "$proj" && bash "$mut" "$TMP/served.log" 2>&1); rc=$?
    if grep -q '^  # timestamps kept$' "$mut" && ! listing_ok "$out" "$rc"; then
      report 0 'negative control: keeping timestamps loses the served log'
    else report 1 'negative control: keeping timestamps loses the served log' "mutant applied: $(grep -c 'timestamps kept' "$mut"); $out"; fi
  else report 1 'negative control: keeping timestamps loses the served log' 'mutant did not build'; fi
fi
if harness_leg 'a log from another revision names nothing it cannot see #dune'; then
  # Read from a checkout where the stanza has moved down a line: the lines
  # dune quoted no longer match, so the name comes from the excerpt alone
  # (which elides the target) rather than from whatever now sits there.
  moved=$TMP/moved
  mkdir -p "$moved/t" && { printf '; a line added since\n'; cat "$proj/t/dune"; } >"$moved/t/dune"
  out=$(cd "$moved" && subject "$TMP/red.log" 2>&1); rc=$?
  if [ "$rc" = 0 ] && [ "$(printf '%s\n' "$out" | sed -n '2p')" = '  t/dune:6-17  (rule)' ]; then
    report 0 'a log from another revision names nothing it cannot see'
  else report 1 'a log from another revision names nothing it cannot see' "exit $rc; $out"; fi
fi
if harness_leg 'a dune error without a location is listed #dune'; then
  brc=$alias_brc
  out=$(cd "$proj" && subject "$TMP/alias.log" 2>&1); rc=$?
  if [ "$brc" = 1 ] && [ "$rc" = 0 ] && [ "$(printf '%s\n' "$out" | sed -n '1p')" = 'dune errors without a location (1):' ] &&
     printf '%s\n' "$out" | sed -n '2p' | grep -q '^  Error: .*runtest-nonesuch'; then
    report 0 'a dune error without a location is listed'
  else report 1 'a dune error without a location is listed' "build exit $brc, exit $rc; $out"; fi
fi
if harness_leg 'locationless errors beside located failures #dune'; then
  # One log holding dune's locationless error, then the red build, then an
  # `Error:` line a failing action printed: the first stays listed beside
  # the failed rule, the last is the action's output and is not.
  { cat "$TMP/alias.log" "$TMP/red.log"; echo 'Error: printed by the failing test itself'; } >"$TMP/mixed.log"
  out=$(cd "$proj" && subject "$TMP/mixed.log" 2>&1); rc=$?
  if [ "$rc" = 0 ] && listing_ok "$(printf '%s\n' "$out" | sed -n '1,3p')" 0 &&
     [ "$(printf '%s\n' "$out" | sed -n '4p')" = 'dune errors without a location (1):' ] &&
     printf '%s\n' "$out" | sed -n '5p' | grep -q '^  Error: .*runtest-nonesuch' &&
     [ "$(printf '%s\n' "$out" | wc -l | tr -d ' ')" = 5 ]; then
    report 0 'locationless errors beside located failures'
  else report 1 'locationless errors beside located failures' "exit $rc; $out"; fi
fi
if harness_leg 'a golden mismatch and a compile error are told apart #dune'; then
  write_fixture 0
  printf 'something else\n' >"$proj/t/passing.expected"
  build "$TMP/golden.log" @runtest; brc=$?
  printf 'let () = print_string 3\n' >"$proj/t/passing.ml"
  build "$TMP/compile.log" @runtest; crc=$?
  out=$(cd "$proj" && subject "$TMP/golden.log" "$TMP/compile.log" 2>&1); rc=$?
  want='Failed dune rules (2), in the order dune reported them:
  t/passing.expected:1  output differs from this golden
  t/passing.ml:1  compile error'
  if [ "$brc" = 1 ] && [ "$crc" = 1 ] && [ "$rc" = 0 ] && [ "$(printf '%s\n' "$out" | sed -n '1,3p')" = "$want" ] &&
     printf '%s\n' "$out" | sed -n '4p' | grep -q '^      Error: '; then
    report 0 'a golden mismatch and a compile error are told apart'
  else report 1 'a golden mismatch and a compile error are told apart' "builds $brc/$crc, exit $rc; $out"; fi
  # The annotation list shows only the message, so the golden mismatch -- which
  # carries no detail -- must name its path there, not only in `file=`
  # (gh-ocannl-1464).
  out=$(cd "$proj" && subject --github "$TMP/golden.log" 2>&1); rc=$?
  ann=$(printf '%s\n' "$out" | grep '^::error ')
  want='::error file=t/passing.expected,line=1,endLine=1,title=Failed dune rule::t/passing.expected:1  output differs from this golden'
  report "$([ "$rc" = 0 ] && [ "$ann" = "$want" ]; echo $?)" '--github names a golden mismatch'"'"'s path in the annotation message' \
    "exit $rc; $out"
  printf 'let () = print_string "ok\\n"\n' >"$proj/t/passing.ml"
  printf 'ok\n' >"$proj/t/passing.expected"
fi
if harness_leg 'a failing test that prints a fixture dune block is listed alone #dune'; then
  # gh-ocannl-1590: the sweep harness's failure output replayed a fixture unit log holding a
  # dune-shaped block, and every listing of that build named a `test/dune` that does not exist.
  # Here a failing rule prints the same block bare, behind the sweep's `rtc ` prefix and behind
  # the `| ` its harness now puts on a replay: only the rule itself is listed. The bare copy is
  # left out because its excerpt omits line 4 of `lines 1-4`, the prefixed ones because a header
  # is read only at the start of its line.
  mkdir -p "$proj/p" || exit 2
  cat >"$proj/p/dune" <<'DUNE'
(rule
 (alias runtest-phantom)
 (deps phantom.log)
 (action
  (run sh -c "cat phantom.log; exit 1")))
DUNE
  block='File "test/dune", lines 1-4, characters 0-0:
1 | (rule
2 | (alias runtest-state-probe)
unit.exe: segmentation fault'
  { printf 'sweep-hold: sleep:idle block inhibitor held\n%s\n' "$block"
    printf '%s\n' "$block" | sed 's/^/rtc /'
    printf '%s\n' "$block" | sed 's/^/| /'; } >"$proj/p/phantom.log"
  build "$TMP/phantom.log" @p/runtest-phantom; brc=$?
  phantom_ok() { # OUTPUT RC
    [ "$2" = 0 ] && [ "$(printf '%s\n' "$1" | sed -n '1,2p')" = 'Failed dune rules (1), in the order dune reported them:
  p/dune:1-5  (rule) alias runtest-phantom' ] &&
      [ "$(printf '%s\n' "$1" | wc -l | tr -d ' ')" = 3 ] &&
      printf '%s\n' "$1" | sed -n '3p' | grep -qE "^      ran: \(cd _build/default/p && [^ ]*sh -c 'cat phantom\.log; exit 1'\)$"
  }
  out=$(cd "$proj" && subject "$TMP/phantom.log" 2>&1); rc=$?
  if [ "$brc" = 1 ] && grep -q '^File "test/dune", lines 1-4' "$TMP/phantom.log" && phantom_ok "$out" "$rc"; then
    report 0 'a failing test that prints a fixture dune block is listed alone'
  else report 1 'a failing test that prints a fixture dune block is listed alone' "build exit $brc, exit $rc; $out"; fi
  # The negative control: without the excerpt anchor the bare block is listed as a second failed
  # rule -- the issue's false report -- while the prefixed copies still are not.
  if mut=$(mutant no-excerpt-anchor '{ sub(/^  if \(is_dune && nx > 0 && .*dropped = 1$/, "  # excerpt anchor removed") } 1'); then
    out=$(cd "$proj" && bash "$mut" "$TMP/phantom.log" 2>&1); rc=$?
    if grep -q '^  # excerpt anchor removed$' "$mut" && ! phantom_ok "$out" "$rc" &&
       [ "$(printf '%s\n' "$out" | sed -n '1p')" = 'Failed dune rules (2), in the order dune reported them:' ] &&
       [ "$(printf '%s\n' "$out" | grep -c 'test/dune')" = 1 ]; then
      report 0 'negative control: without the excerpt anchor the fixture block is a failed rule'
    else report 1 'negative control: without the excerpt anchor the fixture block is a failed rule' \
      "mutant applied: $(grep -c 'excerpt anchor removed' "$mut"); $out"; fi
  else report 1 'negative control: without the excerpt anchor the fixture block is a failed rule' 'mutant did not build'; fi
fi
if harness_leg 'the green build prints nothing #dune'; then
  write_fixture 0
  build "$TMP/green.log" @runtest; brc=$?
  out=$(cd "$proj" && subject "$TMP/green.log" 2>&1); rc=$?
  report "$([ "$brc" = 0 ] && [ "$rc" = 0 ] && [ -z "$out" ]; echo $?)" 'the green build prints nothing' \
    "build exit $brc, exit $rc; $out; $(cat "$TMP/green.log")"
fi
finish
