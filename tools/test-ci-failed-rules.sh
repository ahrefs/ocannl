#!/usr/bin/env bash
# Legs for tools/ci-failed-rules.sh (gh-ocannl-1304), the listing a red CI
# build job ends with. Its input is Dune's own output, so most legs build a
# small real Dune project and read what that build printed: a red build lists
# exactly the rule that failed, and the same project built green prints
# nothing. The rest feed that captured log through what GitHub does to it
# (timestamps, colour, CRLF, a byte-order mark, `gh run view --log` prefixes),
# read it from a checkout of another revision, and pin the CI mode's
# annotations and refusals. A mutant with the timestamp handling removed must
# lose the GitHub-served leg, so that leg is seen to discriminate. The
# quoting legs (gh-ocannl-1831) build rules that fail for a stated reason --
# claims, a Python unittest run, a scan's golden -- and two that must not lend
# their lines to the rule above them, each boundary held by a mutant. The
# last-output legs (gh-ocannl-1927) build rules that die saying nothing the
# reader matches -- a shell error, a long output, a signal, a bare status.
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
if harness_leg '--names gives each entry one short name #canned'; then
  cat >"$TMP/names.log" <<'LOG'
File "t/dune", line 3, characters 7-10:
3 |  (name foo)
           ^^^
FAIL: a claim
File "t/foo.expected", line 1, characters 0-0:
--- a/t/foo.expected
+++ b/t/foo.exe.output
File "t/foo.expected", line 9, characters 0-0:
--- a/t/foo.expected
+++ b/t/foo.exe.output
File "t/dune", lines 5-6, characters 0-40:
5 | (rule
6 |  (target bar.actual))
FAIL: a claim
File "t/dune", lines 8-9, characters 0-40:
8 | (rule
9 |  (alias runtest-baz))
FAIL: a claim
File "t/dune", lines 11-12, characters 0-40:
11 | (rule
12 |  (alias runtest)
FAIL: a claim
File "t/dune", lines 14-15, characters 0-40:
14 | (rule
15 |  (deps x))
FAIL: a claim
File "t/x.ml", line 1, characters 0-1:
1 | y
    ^
Error: Unbound value y
Error: Alias "nope" specified on the command line is empty.
-> required by alias nope
LOG
  out=$(cd "$TMP" && subject --names names.log 2>&1); rc=$?
  want='foo
foo.expected
bar
runtest-baz
runtest
t/dune:14
x.ml
Error: Alias "nope" specified on the command line is empty.'
  report "$([ "$rc" = 0 ] && [ "$out" = "$want" ]; echo $?)" '--names gives each entry one short name' "exit $rc; $out"
  cat >"$TMP/both.log" <<'LOG'
File "t/dune", lines 5-7, characters 0-40:
5 | (rule
6 |  (alias runtest)
7 |  (target qux.actual))
FAIL: a claim
LOG
  out=$(cd "$TMP" && subject --names both.log 2>&1); rc=$?
  report "$([ "$rc" = 0 ] && [ "$out" = qux ]; echo $?)" '--names prefers a target to an alias' "exit $rc; $out"
  # The same short name in two directories is two failures; each golden header once is one.
  cat >"$TMP/collide.log" <<'LOG'
File "a/dune", line 3, characters 7-10:
3 |  (name foo)
           ^^^
FAIL: a claim
File "b/dune", line 3, characters 7-10:
3 |  (name foo)
           ^^^
FAIL: a claim
File "a/bar.expected", line 1, characters 0-0:
--- a/a/bar.expected
+++ b/a/bar.exe.output
File "b/bar.expected", line 1, characters 0-0:
--- a/b/bar.expected
+++ b/b/bar.exe.output
LOG
  out=$(cd "$TMP" && subject --names collide.log 2>&1); rc=$?
  want='a/foo
b/foo
a/bar.expected
b/bar.expected'
  report "$([ "$rc" = 0 ] && [ "$out" = "$want" ]; echo $?)" '--names keeps a name two directories share, each led by its directory' "exit $rc; $out"
  : >"$TMP/green.log"
  out=$(cd "$TMP" && subject --names green.log 2>&1); rc=$?
  report "$([ "$rc" = 0 ] && [ -z "$out" ]; echo $?)" '--names on a green log prints nothing' "exit $rc; $out"
  out=$(cd "$TMP" && subject --names --github names.log 2>&1); rc=$?
  report "$([ "$rc" = 2 ] && [[ $out == *"--names takes neither"* ]]; echo $?)" '--names refuses --github' "exit $rc; $out"
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
  # command. CLICOLOR_FORCE=1: the colour setup-ocaml exports job-wide, so
  # dune wraps the `File` headers, the echoed commands and its `Error:` in
  # escapes there; forced here too, whatever the ambient setting, so a leg's
  # own read of a raw log sees what CI's does (gh-ocannl-1928: two legs that
  # read their log unstripped passed locally and failed in CI).
  (cd "$proj" && CI=true harness_dune_colour dune build --root . "$@") >"$log" 2>&1
}
# A log's lines as the reader sees them: build() forces dune's colour as CI's
# environment does (CLICOLOR_FORCE), which these legs' own reads of a raw log
# must look past.
plain() { harness_plain "$1"; }
first_two='Failed dune rules (1), in the order dune reported them:
  t/dune:6-17  (rule) target bad.actual'
ran_re="^      ran: \(cd _build/default/t && [^ ]*sh -c 'echo noise from the failing test; exit 3'\) > _build/default/t/bad\.actual$"
# The listing of a red build: the header names the rule, its own target field
# names it, and the command is the one dune ran. Its output went to the target,
# so the block holds only dune's line for the status, shown as the last output
# (gh-ocannl-1927). Nothing else is listed.
listing_head_ok() { # OUTPUT RC
  [ "$2" = 0 ] && [ "$(printf '%s\n' "$1" | sed -n '1,2p')" = "$first_two" ] &&
    printf '%s\n' "$1" | sed -n '3p' | grep -qE "$ran_re"
}
listing_ok() { # OUTPUT RC
  listing_head_ok "$1" "$2" && [ "$(printf '%s\n' "$1" | wc -l | tr -d ' ')" = 4 ] &&
    [ "$(printf '%s\n' "$1" | sed -n '4p')" = '      last output: Command exited with code 3.' ]
}

write_fixture 3
build "$TMP/red.log" @runtest; red_brc=$?
# Dune's locationless error, which two legs read: built here, outside both
# gates, so each passes run alone (tools/harness-legs-alone.sh).
build "$TMP/alias.log" @t/runtest-nonesuch; alias_brc=$?
if harness_leg 'the built logs carry CI'"'"'s forced colour #dune'; then
  # The raw logs hold what CI's hold: dune's escapes around the header, so
  # every leg reading one unstripped is tried here as it is there
  # (gh-ocannl-1928), whatever the ambient CLICOLOR_FORCE.
  if [ "$red_brc" = 1 ] && LC_ALL=C grep -q "$(printf '\033')\[[0-9;]*mFile \"t/dune\"" "$TMP/red.log" &&
     plain "$TMP/red.log" | grep -q '^File "t/dune", lines 6-17'; then
    report 0 'the built logs carry CI'"'"'s forced colour'
  else report 1 'the built logs carry CI'"'"'s forced colour' "build exit $red_brc; $(head -c 300 "$TMP/red.log")"; fi
fi
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
  # And as gh 2.x prints it: each ESC as the two characters `^[`.
  LC_ALL=C awk -v esc="$esc" '{ gsub(esc, "^["); print }' "$TMP/gh-view.log" >"$TMP/gh-caret.log"
  # Under a UTF-8 locale too, where an awk reads the BOM as one character.
  ok=0
  for locale in C en_US.UTF-8; do
    for form in served gh-view gh-caret; do
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
  # And without the caret-escape handling gh's rendering lists nothing (the
  # colour sits right before `File`), so that form is seen to discriminate.
  if mut=$(mutant no-caret-escapes '{ sub(/^  gsub\(\/\\\^\\\[\\\[.*$/, "  # caret escapes kept") } 1'); then
    out=$(cd "$proj" && bash "$mut" "$TMP/gh-caret.log" 2>&1); rc=$?
    if grep -q '^  # caret escapes kept$' "$mut" && ! listing_ok "$out" "$rc" &&
       grep -q '\^\[\[0;1mFile "t/dune"' "$TMP/gh-caret.log"; then
      report 0 'negative control: keeping caret escapes loses gh'"'"'s rendering'
    else report 1 'negative control: keeping caret escapes loses gh'"'"'s rendering' "mutant applied: $(grep -c 'caret escapes kept' "$mut"); $out"; fi
  else report 1 'negative control: keeping caret escapes loses gh'"'"'s rendering' 'mutant did not build'; fi
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
  # the failed rule, the last is the action's output, quoted under its rule
  # (gh-ocannl-1831) and not listed.
  { cat "$TMP/alias.log" "$TMP/red.log"; echo 'Error: printed by the failing test itself'; } >"$TMP/mixed.log"
  out=$(cd "$proj" && subject "$TMP/mixed.log" 2>&1); rc=$?
  if [ "$rc" = 0 ] && listing_head_ok "$out" 0 &&
     [ "$(printf '%s\n' "$out" | sed -n '4p')" = '      | Error: printed by the failing test itself' ] &&
     [ "$(printf '%s\n' "$out" | sed -n '5p')" = 'dune errors without a location (1):' ] &&
     printf '%s\n' "$out" | sed -n '6p' | grep -q '^  Error: .*runtest-nonesuch' &&
     [ "$(printf '%s\n' "$out" | wc -l | tr -d ' ')" = 6 ]; then
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
  # is read only at the start of its line. The bare copy still ends the rule's block, as any
  # header does, so the last output line shown is the one above it (gh-ocannl-1927).
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
      [ "$(printf '%s\n' "$1" | wc -l | tr -d ' ')" = 4 ] &&
      printf '%s\n' "$1" | sed -n '3p' | grep -qE "^      ran: \(cd _build/default/p && [^ ]*sh -c 'cat phantom\.log; exit 1'\)$" &&
      [ "$(printf '%s\n' "$1" | sed -n '4p')" = '      last output: sweep-hold: sleep:idle block inhibitor held' ]
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
# Why each rule failed (gh-ocannl-1831): the lines of its own output block that say so, quoted
# under its entry. Four real rules, each built alone into its own log: a test whose claims fail
# (more of them than the 3 quoted), a Python unittest run as `test_orchestrate` prints one (its
# AssertionError a couple of thousand lines below the command), a scan whose golden diff adds its
# FAIL: line, and two rules that are not failures of theirs -- a passing one that prints a FAIL:
# line, and one whose dependency dune cannot build, named long enough that dune wraps its error.
# Built outside the gates, so each leg passes run alone.
mkdir -p "$proj/r" || exit 2
cat >"$proj/r/dune" <<'DUNE'
(rule
 (alias runtest-claims)
 (deps claims.out)
 (action
  (run sh -c "cat claims.out; exit 1")))

(rule
 (alias runtest-orchestrate)
 (deps orchestrate.out)
 (action
  (run sh -c "cat orchestrate.out >&2; exit 1")))

(rule
 (target scan.actual)
 (deps scan.out)
 (action
  (copy scan.out scan.actual)))

(rule
 (alias runtest-scan)
 (action
  (diff scan.expected scan.actual)))

(rule
 (alias runtest-chatty)
 (action
  (run sh -c "echo 'FAIL: printed by a rule that passes'")))

(rule
 (alias runtest-unbuildable)
 (deps a-dependency-whose-name-is-long-enough-that-dune-wraps-its-error-onto-a-second-line.exe)
 (action
  (run ./a-dependency-whose-name-is-long-enough-that-dune-wraps-its-error-onto-a-second-line.exe)))
DUNE
printf '%s\n' 'checking 7 claims' 'FAIL: claim 1: the gradient is finite' 'ok: claim 2' \
  'FAIL: claim 3: the arena is released' 'FAIL: claim 4: two runs agree' 'FAIL: claim 5: the tile fits' \
  'FAIL: claim 6: the cache is hit' 'FAILED: 5 checks did not hold' >"$proj/r/claims.out"
{ i=0; while [ "$i" -lt 2000 ]; do
    printf 'test_cell_%04d (test_orchestrate.CellTest.test_cell_%04d) ... ok\n' "$i" "$i"; i=$((i + 1)); done
  cat <<'PY'
test_parity_flag (test_orchestrate.ParityCheckTest.test_parity_flag) ... FAIL

======================================================================
FAIL: test_parity_flag (test_orchestrate.ParityCheckTest.test_parity_flag)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/runner/work/ocannl/benchmarks/test_orchestrate.py", line 84, in test_parity_flag
    self.assertEqual(flat["parity"], "FAIL")
AssertionError: 'PASS' != 'FAIL'
- PASS
+ FAIL


----------------------------------------------------------------------
Ran 2001 tests in 3.210s

FAILED (failures=1)
PY
} >"$proj/r/orchestrate.out"
# The golden's context holds an `Error:` line and it loses a FAIL: line: neither is what the run
# said, only the line the diff adds is.
printf '%s\n' 'accepted position in use: printed or raised' 'accepted position in use: an Error: payload' \
  'canary held: the scan reads FAIL: markers' >"$proj/r/scan.expected"
printf '%s\n' 'accepted position in use: printed or raised' 'accepted position in use: an Error: payload' \
  'FAIL: arrayjit/lib/backends.ml:120: a label read the scan cannot place' >"$proj/r/scan.out"
build "$TMP/claims.log" @r/runtest-claims; claims_brc=$?
build "$TMP/orchestrate.log" @r/runtest-orchestrate; orch_brc=$?
build "$TMP/scan.log" @r/runtest-scan; scan_brc=$?
build "$TMP/chatty.log" @r/runtest-chatty; chatty_brc=$?
build "$TMP/unbuildable.log" @r/runtest-unbuildable; unb_brc=$?
claims_want='Failed dune rules (1), in the order dune reported them:
  r/dune:1-5  (rule) alias runtest-claims'
claims_quoted='      | FAIL: claim 1: the gradient is finite
      | FAIL: claim 3: the arena is released
      | FAIL: claim 4: two runs agree'
# The claims rule's entry: its two lines, its first 3 reasons, and the count of the other 3.
claims_ok() { # OUTPUT RC MORE
  [ "$2" = 0 ] && [ "$(printf '%s\n' "$1" | sed -n '1,2p')" = "$claims_want" ] &&
    printf '%s\n' "$1" | sed -n '3p' | grep -qE "^      ran: \(cd _build/default/r && [^ ]*sh -c 'cat claims\.out; exit 1'\)$" &&
    [ "$(printf '%s\n' "$1" | sed -n '4,6p')" = "$claims_quoted" ] &&
    [ "$(printf '%s\n' "$1" | sed -n '7p')" = "      ($3 more such lines)" ]
}
if harness_leg 'a failing claim is quoted under its rule #dune'; then
  out=$(cd "$proj" && subject "$TMP/claims.log" 2>&1); rc=$?
  if [ "$claims_brc" = 1 ] && claims_ok "$out" "$rc" 3 && [ "$(printf '%s\n' "$out" | wc -l | tr -d ' ')" = 7 ]; then
    report 0 'a failing claim is quoted under its rule'
  else report 1 'a failing claim is quoted under its rule' "build exit $claims_brc, exit $rc; $out"; fi
  out=$(cd "$proj" && subject --github "$TMP/claims.log" 2>&1); rc=$?
  ann=$(printf '%s\n' "$out" | grep '^::error ')
  if [ "$rc" = 0 ] && [ "$(printf '%s\n' "$ann" | wc -l | tr -d ' ')" = 1 ] &&
     [[ $ann == '::error file=r/dune,line=1,endLine=5,title=Failed dune rule::r/dune:1-5  (rule) alias runtest-claims -- ran: '*"'cat claims.out; exit 1') -- FAIL: claim 1: the gradient is finite | FAIL: claim 3: the arena is released | FAIL: claim 4: two runs agree | (3 more such lines)" ]]; then
    report 0 '--github carries the quoted claims into the annotation'
  else report 1 '--github carries the quoted claims into the annotation' "exit $rc; $out"; fi
fi
if harness_leg 'a Python AssertionError is quoted under its rule #dune'; then
  out=$(cd "$proj" && subject "$TMP/orchestrate.log" 2>&1); rc=$?
  # unittest's verdict and the assertion, over a thousand lines below the command (dune elides
  # the middle of a long output, `...TRUNCATED BY DUNE...`, but keeps its end); not its
  # traceback's source line, which holds "FAIL" without a colon, nor its `FAILED (failures=1)`.
  want="Failed dune rules (1), in the order dune reported them:
  r/dune:7-11  (rule) alias runtest-orchestrate
      | FAIL: test_parity_flag (test_orchestrate.ParityCheckTest.test_parity_flag)
      | AssertionError: 'PASS' != 'FAIL'"
  if [ "$orch_brc" = 1 ] && [ "$rc" = 0 ] && [ "$(printf '%s\n' "$out" | sed -n '1,2p;4,$p')" = "$want" ] &&
     printf '%s\n' "$out" | sed -n '3p' | grep -q "^      ran: (cd _build/default/r && .*orchestrate\.out" &&
     [ "$(grep -c '^test_cell_' "$TMP/orchestrate.log")" -ge 1000 ]; then
    report 0 'a Python AssertionError is quoted under its rule'
  else report 1 'a Python AssertionError is quoted under its rule' "build exit $orch_brc, exit $rc; $out"; fi
fi
if harness_leg 'a scan'"'"'s FAIL: line is quoted under its golden #dune'; then
  out=$(cd "$proj" && subject "$TMP/scan.log" 2>&1); rc=$?
  want='Failed dune rules (1), in the order dune reported them:
  r/scan.expected:1  output differs from this golden
      | +FAIL: arrayjit/lib/backends.ml:120: a label read the scan cannot place'
  if [ "$scan_brc" = 1 ] && [ "$rc" = 0 ] && [ "$out" = "$want" ] &&
     plain "$TMP/scan.log" | grep -q '^ accepted position in use: an Error: payload$' &&
     plain "$TMP/scan.log" | grep -q '^-canary held: the scan reads FAIL: markers$'; then
    report 0 'a scan'"'"'s FAIL: line is quoted under its golden'
  else report 1 'a scan'"'"'s FAIL: line is quoted under its golden' "build exit $scan_brc, exit $rc; $out"; fi
  # The negative control: reading the whole diff quotes its context line and the line it removes.
  if mut=$(mutant no-diff-filter '{ sub(/^  if \(qdiff\) \{.*$/, "  # diff filter removed") } 1'); then
    out=$(cd "$proj" && bash "$mut" "$TMP/scan.log" 2>&1); rc=$?
    if grep -q '^  # diff filter removed$' "$mut" && [ "$out" != "$want" ] &&
       printf '%s\n' "$out" | grep -q '^      | -canary held: the scan reads FAIL: markers$' &&
       printf '%s\n' "$out" | grep -q '^      |  accepted position in use: an Error: payload$'; then
      report 0 'negative control: reading the whole diff quotes what the run did not say'
    else report 1 'negative control: reading the whole diff quotes what the run did not say' \
      "mutant applied: $(grep -c 'diff filter removed' "$mut"); $out"; fi
  else report 1 'negative control: reading the whole diff quotes what the run did not say' 'mutant did not build'; fi
fi
if harness_leg 'another rule'"'"'s output and dune'"'"'s own error are not quoted #dune'; then
  # After the claims block, in one log a passing rule's FAIL: line, behind the `(cd ...)` line
  # CI's display prints above its output; in another dune's error for a target it cannot build,
  # wrapped onto a second line above the `-> required by` chain that names that target. Neither is
  # the claims rule's: its entry reads as built alone, and dune's error is listed whole with its
  # chain.
  cat "$TMP/claims.log" "$TMP/chatty.log" >"$TMP/after-pass.log"
  cat "$TMP/claims.log" "$TMP/unbuildable.log" >"$TMP/after-dune.log"
  pass_ok() { # OUTPUT RC
    claims_ok "$1" "$2" 3 && [ "$(printf '%s\n' "$1" | wc -l | tr -d ' ')" = 7 ]
  }
  dune_ok() { # OUTPUT RC
    claims_ok "$1" "$2" 3 && [ "$(printf '%s\n' "$1" | sed -n '8p')" = 'dune errors without a location (1):' ] &&
      [ "$(printf '%s\n' "$1" | sed -n '9p')" = '  Error: No rule found for r/a-dependency-whose-name-is-long-enough-that-dune-wraps-its-error-onto-a-second-line.exe' ] &&
      printf '%s\n' "$1" | sed -n '10p' | grep -q '^    -> required by alias r/runtest-unbuildable' &&
      [ "$(printf '%s\n' "$1" | wc -l | tr -d ' ')" = 10 ]
  }
  out=$(cd "$proj" && subject "$TMP/after-pass.log" 2>&1); rc=$?
  out2=$(cd "$proj" && subject "$TMP/after-dune.log" 2>&1); rc2=$?
  if [ "$chatty_brc" = 0 ] && [ "$unb_brc" = 1 ] && plain "$TMP/chatty.log" | grep -q '^(cd .*printed by a rule that passes' &&
     plain "$TMP/unbuildable.log" | grep -q '^-> required by alias r/runtest-unbuildable' &&
     plain "$TMP/unbuildable.log" | grep -qx 'Error: No rule found for' &&
     pass_ok "$out" "$rc" && dune_ok "$out2" "$rc2"; then
    report 0 'another rule'"'"'s output and dune'"'"'s own error are not quoted'
  else report 1 'another rule'"'"'s output and dune'"'"'s own error are not quoted' \
    "builds $chatty_brc/$unb_brc, exits $rc/$rc2; $out; $out2; $(cat "$TMP/chatty.log" "$TMP/unbuildable.log")"; fi
  # The negative controls: without the `(cd ...)` boundary the passing rule's command and output
  # lines (both hold its FAIL:) are counted as the claims rule's, and without the chain check, or
  # without holding a wrapped message's lines until its chain, dune's error is, and goes unlisted.
  if mut=$(mutant no-command-boundary '{ sub(/^      if \(s ~ \/\^\\\(cd \/\) qopen = 0$/, "      if (0) qopen = 0 # command boundary removed") } 1'); then
    out=$(cd "$proj" && bash "$mut" "$TMP/after-pass.log" 2>&1); rc=$?
    if grep -q '^      if (0) qopen = 0 # command boundary removed$' "$mut" && ! pass_ok "$out" "$rc" && claims_ok "$out" "$rc" 5; then
      report 0 'negative control: without the command boundary a passing rule'"'"'s line is quoted'
    else report 1 'negative control: without the command boundary a passing rule'"'"'s line is quoted' \
      "mutant applied: $(grep -c 'command boundary removed' "$mut"); $out"; fi
  else report 1 'negative control: without the command boundary a passing rule'"'"'s line is quoted' 'mutant did not build'; fi
  if mut=$(mutant no-chain-check '{ sub(/^    if \(s ~ \/\^-> required by \/\) \{$/, "    if (0) {") } 1'); then
    out=$(cd "$proj" && bash "$mut" "$TMP/after-dune.log" 2>&1); rc=$?
    if grep -q '^    if (0) {$' "$mut" && ! dune_ok "$out" "$rc" && claims_ok "$out" "$rc" 4 &&
       ! printf '%s\n' "$out" | grep -q 'without a location'; then
      report 0 'negative control: without the chain check dune'"'"'s error is quoted as the rule'"'"'s'
    else report 1 'negative control: without the chain check dune'"'"'s error is quoted as the rule'"'"'s' \
      "mutant applied: $(grep -c '^    if (0) {$' "$mut"); $out"; fi
  else report 1 'negative control: without the chain check dune'"'"'s error is quoted as the rule'"'"'s' 'mutant did not build'; fi
  if mut=$(mutant no-wrap-hold '{ sub(/npb < 8 && /, "npb < 0 \\&\\& ") } 1'); then
    out=$(cd "$proj" && bash "$mut" "$TMP/after-dune.log" 2>&1); rc=$?
    if grep -q 'npb < 0 && ' "$mut" && ! dune_ok "$out" "$rc" && claims_ok "$out" "$rc" 4 &&
       ! printf '%s\n' "$out" | grep -q 'without a location'; then
      report 0 'negative control: without holding a wrapped error dune'"'"'s error is quoted as the rule'"'"'s'
    else report 1 'negative control: without holding a wrapped error dune'"'"'s error is quoted as the rule'"'"'s' \
      "mutant applied: $(grep -c 'npb < 0 && ' "$mut"); $out"; fi
  else report 1 'negative control: without holding a wrapped error dune'"'"'s error is quoted as the rule'"'"'s' 'mutant did not build'; fi
fi
# A rule that dies saying nothing the reader matches (gh-ocannl-1927): its entry showed only the
# command. Four real rules, each built alone: one whose program cannot be run, after a banner
# (the shell's own error, worded per platform); one printing more than dune keeps, a banner first
# and its reason last, blank lines after it; one killed by a signal and one exiting 3, both
# silent, so the only line is dune's. Built outside the gates, so each leg passes run alone.
mkdir -p "$proj/l" || exit 2
cat >"$proj/l/dune" <<'DUNE'
(rule
 (alias runtest-missing)
 (action
  (run sh -c "echo 'probe: starting'; ./nonesuch.exe --probe")))

(rule
 (alias runtest-long)
 (deps long.out)
 (action
  (run sh -c "cat long.out; exit 1")))

(rule
 (alias runtest-signal)
 (action
  (run sh -c "kill -SEGV $$")))

(rule
 (alias runtest-status)
 (action
  (run sh -c "exit 3")))
DUNE
{ echo 'unit.exe: banner, version 1'
  i=0; while [ "$i" -lt 2000 ]; do printf 'step %04d of the unit ... ok\n' "$i"; i=$((i + 1)); done
  printf 'unit.exe: the arena ran out at step 2000\n\n  \n'; } >"$proj/l/long.out"
build "$TMP/missing.log" @l/runtest-missing; missing_brc=$?
build "$TMP/long.log" @l/runtest-long; long_brc=$?
build "$TMP/signal.log" @l/runtest-signal; signal_brc=$?
build "$TMP/status.log" @l/runtest-status; status_brc=$?
# An entry of exactly 4 lines: its two, the command, and the last output line matching RE.
last_ok() { # OUTPUT RC SECOND-LINE COMMAND-RE LAST-RE
  [ "$2" = 0 ] && [ "$(printf '%s\n' "$1" | sed -n '1p')" = 'Failed dune rules (1), in the order dune reported them:' ] &&
    [ "$(printf '%s\n' "$1" | sed -n '2p')" = "$3" ] &&
    printf '%s\n' "$1" | sed -n '3p' | grep -qE "^      ran: \(cd _build/default/l && $4\)$" &&
    printf '%s\n' "$1" | sed -n '4p' | grep -qE "^      last output: $5$" &&
    [ "$(printf '%s\n' "$1" | wc -l | tr -d ' ')" = 4 ]
}
missing_ok() { last_ok "$1" "$2" '  l/dune:1-4  (rule) alias runtest-missing' "[^ ]*sh -c .*nonesuch\.exe --probe'" '[^|]*nonesuch\.exe.*'; }
long_ok() { last_ok "$1" "$2" '  l/dune:6-10  (rule) alias runtest-long' "[^ ]*sh -c 'cat long\.out; exit 1'" 'unit\.exe: the arena ran out at step 2000'; }
if harness_leg 'a rule with no matching line shows its last output line #dune'; then
  out=$(cd "$proj" && subject "$TMP/missing.log" 2>&1); rc=$?
  if [ "$missing_brc" = 1 ] && plain "$TMP/missing.log" | grep -qx 'probe: starting' && missing_ok "$out" "$rc"; then
    report 0 'a shell error is shown as the last output line'
  else report 1 'a shell error is shown as the last output line' "build exit $missing_brc, exit $rc; $out"; fi
  out=$(cd "$proj" && subject --github "$TMP/missing.log" 2>&1); rc=$?
  ann=$(printf '%s\n' "$out" | grep '^::error ')
  if [ "$rc" = 0 ] && [ "$(printf '%s\n' "$ann" | wc -l | tr -d ' ')" = 1 ] &&
     [[ $ann == '::error file=l/dune,line=1,endLine=4,title=Failed dune rule::l/dune:1-4  (rule) alias runtest-missing -- ran: (cd _build/default/l && '*"--probe') -- last output: "*'nonesuch.exe'* ]]; then
    report 0 '--github carries the last output line into the annotation'
  else report 1 '--github carries the last output line into the annotation' "exit $rc; $out"; fi
  # The end of an output longer than dune keeps, not its banner and not the blank lines after it.
  out=$(cd "$proj" && subject "$TMP/long.log" 2>&1); rc=$?
  if [ "$long_brc" = 1 ] && plain "$TMP/long.log" | grep -qx 'unit.exe: banner, version 1' &&
     [ "$(grep -c '^step ' "$TMP/long.log")" -ge 1000 ] && long_ok "$out" "$rc"; then
    report 0 'a long output is quoted by its last non-blank line'
  else report 1 'a long output is quoted by its last non-blank line' "build exit $long_brc, exit $rc; $out"; fi
  # A silent death leaves dune's own line, which says how the command ended.
  out=$(cd "$proj" && subject "$TMP/signal.log" 2>&1); rc=$?
  out2=$(cd "$proj" && subject "$TMP/status.log" 2>&1); rc2=$?
  if [ "$signal_brc" = 1 ] && [ "$status_brc" = 1 ] &&
     last_ok "$out" "$rc" '  l/dune:12-15  (rule) alias runtest-signal' "[^ ]*sh -c 'kill -SEGV \\$\\$'" 'Command got signal SEGV\.' &&
     last_ok "$out2" "$rc2" '  l/dune:17-20  (rule) alias runtest-status' "[^ ]*sh -c 'exit 3'" 'Command exited with code 3\.'; then
    report 0 'a silent signal or exit status shows the line dune prints for it'
  else report 1 'a silent signal or exit status shows the line dune prints for it' \
    "builds $signal_brc/$status_brc, exits $rc/$rc2; $out; $out2"; fi
  # The negative controls: keeping the first line quotes the banner, and without the
  # no-matching-line condition a rule that says why (the claims one) gains a second kind of quote.
  if mut=$(mutant first-output-line '{ sub(/^        if \(s !~ \/\^\[ \\t\]\*\$\/\) qlast = s$/, "        if (qlast == \"\") qlast = s # first line kept") } 1'); then
    out=$(cd "$proj" && bash "$mut" "$TMP/long.log" 2>&1); rc=$?
    if grep -q '# first line kept$' "$mut" && ! long_ok "$out" "$rc" &&
       [ "$(printf '%s\n' "$out" | sed -n '4p')" = '      last output: unit.exe: banner, version 1' ]; then
      report 0 'negative control: keeping the first line quotes the banner'
    else report 1 'negative control: keeping the first line quotes the banner' \
      "mutant applied: $(grep -c 'first line kept' "$mut"); $out"; fi
  else report 1 'negative control: keeping the first line quotes the banner' 'mutant did not build'; fi
  if mut=$(mutant last-output-always '{ sub(/\(qn == 0 && detail ~ \/\^ran: \/\) \? qlast : ""$/, "qlast # always") } 1'); then
    out=$(cd "$proj" && bash "$mut" "$TMP/claims.log" 2>&1); rc=$?
    if grep -q 'qlast # always$' "$mut" && claims_ok "$out" "$rc" 3 &&
       [ "$(printf '%s\n' "$out" | sed -n '8p')" = '      last output: FAILED: 5 checks did not hold' ]; then
      report 0 'negative control: without the condition a rule that says why is quoted twice'
    else report 1 'negative control: without the condition a rule that says why is quoted twice' \
      "mutant applied: $(grep -c 'qlast # always' "$mut"); $out"; fi
  else report 1 'negative control: without the condition a rule that says why is quoted twice' 'mutant did not build'; fi
fi
if harness_leg 'the steps after the build lend no last output line #dune'; then
  # `gh run view --log-failed` of a red job: the build step ends with GitHub's `##[error]` line,
  # and the listing step's own lines follow it. When the failed rule printed last, nothing stands
  # between its block and those; the entry reads as from the build log alone.
  { LC_ALL=C awk '{ printf "Build (ubuntu-latest, 5.5.x, main)\tRun opam exec -- dune build\t2026-11-06T19:53:18.%07dZ %s\n", NR, $0 }' "$TMP/missing.log"
    printf 'Build (ubuntu-latest, 5.5.x, main)\tRun opam exec -- dune build\t2026-11-06T19:53:18.9482040Z %s\n' \
      '##[error]Process completed with exit code 1.'
    printf 'Build (ubuntu-latest, 5.5.x, main)\tList the failed dune rules\t2026-11-06T19:53:20.5748380Z %s\n' \
      '##[group]Run tools/ci-failed-rules.sh --github "$RUNNER_TEMP/dune-build.log"' 'shell: /bin/bash --noprofile --norc -e -o pipefail {0}' \
      '##[endgroup]' 'OCANNL host: cpu="fixture" cpus=3'; } >"$TMP/steps.log"
  out=$(cd "$proj" && subject "$TMP/steps.log" 2>&1); rc=$?
  report "$(missing_ok "$out" "$rc"; echo $?)" 'the steps after the build lend no last output line' "exit $rc; $out"
  # The negative control: without the `##[` boundary the last line of the job is the rule's.
  if mut=$(mutant no-step-boundary '{ sub(/^      else if \(s ~ \/\^##\\\[\/\) qopen = 0$/, "      # step boundary removed") } 1'); then
    out=$(cd "$proj" && bash "$mut" "$TMP/steps.log" 2>&1); rc=$?
    if grep -q '^      # step boundary removed$' "$mut" && ! missing_ok "$out" "$rc" &&
       [ "$(printf '%s\n' "$out" | sed -n '4p')" = '      last output: OCANNL host: cpu="fixture" cpus=3' ]; then
      report 0 'negative control: without the step boundary a later step'"'"'s line is quoted'
    else report 1 'negative control: without the step boundary a later step'"'"'s line is quoted' \
      "mutant applied: $(grep -c 'step boundary removed' "$mut"); $out"; fi
  else report 1 'negative control: without the step boundary a later step'"'"'s line is quoted' 'mutant did not build'; fi
fi
if harness_leg 'the green build prints nothing #dune'; then
  write_fixture 0
  build "$TMP/green.log" @runtest; brc=$?
  out=$(cd "$proj" && subject "$TMP/green.log" 2>&1); rc=$?
  report "$([ "$brc" = 0 ] && [ "$rc" = 0 ] && [ -z "$out" ]; echo $?)" 'the green build prints nothing' \
    "build exit $brc, exit $rc; $out; $(cat "$TMP/green.log")"
fi
finish
