#!/usr/bin/env bash
# Tests for tools/test-run.sh: what a finished run says -- the digest's verdicts, promotion,
# slowest actions and BUILD FAILED -- with `fresh` and `repeat`.
#
#   tools/test-test-run-digest.sh          # run every leg of this member
#   tools/test-test-run-digest.sh --keep   # keep the temp dir for inspection
#   tools/test-test-run-digest.sh --only '^81:'   # only the legs whose names match
#   tools/test-test-run-digest.sh --subject FILE   # test FILE, not tools/test-run.sh
#
# One member of the harness of tools/test-run.sh (gh-ocannl-2002), whose
# others `tools/test-test-run.sh --members` lists; that script runs them all,
# or the ones holding the legs an --only names. The conventions every member
# shares -- --subject, how a leg is named, gated and abandoned, what is tested
# and the fixtures the legs run against -- are tools/test-test-run-common.sh's
# header. Leg numbers are the harness's, not this file's: they stayed as they
# were when it was one file, so a member's are not consecutive.
#
# Groups: #repeat (9-25, 38, 83, 85, 84), #digest (35, 36, 87, 68, 69, 71, 72, 73, 80, 92), #fresh (81), #hermetic (70).
#
# Legs:
#   9. `repeat` forces and preserves three identical dune runs while holding the
#      worktree lock for every iteration.
#  10. stdout drift is a distinct red result, with pairwise diff artifacts.
#  11. stderr-only drift is reported separately and remains a green diagnostic.
#  12. a red dune iteration keeps its nonzero exit code even when repeatable.
#  13. `--alone` serializes dune with `-j 1` on every iteration.
#  14. an active repeat is `last`, and `stop last` cancels the whole set after
#      the current iteration rather than launching the remaining ones.
#  15. cancellation during post-loop comparison still publishes a verdict.
#  16. `wait`'s default bounded deadline covers every repeat iteration.
#  17. repeat cancellation state and traps precede publication as `last`.
#  18. cancellation is checked on both sides of supervisor launch.
#  19. cancelling a later iteration preserves an earlier test failure.
#  20. a supervisor killed without its Dune group cannot overlap the next run.
#  21. the dead supervisor pid is cleared before orphan-group reaping.
#  22. orphan cleanup gates on reachability and revalidates identity for KILL.
#  23. a zombie retains its recorded identity while remaining non-live.
#  24. a setsid descendant cannot escape into the next repeat build context.
#  25. repeat isolation flags precede `dune exec`'s argument separator.
#  35. an invocation dune's own parser refuses -- unknown option, unknown
#      subcommand, malformed operand -- digests as INVOCATION REFUSED quoting
#      dune's complaint: `run`/`wait` exit 2 over a RECORDED exit 1, `status`
#      keeps its publication 0, and dune was invoked exactly once; the
#      launch's own prelude in the log (backends, width) does not hide it.
#  36. controls: a red run (`Error:`/`File` lines, exit 1) still digests as
#      FAIL with exit 1 and that fingerprint, also where dune is made to
#      color them (CLICOLOR_FORCE=1) -- including one whose output opens with
#      a `dune:` line, and one that prints a complete nested dune refusal and
#      then goes on (the shape must be alone in the log, not merely first).
#  38. `repeat` stops after a refused first iteration and exits 2 under the
#      same verdict, while a merely red iteration is still repeated in full.
#  83. `repeat` has both bounds (gh-ocannl-1252): a test past the per-test cap
#      ends the set as TIMEOUT (142) naming it, killed; a GPU batch's wedged
#      device ends it as DEVICE UNHEALTHY (69) before any dune; the default is
#      run's (1500s on a GPU batch, none on a CPU one), and a test inside the
#      cap or an answering device lets every iteration run.
#  84. `repeat --direct` (gh-ocannl-1377), against a real dune: one build,
#      then each named test's own rules replayed once per iteration -- a
#      `(test)` stanza's run and a target rule counted, the environment and
#      output dune's own run had, an alias action left to dune's --force, no
#      iteration compiling -- with DIFFERING for an output that moves, REPLAY
#      DIVERGED when the replay can be told from dune, a sandboxed rule and an
#      unreplayable action refused before any iteration, BUILD FAILED for a
#      test that does not build, the per-test cap timing and killing a
#      replayed test, a replayed test's session-escaped child refusing the
#      build tree's reuse, and non-alias operands refused; a mutant without
#      the replay shows dune alone does not re-run a target-producing test,
#      and one closing the replay's inherited descriptors that the escaped
#      child then goes unseen. Its tests see no step directory either: the
#      environment it compares leaves out none of the runner's own variables.
#  85 sits after leg 83 (not applicable off Linux, gh-ocannl-1930: the
#      stand-in is a Linux facility, and so is the PID 1 it stands in for;
#      85b is that probe's own control): under a stand-in for a PID 1 that reaps
#      no orphans -- a child subreaper that waits for its own child only --
#      the zombie-only groups `repeat` and `run` leave are read as gone, not
#      refused: a capped set keeps TIMEOUT, a GPU set passes through the
#      resolution's inert-zombie branch, and `run` keeps its verdict, while
#      the stand-in must have held a zombie and nothing live (gh-ocannl-1376).
#      The negative controls take out the iteration shell's reaping trap and
#      that branch: each set is then refused, exit 2.
#  84 sits after leg 36: a red run carrying, at a line's start, the line a
#      GPU backend's device-init deadline writes before its process dies by
#      SIGALRM is WEDGED DEVICE (exit 1) quoting it; indented, it is FAIL
#      (gh-ocannl-1281).
#  68-69 sit after leg 36: promotion is offered only on a diff dune printed
#      (gh-ocannl-1055).
#  68. a rule whose action failed, its quoted stanza naming a `.corrected`
#      file, digests as `action failed (no diff)` with its fingerprint, and is
#      never sent to `dune promote`.
#  69. a real diff hunk, plain or colored, still offers promotion; a green run
#      offers neither. A run that chose its own diff command (patdiff's
#      doubled headers still count; `-`, on the command line or in
#      DUNE_DIFF_COMMAND, prints none) is told promotion is possible, never
#      "no diff" -- unless the option sits past dune's own `--` -- whatever
#      its log names (an inline-expect rule's names only its `.ml`). A log
#      longer than the digest's scan window is never "nothing to promote",
#      and a repeat set red on drift alone is not called a failed action.
#      Since gh-ocannl-1087 these read the log only as the fallback, for a run
#      that recorded no promotion list (legs 71-73).
#  71-73 sit after leg 69: the run's last phase records dune's own promotion
#      list, and the digest reads it (gh-ocannl-1087).
#  71. the record decides whatever the log shows: a listed file under a
#      suppressed diff is promotion, named; a hunk over an empty list is not;
#      a green run says nothing; `status`/`wait` read the same record; the
#      list's stream follows dune's version (stderr before 3.22), CRs are
#      dropped, a long list is cut only in the digest, and a dune killed by a
#      signal keeps the status an exec'd one would have.
#  72. no record, and the log decides, where the list cannot be taken
#      faithfully: it fails, a line is not a path here, the version is
#      unreadable, the argv names another build directory or root (dune not
#      asked), or the cap is nearly spent; a missing build directory, also
#      under DUNE_BUILD_DIR's name, is an empty record asked of nobody.
#  73. a refused slot or a cancelled run never reads a recorded list.
#  80 sits after leg 73: the run hands its dune a trace file of its own in
#      the run directory, records the five slowest processes of that trace
#      (a synthetic one), and the digest names them with their seconds,
#      longest first, an unfinished one as `open` -- on a green run and on
#      one the cap killed; the trace is deleted once read. The worktree's
#      shared `_build/trace.csexp` is never read nor touched. Nothing is
#      recorded from a trace the reader cannot read, where the argv names its
#      own trace file, or where the first word is not a subcommand known to
#      take one; the runner's option reaches dune exactly once, before its
#      `--` (mutants doubling it or moving it past `--` show in the pinned
#      calls). A reader past its bound is KILLed -- an ambient bound hook can
#      only shorten it -- and one KILL cannot reap (a mutant) is abandoned:
#      either way the verdict publishes, the lock clears, and no record is
#      left. A refused slot or device probe and a setup error never
#      print a record, a cancellation does. Beside it, against a real dune: a
#      manual build running beside a managed run keeps its live trace.
#  92 sits after leg 80b: a red run whose own trace holds a failed build step
#      (a nonzero exit producing only compile artifacts) is BUILD FAILED,
#      exit 65 over a recorded 1 (`wait` too), naming the step and the output
#      an earlier run left for a named test this run never wrote, never one
#      this run's trace produced; a red test, a failed source generator and
#      an unfinished compile stay FAIL (exit 1); an argv naming its own build
#      directory looks in no worktree `_build` (gh-ocannl-1547). 92b: the same
#      against a real dune, a type error beside a red and a passing test.
#  81 sits after leg 80: `fresh` hands dune exactly `build --build-dir=<the
#      run's own> --force --cache=disabled` and the named aliases, records that
#      command, and removes the build directory with the run (the fixture dune
#      makes it, a control shows); its digest gives each named test the
#      seconds of the processes that are its action -- the alias action, also
#      in a subdirectory under a recursive `@` and not under `@@`, a producer
#      of NAME.exe.output, NAME.actual, NAME.ll.actual, NAME.coverage or (for
#      NAME BASE-VARIANT) BASE.VARIANT.actual, as any target of its process,
#      never a compile artifact (NAME.exe, NAME.pp.ml), or else an unclaimed
#      NAME.exe in its directory -- from a synthetic trace; a dune that exits
#      with the cap nearly spent is UNTIMED, not a TIMEOUT. A named test with no such process
#      is NOT RUN (exit 2), the others still timed, also when the trace holds
#      no process at all; a red run is FAIL with its times and not promotable;
#      an unreadable trace is UNTIMED (exit 1); operands other than per-test
#      aliases and -j N are refused before dune runs.

. "$(cd "$(dirname "$0")" && pwd)/test-test-run-common.sh"


shared_through fixture_dune

# This is an ordering invariant, not a timing lottery: once publish_run writes
# `last`, an external stop may arrive on the next instruction. Pin all three
# prerequisites above that line so a future refactor cannot reopen the gap.
leg_9_25() {
completed_line=$(grep -n '^    completed=0$' "$SRC" | cut -d: -f1)
signal_trap_line=$(grep -n "^    trap 'repeat_signal TERM' TERM$" "$SRC" | cut -d: -f1)
exit_trap_line=$(grep -n '^    trap repeat_exit EXIT$' "$SRC" | cut -d: -f1)
publish_line=$(grep -n '^    publish_run || die "cannot publish repeat run ' "$SRC" | cut -d: -f1)
if [ -n "$completed_line" ] && [ -n "$signal_trap_line" ] \
   && [ -n "$exit_trap_line" ] && [ -n "$publish_line" ] \
   && [ "$completed_line" -lt "$publish_line" ] \
   && [ "$signal_trap_line" -lt "$publish_line" ] \
   && [ "$exit_trap_line" -lt "$publish_line" ]; then
  report 0 "repeat: cancellation lifecycle is armed before publication"
else
  report 1 "repeat: cancellation lifecycle is armed before publication" \
    "completed=${completed_line:-missing} signal=${signal_trap_line:-missing} exit=${exit_trap_line:-missing} publish=${publish_line:-missing}"
fi

# A real red remains the useful verdict when a later iteration is cancelled.
# Block only iteration two, stop the managed set there, and prove both the
# retained exit 7 and the cancellation annotation.
repeat_red_cancel_runs=$TMP/repeat-runs-red-cancel
repeat_red_cancel_prefix=$TMP/repeat-red-cancel
mkdir -p "$repeat_red_cancel_runs"
: >"$TMP/repeat-red-cancel.counter"
: >"$TMP/repeat-red-cancel.calls"
REPEAT_TEST_MODE=fail_first \
REPEAT_TEST_COUNTER=$TMP/repeat-red-cancel.counter \
REPEAT_TEST_CALLS=$TMP/repeat-red-cancel.calls \
REPEAT_TEST_WAIT_PREFIX=$repeat_red_cancel_prefix \
REPEAT_TEST_WAIT_AT=2 \
REPEAT_TEST_DIFF_WAIT_PREFIX= \
REPEAT_TEST_REAL_DIFF="$(command -v diff)" \
REPEAT_TEST_ORPHAN_PID= \
REPEAT_TEST_ORPHAN_REAPED= \
OCANNL_TOOL_TEST_RUNS=$repeat_red_cancel_runs \
PATH=$repeat_bin:$PATH \
  "$repeat_root/tools/test-run.sh" repeat 3 build @cheap \
  >"$TMP/repeat-red-cancel.out" 2>"$TMP/repeat-red-cancel.err" &
repeat_pid=$!
if ! await_fixture_ready "$repeat_red_cancel_prefix.ready"; then
  report 1 "repeat: earlier failure survives later cancellation" \
    "setup timeout waiting for iteration two's readiness marker"
  repeat_abandon "$repeat_red_cancel_prefix"
  harness_abandon "a repeat fixture never became ready"
  return
fi
red_cancel_stop=$(OCANNL_TOOL_TEST_RUNS=$repeat_red_cancel_runs \
  "$repeat_root/tools/test-run.sh" stop last 2>"$TMP/repeat-red-cancel-stop.err")
red_cancel_stop_rc=$?
touch "$repeat_red_cancel_prefix.release"
wait "$repeat_pid"
repeat_red_cancel_rc=$?
repeat_pid=
repeat_red_cancel_dir=$(OCANNL_TOOL_TEST_RUNS="$repeat_red_cancel_runs" "$repeat_root/tools/test-run.sh" paths run last 2>/dev/null)
if [ "$red_cancel_stop_rc" = 0 ] \
   && grep -q '^sent TERM to the repeat coordinator; ' <<<"$red_cancel_stop" \
   && [ "$repeat_red_cancel_rc" = 7 ] \
   && [ "$(cat "$repeat_red_cancel_dir/exit" 2>/dev/null)" = 7 ] \
   && grep -q '^repeat result: CANCELLED -- completed 2 of 3 iterations$' "$TMP/repeat-red-cancel.out"; then
  report 0 "repeat: earlier failure survives later cancellation"
else
  report 1 "repeat: earlier failure survives later cancellation" \
    "stop $red_cancel_stop_rc; repeat $repeat_red_cancel_rc: $(cat "$TMP/repeat-red-cancel.out")"
fi

# Kill the first capped supervisor from inside its Dune child, leaving that
# child/group alive. Iteration two refuses to pass unless the coordinator's
# identity-verified reap completed first.
repeat_orphan_runs=$TMP/repeat-runs-orphan
repeat_orphan_marker=$TMP/repeat-orphan-reaped
repeat_orphan_pid_file=$TMP/repeat-orphan-pid
mkdir -p "$repeat_orphan_runs"
: >"$TMP/repeat-orphan.counter"
: >"$TMP/repeat-orphan.calls"
REPEAT_TEST_MODE=orphan_first \
REPEAT_TEST_COUNTER=$TMP/repeat-orphan.counter \
REPEAT_TEST_CALLS=$TMP/repeat-orphan.calls \
REPEAT_TEST_WAIT_PREFIX= \
REPEAT_TEST_WAIT_AT= \
REPEAT_TEST_DIFF_WAIT_PREFIX= \
REPEAT_TEST_REAL_DIFF="$(command -v diff)" \
REPEAT_TEST_ORPHAN_PID=$repeat_orphan_pid_file \
REPEAT_TEST_ORPHAN_REAPED=$repeat_orphan_marker \
OCANNL_TOOL_TEST_RUNS=$repeat_orphan_runs \
PATH=$repeat_bin:$PATH \
  "$repeat_root/tools/test-run.sh" repeat 2 build @cheap \
  >"$TMP/repeat-orphan.out" 2>"$TMP/repeat-orphan.err"
repeat_orphan_rc=$?
repeat_orphan_dir=$(OCANNL_TOOL_TEST_RUNS="$repeat_orphan_runs" "$repeat_root/tools/test-run.sh" paths run last 2>/dev/null)
orphan_pid=$(cat "$repeat_orphan_pid_file" 2>/dev/null)
if [ "$repeat_orphan_rc" = 137 ] \
   && [ -e "$repeat_orphan_marker" ] \
   && [ "$(cat "$repeat_orphan_dir/iteration-2/exit" 2>/dev/null)" = 0 ] \
   && grep -q '^repeat: iteration group .* survived its supervisor; reaping before reuse$' "$TMP/repeat-orphan.out"; then
  report 0 "repeat: surviving iteration group is reaped before reuse"
else
  report 1 "repeat: surviving iteration group is reaped before reuse" \
    "exit $repeat_orphan_rc; marker=$([ -e "$repeat_orphan_marker" ] && echo yes || echo no); output: $(cat "$TMP/repeat-orphan.out")"
fi
if [ -n "$orphan_pid" ] && kill -0 "$orphan_pid" 2>/dev/null; then
  kill -KILL -- "-$orphan_pid" 2>/dev/null
  kill -KILL "$orphan_pid" 2>/dev/null
fi

# A Dune action can leave its recorded process group with setsid. The FIFO
# writer inherited from the launch must still expose that survivor and refuse
# the shared build before iteration two, even though the group reap sees no
# reachable original group. Releasing the fixture afterwards also proves the
# refusal retained (rather than deleted) its build context while it was live.
repeat_escape_runs=$TMP/repeat-runs-session-escape
escape_release=$TMP/repeat-session-escape.release
escape_pid_file=$TMP/repeat-session-escape.pid
mkdir -p "$repeat_escape_runs"
: >"$TMP/repeat-session-escape.counter"
: >"$TMP/repeat-session-escape.calls"
if REPEAT_TEST_MODE=session_escape \
   REPEAT_TEST_COUNTER=$TMP/repeat-session-escape.counter \
   REPEAT_TEST_CALLS=$TMP/repeat-session-escape.calls \
   REPEAT_TEST_WAIT_PREFIX= \
   REPEAT_TEST_WAIT_AT= \
   REPEAT_TEST_DIFF_WAIT_PREFIX= \
   REPEAT_TEST_REAL_DIFF="$(command -v diff)" \
   REPEAT_TEST_ORPHAN_PID= \
   REPEAT_TEST_ORPHAN_REAPED= \
   REPEAT_TEST_ESCAPE_PID=$escape_pid_file \
   REPEAT_TEST_ESCAPE_RELEASE=$escape_release \
   OCANNL_TOOL_TEST_RUNS=$repeat_escape_runs \
   PATH=$repeat_bin:$PATH \
     "$repeat_root/tools/test-run.sh" repeat 2 build @cheap \
     >"$TMP/repeat-session-escape.out" 2>"$TMP/repeat-session-escape.err"; then
  repeat_escape_rc=0
else
  repeat_escape_rc=$?
fi
repeat_escape_dir=$(OCANNL_TOOL_TEST_RUNS="$repeat_escape_runs" "$repeat_root/tools/test-run.sh" paths run last 2>/dev/null)
escape_pid=$(cat "$escape_pid_file" 2>/dev/null)
escape_pgid=$(ppgid "$escape_pid")
recorded_pgid=$(cat "$repeat_escape_dir/iteration-1/pgid" 2>/dev/null)
if [ "$repeat_escape_rc" = 2 ] \
   && [ -n "$escape_pid" ] && kill -0 "$escape_pid" 2>/dev/null \
   && [ -n "$escape_pgid" ] && [ "$escape_pgid" != "$recorded_pgid" ] \
   && [ -d "$repeat_escape_dir/build" ] \
   && [ ! -d "$repeat_escape_dir/iteration-2" ] \
   && grep -q 'left a session-escaped descendant; refusing to reuse' \
        "$TMP/repeat-session-escape.err"; then
  report 0 "repeat: a session-escaped descendant blocks build reuse"
else
  report 1 "repeat: a session-escaped descendant blocks build reuse" \
    "exit $repeat_escape_rc; pid=${escape_pid:-missing}; escaped-pgid=${escape_pgid:-missing}; recorded-pgid=${recorded_pgid:-missing}; stderr=$(cat "$TMP/repeat-session-escape.err")"
fi
touch "$escape_release"
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  kill -0 "$escape_pid" 2>/dev/null || break
  sleep 0.1
done
if kill -0 "$escape_pid" 2>/dev/null; then kill -KILL "$escape_pid" 2>/dev/null; fi
escape_pid=
escape_release=

# Every step is one launch (repeat_launch), so the bracket is: the loop refuses a cancelled
# launch right before calling it, and the launch rechecks right after starting the supervisor.
sed -n '/^    repeat_launch() {/,/^    }/p' "$SRC" >"$TMP/repeat-launch.sh"
call_line=$(grep -n '^      repeat_launch "$iter" "iteration $i"$' "$SRC" | head -1 | cut -d: -f1)
pre_launch_line=$(grep -n '^      \[ -z "$repeat_cancelled" \] || break$' "$SRC" | cut -d: -f1 |
  awk -v call="${call_line:-0}" '$1 < call { last = $1 } END { if (last) print last }')
launch_line=$(grep -n '^        perl -e "$supervisor_perl" -- "$cap" "${step_cmd\[@\]}" \\$' "$TMP/repeat-launch.sh" | head -1 | cut -d: -f1)
supervisor_line=$(grep -n '^      repeat_sup=\$!$' "$TMP/repeat-launch.sh" | cut -d: -f1)
post_launch_line=$(grep -n '^      \[ -z "$repeat_cancelled" \] || kill "-\$repeat_cancelled" "\$repeat_sup" 2>/dev/null$' "$TMP/repeat-launch.sh" | cut -d: -f1)
if [ -n "$pre_launch_line" ] && [ -n "$call_line" ] && [ -n "$launch_line" ] \
   && [ -n "$supervisor_line" ] && [ -n "$post_launch_line" ] \
   && [ "$pre_launch_line" -lt "$call_line" ] \
   && [ "$launch_line" -lt "$supervisor_line" ] \
   && [ "$supervisor_line" -lt "$post_launch_line" ]; then
  report 0 "repeat: cancellation brackets supervisor launch"
else
  report 1 "repeat: cancellation brackets supervisor launch" \
    "pre=${pre_launch_line:-missing} call=${call_line:-missing} launch=${launch_line:-missing} supervisor=${supervisor_line:-missing} post=${post_launch_line:-missing}"
fi

clear_supervisor_line=$(grep -n '^      repeat_sup=$' "$TMP/repeat-launch.sh" | tail -1 | cut -d: -f1)
reap_group_line=$(grep -n '^      reap_repeat_group "$sdir"$' "$TMP/repeat-launch.sh" | cut -d: -f1)
if [ -n "$clear_supervisor_line" ] && [ -n "$reap_group_line" ] \
   && [ "$clear_supervisor_line" -lt "$reap_group_line" ]; then
  report 0 "repeat: dead supervisor pid is cleared before group reap"
else
  report 1 "repeat: dead supervisor pid is cleared before group reap" \
    "clear=${clear_supervisor_line:-missing} reap=${reap_group_line:-missing}"
fi

sed -n '/^    reap_repeat_group() {/,/^    }/p' "$SRC" >"$TMP/reap-repeat-group.sh"
sed -n '/^      if ! group_identity_matches "$iter_dir"; then$/,/^      kill -KILL/p' \
  "$TMP/reap-repeat-group.sh" >"$TMP/reap-before-kill.sh"
if grep -q 'kill -0 -- "-\$pg"' "$TMP/reap-repeat-group.sh" \
   && grep -q 'group_alive "$pg" || return 0' "$TMP/reap-repeat-group.sh" \
   && ! grep -q 'group_alive' "$TMP/reap-before-kill.sh" \
   && grep -B6 'kill -KILL -- "-\$pg"' "$TMP/reap-repeat-group.sh" \
        | grep -q 'leaderless iteration group \$pg survived TERM' \
   && grep -q 'leaderless iteration group \$pg survived KILL' "$TMP/reap-repeat-group.sh"; then
  report 0 "repeat: orphan reap fails closed after TERM and accepts post-KILL zombies"
else
  report 1 "repeat: orphan reap fails closed after TERM and accepts post-KILL zombies" \
    "$(tr '\n' ';' <"$TMP/reap-repeat-group.sh")"
fi

repeat_probe repeat-identical stable 3 build @cheap
if [ "$repeat_rc" = 0 ] && grep -q '^repeat result: IDENTICAL -- ' <<<"$repeat_out" \
   && [ "$(cat "$TMP/repeat-identical.counter")" = 3 ] \
   && [ "$(grep -c '^clean ' "$TMP/repeat-identical.calls")" = 3 ] \
   && [ "$(grep -c -- '--force' "$TMP/repeat-identical.calls")" = 3 ] \
   && [ "$(grep -c -- '--cache=disabled' "$TMP/repeat-identical.calls")" = 3 ] \
   && [ "$(grep -c -- '--build-dir=' "$TMP/repeat-identical.calls")" = 6 ] \
   && [ -n "$repeat_dir" ] \
   && [ ! -e "$repeat_dir/build" ] \
   && [ "$(find "$repeat_dir" \( -name stdout -o -name stderr \) | wc -l | tr -d ' ')" = 6 ]; then
  report 0 "repeat: identical forced runs retain every stdout/stderr"
else
  report 1 "repeat: identical forced runs retain every stdout/stderr" \
    "exit $repeat_rc; output: ${repeat_out:-<nothing>}; stderr: $(cat "$TMP/repeat-identical.err")"
fi

repeat_probe repeat-stdout stdout 3 build @cheap
if [ "$repeat_rc" = 1 ] && grep -q '^repeat result: DIFFERING -- ' <<<"$repeat_out" \
   && [ -s "$repeat_dir/diffs/1-2.stdout" ]; then
  report 0 "repeat: stdout drift is red and pairwise-diffed"
else
  report 1 "repeat: stdout drift is red and pairwise-diffed" \
    "exit $repeat_rc; output: ${repeat_out:-<nothing>}"
fi

repeat_probe repeat-stderr stderr 3 build @cheap
if [ "$repeat_rc" = 0 ] && grep -q '^repeat result: STDERR-ONLY -- ' <<<"$repeat_out" \
   && [ -s "$repeat_dir/diffs/1-2.stderr" ]; then
  report 0 "repeat: stderr-only drift is distinct and diagnostic-green"
else
  report 1 "repeat: stderr-only drift is distinct and diagnostic-green" \
    "exit $repeat_rc; output: ${repeat_out:-<nothing>}"
fi

repeat_probe repeat-red fail 2 build @cheap
if [ "$repeat_rc" = 7 ] && grep -q '^repeat result: IDENTICAL -- ' <<<"$repeat_out"; then
  report 0 "repeat: a repeatable red dune leg keeps its exit code"
else
  report 1 "repeat: a repeatable red dune leg keeps its exit code" \
    "expected exit 7; got $repeat_rc; output: ${repeat_out:-<nothing>}"
fi

repeat_probe repeat-alone stable --alone 2 build @cheap
if [ "$repeat_rc" = 0 ] && [ "$(grep -c -- '-j 1' "$TMP/repeat-alone.calls")" = 2 ] \
   && grep -q 'iteration 1/2 -- dune (alone, -j 1)' <<<"$repeat_out"; then
  report 0 "repeat: --alone serializes every dune iteration"
else
  report 1 "repeat: --alone serializes every dune iteration" \
    "exit $repeat_rc; calls: $(tr '\n' ';' <"$TMP/repeat-alone.calls"); output: ${repeat_out:-<nothing>}"
fi

# The first `--` belongs to Dune, not to repeat. The trailing --force is a
# program argument on purpose: this fails if isolation flags are appended at
# the end or if the splice mistakes a later option-looking argument for Dune's.
repeat_probe repeat-separator stable 2 exec ./prog.exe -- alpha --force
if [ "$repeat_rc" = 0 ] \
   && [ "$(grep -c '^exec ./prog.exe --force --cache=disabled --build-dir=.* -- alpha --force$' \
          "$TMP/repeat-separator.calls")" = 2 ]; then
  report 0 "repeat: isolation flags precede Dune's argument separator"
else
  report 1 "repeat: isolation flags precede Dune's argument separator" \
    "exit $repeat_rc; calls: $(tr '\n' ';' <"$TMP/repeat-separator.calls")"
fi

# An active repeat must replace `last`, and stop must signal the OUTER
# coordinator so it records cancellation and refuses to start iteration two.
repeat_stop_runs=$TMP/repeat-runs-stop
repeat_stop_prefix=$TMP/repeat-stop
mkdir -p "$repeat_stop_runs"
: >"$TMP/repeat-stop.counter"
: >"$TMP/repeat-stop.calls"
REPEAT_TEST_MODE=stable \
REPEAT_TEST_COUNTER=$TMP/repeat-stop.counter \
REPEAT_TEST_CALLS=$TMP/repeat-stop.calls \
REPEAT_TEST_WAIT_PREFIX=$repeat_stop_prefix \
OCANNL_TOOL_TEST_RUNS=$repeat_stop_runs \
PATH=$repeat_bin:$PATH \
  "$repeat_root/tools/test-run.sh" repeat 3 build @cheap \
  >"$TMP/repeat-stop.out" 2>"$TMP/repeat-stop.err" &
repeat_pid=$!
if ! await_fixture_ready "$repeat_stop_prefix.ready"; then
  report 1 "repeat: last resolves active state and stop cancels the whole set" \
    "setup timeout waiting for iteration one's readiness marker"
  repeat_abandon "$repeat_stop_prefix"
  harness_abandon "a repeat fixture never became ready"
  return
fi
status_out=$(OCANNL_TOOL_TEST_RUNS=$repeat_stop_runs \
  "$repeat_root/tools/test-run.sh" status last 2>"$TMP/repeat-stop-status.err")
status_rc=$?
stop_out=$(OCANNL_TOOL_TEST_RUNS=$repeat_stop_runs \
  "$repeat_root/tools/test-run.sh" stop last 2>"$TMP/repeat-stop-stop.err")
stop_rc=$?
touch "$repeat_stop_prefix.release"
wait "$repeat_pid"
repeat_stop_rc=$?
repeat_pid=
if [ "$status_rc" = 3 ] && grep -q '^running: ' <<<"$status_out" \
   && [ "$stop_rc" = 0 ] && grep -q '^sent TERM to the repeat coordinator; ' <<<"$stop_out" \
   && [ "$repeat_stop_rc" = 143 ] \
   && [ "$(cat "$TMP/repeat-stop.counter")" = 1 ] \
   && grep -q '^repeat result: CANCELLED -- completed 1 of 3 iterations$' "$TMP/repeat-stop.out"; then
  report 0 "repeat: last resolves active state and stop cancels the whole set"
else
  report 1 "repeat: last resolves active state and stop cancels the whole set" \
    "status $status_rc: ${status_out:-<nothing>}; stop $stop_rc: ${stop_out:-<nothing>}; repeat $repeat_stop_rc: $(cat "$TMP/repeat-stop.out")"
fi

# Keep the signal traps armed after the iteration loop: this fixture gives the
# repeat its own process group, blocks inside the first pairwise diff, then
# sends TERM to the WHOLE group. The diff dies from that same signal; the exit
# finalizer must still turn the coordinator's trapped cancellation into an
# atomic verdict.
repeat_finalize_runs=$TMP/repeat-runs-finalize
repeat_finalize_prefix=$TMP/repeat-finalize
mkdir -p "$repeat_finalize_runs"
: >"$TMP/repeat-finalize.counter"
: >"$TMP/repeat-finalize.calls"
REPEAT_TEST_MODE=stdout \
REPEAT_TEST_COUNTER=$TMP/repeat-finalize.counter \
REPEAT_TEST_CALLS=$TMP/repeat-finalize.calls \
REPEAT_TEST_WAIT_PREFIX= \
REPEAT_TEST_DIFF_WAIT_PREFIX=$repeat_finalize_prefix \
REPEAT_TEST_REAL_DIFF="$(command -v diff)" \
OCANNL_TOOL_TEST_RUNS=$repeat_finalize_runs \
PATH=$repeat_bin:$PATH \
  perl -MPOSIX -e 'POSIX::setpgid(0, 0); exec @ARGV' \
  "$repeat_root/tools/test-run.sh" repeat 2 build @cheap \
  >"$TMP/repeat-finalize.out" 2>"$TMP/repeat-finalize.err" &
repeat_pid=$!
if ! await_fixture_ready "$repeat_finalize_prefix.ready"; then
  report 1 "repeat: cancellation during finalization still publishes a verdict" \
    "setup timeout waiting for the comparison readiness marker"
  repeat_abandon "$repeat_finalize_prefix"
  harness_abandon "a repeat fixture never became ready"
  return
fi
kill -TERM -- "-$repeat_pid" 2>"$TMP/repeat-finalize-kill.err"
finalize_kill_rc=$?
wait "$repeat_pid"
repeat_finalize_rc=$?
repeat_pid=
repeat_finalize_dir=$(OCANNL_TOOL_TEST_RUNS="$repeat_finalize_runs" "$repeat_root/tools/test-run.sh" paths run last 2>/dev/null)
if [ "$finalize_kill_rc" = 0 ] \
   && [ "$repeat_finalize_rc" = 143 ] \
   && [ -n "$repeat_finalize_dir" ] \
   && [ "$(cat "$repeat_finalize_dir/exit" 2>/dev/null)" = 143 ] \
   && grep -q '^repeat result: CANCELLED -- completed 2 of 2 iterations$' "$TMP/repeat-finalize.out"; then
  report 0 "repeat: cancellation during finalization still publishes a verdict"
else
  report 1 "repeat: cancellation during finalization still publishes a verdict" \
    "group kill $finalize_kill_rc: $(cat "$TMP/repeat-finalize-kill.err"); repeat $repeat_finalize_rc: $(cat "$TMP/repeat-finalize.out")"
fi

# The stored cap belongs to ONE iteration. Drive `wait` against a fabricated
# live coordinator and a no-op sleep until its default deadline expires: cap 2
# across 3 iterations must report 126 seconds, not the old single-cap 122.
wait_runs=$TMP/repeat-runs-wait-budget
wait_dir=$wait_runs/20000101T000000Z-1
mkdir -p "$wait_dir" "$TMP/wait-bin"
wait_dir_real=$(cd "$wait_dir" && pwd -P)
printf 'build @cheap\n' >"$wait_dir/cmd"
printf '2\n' >"$wait_dir/cap"
printf 'repeat\n' >"$wait_dir/mode"
printf '3\n' >"$wait_dir/repeats"
printf '%s\n' "$$" >"$wait_dir/pid"
ps_token "$$" >"$wait_dir/ptoken"
: >"$wait_dir/log"
cat >"$TMP/wait-bin/sleep" <<'EOF'
#!/bin/sh
printf '%s\n' "$1" >>"$REPEAT_TEST_SLEEP_CALLS"
EOF
chmod +x "$TMP/wait-bin/sleep"
: >"$TMP/repeat-wait-sleeps"
wait_out=$(REPEAT_TEST_SLEEP_CALLS=$TMP/repeat-wait-sleeps \
  OCANNL_TOOL_TEST_RUNS=$wait_runs PATH=$TMP/wait-bin:$PATH \
  "$repeat_root/tools/test-run.sh" wait "$wait_dir" 2>"$TMP/repeat-wait.err")
wait_rc=$?
if [ "$wait_rc" = 124 ] \
   && grep -q "^wait timed out after 126s: $wait_dir_real$" <<<"$wait_out" \
   && [ "$(wc -l <"$TMP/repeat-wait-sleeps" | tr -d ' ')" = 26 ]; then
  report 0 "repeat: wait default covers every per-iteration cap"
else
  report 1 "repeat: wait default covers every per-iteration cap" \
    "exit $wait_rc; sleeps $(tr '\n' ',' <"$TMP/repeat-wait-sleeps"); output: ${wait_out:-<nothing>}"
fi
}
harness_leg_run '9-25: repeat mode #repeat' leg_9_25

shared_through native
# ---------------------------------------------------------------------------
# Legs 35-38: an invocation dune's own parser refused is a verdict of its own
# ---------------------------------------------------------------------------
# The guard above knows two words. Everything else dune's CLI refuses -- an
# unknown option, an unknown subcommand, a malformed operand -- prints
# `dune: <complaint>` then `Usage: dune ...`, exits 1 and runs nothing, and the
# digest used to read that 1 as `FAIL (exit 1)` with no error lines: the verdict
# of a red suite, at the moment the reader decides between reading failures and
# fixing the command line (gh-ocannl-944). Each shape is driven through the
# real `run` path against the fixture dune emitting dune's exact stderr, and
# the assertions are the halves of the contract: the distinct verdict quoting
# dune's own complaint (the wrapped operand complaint WHOLE, continuation line
# included), `run`'s status 2 -- the usage code, so a caller branches without
# reading the log -- over a RECORDED status that stays dune's 1, the same two
# readings from `wait` (2) and `status` (0: it reports publication), and dune
# invoked exactly once -- what tells this refusal from the guard's, which
# shares its exit code.
refused_verdict="verdict: INVOCATION REFUSED (dune rejected the arguments; nothing ran) (exit 2)"
leg_35() {
refused_label="a dune-refused invocation is INVOCATION REFUSED: run/wait exit 2, recorded exit 1"
refused_detail=
for probe in "usage_option|build @cheap --frobnicate" \
             "usage_command|frobnicate" \
             "usage_operand|build -j x"; do
  mode=${probe%%|*}
  case $mode in
    usage_option) want1="dune: unknown option '--frobnicate'." want2= ;;
    usage_command) want1="dune: unknown command 'frobnicate', must be one of" want2= ;;
    usage_operand) want1="dune: option '-j': invalid concurrency value" want2="        number" ;;
  esac
  argv_mode=$mode argv_probe "refused-$mode" run ${probe#*|}
  refused_runs=$TMP/argv-runs-refused-$mode
  case $argv_rc in
    2) ;;
    *) refused_detail="$mode: run exited $argv_rc (want 2); stdout: $argv_out"; break ;;
  esac
  case $argv_out in
    *"$refused_verdict"*"dune said:"*"  $want1"*"$want2"*) ;;
    *) refused_detail="$mode: verdict or quoted complaint missing: $argv_out"; break ;;
  esac
  case $argv_out in
    *"no Error/File lines matched"* | *"fingerprint:"*)
      refused_detail="$mode: the red-run digest still printed: $argv_out"; break ;;
  esac
  if [ "$argv_calls" != "${probe#*|}" ]; then
    refused_detail="$mode: dune calls (want exactly the argv): ${argv_calls:-<none>}"; break
  fi
  if [ -z "$argv_dir" ] || [ "$(cat "$argv_dir/exit" 2>/dev/null)" != 1 ] \
     || [ "$(tail -n 1 "$argv_dir/log" 2>/dev/null)" != "exit: 1" ]; then
    refused_detail="$mode: recorded status is not dune's 1: $(cat "$argv_dir/exit" 2>/dev/null; tail -n 1 "$argv_dir/log" 2>/dev/null)"
    break
  fi
  argv_runs=$refused_runs argv_probe "refused-$mode-status" status last
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "" ]; then
    refused_detail="$mode: status last exited $argv_rc (want 0: publication, not the verdict); calls: $argv_calls"; break
  fi
  case $argv_out in
    *"$refused_verdict"*) ;;
    *) refused_detail="$mode: status last lost the verdict: $argv_out"; break ;;
  esac
  argv_runs=$refused_runs argv_probe "refused-$mode-wait" wait last
  if [ "$argv_rc" != 2 ] || [ "$argv_calls" != "" ]; then
    refused_detail="$mode: wait last exited $argv_rc (want 2); calls: $argv_calls"; break
  fi
  case $argv_out in
    *"$refused_verdict"*"  $want1"*) ;;
    *) refused_detail="$mode: wait last lost the verdict: $argv_out"; break ;;
  esac
done
argv_mode= argv_runs=
# The launch's own prelude -- the batch's backends, the width announcement and
# the fleet slot's admission line, all in the log before dune starts -- does
# not hide the refusal (Codex review round 1 on PR #832).
if [ -z "$refused_detail" ]; then
  argv_mode=usage_option
  dxg_probe refused-prelude "$dxg_present" cuda run build @cheap --frobnicate
  argv_mode=
  { [ "$argv_rc" = 2 ] && grep -qF "$refused_verdict" <<<"$argv_out" &&
    grep -q '^test-run: batch: ' "$argv_dir/log" && grep -q '^test-run: capping dune at -j 2' "$argv_dir/log"; } ||
    refused_detail="under the launch prelude: exit $argv_rc (want 2): $argv_out"
fi
# ...but only the launcher's prelude, found by the offset it recorded: a
# program's own `test-run: ` line before a nested refusal is output, and the
# run is an ordinary red one.
if [ -z "$refused_detail" ]; then
  argv_mode=prefixed_refusal
  dxg_probe refused-prefixed "$dxg_present" cuda run build @cheap
  argv_mode=
  { [ "$argv_rc" = 1 ] && grep -q "verdict: FAIL" <<<"$argv_out"; } ||
    refused_detail="a program's own test-run: line: exit $argv_rc (want 1, FAIL): $argv_out"
fi
if [ -z "$refused_detail" ]; then
  report 0 "$refused_label"
else
  report 1 "$refused_label" "$refused_detail"
fi
}
harness_leg_run '35: a dune-refused invocation is INVOCATION REFUSED #digest' leg_35

# The controls: a run that FAILED still says so, with dune's status and the
# fingerprint -- a recogniser that fired on `dune:` alone would turn a test
# whose output opens on that word into "nothing ran", and one that stopped
# reading at `Usage:` would do the same to a `dune exec` program that printed
# a nested dune refusal and then failed (Codex review round 1, P2): both are
# the inverse misreading, so the shape must also be ALONE in the log.
leg_36() {
red_label="a red run still digests as FAIL (exit 1) with its fingerprint"
red_detail=
for mode in red red_color dune_prefixed_red nested_usage_red nested_usage_output; do
  argv_mode=$mode argv_probe "red-$mode" run build @cheap
  case $argv_rc in
    1) ;;
    *) red_detail="$mode: run exited $argv_rc (want 1); stdout: $argv_out"; break ;;
  esac
  case $argv_out in
    *"verdict: FAIL (exit 1)"*) ;;
    *) red_detail="$mode: FAIL verdict missing: $argv_out"; break ;;
  esac
  case $argv_out in
    *"INVOCATION REFUSED"*) red_detail="$mode: a red run read as refused: $argv_out"; break ;;
  esac
  if [ "$mode" = nested_usage_output ]; then
    # No located error to fingerprint: the digest falls through to the log
    # tail, which is the trailing evidence itself.
    case $argv_out in
      *"no Error/File lines matched"*"the program went on to print this"*) ;;
      *) red_detail="$mode: log tail missing: $argv_out"; break ;;
    esac
    argv_runs=$TMP/argv-runs-red-$mode argv_probe "red-$mode-wait" wait last
    if [ "$argv_rc" != 1 ]; then
      red_detail="$mode: wait last exited $argv_rc (want 1)"; break
    fi
    continue
  fi
  # The fingerprint is sorted (`sort -u`), so its two lines are pinned
  # separately rather than in an order the digest never promised.
  case $argv_out in
    *"fingerprint:"*'File "test/operations/fixture.ml", line 3'*) ;;
    *) red_detail="$mode: File fingerprint missing: $argv_out"; break ;;
  esac
  case $argv_out in
    *"fingerprint:"*"Error: This expression has type int"*) ;;
    *) red_detail="$mode: Error fingerprint missing: $argv_out"; break ;;
  esac
  argv_runs=$TMP/argv-runs-red-$mode argv_probe "red-$mode-wait" wait last
  if [ "$argv_rc" != 1 ]; then
    red_detail="$mode: wait last exited $argv_rc (want 1)"; break
  fi
done
argv_mode= argv_runs=
if [ -z "$red_detail" ]; then
  report 0 "$red_label"
else
  report 1 "$red_label" "$red_detail"
fi
}
harness_leg_run '36: a red run still digests as FAIL #digest' leg_36

# Leg 87 (gh-ocannl-1281): a red run whose log carries, at a line's start, the
# line a GPU backend's device-init deadline writes before its process dies by
# SIGALRM is WEDGED DEVICE, keeping dune's 1 and quoting the line -- also when
# 11MB of output follows it, past the digest's 10MB tail; the same
# line indented (a test quoting a child's stderr) leaves the run an ordinary
# FAIL.
if harness_leg '87: a device-init deadline death is WEDGED DEVICE #digest'; then
wedged_label="a red run carrying the device-init deadline's line is WEDGED DEVICE (exit 1), an indented one is FAIL"
wedged_detail=
argv_mode=wedged_init argv_probe wedged-init run build @cheap
case $argv_rc:$argv_out in
  1:*"verdict: WEDGED DEVICE ("*"(exit 1)"*"wedged device (the device_init_deadline's own line):"*"  OCANNL device_init_deadline: hip device initialization"*"journalctl -k"*) ;;
  *) wedged_detail="wedged_init: exit $argv_rc (want 1) or verdict/quote missing: $argv_out" ;;
esac
if [ -z "$wedged_detail" ]; then
  argv_mode=wedged_long argv_probe wedged-long run build @cheap
  case $argv_rc:$argv_out in
    1:*"verdict: WEDGED DEVICE ("*"  OCANNL device_init_deadline: hip device initialization"*) ;;
    *) wedged_detail="wedged_long: the line before 11MB of output was missed: exit $argv_rc: $(printf '%s' "$argv_out" | head -c 2000)" ;;
  esac
fi
if [ -z "$wedged_detail" ]; then
  argv_mode=wedged_quoted argv_probe wedged-quoted run build @cheap
  case $argv_rc:$argv_out in
    *WEDGED* | *"wedged device"*) wedged_detail="wedged_quoted: an indented quote read as a death: $argv_out" ;;
    1:*"verdict: FAIL (exit 1)"*) ;;
    *) wedged_detail="wedged_quoted: exit $argv_rc (want 1) or FAIL missing: $argv_out" ;;
  esac
fi
argv_mode=
if [ -z "$wedged_detail" ]; then
  report 0 "$wedged_label"
else
  report 1 "$wedged_label" "$wedged_detail"
fi
fi # leg 87

# Legs 68-69 (gh-ocannl-1055): promotion is offered only on a diff dune printed.
# A rule whose action failed names its `.corrected` file in the quoted stanza,
# and the digest used to send that reader to `dune promote`; it must instead say
# the action failed with no diff, and still show the fingerprint. The other
# half: a real hunk, plain or colored, is still promotion -- and a green run is
# neither.
leg_68() {
nodiff_label="a failed action with no diff hunk is 'action failed (no diff)', never promotion"
argv_mode=action_no_diff argv_probe nodiff-action run build @cheap
nodiff_detail=
[ "$argv_rc" = 1 ] || nodiff_detail="run exited $argv_rc (want 1)"
[ -n "$nodiff_detail" ] || case $argv_out in
  *"verdict: FAIL (exit 1)"*"action failed (no diff)"*) ;;
  *) nodiff_detail="no 'action failed (no diff)' under a FAIL verdict" ;;
esac
[ -n "$nodiff_detail" ] || case $argv_out in
  *"promotion diffs present"*) nodiff_detail="promotion offered without a diff" ;;
esac
[ -n "$nodiff_detail" ] || case $argv_out in
  *"fingerprint:"*'File "test/einsum/dune", lines 197-217'*) ;;
  *) nodiff_detail="the stanza's location is not fingerprinted" ;;
esac
if [ -z "$nodiff_detail" ]; then
  report 0 "$nodiff_label"
else
  report 1 "$nodiff_label" "$nodiff_detail; stdout: $argv_out"
fi
}
harness_leg_run '68: a failed action with no diff is never promotion #digest' leg_68
leg_69() {
hunk_label="a printed diff hunk, plain or colored, is still promotion; a green run is neither"
hunk_detail=
for mode in diff_hunk diff_hunk_color stable; do
  argv_mode=$mode argv_probe "hunk-$mode" run build @cheap
  case $mode:$argv_rc:$argv_out in
    stable:0:*"promotion diffs present"* | stable:0:*"no diff)"*)
      hunk_detail="$mode: a green run was told about promotion" ;;
    stable:0:*) ;;
    diff_hunk*:1:*"action failed (no diff)"*)
      hunk_detail="$mode: a diff read as no diff" ;;
    diff_hunk*:1:*"promotion diffs present"*) ;;
    *) hunk_detail="$mode: exit $argv_rc; no promotion line" ;;
  esac
  [ -z "$hunk_detail" ] || { hunk_detail="$hunk_detail; stdout: $argv_out"; break; }
done
argv_mode=
if [ -z "$hunk_detail" ]; then
  report 0 "$hunk_label"
else
  report 1 "$hunk_label" "$hunk_detail"
fi
# A run that chose its own diff presentation: patdiff's doubled headers are
# still a hunk; `--diff-command -` (or DUNE_DIFF_COMMAND=- at launch) prints
# none for a real promotion, and the digest says promotion is POSSIBLE rather
# than "no diff" -- which the same log without the choice does say.
custom_label="a run that chose its own diff command is never told 'no diff' over a pending promotion"
custom_detail=
for probe in "patdiff_hunk|present|build --diff-command patdiff @cheap" \
             "diff_suppressed|possible|build --diff-command - @cheap" \
             "diff_suppressed|possible|build --diff-command=- @cheap" \
             "diff_suppressed|possible|build --diff - @cheap" \
             "diff_suppressed|env|build @cheap" \
             "diff_suppressed|nodiff|build @cheap" \
             "inline_suppressed|possible|build --diff-command - @cheap" \
             "action_no_diff|possible|build --diff-command - @cheap" \
             "diff_suppressed|nodiff|exec ./prog.exe -- --diff-command -"; do
  mode=${probe%%|*}; rest=${probe#*|}; want=${rest%%|*}
  if [ "$want" = env ]; then export DUNE_DIFF_COMMAND=-; want=possible; fi
  # shellcheck disable=SC2086
  argv_mode=$mode argv_probe "custom-$mode-$want" run ${rest#*|}
  unset DUNE_DIFF_COMMAND
  case $want:$argv_out in
    present:*"promotion diffs present"*) ;;
    possible:*"promotion diffs possible"*) ;;
    nodiff:*"action failed (no diff)"*) ;;
    *) custom_detail="$mode (${rest#*|}): want $want: $argv_out"; break ;;
  esac
  case $want:$argv_out in
    possible:*"no diff)"* | present:*"no diff)"*)
      custom_detail="$mode (${rest#*|}): also said no diff: $argv_out"; break ;;
  esac
done
argv_mode=
if [ -z "$custom_detail" ]; then
  report 0 "$custom_label"
else
  report 1 "$custom_label" "$custom_detail"
fi
# A log longer than the digest's scan window may hold a hunk above it: the
# digest must say it did not look there, never "nothing to promote".
argv_mode=diff_then_noise argv_probe noise-hunk run build @cheap
argv_mode=
noise_label="a log past the scanned tail is never told 'nothing to promote'"
if [ "$argv_rc" = 1 ] \
   && case $argv_out in *"the log is longer"*) true ;; *) false ;; esac \
   && case $argv_out in *"(no diff)"* | *"nothing to promote;"*) false ;; *) true ;; esac; then
  report 0 "$noise_label"
else
  report 1 "$noise_label" "exit $argv_rc: $(printf '%s' "$argv_out" | head -c 2000)"
fi
# A repeat set's red can be drift between iterations that each passed: its
# digest must not blame a failed action.
repeat_probe repeat-drift-nodiff stdout 2 build @cheap
argv_runs=$TMP/repeat-runs-repeat-drift-nodiff argv_probe repeat-drift-nodiff-wait wait last
argv_runs=
drift_label="a repeat set's drift is not called a failed action"
if [ "$repeat_rc" = 1 ] && [ "$argv_rc" = 1 ] \
   && case $argv_out in *"verdict: FAIL"*) true ;; *) false ;; esac \
   && case $repeat_out$argv_out in *"no diff)"*) false ;; *) true ;; esac; then
  report 0 "$drift_label"
else
  report 1 "$drift_label" "repeat exit $repeat_rc: $repeat_out; wait exit $argv_rc: $argv_out"
fi
}
harness_leg_run '69: a printed or possible diff is promotion; a long log or drift is not no-diff #digest' leg_69

# Legs 71-73 (gh-ocannl-1087): the run's last phase records dune's own
# promotion list (`dune promotion list`) as `promotions`, and the digest reads
# that record in place of the log -- the log inference of legs 68-69 is only
# the fallback for runs that recorded none. The fixture dune answers the list
# from a file (REPEAT_TEST_PROMOTIONS), on the stream its faked version uses.
promo_lists=$TMP/promotion-lists
mkdir -p "$promo_lists" "$repeat_root/test/operations"
printf 'test/operations/fixture.expected\n' >"$promo_lists/one"
: >"$promo_lists/empty"
printf 'test/operations/fixture.expected\r\ntest/operations/other.expected\r\n' >"$promo_lists/crlf"
printf 'test/operations/fixture.expected\ntest/operations/other.expected\n' >"$promo_lists/crlf-want"
for ((i = 1; i <= 23; i++)); do printf 'test/operations/f%02d.expected\n' "$i"; done >"$promo_lists/many"
printf 'no/such/dir/fixture.expected\n' >"$promo_lists/absent-dir"
printf 'test/operations/fixture.expected\n/abs/elsewhere.expected\n' >"$promo_lists/absolute"
promo_present="promotion diffs present -- dune's promotion list at the run's end names"
promo_empty="dune's promotion list at the run's end is empty, nothing to promote"
promo_calls=$TMP/promotion-calls
promo_any() { # <run dir>; 0 iff the record, or any scratch file of its, is there
  local f
  for f in "$1"/promotions*; do [ -e "$f" ] && return 0; done
  return 1
}
promo_probe() { # tag list-name mode subcommand [argv...]; the list file may be "unset"
  local tag=$1 list=$2 mode=$3
  shift 3
  : >"$promo_calls"
  if [ "$list" = unset ]; then
    REPEAT_TEST_PROMOTION_CALLS=$promo_calls argv_mode=$mode argv_probe "$tag" "$@"
  else
    REPEAT_TEST_PROMOTIONS=$promo_lists/$list REPEAT_TEST_PROMOTION_CALLS=$promo_calls \
      argv_mode=$mode argv_probe "$tag" "$@"
  fi
  argv_mode=
}

# Leg 71: the record decides, whatever the log shows. A log with no hunk (a
# suppressed diff, on the command line too) over a nonempty list is promotion,
# listed file by file, never "possible" or "no diff"; a hunk over an empty list
# is not promotion; a green run with an empty list says nothing. `status` and
# `wait` read the same record. dune before 3.22 prints the list on stderr, and
# a line on the other stream is not part of it either way; CRs are dropped; a
# long list is cut at 20 in the digest, never in the record. A dune killed by
# a signal leaves the status an exec'd one would have, and that verdict does
# not read the record.
leg_71() {
rec_label="dune's recorded promotion list decides the digest, whatever the log shows"
rec_detail=
promo_probe rec-suppressed one diff_suppressed run build @cheap
case $argv_rc:$argv_out in
  1:*"$promo_present 1 file(s)"*"  test/operations/fixture.expected"*) ;;
  *) rec_detail="suppressed diff over a listed file: exit $argv_rc: $argv_out" ;;
esac
[ -n "$rec_detail" ] || case $argv_out in
  *"no diff)"* | *"promotions possible"* | *"diffs possible"* | *"$promo_empty"*)
    rec_detail="suppressed diff: the log was read as well: $argv_out" ;;
esac
[ -n "$rec_detail" ] || cmp -s "$argv_dir/promotions" "$promo_lists/one" ||
  rec_detail="suppressed diff: the record is not dune's list: $(cat "$argv_dir/promotions" 2>&1)"
[ -n "$rec_detail" ] || [ "$(cat "$promo_calls")" = "promotion list --diff-command=diff --trace-file=$argv_dir/promotions.trace" ] ||
  rec_detail="the list was not asked for once, with its own trace file: $(cat "$promo_calls")"
[ -n "$rec_detail" ] || { [ ! -e "$argv_dir/promotions.trace" ] && [ ! -e "$argv_dir/promotions.tmp" ] &&
                          [ ! -e "$argv_dir/promotions.list" ]; } ||
  rec_detail="the recorder left its scratch files: $(ls "$argv_dir" | tr '\n' ' ')"
if [ -z "$rec_detail" ]; then
  argv_runs=$TMP/argv-runs-rec-suppressed argv_probe rec-suppressed-status status last
  case $argv_rc:$argv_out in
    0:*"$promo_present 1 file(s)"*) ;;
    *) rec_detail="status last: exit $argv_rc: $argv_out" ;;
  esac
  argv_runs=$TMP/argv-runs-rec-suppressed argv_probe rec-suppressed-wait wait last
  case $argv_rc:$argv_out in
    1:*"$promo_present 1 file(s)"*) ;;
    *) rec_detail="${rec_detail:+$rec_detail; }wait last: exit $argv_rc: $argv_out" ;;
  esac
  argv_runs=
fi
if [ -z "$rec_detail" ]; then
  promo_probe rec-diff-command one diff_suppressed run build --diff-command - @cheap
  case $argv_out in
    *"$promo_present 1 file(s)"*) ;;
    *) rec_detail="a chosen diff command over a listed file: $argv_out" ;;
  esac
fi
if [ -z "$rec_detail" ]; then
  promo_probe rec-hunk-empty empty diff_hunk run build @cheap
  case $argv_rc:$argv_out in
    *"promotion diffs present"*) rec_detail="a hunk over an empty list offered promotion: $argv_out" ;;
    1:*"action failed -- $promo_empty"*) ;;
    *) rec_detail="a hunk over an empty list: exit $argv_rc: $argv_out" ;;
  esac
  [ -n "$rec_detail" ] || { [ -f "$argv_dir/promotions" ] && [ ! -s "$argv_dir/promotions" ]; } ||
    rec_detail="an empty list is not recorded as an empty file"
fi
if [ -z "$rec_detail" ]; then
  promo_probe rec-green empty stable run build @cheap
  case $argv_rc:$argv_out in
    0:*promot*) rec_detail="a green run with an empty list mentioned promotion: $argv_out" ;;
    0:*) ;;
    *) rec_detail="a green run: exit $argv_rc: $argv_out" ;;
  esac
fi
if [ -z "$rec_detail" ]; then
  REPEAT_TEST_DUNE_VERSION=3.20.2 REPEAT_TEST_PROMOTION_NOISE='chatter on stdout' \
    promo_probe rec-old-stream one diff_suppressed run build @cheap
  cmp -s "$argv_dir/promotions" "$promo_lists/one" ||
    rec_detail="dune 3.20 (stderr): recorded $(cat "$argv_dir/promotions" 2>&1)"
  REPEAT_TEST_PROMOTION_NOISE='Warning: chatter on stderr' \
    promo_probe rec-new-stream one diff_suppressed run build @cheap
  cmp -s "$argv_dir/promotions" "$promo_lists/one" ||
    rec_detail="${rec_detail:+$rec_detail; }dune 3.24 (stdout): recorded $(cat "$argv_dir/promotions" 2>&1)"
  promo_probe rec-crlf crlf diff_suppressed run build @cheap
  cmp -s "$argv_dir/promotions" "$promo_lists/crlf-want" ||
    rec_detail="${rec_detail:+$rec_detail; }CRLF list: recorded $(od -c "$argv_dir/promotions" 2>&1 | head -3)"
fi
if [ -z "$rec_detail" ]; then
  promo_probe rec-many many diff_suppressed run build @cheap
  case $argv_out in
    *"$promo_present 23 file(s)"*"  test/operations/f20.expected"*"  ... and 3 more, in $argv_dir/promotions"*) ;;
    *) rec_detail="a long list: $argv_out" ;;
  esac
  case $argv_out in
    *f21.expected*) rec_detail="a long list is not cut at 20: $argv_out" ;;
  esac
  [ -n "$rec_detail" ] || cmp -s "$argv_dir/promotions" "$promo_lists/many" ||
    rec_detail="a long list is cut in the record itself"
fi
if [ -z "$rec_detail" ]; then
  promo_probe rec-self-kill one self_kill run build @cheap
  case $argv_rc:$argv_out in
    *"promotion diffs present"*) rec_detail="a KILLED run read the record: $argv_out" ;;
    137:*"verdict: KILLED"*) ;;
    *) rec_detail="a dune killed by SIGKILL: exit $argv_rc (want 137): $argv_out" ;;
  esac
  [ -n "$rec_detail" ] || [ "$(cat "$argv_dir/exit" 2>/dev/null)" = 137 ] ||
    rec_detail="a dune killed by SIGKILL: recorded $(cat "$argv_dir/exit" 2>&1)"
fi
if [ -z "$rec_detail" ]; then
  report 0 "$rec_label"
else
  report 1 "$rec_label" "$rec_detail"
fi
}
harness_leg_run '71: dune'\''s recorded promotion list decides the digest #digest' leg_71

# Leg 72: no record where the list cannot be taken faithfully, and the log
# decides as before: the list failing, a line that is not a path here (a
# directory this worktree lacks, an absolute path), an unreadable version, an
# argv naming another build directory or root (dune not even asked; past
# dune's own `--` the words are a program's, and the list is taken), and a cap
# too nearly spent to ask. A missing build directory is an empty record,
# asked of nobody -- under DUNE_BUILD_DIR's name too.
leg_72() {
norec_label="no record where dune's list cannot be taken faithfully; the log decides as before"
norec_detail=
for probe in "unset|diff_hunk|present|" \
             "absent-dir|diff_suppressed|nodiff|" \
             "absolute|diff_suppressed|nodiff|" \
             "one|diff_suppressed|nodiff|fail"; do
  IFS='|' read -r list mode want version <<<"$probe"
  REPEAT_TEST_DUNE_VERSION=$version promo_probe "norec-$list-$mode" "$list" "$mode" run build @cheap
  case $want:$argv_out in
    present:*"promotion diffs present -- inspect the log"*) ;;
    nodiff:*"action failed (no diff)"*) ;;
    *) norec_detail="$list/$mode/${version:-default}: want the log's $want: $argv_out"; break ;;
  esac
  if promo_any "$argv_dir"; then
    norec_detail="$list/$mode/${version:-default}: recorded anyway: $(ls "$argv_dir" | tr '\n' ' ')"; break
  fi
done
if [ -z "$norec_detail" ]; then
  for argv in "build --build-dir=elsewhere @cheap" "build --bu elsewhere @cheap" "build --root . @cheap"; do
    # shellcheck disable=SC2086
    promo_probe norec-build-root one diff_suppressed run $argv
    if [ -e "$argv_dir/promotions" ] || [ -s "$promo_calls" ]; then
      norec_detail="$argv: recorded or asked: $(cat "$promo_calls")"; break
    fi
    case $argv_out in
      *"action failed (no diff)"*) ;;
      *) norec_detail="$argv: the log did not decide: $argv_out"; break ;;
    esac
  done
fi
if [ -z "$norec_detail" ]; then
  promo_probe norec-past-separator one stable run exec ./prog.exe -- --build-dir=elsewhere
  cmp -s "$argv_dir/promotions" "$promo_lists/one" ||
    norec_detail="--build-dir past dune's own --: not recorded ($(cat "$promo_calls"))"
fi
if [ -z "$norec_detail" ]; then
  promo_probe norec-cap-spent one stable run --cap 5 build @cheap
  if [ "$argv_rc" != 0 ] || [ -e "$argv_dir/promotions" ] || [ -s "$promo_calls" ]; then
    norec_detail="a nearly spent cap: exit $argv_rc; asked: $(cat "$promo_calls"); $(ls "$argv_dir" | tr '\n' ' ')"
  fi
fi
if [ -z "$norec_detail" ]; then
  mv "$repeat_root/_build" "$repeat_root/_build.away"
  promo_probe norec-no-build-dir one diff_suppressed run build @cheap
  mv "$repeat_root/_build.away" "$repeat_root/_build"
  { [ -f "$argv_dir/promotions" ] && [ ! -s "$argv_dir/promotions" ] && [ ! -s "$promo_calls" ] \
    && [ ! -e "$repeat_root/_build.away" ]; } ||
    norec_detail="no build directory: asked $(cat "$promo_calls"); record: $(cat "$argv_dir/promotions" 2>&1)"
  case $argv_out in
    *"action failed -- $promo_empty"*) ;;
    *) norec_detail="${norec_detail:+$norec_detail; }no build directory: $argv_out" ;;
  esac
fi
if [ -z "$norec_detail" ]; then
  DUNE_BUILD_DIR=$TMP/no-such-build-dir promo_probe norec-env-absent one stable run build @cheap
  { [ -f "$argv_dir/promotions" ] && [ ! -s "$argv_dir/promotions" ] && [ ! -s "$promo_calls" ]; } ||
    norec_detail="DUNE_BUILD_DIR naming no directory: asked $(cat "$promo_calls")"
  mkdir -p "$TMP/other-build-dir"
  mv "$repeat_root/_build" "$repeat_root/_build.away"
  DUNE_BUILD_DIR=$TMP/other-build-dir promo_probe norec-env-present one stable run build @cheap
  mv "$repeat_root/_build.away" "$repeat_root/_build"
  cmp -s "$argv_dir/promotions" "$promo_lists/one" ||
    norec_detail="${norec_detail:+$norec_detail; }DUNE_BUILD_DIR naming a directory, _build absent: not asked"
fi
if [ -z "$norec_detail" ]; then
  report 0 "$norec_label"
else
  report 1 "$norec_label" "$norec_detail"
fi
}
harness_leg_run '72: no record where the list cannot be taken faithfully #digest' leg_72

# Leg 73: a verdict dune did not give never reads the record -- the list a run
# that ran nothing finds is an older build's. Forged from a recorded run with
# a stale list: a refused slot (exit 1 under a `slot` record and the slot's
# refusal line) and a cancellation (143).
leg_73() {
stale_label="a refused slot or a cancelled run never reads a recorded list"
stale_detail=
promo_probe stale-source one diff_suppressed run build @cheap
stale_src=$argv_dir
stale_runs=$(dirname "$stale_src")
for probe in "slot|1|SLOT REFUSED|1" "cancel|143|CANCELLED|2"; do
  IFS='|' read -r kind code want n <<<"$probe"
  forged=$stale_runs/20000101T00000${n}Z-99$n
  rm -rf "$forged"
  cp -R "$stale_src" "$forged"
  printf '%s\n' "$code" >"$forged/exit"
  if [ "$kind" = slot ]; then
    printf '%s\n' "$TMP/no-fleet-worker.sh" >"$forged/slot"
    printf 'EXECUTION SLOT REFUSED fixture: no slot before the deadline\nexit: 1\n' >"$forged/log"
  else
    printf 'exit: 143\n' >"$forged/log"
  fi
  out=$(OCANNL_TOOL_TEST_RUNS=$stale_runs "$repeat_root/tools/test-run.sh" status "$forged" 2>&1)
  case $out in
    *promot*) stale_detail="$kind: the record was read: $out"; break ;;
    *"verdict: $want"*) ;;
    *) stale_detail="$kind: want $want: $out"; break ;;
  esac
done
if [ -z "$stale_detail" ]; then
  report 0 "$stale_label"
else
  report 1 "$stale_label" "$stale_detail"
fi
}
harness_leg_run '73: a refused slot or a cancelled run never reads a recorded list #digest' leg_73

# Leg 80: the run records its slowest actions from its own dune trace, and the
# digest names them. The trace is synthetic -- a real one records dune's
# environment (gh-ocannl-1280) -- with seven processes, one never finished, so
# the five kept are known exactly and the open one is a lower bound up to the
# trace's last timestamp. The reader is the shipping tools/action-durations.sh,
# staged beside the fixture's copy of the script.
leg_80() {
slow_label="the digest names the batch's five slowest actions from its own dune trace"
if ! command -v python3 >/dev/null 2>&1; then
  skip "$slow_label" "no python3 on PATH (tools/action-durations.sh needs it)"
else
  cp "$HERE/action-durations.sh" "$repeat_root/tools/action-durations.sh"
  slow_trace=$TMP/slowest-trace.csexp
  python3 - "$slow_trace" "$repeat_root/_build" <<'PY'
import sys
path, build = sys.argv[1], sys.argv[2]
S, T0 = 1_000_000_000, 1_791_291_000 * 1_000_000_000
def atom(s):
    s = str(s).encode()
    return str(len(s)).encode() + b':' + s
def sexp(x):
    return b'(' + b''.join(sexp(e) if isinstance(e, list) else atom(e) for e in x) + b')'
out = [sexp(['config', 'init', str(T0), ['build_dir', build]])]
for pid, start, dur, prog, target in [
        (11, 1, 21.5, 'fsm_transformer.exe', 'test/training/fsm_transformer.actual'),
        (12, 2, 6.72, 'env_var_deps.exe', 'test/operations/env_var_deps.actual'),
        (13, 3, 1.96, 'small.exe', 'test/operations/small.actual'),
        (14, 4, 3.25, 'ocamlopt.opt', 'arrayjit/lib/x.cmx'),
        (15, 5, 5.43, 'sh', 'test/operations/sh.actual'),
        (16, 10, None, 'hung_probe.exe', 'test/operations/hung_probe.exe.output'),
        (17, 6, 0.5, 'ppx.exe', 'tensor/y.pp.ml')]:
    t = T0 + start * S
    f = [['pid', str(pid)], ['prog', '/opt/bin/' + prog], ['dir', build + '/default'],
         ['target_files', [build + '/default/' + target]]]
    out.append(sexp(['process', 'start', str(t)] + f))
    if dur is not None:
        out.append(sexp(['process', 'finish', [str(t), str(int(round(dur * S)))]] + f + [['exit', '0']]))
open(path, 'wb').write(b''.join(out))
PY
  # The seconds and program of each row the digest prints, in its order.
  slow_rows() {
    printf '%s\n' "$1" | awk '/^slowest actions/ { f = 1; next }
      f && /^  / { if ($1 ~ /^[0-9]+\.[0-9][0-9]$/) print $1, $2, $3; next }
      f { exit }'
  }
  slow_want='21.50 0 fsm_transformer.exe
12.50 open hung_probe.exe
6.72 0 env_var_deps.exe
5.43 0 sh
3.25 0 ocamlopt.opt'
  slow_detail=
  slow_args=$TMP/slowest-trace-args
  slow_physical() { (cd "$(dirname "$1")" 2>/dev/null && printf '%s/%s\n' "$(pwd -P)" "$(basename "$1")"); }
  : >"$slow_args"
  REPEAT_TEST_TRACE=$slow_trace REPEAT_TEST_TRACE_ARGS=$slow_args argv_probe slow-pass run build @cheap
  case $argv_rc:$argv_out in
    0:*"slowest actions (this run's dune trace, by tools/action-durations.sh):"*"  processes=7  open=1"*) ;;
    *) slow_detail="a green run: exit $argv_rc, no slowest-actions table: $argv_out" ;;
  esac
  [ -n "$slow_detail" ] || [ "$(slow_rows "$argv_out")" = "$slow_want" ] ||
    slow_detail="a green run: rows $(slow_rows "$argv_out" | tr '\n' ';') (want the five slowest, longest first)"
  # The trace was the run's own file, handed to dune once, as the second word,
  # and is gone once read; no other trace option reached dune.
  [ -n "$slow_detail" ] || [ "$(cat "$slow_args")" = "2 before $(slow_physical "$argv_dir/trace.csexp")" ] ||
    slow_detail="a green run: dune was not handed exactly the run's own trace file, once: $(cat "$slow_args")"
  [ -n "$slow_detail" ] || [ "$argv_calls" = "build @cheap" ] ||
    slow_detail="a green run: dune's other words changed: $argv_calls"
  [ -n "$slow_detail" ] || { [ -s "$argv_dir/slowest" ] && [ ! -e "$argv_dir/slowest.tmp" ] &&
    [ ! -e "$argv_dir/trace.csexp" ] && [ ! -e "$argv_dir/slowest.err" ] &&
    [ ! -e "$argv_dir/slowest-skipped" ]; } ||
    slow_detail="a green run: the record is not published cleanly: $(ls "$argv_dir" | tr '\n' ' ')"
  # A run the cap kills records it too: the TIMEOUT digest shows how far the rest got.
  if [ -z "$slow_detail" ]; then
    REPEAT_TEST_TRACE=$slow_trace REPEAT_TEST_ACTION_SECS=30 argv_mode=slow_dune \
      argv_probe slow-timeout run --cap 2 build @cheap
    argv_mode=
    case $argv_rc:$argv_out in
      142:*"verdict: TIMEOUT"*"slowest actions"*) ;;
      *) slow_detail="a capped run: exit $argv_rc (want 142 with the table): $argv_out" ;;
    esac
    [ -n "$slow_detail" ] || [ "$(slow_rows "$argv_out")" = "$slow_want" ] ||
      slow_detail="a capped run: rows $(slow_rows "$argv_out" | tr '\n' ';')"
  fi
  # The worktree's shared trace -- which a concurrent manual dune may be
  # writing, under a build lock the launch does not hold -- is never read and
  # never touched: a run whose dune writes no trace records nothing, and the
  # shared one keeps every byte.
  if [ -z "$slow_detail" ]; then
    cp "$slow_trace" "$repeat_root/_build/trace.csexp"
    argv_probe slow-shared run build @cheap
    case $argv_out in
      *"slowest actions"*) slow_detail="the shared trace was read: $argv_out" ;;
    esac
    [ -n "$slow_detail" ] || { [ ! -e "$argv_dir/slowest" ] && [ ! -e "$argv_dir/slowest-skipped" ] &&
      cmp -s "$slow_trace" "$repeat_root/_build/trace.csexp"; } ||
      slow_detail="the shared trace: recorded from, or not left as it was"
    rm -f "$repeat_root/_build/trace.csexp"
  fi
  # A trace the reader cannot read publishes nothing, scratch file included.
  if [ -z "$slow_detail" ]; then
    printf 'not a trace\n' >"$TMP/slowest-garbage.csexp"
    REPEAT_TEST_TRACE=$TMP/slowest-garbage.csexp argv_probe slow-garbage run build @cheap
    case $argv_rc:$argv_out in
      0:*"slowest actions"*) slow_detail="an unreadable trace printed a table: $argv_out" ;;
      0:*"no slowest-actions record: action-durations.sh: "*"is not a canonical-S-expression dune trace"*) ;;
      0:*) slow_detail="an unreadable trace: the digest does not say why there is no record: $argv_out" ;;
      *) slow_detail="an unreadable trace: exit $argv_rc: $argv_out" ;;
    esac
    [ -n "$slow_detail" ] || { [ ! -e "$argv_dir/slowest" ] && [ ! -e "$argv_dir/slowest.tmp" ] &&
      [ ! -e "$argv_dir/trace.csexp" ]; } ||
      slow_detail="an unreadable trace left a record: $(ls "$argv_dir" | tr '\n' ' ')"
  fi
  # Where the argv names a trace file of its own -- either spelling -- or its
  # first word is not a subcommand known to take one (a global option ahead of
  # it included, which reaches dune only past the runner's own `--`: a leading
  # option is refused before dune, gh-ocannl-1797), dune is handed none: the caller's option is the only one,
  # verbatim, and nothing is recorded. Past dune's own `--` the word is a
  # program's argument: the runner's own still goes before the separator, once,
  # and the program's reaches it untouched.
  if [ -z "$slow_detail" ]; then
    for probe in "build --trace-file $TMP/t.csexp @cheap|2 before --trace-file" \
                 "build --trace-file=$TMP/t.csexp @cheap|2 before $TMP/t.csexp" \
                 "describe @cheap|" "-- --display=short build @cheap|" \
                 "exec ./prog.exe -- --trace-file=$TMP/p.csexp|OWN
5 after $TMP/p.csexp"; do
      want=${probe#*|} probe=${probe%%|*}
      : >"$slow_args"
      # shellcheck disable=SC2086
      REPEAT_TEST_TRACE_ARGS=$slow_args argv_probe slow-elsewhere run $probe
      want=${want/OWN/2 before $(slow_physical "$argv_dir/trace.csexp")}
      if [ "$(cat "$slow_args")" != "$want" ] || [ "$argv_calls" != "${probe#-- }" ] || [ -e "$argv_dir/slowest" ]; then
        slow_detail="$probe: trace options $(tr '\n' ';' <"$slow_args") (want ${want:-none}); calls: $argv_calls"
        break
      fi
    done
  fi
  # The observer above is not normalizing a defect away: a runner that hands
  # dune its option twice, or past the separator, shows in the calls the legs
  # pin and in the observed options. Each mutant replaces the shipping copy in
  # the fixture root for one run.
  slow_mutant() { # tag from to run-argv... -> runs, once, the script with FROM replaced by TO
    local tag=$1
    SLOW_FROM=$2 SLOW_TO=$3 perl -pe 's/\Q$ENV{SLOW_FROM}\E/$ENV{SLOW_TO}/' "$SRC" >"$repeat_root/tools/test-run.sh"
    shift 3
    if cmp -s "$SRC" "$repeat_root/tools/test-run.sh"; then
      slow_detail="mutant $tag: the substitution matched nothing in $SRC"
      cp "$SRC" "$repeat_root/tools/test-run.sh"
      return 1
    fi
    argv_probe "$tag" "$@"
    cp "$SRC" "$repeat_root/tools/test-run.sh"
  }
  slow_inject='set -- "$trace_sub" "--trace-file=$trace_file" "$@" ;;'
  if [ -z "$slow_detail" ]; then
    : >"$slow_args"
    REPEAT_TEST_TRACE_ARGS=$slow_args slow_mutant slow-mutant-twice "$slow_inject" \
      'set -- "$trace_sub" "--trace-file=$trace_file" "--trace-file=$trace_file" "$@" ;;' run build @cheap
    [ -n "$slow_detail" ] || { [ "$(grep -c '' "$slow_args")" = 2 ] && [ "$argv_calls" != "build @cheap" ]; } ||
      slow_detail="a doubled option went unseen: options $(tr '\n' ';' <"$slow_args"); calls: $argv_calls"
  fi
  if [ -z "$slow_detail" ]; then
    : >"$slow_args"
    REPEAT_TEST_TRACE_ARGS=$slow_args slow_mutant slow-mutant-after "$slow_inject" \
      'set -- "$trace_sub" "$@" "--trace-file=$trace_file" ;;' run exec ./prog.exe -- x
    [ -n "$slow_detail" ] || { grep -q '^5 after ' "$slow_args" && [ "$argv_calls" != "exec ./prog.exe -- x" ]; } ||
      slow_detail="an option past the separator went unseen: options $(tr '\n' ';' <"$slow_args"); calls: $argv_calls"
  fi
  # A reader past its bound is KILLed and the verdict still publishes, even
  # when the caller's environment carries an out-of-range bound (an ambient
  # hook can only shorten it) and a retired escalation knob: the verdict is
  # never later than the 30-second bound plus the reap. A reader KILL cannot
  # reap -- stood in for by a mutant whose escalation signals nothing -- is
  # abandoned after the bounded reap: the verdict publishes, the worktree
  # lock clears, and no record is left.
  if [ -z "$slow_detail" ]; then
    slow_reader=$TMP/slowest-stuck-reader.sh
    printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$$" >"$SLOW_READER_PID"\nexec sleep 120\n' >"$slow_reader"
    for kind in bounded ambient unreapable; do
      rm -f "$TMP/slowest-reader.pid"
      case $kind in
        bounded) slow_cap=1 slow_limit=20 ;;
        ambient) slow_cap=99999 slow_limit=50 ;;
        unreapable) slow_cap=1 slow_limit=20 ;;
      esac
      slow_t0=$SECONDS
      if [ "$kind" = unreapable ]; then
        SLOW_READER_PID=$TMP/slowest-reader.pid OCANNL_TOOL_SLOWEST_READER=$slow_reader \
          OCANNL_TOOL_SLOWEST_CAP=$slow_cap REPEAT_TEST_TRACE=$slow_trace \
          slow_mutant "slow-stuck-$kind" 'kill("KILL", -$c) or kill("KILL", $c);' 'kill(0, -$c) or kill(0, $c);' \
          run build @cheap
      else
        SLOW_READER_PID=$TMP/slowest-reader.pid OCANNL_TOOL_SLOWEST_READER=$slow_reader \
          OCANNL_TOOL_SLOWEST_CAP=$slow_cap OCANNL_TOOL_SLOWEST_KILL=0 REPEAT_TEST_TRACE=$slow_trace \
          argv_probe "slow-stuck-$kind" run build @cheap
      fi
      [ -z "$slow_detail" ] || break
      slow_secs=$((SECONDS - slow_t0))
      slow_pid=$(cat "$TMP/slowest-reader.pid" 2>/dev/null)
      slow_lock=$(OCANNL_TOOL_TEST_RUNS=$TMP/argv-runs-slow-stuck-$kind "$repeat_root/tools/test-run.sh" lock-status 2>&1)
      slow_alive=no
      [ -n "$slow_pid" ] && kill -0 "$slow_pid" 2>/dev/null && slow_alive=yes
      [ -z "$slow_pid" ] || kill -KILL "$slow_pid" 2>/dev/null
      case $kind:$slow_alive in
        bounded:yes | ambient:yes) slow_detail="$kind: the reader survived its bound" ;;
        unreapable:no) slow_detail="unreapable: the stand-in reader died, so the unreapable path was not taken" ;;
      esac
      [ -n "$slow_detail" ] || [ -n "$slow_pid" ] || slow_detail="$kind: the stand-in reader never ran"
      [ -n "$slow_detail" ] || case $argv_rc:$argv_out in
        0:*"verdict: pass"*"slowest actions"*) slow_detail="$kind: a stuck reader left a table: $argv_out" ;;
        0:*"verdict: pass"*"no slowest-actions record: the reader did not finish within "*s) ;;
        0:*"verdict: pass"*) slow_detail="$kind: the digest does not say the reader timed out: $argv_out" ;;
        *) slow_detail="$kind: exit $argv_rc (want the run's own 0): $argv_out" ;;
      esac
      [ -n "$slow_detail" ] || [ "$(cat "$argv_dir/exit" 2>/dev/null)" = 0 ] ||
        slow_detail="$kind: the verdict was not published"
      [ -n "$slow_detail" ] || [ "$slow_lock" = idle ] ||
        slow_detail="$kind: the worktree lock is still held: $slow_lock"
      [ -n "$slow_detail" ] || [ "$slow_secs" -le "$slow_limit" ] ||
        slow_detail="$kind: took ${slow_secs}s (limit ${slow_limit}s)"
      [ -n "$slow_detail" ] || { [ ! -e "$argv_dir/slowest" ] && [ ! -e "$argv_dir/trace.csexp" ] &&
        [ ! -e "$argv_dir/slowest.err" ]; } ||
        slow_detail="$kind: left $(ls "$argv_dir" | tr '\n' ' ')"
      [ -n "$slow_detail" ] || case $kind:$(cat "$argv_dir/slowest-skipped" 2>/dev/null) in
        ambient:"the reader did not finish within 30s" | bounded:"the reader did not finish within 1s" | \
          unreapable:"the reader did not finish within 1s") ;;
        *) slow_detail="$kind: the skip record says: $(cat "$argv_dir/slowest-skipped" 2>&1)" ;;
      esac
      [ -z "$slow_detail" ] || break
    done
  fi
  # A verdict no batch produced never prints a record: forged from the green
  # run's, a refused slot, a refused device probe (whose trace is its own
  # build's) and a setup error; a cancellation still prints it.
  if [ -z "$slow_detail" ]; then
    slow_src=$TMP/argv-runs-slow-pass
    slow_from=$(OCANNL_TOOL_TEST_RUNS=$slow_src "$repeat_root/tools/test-run.sh" paths run last)
    for probe in "slot|1|SLOT REFUSED|1|no" "probe|69|DEVICE UNHEALTHY|2|no" \
                 "setup|127|ERROR|3|no" "cancel|143|CANCELLED|4|yes"; do
      IFS='|' read -r kind code want n shown <<<"$probe"
      forged=$slow_src/20000101T00000${n}Z-98$n
      rm -rf "$forged"
      cp -R "$slow_from" "$forged"
      printf '%s\n' "$code" >"$forged/exit"
      if [ "$kind" = slot ]; then
        printf '%s\n' "$TMP/no-fleet-worker.sh" >"$forged/slot"
        printf 'EXECUTION SLOT REFUSED fixture: no slot before the deadline\nexit: 1\n' >"$forged/log"
      else
        printf 'exit: %s\n' "$code" >"$forged/log"
      fi
      out=$(OCANNL_TOOL_TEST_RUNS=$slow_src "$repeat_root/tools/test-run.sh" status "$forged" 2>&1)
      case $shown:$out in
        *:*"verdict: $want"*) ;;
        *) slow_detail="$kind: want $want: $out"; break ;;
      esac
      case $shown:$out in
        yes:*"slowest actions"* | no:*) ;;
        *) slow_detail="$kind: the record was not printed: $out"; break ;;
      esac
      case $shown:$out in
        no:*"slowest actions"*) slow_detail="$kind: the record was printed: $out"; break ;;
      esac
    done
  fi
  if [ -z "$slow_detail" ]; then
    report 0 "$slow_label"
  else
    report 1 "$slow_label" "$slow_detail"
  fi
fi
}
harness_leg_run '80: the digest names the batch'\''s slowest actions #digest' leg_80

# Leg 81: `fresh` runs the named per-test aliases once, in a build directory of
# the run's own, forced and uncached, and the digest gives each its wall time
# from the run's dune trace -- or refuses it as NOT RUN where no process of the
# trace is its action. The trace is synthetic, as in leg 80; the reader takes
# the build directory from the trace's own `config init`, so a made-up one
# serves whatever directory the run hands dune. alpha is a `(test)` stanza's
# producer (`DIR/NAME.exe.output`) plus, under the recursive `@`, the same
# alias's action in a subdirectory; beta is an alias action named with `@@`,
# its directory alone, so the subdirectory twin planted for it does not count;
# gamma's producer is `NAME.actual` under a program of another name, which
# therefore no other test may claim by that program; epsilon-c_fwd writes
# `NAME.ll.actual` and zeta-extension `zeta.extension.actual`, both from a
# program of another name; theta-extension's output is the SECOND target of
# its process; iota's producer writes `iota.ml` and `iota.coverage`, while
# its link (`iota.exe`) and preprocessing (`iota.pp.ml`) are compile
# artifacts and count for nobody; delta writes an output of another name, so
# only its program finds it; the compile and a bystander test count for
# nobody either.
leg_81() {
fresh_label="fresh: forced, uncached, own build directory; per-test times; NOT RUN refused"
if ! command -v python3 >/dev/null 2>&1; then
  skip "$fresh_label" "no python3 on PATH (tools/action-durations.sh needs it)"
else
  cp "$HERE/action-durations.sh" "$repeat_root/tools/action-durations.sh"
  cat >"$TMP/fresh-trace.py" <<'PY'
import sys
path, build, skip = sys.argv[1], sys.argv[2], sys.argv[3]
S, T0 = 1_000_000_000, 1_791_291_000 * 1_000_000_000
def atom(s):
    s = str(s).encode()
    return str(len(s)).encode() + b':' + s
def sexp(x):
    return b'(' + b''.join(sexp(e) if isinstance(e, list) else atom(e) for e in x) + b')'
out = [sexp(['config', 'init', str(T0), ['build_dir', build]])]
act = build + '/.actions/default/'
for pid, start, dur, prog, target, owner in [
        (11, 1, 12.34, 'alpha.exe', build + '/default/test/operations/alpha.exe.output', 'alpha'),
        (12, 2, 1.5, 'sh', act + 'test/operations/sub/runtest-alpha-' + 'a' * 32, 'alpha'),
        (13, 3, 3.25, 'beta.exe', act + 'test/einsum/slow-beta-' + 'b' * 32, 'beta'),
        (14, 4, 7.0, 'beta.exe', act + 'test/einsum/deeper/slow-beta-' + 'c' * 32, 'twin'),
        (15, 5, 4.5, 'shared.exe', build + '/default/test/operations/gamma.actual', 'gamma'),
        (16, 6, 9.75, 'bystander.exe', build + '/default/test/operations/bystander.exe.output', 'other'),
        (17, 7, 2.0, 'ocamlopt.opt', build + '/default/tensor/x.cmx', 'other'),
        (18, 8, 2.25, 'delta.exe', build + '/default/test/operations/backend-0-0.log.actual', 'delta'),
        (19, 9, 1.75, 'epsilon.exe', build + '/default/test/operations/epsilon-c_fwd.ll.actual', 'epsilon'),
        (20, 10, 0.5, 'zeta.exe', build + '/default/test/operations/zeta.extension.actual', 'zeta'),
        (21, 11, 1.25, 'theta.exe', [build + '/default/test/operations/theta-unoptimized.ll.actual',
                                     build + '/default/test/operations/theta.extension.actual'], 'theta'),
        (22, 12, 0.75, 'iota_extract.exe', [build + '/default/test/operations/iota.ml',
                                            build + '/default/test/operations/iota.coverage'], 'iota'),
        (23, 13, 5.0, 'ocamlopt.opt', build + '/default/test/operations/iota.exe', 'other'),
        (24, 14, 5.0, 'ppx.exe', build + '/default/test/operations/iota.pp.ml', 'other')]:
    if skip in (owner, 'all'):
        continue
    t = T0 + start * S
    f = [['pid', str(pid)], ['prog', '/opt/bin/' + prog], ['dir', build + '/default'],
         ['target_files', target if isinstance(target, list) else [target]]]
    out.append(sexp(['process', 'start', str(t)] + f))
    out.append(sexp(['process', 'finish', [str(t), str(int(round(dur * S)))]] + f + [['exit', '0']]))
open(path, 'wb').write(b''.join(out))
PY
  for fresh_skip in '' beta all; do
    python3 "$TMP/fresh-trace.py" "$TMP/fresh-${fresh_skip:-full}.trace" /fake/build "$fresh_skip"
  done
  fresh_probe() { # tag [argv...]: `fresh` through the fixture dune, its trace FRESH_TRACE's
    local tag=$1
    shift
    REPEAT_TEST_TRACE=${FRESH_TRACE:-$TMP/fresh-full.trace} REPEAT_TEST_MKBUILD=1 \
      argv_probe "$tag" fresh "$@"
  }
  fresh_rows() { # digest -> its per-test rows, one line each, blanks squeezed
    printf '%s\n' "$1" | awk '/^per-test wall time/ { f = 1; next }
      f && /^  / { if ($1 != "seconds") print; next }
      f { exit }' | sed 's/^ *//; s/  */ /g'
  }
  fresh_detail=
  fresh_aliases='@test/operations/runtest-alpha @@test/einsum/slow-beta @test/operations/runtest-gamma @test/operations/runtest-delta @test/operations/runtest-epsilon-c_fwd @test/operations/runtest-zeta-extension @test/operations/runtest-theta-extension @test/operations/runtest-iota'
  # 1. Every named test ran: exit 0, the times attributed, the command forced,
  # uncached and in the run's own build directory -- recorded as such -- and
  # that directory gone with the run.
  # shellcheck disable=SC2086  # the aliases are word-split on purpose
  fresh_probe fresh-pass $fresh_aliases -j 4
  case $argv_rc:$argv_out in
    0:*"verdict: pass (exit 0)"*"per-test wall time (fresh: own build directory, --force, --cache=disabled; this run's dune trace):"*) ;;
    *) fresh_detail="all ran: exit $argv_rc (want 0 with the table): $argv_out" ;;
  esac
  fresh_want='13.84 2 0 action @test/operations/runtest-alpha
3.25 1 0 action @@test/einsum/slow-beta
4.50 1 0 action @test/operations/runtest-gamma
2.25 1 0 program @test/operations/runtest-delta
1.75 1 0 action @test/operations/runtest-epsilon-c_fwd
0.50 1 0 action @test/operations/runtest-zeta-extension
1.25 1 0 action @test/operations/runtest-theta-extension
0.75 1 0 action @test/operations/runtest-iota'
  [ -n "$fresh_detail" ] || [ "$(fresh_rows "$argv_out")" = "$fresh_want" ] ||
    fresh_detail="all ran: rows $(fresh_rows "$argv_out" | tr '\n' ';') (want $(printf '%s' "$fresh_want" | tr '\n' ';'))"
  [ -n "$fresh_detail" ] ||
    [ "$argv_calls" = "build --build-dir=$argv_dir/build --force --cache=disabled $fresh_aliases -j 4" ] ||
    fresh_detail="all ran: dune was not handed the fresh argv: $argv_calls"
  case $fresh_detail:$argv_out in
    :*"command: dune build --build-dir=$argv_dir/build --force --cache=disabled $fresh_aliases -j 4"*) ;;
    :*) fresh_detail="all ran: the recorded command does not name the build directory: $argv_out" ;;
  esac
  [ -n "$fresh_detail" ] || { [ ! -e "$argv_dir/build" ] && [ -s "$argv_dir/timings" ] &&
    [ "$(cat "$argv_dir/mode")" = fresh ] && [ ! -e "$argv_dir/trace.csexp" ] &&
    [ ! -e "$argv_dir/fresh-processes.tsv" ] && [ ! -e "$argv_dir/timings-skipped" ]; } ||
    fresh_detail="all ran: the run directory is not as left: $(ls "$argv_dir" | tr '\n' ' ')"
  # The negative control for the build directory's removal: the fixture made
  # it, so its absence is the supervisor's doing.
  if [ -z "$fresh_detail" ]; then
    REPEAT_TEST_MKBUILD=1 argv_probe fresh-mk-control run build --build-dir="$TMP/fresh-mk-control" @x
    [ -d "$TMP/fresh-mk-control/default" ] ||
      fresh_detail="control: the fixture dune made no build directory, so its removal proves nothing"
  fi
  # 2. A named test no process is the action of is NOT RUN, exit 2, while the
  # others keep their times -- a green dune is not a fresh time.
  if [ -z "$fresh_detail" ]; then
    # shellcheck disable=SC2086
    FRESH_TRACE=$TMP/fresh-beta.trace fresh_probe fresh-skip $fresh_aliases
    case $argv_rc:$argv_out in
      2:*"verdict: NOT RUN (no process of this run is the action of 1 named test(s): no fresh time for them) (exit 2)"*"a test NOT RUN"*) ;;
      *) fresh_detail="one skipped: exit $argv_rc (want 2, NOT RUN): $argv_out" ;;
    esac
    [ -n "$fresh_detail" ] || [ "$(fresh_rows "$argv_out" | tr '\n' ';')" = \
      "13.84 2 0 action @test/operations/runtest-alpha;NOT RUN 0 - - @@test/einsum/slow-beta;4.50 1 0 action @test/operations/runtest-gamma;2.25 1 0 program @test/operations/runtest-delta;1.75 1 0 action @test/operations/runtest-epsilon-c_fwd;0.50 1 0 action @test/operations/runtest-zeta-extension;1.25 1 0 action @test/operations/runtest-theta-extension;0.75 1 0 action @test/operations/runtest-iota;" ] ||
      fresh_detail="one skipped: rows $(fresh_rows "$argv_out" | tr '\n' ';')"
  fi
  # A program another named test's action already claimed is not a second
  # test's: shared.exe is gamma's producer, so runtest-shared is NOT RUN.
  if [ -z "$fresh_detail" ]; then
    fresh_probe fresh-claimed @test/operations/runtest-gamma @test/operations/runtest-shared
    case $argv_rc:$(fresh_rows "$argv_out" | tr '\n' ';') in
      "2:4.50 1 0 action @test/operations/runtest-gamma;NOT RUN 0 - - @test/operations/runtest-shared;") ;;
      *) fresh_detail="a claimed program: exit $argv_rc (want 2, shared NOT RUN): $argv_out" ;;
    esac
  fi
  # 3. A trace with no process at all (all served from elsewhere): NOT RUN,
  # not UNTIMED.
  if [ -z "$fresh_detail" ]; then
    FRESH_TRACE=$TMP/fresh-all.trace fresh_probe fresh-none @test/operations/runtest-alpha
    case $argv_rc:$argv_out in
      2:*"verdict: NOT RUN (no process of this run is the action of 1 named test(s)"*"NOT RUN"*"@test/operations/runtest-alpha"*) ;;
      *) fresh_detail="no process: exit $argv_rc (want 2, NOT RUN): $argv_out" ;;
    esac
  fi
  # 4. A red run is FAIL, exit 1, with its times, and never sent to promote
  # from the worktree; a trace the reader refuses is UNTIMED, exit 1.
  if [ -z "$fresh_detail" ]; then
    # shellcheck disable=SC2086
    argv_mode=red fresh_probe fresh-red $fresh_aliases
    argv_mode=
    case $argv_rc:$argv_out in
      1:*"verdict: FAIL (exit 1)"*"per-test wall time"*"not promotable"*) ;;
      *) fresh_detail="a red run: exit $argv_rc (want 1, FAIL, times, not promotable): $argv_out" ;;
    esac
  fi
  if [ -z "$fresh_detail" ]; then
    printf 'not a trace\n' >"$TMP/fresh-garbage.trace"
    FRESH_TRACE=$TMP/fresh-garbage.trace fresh_probe fresh-untimed @test/operations/runtest-alpha
    case $argv_rc:$argv_out in
      1:*"verdict: UNTIMED"*"no per-test times: action-durations.sh: "*"is not a canonical-S-expression dune trace"*) ;;
      *) fresh_detail="an unreadable trace: exit $argv_rc (want 1, UNTIMED): $argv_out" ;;
    esac
  fi
  # A dune that exits with the run's cap nearly spent leaves no time to read
  # the trace in: UNTIMED, never a TIMEOUT over a dune that passed.
  if [ -z "$fresh_detail" ]; then
    REPEAT_TEST_ACTION_SECS=5 argv_mode=slow_dune fresh_probe fresh-nearcap --cap 10 @test/operations/runtest-alpha
    argv_mode=
    case $argv_rc:$argv_out in
      1:*"verdict: UNTIMED"*"no per-test times: the run's cap was nearly spent when dune exited"*) ;;
      *) fresh_detail="near the cap: exit $argv_rc (want 1, UNTIMED): $argv_out" ;;
    esac
  fi
  # 5. Operands that are not per-test aliases are refused before dune runs,
  # with the usage code.
  if [ -z "$fresh_detail" ]; then
    for probe in "none:" "aggregate:@test/operations/slow" "rootless:@runtest-alpha" \
                 "dotdot:@test/../x/runtest-alpha" "dot:@./test/runtest-alpha" \
                 "twice:@test/operations/runtest-alpha @test/operations/runtest-alpha" \
                 "option:--force @test/operations/runtest-alpha" "word:build @test/operations/runtest-alpha" \
                 "zero-width:@test/operations/runtest-alpha -j 0" \
                 "after:@test/operations/runtest-alpha --cap 900"; do
      # shellcheck disable=SC2086  # the operands are word-split on purpose
      fresh_probe "fresh-refuse-${probe%%:*}" ${probe#*:}
      if [ "$argv_rc" != 2 ] || [ -n "$argv_calls" ]; then
        fresh_detail="'${probe#*:}': exit $argv_rc, calls '$argv_calls' (want 2, dune never run): $argv_err"
        break
      fi
    done
  fi
  # The control: the other spelling of the width reaches dune as -j N.
  if [ -z "$fresh_detail" ]; then
    fresh_probe fresh-jcompact @test/operations/runtest-alpha -j2
    case $argv_rc:$argv_calls in
      0:"build --build-dir=$argv_dir/build --force --cache=disabled @test/operations/runtest-alpha -j 2") ;;
      *) fresh_detail="-j2: exit $argv_rc, calls $argv_calls" ;;
    esac
  fi
  if [ -z "$fresh_detail" ]; then
    report 0 "$fresh_label"
  else
    report 1 "$fresh_label" "$fresh_detail"
  fi
fi
}
harness_leg_run '81: fresh runs per-test aliases forced, uncached and timed #fresh' leg_81

# The same contract against a real dune, where the shared trace is live: a
# manual `dune build` is mid-rule, holding dune's build lock, when a managed
# run starts in the same tree. The manual build must end with its own trace
# whole -- readable, naming its slow rule -- whatever the managed run's dune
# made of the lock. (Deleting the shared trace at launch, as this record's
# first version did, left the manual build writing into an unlinked file.)
# A host with no dune skips it, as leg 79's trace leg does.
leg_80_live() {
live_label="a manual dune running beside a managed run keeps its live trace (real dune)"
live_real=
! harness_real_dune || live_real=$harness_dune
if [ -z "$live_real" ] || ! command -v python3 >/dev/null 2>&1; then
  skip "$live_label" "${harness_dune_absent:-no python3 on PATH}"
else
  live_root=$TMP/live-trace
  mkdir -p "$live_root/tools" "$live_root/scripts"
  cp "$SRC" "$live_root/tools/test-run.sh"
  stage_sourced "$live_root" || { harness_abandon "$stage_error"; return; }
  cp "$HERE/action-durations.sh" "$live_root/tools/action-durations.sh"
  chmod +x "$live_root/tools/test-run.sh"
  printf '(lang dune 3.20)\n' >"$live_root/dune-project"
  printf '(rule\n (with-stdout-to slow.txt (run sh -c "touch %s; sleep 4; echo slow")))\n(rule\n (with-stdout-to quick.txt (run sh -c "echo quick")))\n' \
    "$TMP/live-trace.started" >"$live_root/dune"
  live_detail=
  # Both builds get dune's own directory first on PATH: a dune found only
  # through tools/opam-env.sh finds its switch's ocamlc there and nowhere else.
  (cd "$live_root" && PATH=$(dirname "$live_real"):$PATH exec "$live_real" build --cache=disabled ./slow.txt) \
    >"$TMP/live-manual.out" 2>&1 &
  live_pid=$!
  if ! await_fixture_ready "$TMP/live-trace.started" 300; then
    live_detail="the manual build never started its slow rule: $(tail -5 "$TMP/live-manual.out")"
  else
    (cd "$live_root" && PATH=$(dirname "$live_real"):$PATH OCANNL_TOOL_TEST_RUNS=$TMP/live-runs \
      "$live_root/tools/test-run.sh" run build --cache=disabled ./quick.txt) >"$TMP/live-managed.out" 2>&1
  fi
  wait "$live_pid"
  live_rc=$?
  if [ -z "$live_detail" ] && [ "$live_rc" != 0 ]; then
    live_detail="the manual build exited $live_rc: $(tail -5 "$TMP/live-manual.out")"
  fi
  if [ -z "$live_detail" ]; then
    live_table=$("$HERE/action-durations.sh" -n 0 "$live_root/_build/trace.csexp" 2>&1)
    case $live_table in
      *slow.txt*) ;;
      *) live_detail="the manual build's trace lost its slow rule: $live_table; managed run: $(tail -8 "$TMP/live-managed.out")" ;;
    esac
  fi
  if [ -z "$live_detail" ]; then
    report 0 "$live_label"
  else
    report 1 "$live_label" "$live_detail"
  fi
fi
}
harness_leg_run '80b: a manual dune beside a managed run keeps its live trace #digest' leg_80_live

shared_through synthetic_trace

# Leg 92: a build failure is not a failed test (gh-ocannl-1547). dune exits 1
# for both; a red run whose own trace holds a failed build step -- a nonzero
# exit producing only compile artifacts -- is BUILD FAILED, exit 65 (the
# recorded status stays dune's 1), naming the step, and an output an earlier
# run left for a named test that this run never wrote is named as not this
# run's -- while one this run's trace produced is not. Controls: a red run
# whose failed processes are a test's action, a source generator and an
# unfinished compile stays FAIL (exit 1) and names no stale output; an argv
# naming its own build directory is still BUILD FAILED but looks in no
# worktree `_build`. Against a real dune too (92b, below).
leg_92() {
bf_label="a red run whose own trace holds a failed build step is BUILD FAILED (exit 65), not FAIL"
if ! command -v python3 >/dev/null 2>&1; then
  skip "$bf_label" "no python3 on PATH (tools/action-durations.sh needs it)"
  return
fi
cp "$HERE/action-durations.sh" "$repeat_root/tools/action-durations.sh"
bf_build=$repeat_root/_build
mkdir -p "$bf_build/default/t"
synthetic_trace "$TMP/bf-broken.csexp" "$bf_build" \
  'ocamlopt.opt|2|t/.bad.eobjs/native/dune__exe__Bad.cmx,t/.bad.eobjs/native/dune__exe__Bad.o' \
  'ocamlopt.opt|0|t/good.exe' 'good.exe|0|t/good.exe.output' 'red.exe|1|@t/runtest-red'
synthetic_trace "$TMP/bf-red.csexp" "$bf_build" \
  'ocamlopt.opt|0|t/bad.exe' 'red.exe|1|@t/runtest-red' 'bad.exe|3|t/bad.exe.output' \
  'gen.exe|1|t/generated.ml' 'ocamlopt.opt|open|t/.slow.eobjs/native/dune__exe__Slow.cmx'
# What an earlier run left: bad's output, which the broken build never rewrote,
# and good's, which this run's trace says it wrote.
printf 'earlier\n' >"$bf_build/default/t/bad.exe.output"
printf 'earlier\n' >"$bf_build/default/t/good.exe.output"
bf_detail=
REPEAT_TEST_TRACE=$TMP/bf-broken.csexp argv_mode=red argv_probe bf-broken run build @t/runtest-bad @t/runtest-good
argv_mode=
case $argv_rc:$argv_out in
  65:*"verdict: BUILD FAILED (a compile step failed: the tests needing it never ran, not judged) (exit 65)"*) ;;
  *) bf_detail="a broken build: exit $argv_rc (want 65 under BUILD FAILED): $argv_out" ;;
esac
bf_steps=$(printf '%s\n' "$argv_out" | awk '/^build steps that failed/ { f = 1; next } f && /^  / { print; next } f { exit }')
[ -n "$bf_detail" ] || [ "$bf_steps" = "  ocamlopt.opt   t/.bad.eobjs/native/dune__exe__Bad.cmx" ] ||
  bf_detail="a broken build: the failed steps listed are not the compile alone: ${bf_steps:-<none>}"
bf_stale=$(printf '%s\n' "$argv_out" | awk '/^NOT this run.s output/ { f = 1; next } f && /^  / { print; next } f { exit }')
case $bf_stale in
  "  "*/_build/default/t/bad.exe.output) ;;
  *) [ -n "$bf_detail" ] || bf_detail="a broken build: the stale outputs named are not bad's alone: ${bf_stale:-<none>}" ;;
esac
[ -n "$bf_detail" ] || [ "$(cat "$argv_dir/exit" 2>/dev/null)" = 1 ] ||
  bf_detail="a broken build: the recorded status is not dune's own 1: $(cat "$argv_dir/exit" 2>/dev/null)"
# The fingerprint still shows the compiler's error.
case $bf_detail:$argv_out in
  :*"fingerprint:"*"Error: This expression has type int"*) ;;
  :*) bf_detail="a broken build: the compiler's error is not in the fingerprint: $argv_out" ;;
esac
# `wait` exits with the same substitution; `status` with its publication 0.
if [ -z "$bf_detail" ]; then
  bf_wait=$(OCANNL_TOOL_TEST_RUNS=$TMP/argv-runs-bf-broken "$repeat_root/tools/test-run.sh" wait last 2>&1)
  bf_wait_rc=$?
  OCANNL_TOOL_TEST_RUNS=$TMP/argv-runs-bf-broken "$repeat_root/tools/test-run.sh" status last >/dev/null 2>&1
  bf_status_rc=$?
  case $bf_wait_rc:$bf_status_rc:$bf_wait in
    65:0:*"verdict: BUILD FAILED"*) ;;
    *) bf_detail="wait/status: exits $bf_wait_rc/$bf_status_rc (want 65/0): $bf_wait" ;;
  esac
fi
if [ -z "$bf_detail" ]; then
  REPEAT_TEST_TRACE=$TMP/bf-red.csexp argv_mode=red argv_probe bf-red run build @t/runtest-bad @t/runtest-red
  argv_mode=
  case $argv_rc:$argv_out in
    1:*"verdict: FAIL (exit 1)"*) ;;
    *) bf_detail="a red test, generator and unfinished compile: exit $argv_rc (want 1 under FAIL): $argv_out" ;;
  esac
  case $argv_out in
    *"build steps that failed"* | *"NOT this run"*)
      [ -n "$bf_detail" ] || bf_detail="a red test named build steps or stale outputs: $argv_out" ;;
  esac
  [ -n "$bf_detail" ] || [ ! -e "$argv_dir/build-failures" ] ||
    bf_detail="a red test recorded build failures: $(cat "$argv_dir/build-failures")"
fi
if [ -z "$bf_detail" ]; then
  REPEAT_TEST_TRACE=$TMP/bf-broken.csexp argv_mode=red \
    argv_probe bf-elsewhere run build --build-dir=$TMP/bf-elsewhere @t/runtest-bad
  argv_mode=
  case $argv_rc:$argv_out in
    65:*"NOT this run"*) bf_detail="a build directory of its own: the worktree's _build was read: $argv_out" ;;
    65:*"verdict: BUILD FAILED"*) ;;
    *) bf_detail="a build directory of its own: exit $argv_rc (want 65): $argv_out" ;;
  esac
fi
rm -f "$bf_build/default/t/bad.exe.output" "$bf_build/default/t/good.exe.output"
if [ -z "$bf_detail" ]; then report 0 "$bf_label"; else report 1 "$bf_label" "$bf_detail"; fi
}
harness_leg_run '92: a build failure is BUILD FAILED, not a failed test #digest' leg_92

# The same against a real dune: a test whose source does not type-check beside
# one that fails and one that passes. A first, green run leaves the broken
# test's output; the broken run names it, and not the passing test's, which
# this run rewrote. A host with no dune skips it, as leg 80b does.
leg_92_real() {
bfr_label="a real type error is BUILD FAILED naming the stale output; a red test alone stays FAIL (real dune)"
bfr_real=
! harness_real_dune || bfr_real=$harness_dune
if [ -z "$bfr_real" ] || ! command -v python3 >/dev/null 2>&1; then
  skip "$bfr_label" "${harness_dune_absent:-no python3 on PATH}"
  return
fi
bfr_root=$TMP/build-failed-real
mkdir -p "$bfr_root/tools" "$bfr_root/scripts" "$bfr_root/t"
cp "$SRC" "$bfr_root/tools/test-run.sh"
stage_sourced "$bfr_root" || { harness_abandon "$stage_error"; return; }
cp "$HERE/action-durations.sh" "$bfr_root/tools/action-durations.sh"
chmod +x "$bfr_root/tools/test-run.sh"
printf '(lang dune 3.20)\n' >"$bfr_root/dune-project"
printf '(test (name bad) (modules bad))\n(test (name good) (modules good))\n(test (name red) (modules red))\n' \
  >"$bfr_root/t/dune"
printf 'let () = print_endline "one"\n' >"$bfr_root/t/bad.ml"
printf 'one\n' >"$bfr_root/t/bad.expected"
printf 'let () = print_endline "v1"\n' >"$bfr_root/t/good.ml"
printf 'v1\n' >"$bfr_root/t/good.expected"
printf 'let () = exit 1\n' >"$bfr_root/t/red.ml"
bfr_run() { # tag dune-args...
  local tag=$1
  shift
  (cd "$bfr_root" && PATH=$(dirname "$bfr_real"):$PATH OCANNL_TOOL_TEST_RUNS=$TMP/bfr-runs \
    harness_dune_colour "$bfr_root/tools/test-run.sh" run build --cache=disabled "$@") >"$TMP/bfr-$tag.out" 2>&1
  bfr_rc=$?
  bfr_out=$(cat "$TMP/bfr-$tag.out")
}
bfr_detail=
bfr_run green @t/runtest-bad @t/runtest-good
[ "$bfr_rc" = 0 ] || bfr_detail="the first, green run: exit $bfr_rc: $bfr_out"
if [ -z "$bfr_detail" ]; then
  printf 'let () = print_endline 1\n' >"$bfr_root/t/bad.ml"
  printf 'let () = print_endline "v2"\n' >"$bfr_root/t/good.ml"
  printf 'v2\n' >"$bfr_root/t/good.expected"
  bfr_run broken @t/runtest-bad @t/runtest-good @t/runtest-red
  case $bfr_rc:$bfr_out in
    65:*"verdict: BUILD FAILED"*"build steps that failed"*"dune__exe__Bad"*"NOT this run's output"*/_build/default/t/bad.exe.output*) ;;
    *) bfr_detail="a type error: exit $bfr_rc (want 65 naming the step and bad's stale output): $bfr_out" ;;
  esac
  bfr_dir=$(OCANNL_TOOL_TEST_RUNS=$TMP/bfr-runs "$bfr_root/tools/test-run.sh" paths run last)
  if ! LC_ALL=C grep -q "$(printf '\033')\[" "$bfr_dir/log"; then
    bfr_detail="the real build failure log did not carry forced colour: $bfr_dir/log"
  fi
  bfr_stale=$(printf '%s\n' "$bfr_out" | awk '/^NOT this run.s output/ { f = 1; next } f && /^  / { print; next } f { exit }')
  case $bfr_detail:$bfr_stale in
    :*good.exe.output*) bfr_detail="a type error: the rewritten good.exe.output was named stale: $bfr_out" ;;
  esac
fi
if [ -z "$bfr_detail" ]; then
  printf 'let () = print_endline "one"\n' >"$bfr_root/t/bad.ml"
  bfr_run red @t/runtest-bad @t/runtest-red
  case $bfr_rc:$bfr_out in
    1:*"verdict: FAIL (exit 1)"*) ;;
    *) bfr_detail="a red test alone: exit $bfr_rc (want 1 under FAIL): $bfr_out" ;;
  esac
fi
if [ -z "$bfr_detail" ]; then report 0 "$bfr_label"; else report 1 "$bfr_label" "$bfr_detail"; fi
}
harness_leg_run '92b: a real build failure is BUILD FAILED, not a failed test #digest' leg_92_real

# `repeat` has its own vocabulary and the same trap: N refused iterations are
# byte-identical, so an unguarded set would report IDENTICAL over nothing.
# One iteration decides it (every iteration runs the same argv), the set stops
# there, the coordinator exits 2 over a recorded 1, and `wait` on the set
# digests it under the same verdict. The control: a merely red iteration is
# still repeated in full, with dune's status.
leg_38() {
repeat_probe repeat-refused usage_option 3 build @cheap
repeat_refused_label="repeat stops after a refused first iteration and exits 2 over a recorded 1"
if [ "$repeat_rc" = 2 ] && [ "$(cat "$TMP/repeat-refused.counter")" = 1 ] \
   && [ "$(cat "$repeat_dir/exit" 2>/dev/null)" = 1 ] \
   && case $repeat_out in
        *"repeat result: INVOCATION REFUSED"*"dune said:"*"  dune: unknown option '--frobnicate'."*) true ;;
        *) false ;;
      esac; then
  argv_runs=$TMP/repeat-runs-repeat-refused argv_probe repeat-refused-wait wait last
  argv_runs=
  if [ "$argv_rc" = 2 ] && case $argv_out in *"$refused_verdict"*) true ;; *) false ;; esac; then
    report 0 "$repeat_refused_label"
  else
    report 1 "$repeat_refused_label" "wait last on the set: exit $argv_rc; stdout: $argv_out"
  fi
else
  report 1 "$repeat_refused_label" \
    "exit $repeat_rc; iterations $(cat "$TMP/repeat-refused.counter"); recorded $(cat "$repeat_dir/exit" 2>/dev/null); stdout: $repeat_out"
fi
repeat_probe repeat-red-error red 2 build @cheap
if [ "$repeat_rc" = 1 ] && [ "$(cat "$TMP/repeat-red-error.counter")" = 2 ] \
   && case $repeat_out in
        *"INVOCATION REFUSED"*) false ;;
        *"repeat result: IDENTICAL"*) true ;;
        *) false ;;
      esac; then
  report 0 "repeat: a red iteration is repeated in full and keeps dune's status"
else
  report 1 "repeat: a red iteration is repeated in full and keeps dune's status" \
    "exit $repeat_rc; iterations $(cat "$TMP/repeat-red-error.counter"); stdout: $repeat_out"
fi
}
harness_leg_run '38: repeat stops after a refused first iteration #repeat' leg_38

shared_through renamed_dune
leg_83() {
rb_label="repeat bounds: a test past the per-test cap ends the set as TIMEOUT naming it, a wedged device ends it as DEVICE UNHEALTHY; inside the cap, an answering device and a CPU batch's default do not"
renamed_dune_readable
rb_detail=
if [ -z "${test_cap_named_ok:-}" ]; then
  skip "$rb_label" "a process renamed through \$0 does not read back as dune in ps here"
else
  export REPEAT_TEST_ACTION_PID=$TMP/rb-action.pid
  rb_start=$SECONDS
  argv_mode=action argv_probe rb-hung repeat --test-cap 2 3 build @cheap
  rb_took=$((SECONDS - rb_start))
  rb_action=$(cat "$REPEAT_TEST_ACTION_PID" 2>/dev/null)
  { [ "$argv_rc" = 142 ] &&
    grep -qF "repeat result: TIMEOUT -- in iteration 1 a test outlived the per-test cap (sleep 60), and the set stopped there (1 of 3 iterations started)" <<<"$argv_out"; } ||
    rb_detail="hung: exit $argv_rc (want 142): $argv_out"
  [ -n "$rb_detail" ] || [ "$rb_took" -lt 20 ] || rb_detail="hung: the set took ${rb_took}s under a 2s test cap"
  [ -n "$rb_detail" ] || [ "$(grep -c '^clean ' "$TMP/rb-hung.calls")" = 1 ] ||
    rb_detail="hung: the set went on past the hung iteration: $(cat "$TMP/rb-hung.calls")"
  [ -n "$rb_detail" ] || [ -n "$rb_action" ] || rb_detail="hung: the test never started"
  if [ -z "$rb_detail" ]; then
    case $(pstate "$rb_action") in
      '' | Z) ;;
      *) rb_detail="hung: the test (pid $rb_action) survived the set"; kill -KILL "$rb_action" 2>/dev/null ;;
    esac
  fi
  [ -n "$rb_detail" ] || grep -q "^test-run: per-test cap: pid $rb_action has run" "$argv_dir/iteration-1/stdout" ||
    rb_detail="hung: the iteration's output does not name the test: $(cat "$argv_dir/iteration-1/stdout")"
  # The run's digest reads the record the iteration left, as for a run's own.
  if [ -z "$rb_detail" ]; then
    rb_status=$(OCANNL_TOOL_TEST_RUNS=$TMP/argv-runs-rb-hung "$repeat_root/tools/test-run.sh" status last 2>&1)
    case $rb_status in
      *"verdict: TIMEOUT (a test outlived the per-test cap; run was killed, not judged) (exit 142)"*"hung test: sleep 60"*) ;;
      *) rb_detail="hung: the digest does not read the record: $rb_status" ;;
    esac
  fi
  [ -n "$rb_detail" ] ||
    OCANNL_TOOL_TEST_RUNS=$TMP/argv-runs-rb-hung "$repeat_root/tools/test-run.sh" idle >/dev/null 2>&1 ||
    rb_detail="hung: the worktree is not idle after the set"
  # Inside the cap, every iteration runs and the set compares as before.
  if [ -z "$rb_detail" ]; then
    REPEAT_TEST_ACTION_SECS=1 argv_mode=action argv_probe rb-inside repeat --test-cap 4 2 build @cheap
    { [ "$argv_rc" = 0 ] && grep -q '^repeat result: IDENTICAL -- ' <<<"$argv_out" &&
      [ ! -e "$argv_dir/test-cap" ]; } || rb_detail="inside: exit $argv_rc: $argv_out"
  fi
  # The default is run's: 1500s for a GPU batch (resolved here under the fake slot), said in the
  # log; none for a CPU one. OCANNL_TOOL_PER_TEST_CAP is explicit, and on a GPU batch the hung test
  # is blamed on the device.
  if [ -z "$rb_detail" ]; then
    REPEAT_TEST_ACTION_SECS=1 argv_mode=action slot_probe rb-gpu-default hold hip cc repeat 2 build @cheap
    { [ "$argv_rc" = 0 ] && [ -z "$slot_calls" ] &&
      grep -qF "test-run: repeat: per-test cap: 1500s for each process dune starts (the default for a GPU batch)" "$argv_dir/log"; } ||
      rb_detail="gpu default: exit $argv_rc; slot calls: ${slot_calls:-<none>}; log: $(cat "$argv_dir/log")"
  fi
  if [ -z "$rb_detail" ]; then
    REPEAT_TEST_ACTION_SECS=1 argv_mode=action slot_probe rb-cpu-default hold cc cc repeat 2 build @cheap
    { [ "$argv_rc" = 0 ] && grep -qF "test-run: repeat: per-test cap: none (no default: a CPU batch)" "$argv_dir/log"; } ||
      rb_detail="cpu default: exit $argv_rc; log: $(cat "$argv_dir/log")"
  fi
  if [ -z "$rb_detail" ]; then
    OCANNL_TOOL_PER_TEST_CAP=2 argv_mode=action slot_probe rb-gpu-hung hold hip cc repeat 2 build @cheap
    { [ "$argv_rc" = 142 ] && grep -q "a wedged device hangs every test after it" "$argv_dir/iteration-1/stdout"; } ||
      rb_detail="gpu hung: exit $argv_rc: $argv_out"
    rb_action=$(cat "$REPEAT_TEST_ACTION_PID" 2>/dev/null)
    case $(pstate "${rb_action:-0}") in
      '' | Z) ;;
      *) kill -KILL "$rb_action" 2>/dev/null
         [ -n "$rb_detail" ] || rb_detail="gpu hung: the test (pid $rb_action) survived the set" ;;
    esac
  fi
  unset REPEAT_TEST_ACTION_PID
  # The device probe: a query that never answers ends the set before the first iteration's dune
  # -- its clean included -- and one that answers lets each iteration run, asked once each.
  if [ -z "$rb_detail" ]; then
    probe_hang_run pid rb-probe-hang hip cc hip "" repeat 3 build @cheap
    { [ "$argv_rc" = 69 ] && [ -z "$argv_calls" ] && [ "$(grep -c . "$TMP/rb-probe-hang.probe")" = 1 ] &&
      grep -qF "repeat result: DEVICE UNHEALTHY -- iteration 1's device probe got no answer, and the set stopped there (1 of 3 iterations started)" <<<"$argv_out" &&
      grep -qF "hip: NO ANSWER within ${probe_cap}s" "$argv_dir/device-probe"; } ||
      rb_detail="probe hang: exit $argv_rc; dune calls: ${argv_calls:-<none>}; probes: $probe_calls; output: $argv_out"
    rb_probe_pid=$(cat "$TMP/rb-probe-hang.probe.pid" 2>/dev/null)
    [ -n "$rb_detail" ] || [ -n "$rb_probe_pid" ] ||
      rb_detail="probe hang: the stand-in recorded no pid within ${probe_cap}s"
    case ${rb_probe_pid:+$(pstate "$rb_probe_pid")} in
      '' | Z) ;;
      *) kill -KILL "$rb_probe_pid" 2>/dev/null
         [ -n "$rb_detail" ] || rb_detail="probe hang: the query (pid $rb_probe_pid) survived its bound" ;;
    esac
  fi
  if [ -z "$rb_detail" ]; then
    probe_run rb-probe-answer hip cc "" "" repeat 2 build @cheap
    { [ "$argv_rc" = 0 ] && grep -q '^repeat result: IDENTICAL -- ' <<<"$argv_out" &&
      [ "$(grep -c '^clean ' "$TMP/rb-probe-answer.calls")" = 2 ] &&
      [ "$(grep -c . "$TMP/rb-probe-answer.probe")" = 2 ] &&
      grep -qF "hip: answered in" "$argv_dir/iteration-2/device-probe"; } ||
      rb_detail="probe answer: exit $argv_rc; dune calls: ${argv_calls:-<none>}; probes: $probe_calls; output: $argv_out"
  fi
  # A resolution the cap cuts short runs no iteration and keeps the cap's status: TIMEOUT, 142,
  # never a usage error (local Codex review).
  if [ -z "$rb_detail" ]; then
    FAKE_REACH_SLEEP=5 slot_probe rb-resolve-cap hold hip cc repeat --cap 1 2 build @cheap
    rb_status=$(OCANNL_TOOL_TEST_RUNS=$TMP/argv-runs-rb-resolve-cap "$repeat_root/tools/test-run.sh" status last 2>&1)
    { [ "$argv_rc" = 142 ] && [ -z "$argv_calls" ] &&
      grep -qF "the bounds were not resolved (exit 142" <<<"$argv_out" &&
      grep -qF "verdict: TIMEOUT (cap expired; run was killed, not judged) (exit 142)" <<<"$rb_status"; } ||
      rb_detail="resolution past the cap: exit $argv_rc; dune calls: ${argv_calls:-<none>}; output: $argv_out; status: $rb_status"
  fi
  if [ -z "$rb_detail" ]; then report 0 "$rb_label"; else report 1 "$rb_label" "$rb_detail"; fi
fi
}
harness_leg_run '83: repeat has both bounds #repeat' leg_83

# ---------------------------------------------------------------------------
# Leg 85: zombie-only groups under a parent that reaps no orphans (gh-ocannl-1376)
# ---------------------------------------------------------------------------
# Three review rounds of staging#1172 each found a process group holding nothing
# but zombies, left behind by a `repeat` whose members outlived their parents,
# that reap_repeat_group refused (exit 2) where the set should have kept its
# verdict: the iteration shell dying on the supervisor's TERM before its
# per-test-cap watcher, and the bounds resolution's process-substitution child,
# which its shell never reaps. Each reproduced only in the reviewer's container,
# whose PID 1 never reaps orphans; macOS and the Ubuntu runners do, and the
# zombie is gone before anything looks. So these runs go through a stand-in for
# that PID 1: a Linux child subreaper (prctl PR_SET_CHILD_SUBREAPER) that waits
# for its own child only, so every orphan handed to it stays a zombie in the
# group it left. When its child exits it records what it holds, KILLs
# anything still live (a leak, which fails the case), and only then reaps all
# of it, so its zombies never outlive the harness. Under it, `repeat` past
# the per-test cap keeps TIMEOUT (142), a GPU batch's set (resolved bounds, the
# default cap) passes, and `run` keeps its verdict both ways: nothing is
# refused, and across the cases the stand-in must have held a zombie, or it
# reproduced nothing. The negative controls take out the two fixes -- the
# iteration shell's reaping trap, the resolution's inert-zombie branch -- and
# each set must then be refused, exit 2, under the same stand-in.
#
# Off Linux the leg is not applicable (gh-ocannl-1930): the stand-in is a
# Linux prctl read back through Linux's /proc, and what it stands in for -- a
# container's PID 1 that reaps nothing -- is a Linux arrangement; macOS's PID 1
# reaps every orphan, so no zombie-only group outlives its parent there and no
# coverage is lost. subreaper_kernel_probe prints `absent` when UNAME named a
# kernel other than Linux, and that answer alone makes the leg not applicable:
# a uname that is missing or says nothing leaves the leg to run, or to skip
# (leg 85b is the probe's control). On Linux a missing python3 or a refused
# prctl is still a skip: coverage that host lost.
subreaper_kernel_probe() { # UNAME
  local kernel
  kernel=$("$1" -s 2>/dev/null) || return 0
  [ -z "$kernel" ] || [ "$kernel" = Linux ] || echo absent
}
reap_label="zombie-only groups under a parent that reaps no orphans: repeat keeps TIMEOUT and a GPU set passes, run keeps its verdict, nothing refused"
reap_nc_trap_label="negative control: without the iteration shell's reaping trap, a capped repeat under that parent is refused"
reap_nc_resolve_label="negative control: without the resolution's inert-zombie branch, a GPU set under that parent is refused"
reap_stub=$TMP/subreaper.py
cat >"$reap_stub" <<'SUBREAPER'
import ctypes, os, signal, sys
# The nearest subreaper ancestor inherits every orphan below it; this one waits
# for its own child only, as a container's PID 1 that reaps nothing would.
if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
    sys.exit(97)
if len(sys.argv) == 1:
    sys.exit(0)  # the capability probe
pid = os.fork()
if pid == 0:
    os.execvp(sys.argv[1], sys.argv[1:])
_, st = os.waitpid(pid, 0)
held = []
for d in os.listdir('/proc'):
    if not d.isdigit():
        continue
    try:
        with open('/proc/%s/stat' % d) as f:
            fields = f.read().rsplit(')', 1)[1].split()
    except OSError:
        continue
    if int(fields[1]) == os.getpid():
        held.append((int(d), fields[0], fields[2]))
with open(os.environ['SUBREAPER_RECORD'], 'a') as f:
    for p, state, pgid in held:
        f.write('%d %s %s\n' % (p, state, pgid))
for p, state, _ in held:
    if state != 'Z':
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
# Recorded, it reaps everything it adopted: under a PID 1 that reaps nothing
# (where this leg matters most) its zombies would otherwise outlive the harness.
while True:
    try:
        os.waitpid(-1, 0)
    except ChildProcessError:
        break
    except InterruptedError:
        continue
sys.exit(os.WEXITSTATUS(st) if os.WIFEXITED(st) else 128 + os.WTERMSIG(st))
SUBREAPER
reap_held= reap_got=
reap_case() { # tag want-exit -- argv_probe/slot_probe call...: under the stand-in; 0 iff kept and nothing leaked
  local tag=$1 want=$2
  shift 3
  : >"$TMP/$tag.held"
  fixture_launcher=(python3 "$reap_stub")
  SUBREAPER_RECORD=$TMP/$tag.held "$@"
  fixture_launcher=()
  reap_held="$reap_held$(cat "$TMP/$tag.held")"$'\n'
  reap_got="exit $argv_rc (want $want); held (pid state pgid): $(tr '\n' ';' <"$TMP/$tag.held"); output: $argv_out $argv_err"
  [ "$argv_rc" = "$want" ] || return 1
  if grep -q 'refusing to reuse' <<<"$argv_out$argv_err"; then return 1; fi
  if grep -q ' [^Z] ' "$TMP/$tag.held"; then return 1; fi
}
reap_swap() { # FROM TO: repeat_root's script is the shipping one with its one line holding FROM made TO
  FROM=$1 TO=$2 perl -ne 'if (index($_, $ENV{FROM}) >= 0) { $n++; print "$ENV{TO}\n"; next }
    print; END { exit($n == 1 ? 0 : 1) }' "$SRC" >"$repeat_root/tools/test-run.sh" || return 1
  if cmp -s "$SRC" "$repeat_root/tools/test-run.sh"; then return 1; fi
  bash -n "$repeat_root/tools/test-run.sh"
}
reap_refused() { # label: the last case was refused, exit 2, as reap_repeat_group refuses
  if [ "$argv_rc" = 2 ] && grep -q 'refusing to reuse' <<<"$argv_out$argv_err"; then
    report 0 "$1"
  else
    report 1 "$1" "$reap_got"
  fi
}
leg_85() {
local why= detail=
if [ "$(subreaper_kernel_probe uname)" = absent ]; then
  harness_not_applicable "this kernel is $(uname -s), not Linux: it has no child subreaper (prctl PR_SET_CHILD_SUBREAPER) to stand in for a PID 1 that reaps no orphans, the Linux container case this leg pins"
  return
fi
renamed_dune_readable
if ! command -v python3 >/dev/null 2>&1 || ! python3 "$reap_stub" >/dev/null 2>&1; then
  why="needs a child subreaper (prctl PR_SET_CHILD_SUBREAPER, through python3), which this host did not give"
elif [ -z "${test_cap_named_ok:-}" ] || [ "$have_state" != 1 ]; then
  why="a process renamed through \$0 does not read back as dune in ps here, or no state reader"
fi
if [ -n "$why" ]; then
  skip "$reap_label" "$why"
  skip "$reap_nc_trap_label" "$why"
  skip "$reap_nc_resolve_label" "$why"
  return
fi
reap_held=
export REPEAT_TEST_ACTION_PID=$TMP/reap-action.pid
argv_mode=action reap_case reap-repeat-hung 142 -- argv_probe reap-repeat-hung repeat --test-cap 2 2 build @cheap ||
  detail="repeat past the cap: $reap_got"
if [ -z "$detail" ]; then
  argv_mode=action REPEAT_TEST_ACTION_SECS=1 reap_case reap-repeat-gpu 0 -- \
    slot_probe reap-repeat-gpu hold hip cc repeat 2 build @cheap ||
    detail="a GPU set: $reap_got"
  # ...through the resolution's inert-zombie branch, the shape round 3 found.
  [ -n "$detail" ] || grep -q '^repeat: the resolution left only zombies in group [0-9]*: inert, not reaped$' "$argv_dir/log" ||
    detail="a GPU set: its log does not say the resolution left only zombies: $(cat "$argv_dir/log")"
fi
if [ -z "$detail" ]; then
  argv_mode=action reap_case reap-run-hung 142 -- argv_probe reap-run-hung run --test-cap 2 build @cheap ||
    detail="run past the cap: $reap_got"
fi
if [ -z "$detail" ]; then
  argv_mode=action REPEAT_TEST_ACTION_SECS=1 reap_case reap-run-gpu 0 -- \
    slot_probe reap-run-gpu hold hip cc run build @cheap ||
    detail="a GPU run: $reap_got"
fi
# The stand-in reproduced the shape only if it was handed something to keep.
if [ -z "$detail" ] && ! grep -q ' Z ' <<<"$reap_held"; then
  detail="the stand-in held no zombie in any case: no case met a parent that reaps no orphans"
fi
if [ -z "$detail" ]; then report 0 "$reap_label"; else report 1 "$reap_label" "$detail"; fi
# The controls: the shipping script with one fix taken out, put back after.
if reap_swap '{ kill "$watcher"; wait "$watcher"; } 2>/dev/null; exit 143' '        :'; then
  argv_mode=action reap_case reap-nc-trap 2 -- argv_probe reap-nc-trap repeat --test-cap 2 2 build @cheap
  reap_refused "$reap_nc_trap_label"
else
  report 1 "$reap_nc_trap_label" "the iteration shell's trap could not be taken out of $SRC"
fi
if reap_swap '! group_identity_matches "$resolve_dir" && ! group_alive "$resolve_pg"; then' '       false; then'; then
  argv_mode=action REPEAT_TEST_ACTION_SECS=1 reap_case reap-nc-resolve 2 -- \
    slot_probe reap-nc-resolve hold hip cc repeat 2 build @cheap
  reap_refused "$reap_nc_resolve_label"
else
  report 1 "$reap_nc_resolve_label" "the resolution's inert-zombie branch could not be taken out of $SRC"
fi
cp "$SRC" "$repeat_root/tools/test-run.sh"
unset REPEAT_TEST_ACTION_PID
}
harness_leg_run '85: zombie-only groups under a parent that reaps no orphans #repeat' leg_85

leg_85b() {
  local d=$TMP/85b ok=1
  mkdir -p "$d"
  # Stand-ins for the answers a uname gives: Linux, another kernel, nothing, a failure.
  printf '#!/bin/sh\necho Linux\n' >"$d/linux"
  printf '#!/bin/sh\necho Darwin\n' >"$d/darwin"
  printf '#!/bin/sh\nexit 0\n' >"$d/silent"
  printf '#!/bin/sh\necho Darwin\nexit 1\n' >"$d/failing"
  chmod +x "$d/linux" "$d/darwin" "$d/silent" "$d/failing"
  for stand_in in "$d/linux" "$d/silent" "$d/failing" "$d/missing"; do
    [ "$(subreaper_kernel_probe "$stand_in")" != absent ] || ok=0
  done
  [ "$(subreaper_kernel_probe "$d/darwin")" = absent ] || ok=0
  if [ "$ok" = 1 ]; then
    report 0 "85b: only a kernel that named itself, and not as Linux, makes 85 not applicable (negative control)"
  else
    report 1 "85b: only a kernel that named itself, and not as Linux, makes 85 not applicable (negative control)" \
      "the probe answered a Linux, silent, failing or missing uname as absent, or a Darwin one as present"
  fi
}
harness_leg_run '85b: only a kernel that named itself, and not as Linux, makes 85 not applicable #repeat' leg_85b

# ---------------------------------------------------------------------------
# Leg 84: repeat --direct builds once and replays the rules (gh-ocannl-1377)
# ---------------------------------------------------------------------------
# Against a real dune, since what a replay owes is dune's own execution of the
# same rules: a fixture project whose one program counts its runs in a file
# outside the build tree, and prints the environment it was given. A direct set
# builds once and re-executes each named test once per iteration (the counts --
# which dune's --force alone never moves for a target-producing rule, the
# negative control below), with the environment, directory and output dune's
# own run had (the build phase's snapshot equals the first replay's outputs); an
# alias action is dune's to re-run; no iteration's dune compiles anything. A test
# whose output moves is DIFFERING; one whose output tells the replay from dune
# (an action's DUNE_ACTION_TRACE_DIR, which dune alone sets) is REPLAY DIVERGED
# -- the witness's own negative control; a sandboxed rule and an action form
# outside the replayed subset are refused before any iteration, a test the
# build phase could not build is BUILD FAILED, a replayed test past the
# per-test cap ends the set as TIMEOUT and is killed, and operands other than
# per-test aliases are refused. A host with no dune or no python3 skips it.
leg_84() {
direct_label="repeat --direct: one build, then each test's own rules replayed as dune ran them (real dune)"
direct_nc_label="negative control: with the replay cut out, dune's --force alone runs no target-producing test again (real dune)"
direct_fd_label="negative control: with the replay's inherited descriptors closed, a session-escaped descendant goes unseen (real dune)"
direct_real=
! harness_real_dune || direct_real=$harness_dune
if [ -z "$direct_real" ] || ! command -v python3 >/dev/null 2>&1; then
  skip "$direct_label" "${harness_dune_absent:-no python3 on PATH}"
  skip "$direct_nc_label" "${harness_dune_absent:-no python3 on PATH}"
  skip "$direct_fd_label" "${harness_dune_absent:-no python3 on PATH}"
else
  direct_root=$TMP/direct-repo
  mkdir -p "$direct_root/tools" "$direct_root/scripts" "$direct_root/t"
  cp "$SRC" "$direct_root/tools/test-run.sh"
  stage_sourced "$direct_root" || { harness_abandon "$stage_error"; return; }
  cp "$HERE/action-durations.sh" "$HERE/replay-rules.sh" "$direct_root/tools/"
  chmod +x "$direct_root/tools/test-run.sh"
  printf '(lang dune 3.20)\n' >"$direct_root/dune-project"
  cat >"$direct_root/t/dune" <<'DUNE'
(test
 (name det)
 (modules det)
 (deps cfg)
 (libraries unix))
(rule
 (target envp.actual)
 (deps cfg)
 (action (with-stdout-to envp.actual (progn (echo "a" "b\n") (run %{dep:det.exe} env)))))
(rule (alias runtest-envp) (action (diff envp.actual envp.actual)))
(rule (alias runtest-inalias) (action (run %{dep:det.exe} inalias)))
(rule (target flaky.actual) (action (with-stdout-to flaky.actual (run %{dep:det.exe} flaky))))
(rule (alias runtest-flaky) (action (diff flaky.actual flaky.actual)))
(rule (target leak.actual) (action (with-stdout-to leak.actual (run %{dep:det.exe} leak))))
(rule (alias runtest-leak) (action (diff leak.actual leak.actual)))
(rule (target slow.actual) (action (with-stdout-to slow.actual (run %{dep:det.exe} slow))))
(rule (alias runtest-slow) (action (diff slow.actual slow.actual)))
(rule
 (target sb.actual)
 (deps (sandbox always))
 (action (with-stdout-to sb.actual (run %{dep:det.exe} sb))))
(rule (alias runtest-sb) (action (diff sb.actual sb.actual)))
(rule (target sys.actual) (action (with-stdout-to sys.actual (system "echo sys"))))
(rule (alias runtest-sys) (action (diff sys.actual sys.actual)))
(rule (target escape.actual) (action (with-stdout-to escape.actual (run %{dep:det.exe} escape))))
(rule (alias runtest-escape) (action (diff escape.actual escape.actual)))
(executable (name broken) (modules broken))
(rule (target broken.actual) (action (with-stdout-to broken.actual (run %{dep:broken.exe}))))
(rule (alias runtest-broken) (action (diff broken.actual broken.actual)))
DUNE
  # The counter lives outside the build tree, so a run is counted wherever it happens.
  cat >"$direct_root/t/det.ml" <<'ML'
let count tag =
  let f = Sys.getenv "DIRECT_COUNTER" ^ "." ^ tag in
  let n = try List.length (In_channel.with_open_text f In_channel.input_lines) with Sys_error _ -> 0 in
  Out_channel.with_open_gen [ Open_append; Open_creat ] 0o644 f (fun oc -> output_string oc "run\n");
  n + 1

let () =
  match Sys.argv with
  | [| _ |] -> ignore (count "det"); print_endline "det"
  | [| _; "env" |] ->
    ignore (count "env");
    (* Dune gives each action a TMPDIR and a trace directory of its own: everything else is what
       the replay owes -- and a step's own directory (OCANNL_TOOL_TESTRUN_RD), which differs
       between the build phase and every replay, reaches neither (gh-ocannl-1448). *)
    let mine k = List.mem k [ "TMPDIR"; "DUNE_ACTION_TRACE_DIR" ] in
    Unix.environment () |> Array.to_list
    |> List.filter (fun kv -> not (mine (List.hd (String.split_on_char '=' kv))))
    |> List.sort compare |> List.iter print_endline;
    print_endline (Sys.getcwd ())
  | [| _; "inalias" |] -> ignore (count "inalias")
  | [| _; "flaky" |] -> Printf.printf "run %d\n" (count "flaky")
  | [| _; "leak" |] ->
    print_endline (if Sys.getenv_opt "DUNE_ACTION_TRACE_DIR" = None then "replayed" else "dune")
  | [| _; "slow" |] -> if count "slow" > 1 then Unix.sleep 60; print_endline "slow"
  | [| _; "escape" |] ->
    (* Replayed (not the build phase's run): a child leaves the session and outlives the test. *)
    if count "escape" > 1 then begin
      flush stdout;
      if Unix.fork () = 0 then begin
        ignore (Unix.setsid ());
        Out_channel.with_open_text (Sys.getenv "DIRECT_COUNTER" ^ ".escape-pid") (fun oc ->
          output_string oc (string_of_int (Unix.getpid ())));
        let release = Sys.getenv "DIRECT_COUNTER" ^ ".escape-release" in
        let t0 = Unix.gettimeofday () in
        while (not (Sys.file_exists release)) && Unix.gettimeofday () -. t0 < 30. do Unix.sleepf 0.1 done;
        Unix._exit 0
      end
    end;
    print_endline "escape"
  | _ -> print_endline "sb"
ML
  printf 'let () = print_endline (1 + "broken")\n' >"$direct_root/t/broken.ml"
  printf 'det\n' >"$direct_root/t/det.expected"
  printf 'cfg\n' >"$direct_root/t/cfg"
  direct_detail=
  direct_probe() { # tag script repeat-args... -> direct_rc, direct_out, direct_dir
    local tag=$1 script=$2
    shift 2
    rm -rf "$TMP/direct-runs-$tag"
    direct_out=$(cd "$direct_root" && PATH=$(dirname "$direct_real"):$PATH \
      DIRECT_COUNTER=$TMP/direct-count-$tag OCANNL_TOOL_TEST_RUNS=$TMP/direct-runs-$tag \
      "${BASH:-bash}" "$script" repeat "$@" 2>&1)
    direct_rc=$?
    direct_dir=$(ls -d "$TMP/direct-runs-$tag"/2*/ 2>/dev/null | head -n 1)
    direct_dir=${direct_dir%/}
  }
  direct_count() { grep -c . "$TMP/direct-count-$1.$2" 2>/dev/null || echo 0; }
  # Three iterations of a (test) stanza's run, a target rule reading the environment, and an
  # alias action: each test ran in the build phase and once per iteration, the replay did what
  # dune's own run did, and no iteration's dune compiled anything.
  direct_probe green "$direct_root/tools/test-run.sh" --direct 3 @t/runtest-det @t/runtest-envp @t/runtest-inalias
  { [ "$direct_rc" = 0 ] &&
    grep -qF "repeat: dune's own run vs replay 1: identical" <<<"$direct_out" &&
    grep -q '^repeat result: IDENTICAL -- the replayed outputs' <<<"$direct_out"; } ||
    direct_detail="green: exit $direct_rc: $direct_out"
  for direct_tag in det env inalias; do
    [ -n "$direct_detail" ] || [ "$(direct_count green "$direct_tag")" = 4 ] ||
      direct_detail="green: $direct_tag ran $(direct_count green "$direct_tag") times, not once in the build and once per iteration"
  done
  [ -n "$direct_detail" ] || { grep -q '^== t/envp.actual' "$direct_dir/iteration-3/outputs" &&
    grep -q '^a b$' "$direct_dir/iteration-3/outputs" && grep -q '^INSIDE_DUNE=' "$direct_dir/iteration-3/outputs" &&
    grep -q '^== t/det.exe.output' "$direct_dir/iteration-3/outputs"; } ||
    direct_detail="green: the replayed outputs are not recorded: $(head -c 400 "$direct_dir/iteration-3/outputs" 2>&1)"
  [ -n "$direct_detail" ] || ! grep -qF 'inalias' "$direct_dir/plan.json" ||
    direct_detail="green: the alias action was planned for a replay, not left to dune: $(cat "$direct_dir/plan.json")"
  # A compile is a process making a compile artifact (`ocamlc.opt (in .)` is dune asking the
  # compiler for its configuration, which every dune invocation does).
  [ -n "$direct_detail" ] || ! cut -f2 "$direct_dir/iteration-2/dune-processes" | grep -qE '\.(cm[a-z]*|o|a|exe)( \+[0-9]+)?$' ||
    direct_detail="green: an iteration's dune compiled: $(cat "$direct_dir/iteration-2/dune-processes")"
  [ -n "$direct_detail" ] || [ ! -e "$direct_dir/build" ] ||
    direct_detail="green: the build directory survived the set"
  # An output that moves between runs.
  if [ -z "$direct_detail" ]; then
    direct_probe flaky "$direct_root/tools/test-run.sh" --direct 2 @t/runtest-flaky
    { [ "$direct_rc" = 1 ] && grep -q '^repeat result: DIFFERING -- ' <<<"$direct_out" &&
      grep -q '^+run 3$' "$direct_dir/diffs/1-2.outputs"; } || direct_detail="flaky: exit $direct_rc: $direct_out"
  fi
  # The witness's negative control: replays that agree with each other but not with dune.
  if [ -z "$direct_detail" ]; then
    direct_probe leak "$direct_root/tools/test-run.sh" --direct 2 @t/runtest-leak
    { [ "$direct_rc" = 1 ] && grep -q '^repeat result: REPLAY DIVERGED -- ' <<<"$direct_out" &&
      grep -q '^-dune$' "$direct_dir/diffs/build-1.outputs"; } || direct_detail="leak: exit $direct_rc: $direct_out"
  fi
  # What a replay cannot reproduce is refused before any iteration.
  if [ -z "$direct_detail" ]; then
    direct_probe sb "$direct_root/tools/test-run.sh" --direct 2 @t/runtest-sb
    { [ "$direct_rc" = 2 ] && grep -q '^repeat result: DIRECT REFUSED -- @t/runtest-sb runs t/sb.actual in a dune sandbox' <<<"$direct_out" &&
      [ ! -e "$direct_dir/iteration-1" ]; } || direct_detail="sandbox: exit $direct_rc: $direct_out"
  fi
  if [ -z "$direct_detail" ]; then
    direct_probe sys "$direct_root/tools/test-run.sh" --direct 2 @t/runtest-sys
    { [ "$direct_rc" = 2 ] && grep -q '^repeat result: DIRECT REFUSED -- the rule for t/sys.actual uses `system`' <<<"$direct_out" &&
      [ ! -e "$direct_dir/iteration-1" ]; } || direct_detail="system: exit $direct_rc: $direct_out"
  fi
  if [ -z "$direct_detail" ]; then
    direct_probe broken "$direct_root/tools/test-run.sh" --direct 2 @t/runtest-broken
    { [ "$direct_rc" = 1 ] && grep -q '^repeat result: BUILD FAILED -- the build phase exited 1, and no process' <<<"$direct_out" &&
      [ ! -e "$direct_dir/iteration-1" ]; } || direct_detail="broken: exit $direct_rc: $direct_out"
  fi
  # A replayed test past the per-test cap ends the set, killed (the watcher times the replay's
  # processes as it times dune's).
  if [ -z "$direct_detail" ]; then
    direct_start=$SECONDS
    direct_probe slow "$direct_root/tools/test-run.sh" --direct --test-cap 2 2 @t/runtest-slow
    { [ "$direct_rc" = 142 ] && [ $((SECONDS - direct_start)) -lt 40 ] &&
      grep -q '^repeat result: TIMEOUT -- in iteration 1 a test outlived the per-test cap (.*det.exe slow)' <<<"$direct_out"; } ||
      direct_detail="slow: exit $direct_rc after $((SECONDS - direct_start))s: $direct_out"
    direct_left=$(ps -A -o pid= -o args= 2>/dev/null | grep -F 'det.exe slow' | grep -v grep)
    [ -z "$direct_left" ] || { direct_detail="slow: the test survived the set: $direct_left"
      kill -KILL $(awk '{print $1}' <<<"$direct_left") 2>/dev/null; }
  fi
  # A replayed test's child that leaves the session holds the descendant witness as dune's
  # actions do, so the set refuses to reuse the build tree under it.
  direct_escape() { # tag script -> sets direct_escaped (the escaped pid, released and gone)
    direct_probe "$1" "$2" --direct 2 @t/runtest-escape
    direct_escaped=$(cat "$TMP/direct-count-$1.escape-pid" 2>/dev/null)
    : >"$TMP/direct-count-$1.escape-release"
    if [ -n "$direct_escaped" ]; then
      for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        kill -0 "$direct_escaped" 2>/dev/null || break
        sleep 0.2
      done
      kill -KILL "$direct_escaped" 2>/dev/null
    fi
  }
  if [ -z "$direct_detail" ]; then
    direct_escape escape "$direct_root/tools/test-run.sh"
    { [ "$direct_rc" = 2 ] && [ -n "$direct_escaped" ] &&
      grep -qF 'iteration 1 left a session-escaped descendant; refusing to reuse' <<<"$direct_out" &&
      [ ! -e "$direct_dir/iteration-2" ]; } || direct_detail="escape: exit $direct_rc, escaped pid ${direct_escaped:-none}: $direct_out"
  fi
  if [ -z "$direct_detail" ]; then
    direct_probe usage "$direct_root/tools/test-run.sh" --direct 2 runtest
    { [ "$direct_rc" = 2 ] && grep -qF 'repeat --direct takes per-test aliases and -j N only, not: runtest' <<<"$direct_out"; } ||
      direct_detail="usage: exit $direct_rc: $direct_out"
  fi
  if [ -z "$direct_detail" ]; then
    direct_probe misplaced "$direct_root/tools/test-run.sh" 2 @t/runtest-det --direct
    { [ "$direct_rc" = 2 ] && grep -qF -- '--direct belongs before the dune arguments' <<<"$direct_out"; } ||
      direct_detail="misplaced: exit $direct_rc: $direct_out"
  fi
  if [ -z "$direct_detail" ]; then report 0 "$direct_label"; else report 1 "$direct_label" "$direct_detail"; fi
  # The counts are what an iteration re-executing the test means: with the replay cut out of the
  # step, the iterations' dune --force re-runs the alias action alone, and the test stanza's run
  # and the target rule stay at their build-phase run.
  direct_mutant=$(mutant no-direct-replay '
    /^        if \[ "\$pre" = replay \]; then$/ && !done { sub(/"\$pre" = replay/, "\"$pre\" = none"); done = 1 }
    { print }')
  if [ -z "$direct_mutant" ] || ! grep -q '"$pre" = none' "$direct_mutant"; then
    report 1 "$direct_nc_label" "the mutant was not made"
  else
    cp "$direct_mutant" "$direct_root/tools/test-run-mutant.sh"
    direct_probe mutant "$direct_root/tools/test-run-mutant.sh" --direct 2 @t/runtest-det @t/runtest-envp @t/runtest-inalias
    if [ "$(direct_count mutant det)" = 1 ] && [ "$(direct_count mutant env)" = 1 ] &&
       [ "$(direct_count mutant inalias)" = 3 ]; then
      report 0 "$direct_nc_label"
    else
      report 1 "$direct_nc_label" "counts det=$(direct_count mutant det) env=$(direct_count mutant env) inalias=$(direct_count mutant inalias): $direct_out"
    fi
  fi
  # The witness reaches a replayed action only because the replay passes on what it inherited.
  direct_fd_root=$TMP/direct-repo-fd
  rm -rf "$direct_fd_root"
  mkdir -p "$direct_fd_root"
  cp -R "$direct_root/tools" "$direct_root/scripts" "$direct_root/t" "$direct_root/dune-project" "$direct_fd_root/"
  rm -f "$direct_fd_root/tools/test-run-mutant.sh"
  perl -0pe 's/stderr=err,\s*pass_fds=INHERITED\)/stderr=err)/' "$HERE/replay-rules.sh" \
    >"$direct_fd_root/tools/replay-rules.sh"
  if grep -q 'pass_fds=INHERITED' "$direct_fd_root/tools/replay-rules.sh" || ! bash -n "$direct_fd_root/tools/replay-rules.sh"; then
    report 1 "$direct_fd_label" "the mutant was not made"
  else
    direct_escape escape-fd "$direct_fd_root/tools/test-run.sh"
    if [ -n "$direct_escaped" ] && [ "$direct_rc" != 2 ] &&
       ! grep -qF 'session-escaped descendant' <<<"$direct_out"; then
      report 0 "$direct_fd_label"
    else
      report 1 "$direct_fd_label" "exit $direct_rc, escaped pid ${direct_escaped:-none}: $direct_out"
    fi
  fi
fi
}
harness_leg_run '84: repeat --direct builds once and replays the rules #repeat' leg_84

test_run_close
