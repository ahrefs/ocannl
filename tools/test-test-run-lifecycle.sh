#!/usr/bin/env bash
# Tests for tools/test-run.sh: the attached `run` and detached `start` lifecycle with its
# signal legs, and the source a run records at launch.
#
#   tools/test-test-run-lifecycle.sh          # run every leg of this member
#   tools/test-test-run-lifecycle.sh --keep   # keep the temp dir for inspection
#   tools/test-test-run-lifecycle.sh --only '^97:'   # only the legs whose names match
#   tools/test-test-run-lifecycle.sh --subject FILE   # test FILE, not tools/test-run.sh
#
# One member of the harness of tools/test-run.sh (gh-ocannl-2002), whose
# others `tools/test-test-run.sh --members` lists; that script runs them all,
# or the ones holding the legs an --only names. The conventions every member
# shares -- --subject, how a leg is named, gated and abandoned, what is tested
# and the fixtures the legs run against -- are tools/test-test-run-common.sh's
# header. Leg numbers are the harness's, not this file's: they stayed as they
# were when it was one file, so a member's are not consecutive.
#
# Groups: #lifecycle (65, 97, 40-47, 74), #source (59-63, 66, 104), #hermetic (70).
#
# Legs:
#  40-47. run/start lifecycle: completion, signals (a group TERM landing inside
#      the attached launcher's own liveness probe included, with its
#      unsettled-probe control; gh-ocannl-1362), stop, competing launches,
#      orphan/background groups, publication window, last and legacy locks,
#      and byte-identical source-tree preservation. Each signal reaches a
#      launcher that inherited it ignored, as from a background `&`,
#      `bg-run.sh spawn` or `nohup`, and reset it, so the legs pass whatever
#      this harness inherited (gh-ocannl-1236); a fixture that never becomes
#      ready fails and ends its leg, never the harness.
#  74. fork TERM is deferred until the parent owns its child pid and the child
#      resets its handlers; no-mask and inherited-handler controls prove both.
#  65. a `stop`, or a signal to the launcher alone, while the backends
#      resolve (the published run's first phase) cancels the run: exit 143,
#      dune never started and the log says so, the worktree idle; a hung
#      reader is cut off by the cap, a reader's leftover reaped with the run.
#  97 sits after leg 65: a TERM to the supervisor, while the backends resolve
#      or while a TERM-ignoring dune runs, each leaving a session-escaped
#      descendant holding the lock, settles inside the fleet wrappers' 10 s
#      grace (gh-ocannl-1808): verdict 143, the log ending `exit: 143`, the
#      escapee and the group gone, the lock idle; the slowest-actions reader
#      runs after, bounded to 3 s; a `fresh` run's own build directory goes
#      with it all the same. A HUP to the attached launcher of a `run`
#      in the foreground, relayed as TERM, settles the same way and the
#      launcher returns 143, CANCELLED (gh-ocannl-1906). Mutation controls:
#      no lock census leaves the escapee holding the lock; the reader before
#      the verdict misses the 10 s.
#  59-63 sit after leg 47: the source a run tested, recorded at launch
#      (gh-ocannl-992), against a committed fixture checkout.
#  59. a clean checkout records its HEAD as `head` and an empty `dirty`.
#  60. a dirty one lists a modified tracked path and an untracked one (the
#      latter although the checkout's config hides untracked files), and the
#      launch leaves the checkout's index as it found it.
#  61. an inherited GIT_DIR/GIT_WORK_TREE naming another repository does not
#      redirect the record.
#  62. outside Git, and in a copy sitting below a checkout's top level, neither
#      file is recorded and the run launches as before.
#  63. a commit landing between the HEAD read and the status is re-read: the
#      record never pairs the older HEAD with the newer tree's status.
#  66 sits after leg 63: the digest's `source:` line reads the record -- the
#      commit and `(clean)`, or `+ N uncommitted paths`; nothing unrecorded.
# 104 sits after leg 66: a checkout with unmerged paths is refused
#      (gh-ocannl-1991). The fixture checkout runs green before the merge (the
#      control); stopped on a conflict, `run`, `start` and `repeat` each exit
#      2 naming the unmerged path alone, dune not started, the digest and
#      `wait last` under UNMERGED REFUSED, and `plan` says run would refuse;
#      OCANNL_TOOL_TEST_UNMERGED=allow runs it, the `source:` line counting
#      the unmerged path; resolved and staged, it runs as any dirty tree.

. "$(cd "$(dirname "$0")" && pwd)/test-test-run-common.sh"

shared_through native

# Leg 65: the run's first phase. The backends resolve in the supervisor's
# child, after the run is published (gh-ocannl-1106), so a `stop` of the run
# reaches its supervisor as for any run: the child and the reader it waits on
# are reaped, the launch exits 143 with the verdict recorded, dune never
# started and the log says so, and the worktree is idle again. A reader held
# asleep stands for a slow build.
leg_65() {
plan_stop_runs=$TMP/argv-runs-plan-stop
plan_stop_detail=
mkdir -p "$plan_stop_runs"
: >"$TMP/plan-stop.counter"
: >"$TMP/plan-stop.calls"
: >"$TMP/plan-stop.reach"
FAKE_REACH_CALLS=$TMP/plan-stop.reach FAKE_REACH_SLEEP=30 \
REPEAT_TEST_MODE=stable REPEAT_TEST_COUNTER=$TMP/plan-stop.counter REPEAT_TEST_CALLS=$TMP/plan-stop.calls \
REPEAT_TEST_WAIT_PREFIX= REPEAT_TEST_WAIT_AT= REPEAT_TEST_ORPHAN_PID= REPEAT_TEST_ORPHAN_REAPED= \
REPEAT_TEST_DIFF_WAIT_PREFIX= REPEAT_TEST_REAL_DIFF="$(command -v diff)" \
OCANNL_TOOL_TEST_RUNS=$plan_stop_runs OCANNL_TOOL_DXG_DEVICE=$dxg_present OCANNL_BACKEND=cuda \
OCANNL_TOOL_KFD_TOPOLOGY=$kfd_absent OCANNL_TOOL_NVIDIA_DEVICE=$nv_absent PATH=$repeat_bin:$PATH \
  "$repeat_root/tools/test-run.sh" run build @cheap >"$TMP/plan-stop.out" 2>"$TMP/plan-stop.err" &
plan_stop_pid=$!
plan_stop_marker=
for _ in $(seq 1 100); do
  plan_stop_marker=$(ls "$plan_stop_runs"/2*/pid 2>/dev/null | head -n 1)
  [ -n "$plan_stop_marker" ] && [ -s "$TMP/plan-stop.reach" ] && break
  sleep 0.1
done
if [ -z "$plan_stop_marker" ] || [ ! -s "$TMP/plan-stop.reach" ]; then
  plan_stop_detail="the launch never recorded its supervisor: $(cat "$TMP/plan-stop.err")"
  kill -TERM "$plan_stop_pid" 2>/dev/null
  wait "$plan_stop_pid" 2>/dev/null
else
  plan_stop_run=$(dirname "$plan_stop_marker")
  plan_stop_out=$(OCANNL_TOOL_TEST_RUNS=$plan_stop_runs "$repeat_root/tools/test-run.sh" stop "$plan_stop_run" 2>&1)
  wait "$plan_stop_pid"
  plan_stop_rc=$?
  [ "$plan_stop_rc" = 143 ] || plan_stop_detail="the launch exited $plan_stop_rc (want 143): $(cat "$TMP/plan-stop.err")"
  [ -n "$plan_stop_detail" ] || case $plan_stop_out in
    *"sent TERM; confirm with"*) ;;
    *) plan_stop_detail="stop said: $plan_stop_out" ;;
  esac
  [ -n "$plan_stop_detail" ] || { [ "$(cat "$plan_stop_run/exit" 2>/dev/null)" = 143 ] &&
    grep -q "cancelled while the launch resolved the batch's backends; dune was not started" "$plan_stop_run/log"; } ||
    plan_stop_detail="the cancelled run's verdict or log does not say so: $(cat "$plan_stop_run/log" 2>/dev/null)"
  [ -n "$plan_stop_detail" ] || [ ! -s "$TMP/plan-stop.calls" ] ||
    plan_stop_detail="dune ran: $(cat "$TMP/plan-stop.calls")"
  [ -n "$plan_stop_detail" ] ||
    OCANNL_TOOL_TEST_RUNS=$plan_stop_runs "$repeat_root/tools/test-run.sh" idle 2>/dev/null ||
    plan_stop_detail="the worktree is not idle after the stop"
fi
# The resolution shares the run's cap: a reader that hangs is cut off, the
# batch reads as unread (every backend, so the dxg cap), and the launch goes
# on instead of stranding (Codex review round 1 on PR #832). The suite's own
# verdict under what is left of a 3-second cap is not the point here.
if [ -z "$plan_stop_detail" ]; then
  plan_hang_start=$SECONDS
  export FAKE_REACH_SLEEP=30
  dxg_probe plan-hang "$dxg_present" cc run --cap 3 build @cheap
  unset FAKE_REACH_SLEEP
  plan_hang_took=$((SECONDS - plan_hang_start))
  [ "$plan_hang_took" -lt 20 ] ||
    plan_stop_detail="a hung reader held the launch ${plan_hang_took}s under --cap 3"
  # The resolution spent the whole cap, so the verdict is the cap's and dune
  # never starts on a fresh budget (Codex review round 2 on PR #832).
  [ -n "$plan_stop_detail" ] || { [ "$argv_rc" = 142 ] && [ -z "$argv_calls" ] &&
    grep -q "verdict: TIMEOUT" <<<"$argv_out"; } ||
    plan_stop_detail="a hung reader spending the cap: exit $argv_rc (want 142); calls: ${argv_calls:-<none>}; stdout: $argv_out"
  [ -n "$plan_stop_detail" ] || { grep -q "the cap expired while the launch resolved the batch's backends; dune was not started" "$argv_dir/log" &&
    [ "$(cat "$argv_dir/exit" 2>/dev/null)" = 142 ]; } ||
    plan_stop_detail="a hung reader: the log or the verdict does not say so: $(cat "$argv_dir/log" 2>/dev/null)"
fi
# A reader that exits leaving a background descendant behind does not leave it
# holding the worktree lock: it is in the run's process group, which the
# supervisor reaps with dune's (Codex review round 5 on PR #832).
if [ -z "$plan_stop_detail" ]; then
  export FAKE_REACH_LEAVE=30
  argv_runs=$TMP/argv-runs-plan-leave dxg_probe plan-leave "$dxg_present" cuda run build @cheap
  unset FAKE_REACH_LEAVE
  argv_runs=
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build -j 2 @cheap" ] &&
    OCANNL_TOOL_TEST_RUNS=$TMP/argv-runs-plan-leave "$repeat_root/tools/test-run.sh" idle 2>/dev/null; } ||
    plan_stop_detail="a reader's leftover descendant: exit $argv_rc; calls: ${argv_calls:-<none>}; or the worktree is not idle after the run"
fi
# A signal to the launcher alone -- not the terminal's group, not a stop --
# ends the resolution at once: the launcher relays it to the supervisor, as it
# does while dune runs (Codex review round 2 on PR #832). A reader that
# ignores TERM gets the supervisor's grace and then KILL, so it cannot keep
# the worktree locked after the run was cancelled (Codex review round 3).
for plan_sig_kind in plain stubborn; do
  [ -z "$plan_stop_detail" ] || break
  plan_sig_ignore=
  [ "$plan_sig_kind" = plain ] || plan_sig_ignore=1
  plan_sig_runs=$TMP/argv-runs-plan-sig-$plan_sig_kind
  mkdir -p "$plan_sig_runs"
  : >"$TMP/plan-sig.counter"
  : >"$TMP/plan-sig.calls"
  plan_sig_start=$SECONDS
  : >"$TMP/plan-sig.reach"
  FAKE_REACH_CALLS=$TMP/plan-sig.reach FAKE_REACH_SLEEP=30 FAKE_REACH_IGNORE_TERM=$plan_sig_ignore \
  REPEAT_TEST_MODE=stable REPEAT_TEST_COUNTER=$TMP/plan-sig.counter REPEAT_TEST_CALLS=$TMP/plan-sig.calls \
  REPEAT_TEST_WAIT_PREFIX= REPEAT_TEST_WAIT_AT= REPEAT_TEST_ORPHAN_PID= REPEAT_TEST_ORPHAN_REAPED= \
  REPEAT_TEST_DIFF_WAIT_PREFIX= REPEAT_TEST_REAL_DIFF="$(command -v diff)" \
  OCANNL_TOOL_TEST_RUNS=$plan_sig_runs OCANNL_TOOL_DXG_DEVICE=$dxg_present OCANNL_BACKEND=cuda \
  OCANNL_TOOL_KFD_TOPOLOGY=$kfd_absent OCANNL_TOOL_NVIDIA_DEVICE=$nv_absent PATH=$repeat_bin:$PATH \
    "$repeat_root/tools/test-run.sh" run build @cheap >"$TMP/plan-sig.out" 2>"$TMP/plan-sig.err" &
  plan_sig_pid=$!
  plan_sig_marker=
  for _ in $(seq 1 100); do
    plan_sig_marker=$(ls "$plan_sig_runs"/2*/pid 2>/dev/null | head -n 1)
    [ -n "$plan_sig_marker" ] && [ -s "$TMP/plan-sig.reach" ] && break
    sleep 0.1
  done
  if [ -z "$plan_sig_marker" ] || [ ! -s "$TMP/plan-sig.reach" ]; then
    plan_stop_detail="the signalled launch never recorded its supervisor: $(cat "$TMP/plan-sig.err")"
    kill -TERM "$plan_sig_pid" 2>/dev/null
    wait "$plan_sig_pid" 2>/dev/null
  else
    kill -TERM "$plan_sig_pid"
    wait "$plan_sig_pid"
    plan_sig_rc=$?
    plan_sig_took=$((SECONDS - plan_sig_start))
    { [ "$plan_sig_rc" = 143 ] && [ "$plan_sig_took" -lt 15 ] && [ ! -s "$TMP/plan-sig.calls" ] &&
      [ "$(cat "$(dirname "$plan_sig_marker")/exit" 2>/dev/null)" = 143 ] &&
      grep -q "dune was not started" "$(dirname "$plan_sig_marker")/log" &&
      OCANNL_TOOL_TEST_RUNS=$plan_sig_runs "$repeat_root/tools/test-run.sh" idle 2>/dev/null; } ||
      plan_stop_detail="a TERM to the launcher alone ($plan_sig_kind reader): not idle, or exit $plan_sig_rc (want 143) after ${plan_sig_took}s; calls: $(cat "$TMP/plan-sig.calls"); stderr: $(cat "$TMP/plan-sig.err")"
  fi
done
if [ -z "$plan_stop_detail" ]; then
  report 0 "plan: a stop or a signal while the backends resolve cancels the run before dune, and the worktree is idle"
else
  report 1 "plan: a stop or a signal while the backends resolve cancels the run before dune, and the worktree is idle" "$plan_stop_detail"
fi
}
harness_leg_run '65: a stop while the backends resolve cancels the run #lifecycle' leg_65

# Leg 97: a cancellation fits the fleet wrappers' grace (gh-ocannl-1808).
# `execution slot` and `bg-run.sh stop` KILL ten seconds after their TERM, so
# the supervisor, TERMed, publishes its verdict and frees the worktree lock
# inside that: the run's group reaped on a short grace, then every other
# holder of the lock -- here a session-escaped descendant, TERM ignored,
# still holding inherited fd 9 -- found by the census and KILLed, then the
# verdict, then the lock, and only after them the slowest-actions reader,
# bounded (a stuck one stands in). The phases: TERM while the batch's
# backends resolve (the reader that spawned the escapee asleep), while a
# TERM-ignoring dune runs, and while a `fresh` one does; and a HUP to the
# launcher of a `run` started in the foreground while that dune runs -- a
# closed terminal, which the launcher relays to the supervisor as TERM
# (gh-ocannl-1906), so the same census has to reap the escapee there. Within
# ten seconds of the signal: the verdict is 143, the log ends with `exit:
# 143`, the escapee and the group are gone, the lock is idle; the HUPed
# launcher itself returns 143 with the cancellation. Two mutation controls,
# each one line of the shipping supervisor replaced: a census cut out leaves
# the escapee holding the lock in every phase but `fresh`'s, and the
# cancellation finished in the old order (reader before verdict) publishes no
# verdict within ten seconds of a TERM to dune.
leg_97() {
local label="a TERMed run publishes its verdict and frees the lock within ten seconds, the reader after"
local nc_census="negative control: a supervisor without the lock census leaves the escapee holding the lock"
local nc_order="negative control: the reader before the verdict misses the ten seconds"
local skip_why= census_line order_line
case $(uname -s) in MSYS* | MINGW*) skip_why="POSIX sessions and process groups (setsid) are not native here" ;; esac
if [ -z "$skip_why" ] && [ "$have_state" = 0 ]; then
  skip_why="no way to read a process's state, so a reaped escapee cannot be told from a live one"
elif [ -z "$skip_why" ] && [ ! -d /proc/self/fdinfo ] && ! command -v lsof >/dev/null 2>&1; then
  skip_why="neither /proc fdinfo nor lsof here: the supervisor has no lock census"
fi
if [ -n "$skip_why" ]; then
  skip "$label" "$skip_why"; skip "$nc_census" "$skip_why"; skip "$nc_order" "$skip_why"
  return
fi
census_line='    $reap_holders->() if $cancel;'
order_line='    $finish->($code, $cancel);'
local slow_reader=$TMP/term-grace-reader.sh
printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$$" >"$TERM_GRACE_READER_PID"\nexec sleep 120\n' >"$slow_reader"
local trace=$TMP/term-grace-trace.csexp
printf 'not a trace; the stand-in reader never reads it\n' >"$trace"
local detail= census_detail= order_detail=
local phase variant tag runs pre esc run_dir pg sup t0 i ok took last launcher launch_rc
for variant in fix census order; do
  for phase in resolve dune fresh hup; do
    case $variant/$phase in order/resolve | order/hup | census/fresh | order/fresh) continue ;; esac
    tag=term-grace-$variant-$phase
    runs=$TMP/argv-runs-$tag
    pre=$TMP/$tag
    esc=$pre.escape
    rm -rf "$runs"; mkdir -p "$runs"
    rm -f "$esc" "$pre".* 2>/dev/null
    : >"$pre.calls"; : >"$pre.counter"; : >"$pre.reach"
    cp "$SRC" "$repeat_root/tools/test-run.sh"
    case $variant in
      census | order)
        [ "$variant" = census ] && last=$census_line || last=$order_line
        awk -v line="$last" -v v="$variant" '
          $0 == line { n++; print (v == "census" ? "    1;" : "    $finish->($code, 0);"); next }
          { print }
          END { exit n == 1 ? 0 : 1 }' "$SRC" >"$repeat_root/tools/test-run.sh" ||
          { report 1 "$label" "the $variant control line was not found once in $SRC"; cp "$SRC" "$repeat_root/tools/test-run.sh"; return; }
        ;;
    esac
    local mode=stable reach_sleep= reach_escape= launch=(start build @cheap) attach=()
    if [ "$phase" = resolve ]; then reach_sleep=30 reach_escape=$esc; else mode=term_escape; fi
    [ "$phase" != fresh ] || launch=(fresh @test/operations/runtest-alpha)
    # The foreground launcher traps HUP only if it did not start with HUP
    # ignored, which a harness under nohup or bg-run.sh would hand it.
    [ "$phase" != hup ] || { launch=(run build @cheap); attach=(perl -e '$SIG{HUP} = "DEFAULT"; exec @ARGV'); }
    FAKE_REACH_CALLS=$pre.reach FAKE_REACH_SLEEP=$reach_sleep FAKE_REACH_ESCAPE=$reach_escape \
    REPEAT_TEST_MODE=$mode REPEAT_TEST_COUNTER=$pre.counter REPEAT_TEST_CALLS=$pre.calls \
    REPEAT_TEST_ESCAPE_PID=$esc REPEAT_TEST_ESCAPE_READY=$pre.ready REPEAT_TEST_TRACE=$trace \
    REPEAT_TEST_WAIT_PREFIX= REPEAT_TEST_WAIT_AT= REPEAT_TEST_ORPHAN_PID= REPEAT_TEST_ORPHAN_REAPED= \
    REPEAT_TEST_DIFF_WAIT_PREFIX= REPEAT_TEST_REAL_DIFF="$(command -v diff)" \
    TERM_GRACE_READER_PID=$pre.reader OCANNL_TOOL_SLOWEST_READER=$slow_reader REPEAT_TEST_MKBUILD=1 \
    OCANNL_TOOL_TEST_RUNS=$runs OCANNL_TOOL_DXG_DEVICE=$dxg_present OCANNL_BACKEND=cuda \
    OCANNL_TOOL_KFD_TOPOLOGY=$kfd_absent OCANNL_TOOL_NVIDIA_DEVICE=$nv_absent PATH=$repeat_bin:$PATH \
      ${attach[@]+"${attach[@]}"} "$repeat_root/tools/test-run.sh" "${launch[@]}" >"$pre.out" 2>"$pre.err" &
    launcher=$!
    # A `start` returns once the batch has resolved, and a `run` stays
    # attached, so neither is waited on here: the run is found by its
    # supervisor's record.
    ok= run_dir=
    for ((i = 0; i < 300; i++)); do
      run_dir=$(ls -d "$runs"/2*/ 2>/dev/null | head -n 1)
      run_dir=${run_dir%/}
      if [ -n "$run_dir" ] && [ -s "$esc" ] && [ -s "$run_dir/pid" ] &&
         { [ "$phase" = resolve ] && [ -s "$pre.reach" ] || [ -e "$pre.ready" ]; } &&
         { [ "$phase" != fresh ] || [ -d "$run_dir/build" ]; }; then ok=1; break; fi
      sleep 0.05
    done
    [ -s "$esc" ] && ps_token "$(cat "$esc")" >"$esc.token"
    if [ -z "$ok" ] || [ -z "$run_dir" ]; then
      detail="$variant/$phase: the fixture never got ready: $(cat "$pre.out" "$pre.err" "$run_dir/log" 2>/dev/null)"
      term_grace_cleanup "$run_dir" "$esc" "$pre.reader"
      kill -KILL "$launcher" 2>/dev/null; wait "$launcher" 2>/dev/null
      break 2
    fi
    sup=$(cat "$run_dir/pid")
    pg=$(cat "$run_dir/pgid" 2>/dev/null)
    t0=$SECONDS
    if [ "$phase" = hup ]; then kill -HUP "$launcher"; else kill -TERM "$sup"; fi
    took=
    # Ten seconds, measured from the TERM: 100 polls, each at least 0.1s.
    for ((i = 0; i < 100; i++)); do
      if [ -f "$run_dir/exit" ] && term_grace_dead "$(cat "$esc")" &&
         { [ -z "$pg" ] || ! group_alive "$pg"; } &&
         [ "$(OCANNL_TOOL_TEST_RUNS=$runs "$repeat_root/tools/test-run.sh" lock-status 2>/dev/null)" = idle ]; then
        took=$((SECONDS - t0)); break
      fi
      sleep 0.1
    done
    [ -n "$took" ] && [ $((SECONDS - t0)) -gt 10 ] && took=
    # The attached launcher returns once the supervisor has gone, digesting
    # its verdict.
    launch_rc=
    if [ "$phase" = hup ]; then
      for ((i = 0; i < 100; i++)); do term_grace_dead "$launcher" && break; sleep 0.1; done
      if term_grace_dead "$launcher"; then wait "$launcher"; launch_rc=$?; fi
    fi
    case $variant in
      fix)
        if [ -z "$took" ]; then
          detail="$variant/$phase: not settled 10s after the TERM: exit=$(cat "$run_dir/exit" 2>/dev/null || echo none) escapee=$(pstate "$(cat "$esc")") group=$(group_alive "$pg" && echo alive || echo gone) lock=$(OCANNL_TOOL_TEST_RUNS=$runs "$repeat_root/tools/test-run.sh" lock-status 2>&1); log: $(tail -n 5 "$run_dir/log")"
        elif [ "$(cat "$run_dir/exit")" != 143 ] || [ "$(tail -n 1 "$run_dir/log")" != "exit: 143" ]; then
          detail="$variant/$phase: verdict $(cat "$run_dir/exit"), log ends: $(tail -n 1 "$run_dir/log")"
        elif [ "$phase" = resolve ] && { [ -s "$pre.calls" ] || ! grep -q "dune was not started" "$run_dir/log"; }; then
          detail="$variant/$phase: dune ran, or the log does not say it did not: calls $(cat "$pre.calls"); $(cat "$run_dir/log")"
        elif [ "$phase" = hup ] && { [ "$launch_rc" != 143 ] || ! grep -q 'verdict: CANCELLED' "$pre.out"; }; then
          detail="$variant/$phase: the HUPed launcher returned ${launch_rc:-nothing within ten seconds} (want 143, CANCELLED): $(tail -n 3 "$pre.out" "$pre.err" 2>/dev/null)"
        fi
        # The supervisor itself outlives the verdict only by its bounded tail:
        # the stuck reader is KILLed past three seconds, and says so.
        for ((i = 0; i < 100; i++)); do term_grace_dead "$sup" && break; sleep 0.1; done
        if [ -z "$detail" ] && ! term_grace_dead "$sup"; then
          detail="$variant/$phase: the supervisor outlived its verdict by more than ten seconds"
        elif [ -z "$detail" ] && [ "$phase" = fresh ] &&
             { [ -e "$run_dir/build" ] || [ -e "$run_dir/build.discard" ]; }; then
          detail="$variant/$phase: the fresh build directory outlived the supervisor: $(ls "$run_dir" | tr '\n' ' ')"
        elif [ -z "$detail" ] && [ "$phase" != resolve ]; then
          { [ "$(cat "$run_dir/slowest-skipped" 2>/dev/null)" = "the reader did not finish within 3s" ] &&
            term_grace_dead "$(cat "$pre.reader" 2>/dev/null)" && [ ! -e "$run_dir/trace.csexp" ]; } ||
            detail="$variant/$phase: the reader's tail: skip record '$(cat "$run_dir/slowest-skipped" 2>&1)', reader $(pstate "$(cat "$pre.reader" 2>/dev/null)")"
        fi
        ;;
      census)
        if [ -n "$took" ]; then
          census_detail="$census_detail $phase: settled in ${took}s without the census, so the leg cannot tell it;"
        elif term_grace_dead "$(cat "$esc")"; then
          census_detail="$census_detail $phase: the escapee died without the census;"
        fi ;;
      order)
        [ -f "$run_dir/exit" ] && [ "$((SECONDS - t0))" -le 10 ] &&
          order_detail="the verdict published within ten seconds with the reader first"
        [ -n "$order_detail" ] || [ -f "$run_dir/exit" ] || [ -s "$pre.reader" ] ||
          order_detail="the stand-in reader never ran, so the control proves nothing"
        ;;
    esac
    term_grace_cleanup "$run_dir" "$esc" "$pre.reader"
    kill -KILL "$launcher" 2>/dev/null; wait "$launcher" 2>/dev/null
    [ -z "$detail" ] || break 2
  done
done
cp "$SRC" "$repeat_root/tools/test-run.sh"
if [ -z "$detail" ]; then report 0 "$label"; else report 1 "$label" "$detail"; fi
if [ -n "$detail" ]; then
  skip "$nc_census" "the leg itself failed"; skip "$nc_order" "the leg itself failed"
  return
fi
if [ -z "$census_detail" ]; then report 0 "$nc_census"; else report 1 "$nc_census" "$census_detail"; fi
if [ -z "$order_detail" ]; then report 0 "$nc_order"; else report 1 "$nc_order" "$order_detail"; fi
}
term_grace_dead() { # <pid>: 0 iff gone (or a zombie no one here can reap)
  [ -n "$1" ] || return 0
  case $(pstate "$1") in '' | Z*) return 0 ;; *) return 1 ;; esac
}
term_grace_cleanup() { # <run dir> <escapee pid file> <reader pid file>
  local i
  harness_kill_recorded "$2" "$2.token"
  [ -s "$3" ] && kill -KILL "$(cat "$3")" 2>/dev/null
  [ -n "$1" ] || return 0
  harness_kill_recorded "$1/pgid" "$1/gtoken" group
  for ((i = 0; i < 100; i++)); do
    [ -s "$1/pid" ] && ! term_grace_dead "$(cat "$1/pid")" || break
    sleep 0.1
  done
  harness_kill_recorded "$1/pid" "$1/ptoken"
}
harness_leg_run '97: a TERMed run publishes its verdict and frees the lock inside the wrappers grace #lifecycle' leg_97

shared_through synthetic_trace
# ---------------------------------------------------------------------------
# Legs 40-47: attached run and detached start lifecycle (gh-ocannl-967).
# All fixture work is outside the source tree, including its run store. The
# ready marker is written by Dune AFTER the supervisor/group records exist.
# Direct children remain unreaped until checked; other processes are signalled
# only under their recorded start token. Every fixture wait has a deadline.
# ---------------------------------------------------------------------------
life_root=$TMP/lifecycle-repo
life_runs=$TMP/lifecycle-runs
life_bin=$TMP/lifecycle-bin
life_prefix=$TMP/lifecycle
life_pid= life_run=
mkdir -p "$life_root/tools" "$life_root/scripts" "$life_runs" "$life_bin"
cp "$SRC" "$life_root/tools/test-run.sh"
stage_shared "$life_root"
cp -R "$life_root" "$TMP/lifecycle-before"
cat >"$life_bin/dune" <<'EOF'
#!/usr/bin/env bash
case $1 in
  ok) echo 'built ok' ;;
  hold)
    touch "$LIFECYCLE_PREFIX.ready"
    exec perl -e 'while (!-e "$ARGV[0].release") { select undef, undef, undef, .05 }' "$LIFECYCLE_PREFIX"
    ;;
  background)
    # The child closes the lock as well: lock freedom alone cannot prove the
    # supervisor reaped a descendant that could still modify the build tree.
    perl -e '$SIG{TERM}="IGNORE"; open my $f, ">", "$ARGV[0].child" or die $!;
      print {$f} "$$\n"; close $f; sleep 60' "$LIFECYCLE_PREFIX" 9>&- </dev/null >/dev/null 2>&1 &
    while [ ! -s "$LIFECYCLE_PREFIX.child" ]; do sleep .05; done
    . "$LIFECYCLE_TOKEN_HELPER"
    ps_token "$(cat "$LIFECYCLE_PREFIX.child")" >"$LIFECYCLE_PREFIX.child-token"
    echo 'left a background child'
    ;;
  *) exit 3 ;;
esac
EOF
chmod +x "$life_bin/dune"
life() {
  OCANNL_TOOL_TEST_RUNS=$life_runs LIFECYCLE_PREFIX=$life_prefix \
    LIFECYCLE_TOKEN_HELPER=$TMP/ps_token.sh PATH=$life_bin:$PATH \
    "$life_root/tools/test-run.sh" "$@"
}
life_capture() {
  life "$@" >"$TMP/lifecycle.out" 2>"$TMP/lifecycle.err"
  life_rc=$?
}
life_check() { # expected status, output pattern, label
  if [ "$life_rc" = "$1" ] && grep -q "$2" "$TMP/lifecycle.out"; then
    report 0 "lifecycle: $3"
  else
    report 1 "lifecycle: $3" "exit $life_rc; $(cat "$TMP/lifecycle.out" "$TMP/lifecycle.err")"
  fi
}
life_dead() { case $(pstate "$1") in ''|Z) return 0 ;; *) return 1 ;; esac; }
life_await_dead() {
  local i
  for ((i=0; i<300; i++)); do life_dead "$1" && return 0; sleep .05; done
  return 1
}
life_wait_child() {
  harness_wait_child "$life_pid" life_dead "lifecycle: owned launcher finishes within 15 seconds"
  life_rc=$harness_child_rc
  life_pid=
}
lifecycle_cleanup() {
  local d
  # Recorded supervisors/groups belong to this private fixture. Identity is
  # checked even on failure: mutation controls deliberately leave survivors.
  for d in "$life_runs"/*/; do
    [ -d "$d" ] || continue
    harness_kill_recorded "$d/pgid" "$d/gtoken" group
    harness_kill_recorded "$d/pid" "$d/ptoken"
  done
  harness_kill_recorded "$life_prefix.child" "$life_prefix.child-token"
  if [ -n "${fork_prefix:-}" ]; then
    harness_kill_recorded "$fork_prefix.child" "$fork_prefix.child-token"
  fi
  if [ -n "$life_pid" ]; then kill -KILL "$life_pid" 2>/dev/null; wait "$life_pid" 2>/dev/null; life_pid=; fi
}
life_start() {
  rm -f "$life_prefix.ready" "$life_prefix.release"
  life_capture start --cap 30 hold
  life_check 0 '^started: ' 'start returns after publication'
  life_run=$(life paths run last)
  if ! await_fixture_ready "$life_prefix.ready"; then
    report 1 'lifecycle: Dune reached readiness'; return 1
  fi
}
life_no_survivors() {
  local pg idle
  pg=$(cat "$life_run/pgid")
  idle=$(life lock-status "$life_run"); life_rc=$?
  if [ "$life_rc" = 0 ] && [ "$idle" = idle ] && ! group_alive "$pg"; then
    report 0 "lifecycle: $1 leaves no live group or lock holder"
  else
    report 1 "lifecycle: $1 leaves no live group or lock holder" "group=$pg lock=$life_rc:$idle"
  fi
}
# A fixture that never arrives fails its leg and ends it, never the harness
# (gh-ocannl-1236): the legs after it still run. Each lifecycle leg's body is a
# function so that `return 1` can end it; the leg is reported abandoned, and
# its recorded supervisors and groups are reaped so the lock it held cannot
# fail the legs that follow on the same fixture.
lifecycle_abandoned() { # LEG
  report 1 "lifecycle: $1 abandoned after a fixture failure; its later checks did not run"
  lifecycle_cleanup
}
lifecycle_core() {
life_capture run --cap 30 ok
life_check 0 'verdict: pass' 'attached green run returns its verdict'
life_start || return 1
life_capture status last
life_check 3 '^running: ' 'status observes the detached owner'
life_capture list
life_check 0 'running' 'list observes the detached owner'
life_count_before=$(find "$life_runs" -mindepth 1 -maxdepth 1 -type d | wc -l)
life_capture run --cap 30 ok
if [ "$life_rc" = 2 ] && grep -Fq "status $life_run" "$TMP/lifecycle.err" \
   && [ "$(find "$life_runs" -mindepth 1 -maxdepth 1 -type d | wc -l)" = "$life_count_before" ]; then
  report 0 'lifecycle: competing launch names the active run and creates no run directory'
else
  report 1 'lifecycle: competing launch names the active run and creates no run directory' "$(cat "$TMP/lifecycle.err")"
fi
touch "$life_prefix.release"
life_capture wait last --timeout 15
life_check 0 'verdict: pass' 'wait observes detached completion'
if [ "$(tail -n 1 "$life_run/log")" = 'exit: 0' ] && [ "$(cat "$life_run/exit")" = 0 ]; then
  report 0 'lifecycle: completion publishes both verdict and final log sentinel'
else report 1 'lifecycle: completion publishes both verdict and final log sentinel'; fi
life_no_survivors 'normal completion'
life_start || return 1
life_capture stop last
life_check 0 '^sent TERM; confirm' 'stop signals a live supervisor'
life_capture wait last --timeout 15
life_check 143 'verdict: CANCELLED' 'stop records cancellation'
life_capture stop last
life_check 0 '^already finished:' 'stop on a completed run is idempotent'
life_no_survivors 'stop'
}
leg_40_47() {
lifecycle_core || lifecycle_abandoned '40-47 (start/status/stop)'
}
harness_leg_run '40-47: run/start lifecycle #lifecycle' leg_40_47

# Leg 74: inject a real TERM after fork returns but BEFORE its pid assignment.
# No seam exists in production: only this fixture copy replaces that expression.
# The child records its own identity before pausing, so every control survivor
# is contained even if it never reached the ordinary pgid publication. Every
# copy also cuts the cancellation's lock census (gh-ocannl-1808, leg 97), a
# second defence that would reap the unmasked child too: the legs pin the
# mask alone, and the no-mask control still tells it.
leg_74() {
case $(uname -s) in MSYS*|MINGW*) fork_posix=0 ;; *) fork_posix=1 ;; esac
if [ "$fork_posix" = 1 ] && [ "$have_state" = 1 ]; then
  for fork_case in parent child no-mask inherited-handler; do
    fork_prefix=$TMP/fork-$fork_case
    fork_mode=parent
    case $fork_case in child|inherited-handler) fork_mode=child ;; esac
    awk -v mode="$fork_mode" -v control="$fork_case" '
      /POSIX::sigprocmask/ && control == "no-mask" { dropping=1; next }
      dropping { dropping=0; next }
      $0 == "    $reap_holders->() if $cancel;" { census++; print "    1;"; next }
      /\$SIG\{\$_\} = "DEFAULT" for qw\(ALRM INT TERM HUP\)/ && control == "inherited-handler" { next }
      /^  \$pid = fork\(\);$/ {
        found++
        print "  $pid = do {"
        print "    my $forked = fork();"
        print "    if (defined $forked && !$forked) {"
        print "      $write->(\"$ENV{FORK_PREFIX}.child-token\", census_token($$) . \"\\n\");"
        print "      $write->(\"$ENV{FORK_PREFIX}.child\", \"$$\\n\");"
        if (mode == "child") print "      kill(\"TERM\", $$);"
        print "      select undef, undef, undef, .3;"
        print "    } elsif ($forked) {"
        print "      for (1 .. 100) { last if -s \"$ENV{FORK_PREFIX}.child\"; select undef, undef, undef, .01 }"
        if (mode == "parent") print "      kill(\"TERM\", $$);"
        print "    }"
        print "    $forked;"
        print "  };"
        next
      }
      { print }
      END { if (found != 1 || census != 1) exit 1 }
    ' "$SRC" >"$life_root/tools/test-run.sh" || {
      cp "$SRC" "$life_root/tools/test-run.sh"
      harness_abandon "fork: the seam '  \$pid = fork();' or the census line is not in $SRC exactly once"
      return
    }
    rm -f "$life_prefix.ready" "$life_prefix.release"
    FORK_PREFIX=$fork_prefix OCANNL_TOOL_TEST_RUNS=$life_runs LIFECYCLE_PREFIX=$life_prefix \
      LIFECYCLE_TOKEN_HELPER=$TMP/ps_token.sh PATH=$life_bin:$PATH \
      "$life_root/tools/test-run.sh" run --cap 10 hold >"$TMP/lifecycle.out" 2>"$TMP/lifecycle.err" &
    life_pid=$!
    life_wait_child
    life_run=$(life paths run last)
    fork_child=$(cat "$fork_prefix.child" 2>/dev/null)
    fork_ok=0
    if [ "$life_rc" = 143 ] && [ "$(cat "$life_run/exit" 2>/dev/null)" = 143 ] &&
       [ "$(grep -c '^exit: 143$' "$life_run/log")" = 1 ] &&
       [ -n "$fork_child" ] && life_dead "$fork_child" && [ ! -e "$life_prefix.ready" ] &&
       [ "$(life lock-status "$life_run")" = idle ]; then fork_ok=1; fi
    case $fork_case in
      parent|child)
        report "$((1 - fork_ok))" "fork: $fork_case TERM publishes one verdict, reaps the child and releases the lock" \
          "exit=$life_rc child=${fork_child:-missing}; $(cat "$life_run/log" "$TMP/lifecycle.err")" ;;
      no-mask)
        # Perl safe points really delivered TERM before the assignment: the
        # supervisor ended while its child lived with the inherited lock.
        if [ "$fork_ok" = 0 ] && [ "$life_rc" = 143 ] && [ -n "$fork_child" ] &&
           ! life_dead "$fork_child" && [ "$(life lock-status "$life_run")" = held ]; then
          report 0 'negative control: unmasked fork TERM leaves a live child holding the lock'
        else report 1 'negative control: unmasked fork TERM leaves a live child holding the lock'; fi ;;
      inherited-handler)
        if [ "$fork_ok" = 0 ] && [ "$(grep -c '^exit: 143$' "$life_run/log")" = 2 ]; then
          report 0 'negative control: child inherited handler publishes a second supervisor verdict'
        else report 1 'negative control: child inherited handler publishes a second supervisor verdict' "$(cat "$life_run/log")"; fi ;;
    esac
    lifecycle_cleanup
  done
  cp "$SRC" "$life_root/tools/test-run.sh"
else
  skip 'fork: signal window and negative controls' 'requires POSIX signals and readable process states'
fi
}
harness_leg_run '74: fork TERM before the pid assignment #lifecycle' leg_74

# POSIX group delivery and ignored-on-entry INT are not native Windows signal
# semantics. The general start/status/stop contract above still runs there.
lifecycle_signals() {
# The attached launcher, in a process group of its own, with each signal the
# legs send it ignored at entry and then reset (gh-ocannl-1236). A non-
# interactive bash cannot trap a signal ignored when it started, and callers
# hand one on: a background `&` ignores INT, and `bg-run.sh spawn`, `nohup` and
# `execution slot --bg` ignore HUP. The first perl stands for such a caller,
# whatever this harness itself inherited, so the reset in the second is what
# every run exercises: without it the INT and HUP legs fail in the foreground
# too, instead of only where the caller happened to ignore the signal.
life_launch_grouped() {
  OCANNL_TOOL_TEST_RUNS=$life_runs LIFECYCLE_PREFIX=$life_prefix \
  LIFECYCLE_TOKEN_HELPER=$TMP/ps_token.sh PATH=$life_bin:$PATH \
    perl -e '$SIG{$_} = "IGNORE" for qw(INT HUP TERM); exec @ARGV' \
    perl -MPOSIX -e '$SIG{$_} = "DEFAULT" for qw(INT HUP TERM); POSIX::setpgid(0,0); exec @ARGV' \
    "$life_root/tools/test-run.sh" run --cap 30 hold >"$TMP/lifecycle.out" 2>"$TMP/lifecycle.err" &
  life_pid=$!
}
case $(uname -s) in MSYS*|MINGW*) life_posix=0 ;; *) life_posix=1 ;; esac
if [ "$life_posix" = 1 ] && [ "$have_state" = 1 ] && [ "$have_pgid" = 1 ]; then
  for life_signal in INT HUP GROUP_TERM; do
    rm -f "$life_prefix.ready" "$life_prefix.release"
    life_launch_grouped
    if ! await_fixture_ready "$life_prefix.ready"; then report 1 'lifecycle: signal fixture ready'; return 1; fi
    life_run=$(life paths run last)
    case $life_signal in
      GROUP_TERM) kill -TERM -- "-$life_pid"; life_want=143 ;;
      INT) kill -INT "$life_pid"; life_want=130 ;;
      HUP) kill -HUP "$life_pid"; life_want=143 ;;
    esac
    life_wait_child
    life_check "$life_want" 'verdict: CANCELLED' "attached $life_signal returns cancellation"
    if [ "$(cat "$life_run/exit")" = "$life_want" ] && [ "$(tail -n 1 "$life_run/log")" = "exit: $life_want" ]; then
      report 0 "lifecycle: $life_signal publishes the matching verdict and sentinel"
    else report 1 "lifecycle: $life_signal publishes the matching verdict and sentinel"; fi
    life_no_survivors "$life_signal"
  done

  # gh-ocannl-1362: the group TERM landing INSIDE the attached launcher's own
  # liveness probe, which it kills along with the launcher and the supervisor.
  # Load put it there by chance in CI; this cat shim puts it there every time.
  # The launcher's `cat <run>/resolved`, just before its attached wait, arms
  # it; the next `cat <run>/pid` -- that wait's first probe, in the launcher's
  # process group -- sends TERM to its own group and dies of it. The control
  # runs the launcher with plain sup_alive in that loop: it must report the
  # run dead while the supervisor goes on to publish the cancellation.
  life_real_cat=$(command -v cat)
  for probe_case in settled control; do
    cp "$SRC" "$life_root/tools/test-run.sh"
    if [ "$probe_case" = control ]; then
      probe_n=$(grep -c 'while sup_alive_settled "\$run_dir"; do wait_tick "\$sup"' "$SRC")
      sed 's/while sup_alive_settled "\$run_dir"; do wait_tick "\$sup"/while sup_alive "$run_dir"; do wait_tick "$sup"/' \
        "$SRC" >"$life_root/tools/test-run.sh"
      if [ "$probe_n" != 1 ] ||
         grep -q 'while sup_alive_settled "\$run_dir"; do wait' "$life_root/tools/test-run.sh"; then
        report 1 'lifecycle: probe control extraction' "matches=$probe_n"; return 1
      fi
    fi
    rm -f "$life_prefix.ready" "$life_prefix.release" "$life_prefix.probe-armed" "$life_prefix.probe-fired"
    cat >"$life_bin/cat" <<EOF
#!/usr/bin/env bash
case \${1:-} in
  */resolved) printf '%s\n' "\${1%/resolved}" >"$life_prefix.probe-armed" ;;
  */pid)
    if [ "\$("$life_real_cat" "$life_prefix.probe-armed" 2>/dev/null)/pid" = "\$1" ] &&
       mv "$life_prefix.probe-armed" "$life_prefix.probe-fired" 2>/dev/null; then
      kill -TERM 0
      exit 143
    fi ;;
esac
exec "$life_real_cat" "\$@"
EOF
    chmod +x "$life_bin/cat"
    life_launch_grouped
    life_wait_child
    rm "$life_bin/cat"
    life_run=$(life paths run last)
    if [ ! -e "$life_prefix.probe-fired" ]; then
      report 1 "lifecycle: $probe_case: the group TERM landed inside the launcher's probe" \
        "exit $life_rc; $(cat "$TMP/lifecycle.out" "$TMP/lifecycle.err")"
    elif [ "$probe_case" = settled ]; then
      life_check 143 'verdict: CANCELLED' 'a group TERM inside the attached probe still returns cancellation'
      if [ "$(cat "$life_run/exit")" = 143 ] && [ "$(tail -n 1 "$life_run/log")" = 'exit: 143' ]; then
        report 0 'lifecycle: probe-killing group TERM publishes the matching verdict and sentinel'
      else report 1 'lifecycle: probe-killing group TERM publishes the matching verdict and sentinel'; fi
      life_no_survivors 'probe-killing group TERM'
    else
      # The supervisor was alive all along: it publishes after the launcher
      # gave up on it, which is the CI signature, verdict file and all.
      if [ "$life_rc" = 1 ] && grep -q 'died without recording a verdict' "$TMP/lifecycle.out" &&
         await_fixture_ready "$life_run/exit" && [ "$(cat "$life_run/exit")" = 143 ] &&
         life_await_dead "$(cat "$life_run/pid")"; then
        report 0 'negative control: an unsettled probe reports a live supervisor dead'
      else report 1 'negative control: an unsettled probe reports a live supervisor dead' \
        "exit $life_rc; $(cat "$TMP/lifecycle.out" "$TMP/lifecycle.err")"; fi
      life_no_survivors 'unsettled-probe control'
    fi
    lifecycle_cleanup
  done
  cp "$SRC" "$life_root/tools/test-run.sh"
  life_start || return 1
  life_supervisor=$(cat "$life_run/pid")
  kill -KILL "$life_supervisor"
  life_await_dead "$life_supervisor" || report 1 'lifecycle: killed supervisor becomes non-live'
  life_capture status last
  life_check 1 'died without recording a verdict' 'status identifies a killed supervisor'
  life_capture lock-status last
  life_check 3 '^held$' 'orphan Dune retains the worktree lock'
  life_capture run --cap 30 ok
  if [ "$life_rc" = 2 ] && grep -Fq "status $life_run" "$TMP/lifecycle.err"; then
    report 0 'lifecycle: orphan lock refuses a new launch naming its run'
  else report 1 'lifecycle: orphan lock refuses a new launch naming its run'; fi
  life_capture stop last
  life_check 0 "orphaned process group $(cat "$life_run/pgid")" 'stop names the orphan group'
  life_await_dead "$(cat "$life_run/pgid")" || report 1 'lifecycle: orphan leader terminates'
  life_no_survivors 'orphan recovery'
  life_capture run --cap 30 ok
  life_check 0 'verdict: pass' 'launch succeeds after orphan recovery'

  rm -f "$life_prefix.child"
  # The Dune cap does not bound cleanup after the supervisor starts finishing.
  # Observe the attached launcher under our own deadline: waiting for the
  # child's natural 60-second exit must fail even if the final verdict is 0.
  OCANNL_TOOL_TEST_RUNS=$life_runs LIFECYCLE_PREFIX=$life_prefix \
    LIFECYCLE_TOKEN_HELPER=$TMP/ps_token.sh PATH=$life_bin:$PATH \
    "$life_root/tools/test-run.sh" run --cap 15 background \
    >"$TMP/lifecycle.out" 2>"$TMP/lifecycle.err" &
  life_pid=$!
  life_wait_child
  life_check 0 'verdict: pass' 'background-child run finishes within the fixture deadline'
  life_run=$(life paths run last)
  life_child=$(cat "$life_prefix.child")
  # This assertion is immediate, before cleanup. A descendant that closed its
  # lock is still a live build-tree writer, even when lock-status says idle.
  if life_dead "$life_child"; then report 0 'lifecycle: background child is non-live before run returns'
  else report 1 'lifecycle: background child is non-live before run returns'; fi
  life_no_survivors 'background-child completion'
  lifecycle_cleanup
else
  skip 'lifecycle: attached signals, orphan group and background-child reaping' 'requires POSIX process groups and independent state/pgid readers'
fi

life_last=$(life paths last)
rm -f "$life_last"
mkdir "$life_last"
life_capture run --cap 30 ok
if [ "$life_rc" = 2 ] && grep -q 'not a regular file' "$TMP/lifecycle.err"; then
  report 0 'lifecycle: a directory squatting at last refuses publication'
else report 1 'lifecycle: a directory squatting at last refuses publication'; fi
rmdir "$life_last"
life_capture run --cap 30 ok
life_check 0 'verdict: pass' 'removing the last squatter restores launch'

# Forge only the state that cannot be paused externally: owner published but
# supervisor not recorded. A sleep shim records every 0.1s settle, while still
# sleeping normally; this proves stop traversed the grace rather than guessing
# from wall-clock runtime (a slow census could otherwise fake that proof).
life_run=$life_runs/20000101T000000Z-77777
mkdir "$life_run"
printf 'hold\n' >"$life_run/cmd"
printf '30\n' >"$life_run/cap"
life paths worktree >"$life_run/wt"
life paths runs >"$life_run/runs"
: >"$life_run/log"
life_owner=$(life paths owner)
life_lock=$(life paths lock)
printf '%s\n' "$life_run" >"$life_owner"
printf '%s\n' "$life_run" >"$life_last"
life_hold_lock() {
  rm -f "$life_prefix.lock-ready"
  perl -MFcntl=:flock -e 'open my $f, ">>", $ARGV[0] or die $!;
    flock($f, LOCK_EX|LOCK_NB) or die $!;
    open my $r, ">", $ARGV[1] or die $!; close $r; sleep 60' \
    "$1" "$life_prefix.lock-ready" &
  life_pid=$!
  if ! await_fixture_ready "$life_prefix.lock-ready"; then report 1 'lifecycle: lock fixture ready'; return 1; fi
}
life_hold_lock "$life_lock" || return 1
life_capture status last
life_check 3 'launch in progress' 'status distinguishes the publication-to-record window'
life_real_sleep=$(command -v sleep)
cat >"$life_bin/sleep" <<EOF
#!/usr/bin/env bash
printf '%s\n' "\$1" >>"$TMP/lifecycle-sleeps"
exec "$life_real_sleep" "\$@"
EOF
chmod +x "$life_bin/sleep"
: >"$TMP/lifecycle-sleeps"
life_capture stop last
life_check 0 'held the worktree lock; reaped them' 'stop reaps an abandoned publication'
if [ "$(grep -c '^0.1$' "$TMP/lifecycle-sleeps")" -ge 30 ] && life_dead "$life_pid"; then
  report 0 'lifecycle: stop settles the publication window before reaping its holder'
else report 1 'lifecycle: stop settles the publication window before reaping its holder'; fi
rm "$life_bin/sleep"
life_wait_child
life_capture lock-status last
life_check 0 '^idle$' 'abandoned publication recovery releases the lock'

# Absence of runs is the legacy record discriminator. Ask the public queries
# for its lock/owner paths, rather than reconstructing the retired layout.
life_legacy=$life_runs/20000101T000000Z-legacy
mkdir "$life_legacy"
cp "$life_run/cmd" "$life_run/cap" "$life_run/wt" "$life_legacy/"
life_legacy_lock=$(life paths lock "$life_legacy") || { report 1 'lifecycle: legacy lock query'; return 1; }
life_legacy_owner=$(life paths owner "$life_legacy") || { report 1 'lifecycle: legacy owner query'; return 1; }
printf '/nowhere/legacy-run\n' >"$life_legacy_owner"
# No public query names this retired launcher record; its removal is part of
# the migration's source-tree preservation contract.
printf '1 x\n' >"$life_root/.test-run.lock.launcher"
life_hold_lock "$life_legacy_lock" || return 1
life_capture run --cap 30 ok
if [ "$life_rc" = 2 ] && grep -q 'previous version is still active' "$TMP/lifecycle.err" \
   && grep -q 'status /nowhere/legacy-run' "$TMP/lifecycle.err"; then
  report 0 'lifecycle: held legacy lock refuses launch naming its owner'
else report 1 'lifecycle: held legacy lock refuses launch naming its owner'; fi
kill -TERM "$life_pid"
life_wait_child
life_capture run --cap 30 ok
life_check 0 'verdict: pass' 'free legacy lock permits launch'
if diff -r "$TMP/lifecycle-before" "$life_root" >"$TMP/lifecycle-tree.diff"; then
  report 0 'lifecycle: every launch and recovery leaves the source tree byte-identical'
else report 1 'lifecycle: every launch and recovery leaves the source tree byte-identical' "$(cat "$TMP/lifecycle-tree.diff")"; fi
lifecycle_cleanup
}
leg_40_47_signals() {
lifecycle_signals || lifecycle_abandoned '40-47 (signals, publication window, legacy lock)'
}
harness_leg_run '40-47b: run/start lifecycle (signals, publication window, legacy lock) #lifecycle' leg_40_47_signals

# ---------------------------------------------------------------------------
# Legs 59-63 and 66: the source a run tested, recorded at launch (gh-ocannl-992)
# ---------------------------------------------------------------------------
# A fixture checkout of the staged script, committed, then launched clean and
# dirty; the lifecycle fixture's fake dune does the running. Git's system and
# global configuration are kept out of both the fixture's own git calls and the
# launches (neither variable is one the shipping script drops), so a user's
# hooks, signing or line-ending settings cannot move a verdict.
leg_59_63_66() {
if ! command -v git >/dev/null 2>&1; then
  skip 'checkout: head and dirty recorded at launch (legs 59-63)' 'no git on PATH'
else
  ck_git() { GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null git "$@"; }
  ck_runs=$TMP/checkout-runs
  ck_launch() { # <fixture root>; sets ck_rc and ck_dir, the run it recorded
    GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null OCANNL_TOOL_TEST_RUNS=$ck_runs \
      PATH=${ck_path_extra:+$ck_path_extra:}$life_bin:$PATH "$1/tools/test-run.sh" run --cap 30 ok \
      >"$TMP/checkout.out" 2>"$TMP/checkout.err"
    ck_rc=$?
    ck_dir=$(OCANNL_TOOL_TEST_RUNS=$ck_runs "$1/tools/test-run.sh" paths run last) || ck_dir=
  }
  ck_stage() { # <root>: the script and what it sources, nothing else
    stage_error="cannot copy the script into $1"
    mkdir -p "$1/tools" "$1/scripts" && cp "$SRC" "$1/tools/test-run.sh" || return 1
    stage_sourced "$1" || return 1
    chmod +x "$1/tools/test-run.sh"
  }
  ck_init() { # <root>: a committed checkout with local identity and settings
    ck_git init -q "$1" &&
      ck_git -C "$1" config user.name harness &&
      ck_git -C "$1" config user.email harness@invalid &&
      ck_git -C "$1" config commit.gpgsign false &&
      ck_git -C "$1" config core.autocrlf false &&
      ck_git -C "$1" add -A &&
      ck_git -C "$1" commit -q --no-verify -m fixture
  }
  ck_root=$TMP/checkout-repo
  ck_stage "$ck_root" || { harness_abandon "checkout: fixture root: $stage_error"; return; }
  printf 'first\n' >"$ck_root/NOTES"
  printf 'unchanged\n' >"$ck_root/README"
  ck_init "$ck_root" >/dev/null 2>&1 || { harness_abandon 'checkout: the fixture repository could not be committed'; return; }
  ck_head=$(ck_git -C "$ck_root" rev-parse HEAD)

  # Leg 59: a clean checkout records its HEAD and an EMPTY dirty record.
  ck_launch "$ck_root"
  ck_clean_out=$(cat "$TMP/checkout.out")
  if [ "$ck_rc" = 0 ] && [ -n "$ck_dir" ] \
     && [ "$(cat "$ck_dir/head" 2>/dev/null)" = "$ck_head" ] \
     && [ -f "$ck_dir/dirty" ] && [ ! -s "$ck_dir/dirty" ]; then
    report 0 'checkout: a clean launch records its HEAD and an empty dirty record'
  else
    report 1 'checkout: a clean launch records its HEAD and an empty dirty record' \
      "exit $ck_rc; run ${ck_dir:-?}; head $(cat "$ck_dir/head" 2>&1) (want $ck_head); dirty: $(cat "$ck_dir/dirty" 2>&1)"
  fi

  # Leg 60: a modified tracked file and an untracked one are both on record,
  # the latter although the checkout's own config hides untracked files; and
  # the launch leaves the index as it found it, although README's stat data
  # is stale there -- the refresh an ordinary `git status` would write back.
  printf 'second\n' >>"$ck_root/NOTES"
  printf 'let () = ()\n' >"$ck_root/new_test.ml"
  ck_git -C "$ck_root" config status.showUntrackedFiles no
  touch -t 200001010000 "$ck_root/README"
  ck_index=$(ck_git -C "$ck_root" rev-parse --git-path index)
  case $ck_index in /*) ;; *) ck_index=$ck_root/$ck_index ;; esac
  ck_index_before=$(cksum <"$ck_index")
  ck_launch "$ck_root"
  ck_dirty_out=$(cat "$TMP/checkout.out")
  if [ "$ck_rc" = 0 ] && [ "$(cat "$ck_dir/head" 2>/dev/null)" = "$ck_head" ] \
     && grep -Fqx ' M NOTES' "$ck_dir/dirty" && grep -Fqx '?? new_test.ml' "$ck_dir/dirty" \
     && [ "$(wc -l <"$ck_dir/dirty" | tr -d ' ')" = 2 ]; then
    report 0 'checkout: a dirty launch records every uncommitted path, untracked ones included'
  else
    report 1 'checkout: a dirty launch records every uncommitted path, untracked ones included' \
      "exit $ck_rc; head $(cat "$ck_dir/head" 2>&1) (want $ck_head); dirty: $(cat "$ck_dir/dirty" 2>&1)"
  fi
  if [ "$(cksum <"$ck_index")" = "$ck_index_before" ]; then
    report 0 "checkout: recording the dirty state leaves the checkout's index untouched"
  else
    report 1 "checkout: recording the dirty state leaves the checkout's index untouched" \
      "index checksum $ck_index_before became $(cksum <"$ck_index")"
  fi

  # Leg 61: a launch that inherits another repository's GIT_DIR/GIT_WORK_TREE
  # (as from inside a git hook) still records the checkout it tests.
  ck_other=$TMP/checkout-other
  mkdir -p "$ck_other"
  printf 'other\n' >"$ck_other/NOTES"
  ck_init "$ck_other" >/dev/null 2>&1 || { harness_abandon 'checkout: the second fixture repository could not be committed'; return; }
  GIT_DIR=$ck_other/.git GIT_WORK_TREE=$ck_other ck_launch "$ck_root"
  if [ "$ck_rc" = 0 ] && [ "$(cat "$ck_dir/head" 2>/dev/null)" = "$ck_head" ] \
     && grep -Fqx ' M NOTES' "$ck_dir/dirty"; then
    report 0 "checkout: an inherited GIT_DIR naming another repository does not redirect the record"
  else
    report 1 "checkout: an inherited GIT_DIR naming another repository does not redirect the record" \
      "exit $ck_rc; head $(cat "$ck_dir/head" 2>&1) (want $ck_head, not $(ck_git -C "$ck_other" rev-parse HEAD))"
  fi

  # Leg 62: neither record where the script's root is not a checkout's top
  # level -- outside Git (the lifecycle fixture), and a copy sitting below the
  # checkout above, whose HEAD describes a different tree -- and the run is
  # launched and judged as before.
  ck_detail=
  life_capture run --cap 30 ok
  life_run=$(life paths run last)
  { [ "$life_rc" = 0 ] && [ ! -e "$life_run/head" ] && [ ! -e "$life_run/dirty" ]; } ||
    ck_detail="outside Git: exit $life_rc; $(ls "$life_run" 2>&1 | tr '\n' ' ')"
  ck_stage "$ck_root/nested" || { harness_abandon "checkout: nested fixture root: $stage_error"; return; }
  ck_launch "$ck_root/nested"
  ck_nested_out=$(cat "$TMP/checkout.out")
  { [ "$ck_rc" = 0 ] && [ -n "$ck_dir" ] && [ ! -e "$ck_dir/head" ] && [ ! -e "$ck_dir/dirty" ]; } ||
    ck_detail="$ck_detail${ck_detail:+; }below the top level: exit $ck_rc; $(ls "$ck_dir" 2>&1 | tr '\n' ' ')"
  if [ -z "$ck_detail" ]; then
    report 0 "checkout: a root that is not a checkout's top level records neither, and still runs"
  else
    report 1 "checkout: a root that is not a checkout's top level records neither, and still runs" "$ck_detail"
  fi

  # Leg 63: a commit landing between the HEAD read and the status. A git shim
  # makes one, once, right after the first `git status` of the launch; the
  # record must pair the status with the HEAD it was taken against -- the new
  # commit, after a retake -- never the old commit with the new tree's status.
  ck_shim=$TMP/checkout-shim
  mkdir -p "$ck_shim"
  ck_real_git=$(command -v git)
  cat >"$ck_shim/git" <<EOF
#!/usr/bin/env bash
"$ck_real_git" "\$@"
rc=\$?
if [ "\${1:-}" = status ] && [ ! -e "$TMP/checkout-raced" ]; then
  : >"$TMP/checkout-raced"
  "$ck_real_git" commit -q --allow-empty --no-verify -m race >/dev/null 2>&1
fi
exit \$rc
EOF
  chmod +x "$ck_shim/git"
  rm -f "$TMP/checkout-raced"
  ck_path_extra=$ck_shim ck_launch "$ck_root"
  ck_raced_head=$(ck_git -C "$ck_root" rev-parse HEAD)
  if [ "$ck_rc" = 0 ] && [ -e "$TMP/checkout-raced" ] && [ "$ck_raced_head" != "$ck_head" ] \
     && [ "$(cat "$ck_dir/head" 2>/dev/null)" = "$ck_raced_head" ] \
     && grep -Fqx ' M NOTES' "$ck_dir/dirty"; then
    report 0 'checkout: a commit landing mid-record is re-read, never paired with the older HEAD'
  else
    report 1 'checkout: a commit landing mid-record is re-read, never paired with the older HEAD' \
      "exit $ck_rc; raced: $([ -e "$TMP/checkout-raced" ] && echo yes || echo no); head $(cat "$ck_dir/head" 2>&1) (want $ck_raced_head, was $ck_head)"
  fi

  # Leg 66: the digest quotes the record as its `source:` line, so evidence
  # pasted from it carries the revision: the commit and `(clean)` for leg 59's
  # launch, `+ 2 uncommitted paths` for leg 60's, and no line at all for leg
  # 62's copy below the top level, which recorded nothing.
  ck_detail=
  grep -Fqx "source:  $ck_head (clean)" <<<"$ck_clean_out" ||
    ck_detail="clean: $ck_clean_out"
  grep -Fqx "source:  $ck_head + 2 uncommitted paths" <<<"$ck_dirty_out" ||
    ck_detail="$ck_detail${ck_detail:+; }dirty: $ck_dirty_out"
  { [ -n "$ck_nested_out" ] && ! grep -q '^source:' <<<"$ck_nested_out"; } ||
    ck_detail="$ck_detail${ck_detail:+; }unrecorded: ${ck_nested_out:-<no digest>}"
  if [ -z "$ck_detail" ]; then
    report 0 "checkout: the digest's source line names the commit and the uncommitted count"
  else
    report 1 "checkout: the digest's source line names the commit and the uncommitted count" "$ck_detail"
  fi
  lifecycle_cleanup
fi
}
harness_leg_run '59-63, 66: the source a run tested #source' leg_59_63_66

# ---------------------------------------------------------------------------
# Leg 104: a checkout with unmerged paths is refused (gh-ocannl-1991)
# ---------------------------------------------------------------------------
# A committed fixture checkout of the staged script, as legs 59-63 build one,
# merged with a branch that conflicts in NOTES and adds OTHER: the merge stops
# with `UU NOTES` unmerged and `A  OTHER` staged. The lifecycle fixture's fake
# dune prints `built ok` into the run's log, which is how "dune never started"
# is read; the launch before the merge is the control that it does print it.
leg_104() {
if ! command -v git >/dev/null 2>&1; then
  skip 'unmerged: a checkout with unmerged paths is refused (leg 104)' 'no git on PATH'
  return
fi
um_git() { GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null git "$@"; }
um_runs=$TMP/unmerged-runs
um_root=$TMP/unmerged-repo
um() { # test-run.sh argv...; sets um_rc, um_out (stdout and stderr)
  um_out=$(GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null OCANNL_TOOL_TEST_RUNS=$um_runs \
    PATH=$life_bin:$PATH "$um_root/tools/test-run.sh" "$@" 2>&1)
  um_rc=$?
}
um_last() { um_dir=$(OCANNL_TOOL_TEST_RUNS=$um_runs "$um_root/tools/test-run.sh" paths run last 2>/dev/null) || um_dir=; }
um_ran() { grep -Fqx 'built ok' "$um_dir/log" 2>/dev/null; }
um_check() { # LABEL: reports on $um_detail
  if [ -z "$um_detail" ]; then report 0 "unmerged: $1"; else report 1 "unmerged: $1" "$um_detail"; fi
}
stage_error="cannot copy the script into $um_root"
{ mkdir -p "$um_root/tools" "$um_root/scripts" && cp "$SRC" "$um_root/tools/test-run.sh" &&
    stage_sourced "$um_root" && chmod +x "$um_root/tools/test-run.sh"; } ||
  { harness_abandon "unmerged: fixture root: $stage_error"; return; }
printf 'base\n' >"$um_root/NOTES"
{ um_git init -q "$um_root" &&
    um_git -C "$um_root" config user.name harness &&
    um_git -C "$um_root" config user.email harness@invalid &&
    um_git -C "$um_root" config commit.gpgsign false &&
    um_git -C "$um_root" config core.autocrlf false &&
    um_git -C "$um_root" add -A &&
    um_git -C "$um_root" commit -q --no-verify -m fixture &&
    um_git -C "$um_root" checkout -q -b side &&
    printf 'theirs\n' >"$um_root/NOTES" && printf 'other\n' >"$um_root/OTHER" &&
    um_git -C "$um_root" add -A &&
    um_git -C "$um_root" commit -q --no-verify -m side &&
    um_git -C "$um_root" checkout -q - &&
    printf 'ours\n' >"$um_root/NOTES" &&
    um_git -C "$um_root" commit -q --no-verify -am ours; } >/dev/null 2>&1 ||
  { harness_abandon 'unmerged: the fixture repository could not be committed'; return; }
um_head=$(um_git -C "$um_root" rev-parse HEAD)

# The control: before the merge the same launch runs dune and is green.
um run --cap 30 ok; um_last
um_detail=
{ [ "$um_rc" = 0 ] && um_ran && grep -Fqx "source:  $um_head (clean)" <<<"$um_out"; } ||
  um_detail="exit $um_rc; $um_out"
um_check 'control: the checkout runs green before the merge'
um_control=$um_dir

um_git -C "$um_root" merge -q --no-edit side >/dev/null 2>&1
if [ "$(um_git -C "$um_root" status --porcelain | sort | tr '\n' ';')" != 'A  OTHER;UU NOTES;' ]; then
  harness_abandon "unmerged: the merge did not stop on the one conflict: $(um_git -C "$um_root" status --porcelain | tr '\n' ';')"
  return
fi

# `run`: exit 2, the unmerged path named (and not the staged one), a run of
# its own on record whose dune never started.
um run --cap 30 ok; um_last
um_detail=
{ [ "$um_rc" = 2 ] && [ -n "$um_dir" ] && [ "$um_dir" != "$um_control" ] && ! um_ran &&
    grep -Fqx 'verdict: UNMERGED REFUSED (the checkout has unmerged paths; nothing ran) (exit 2)' <<<"$um_out" &&
    grep -Fqx 'unmerged refused: UU NOTES' <<<"$um_out" &&
    [ "$(grep -c '^unmerged refused: ' <<<"$um_out")" = 1 ] &&
    grep -Fqx 'test-run: UNMERGED REFUSED: UU NOTES' "$um_dir/log" &&
    grep -Fqx "source:  $um_head + 2 uncommitted paths, 1 of them unmerged" <<<"$um_out"; } ||
  um_detail="exit $um_rc; run ${um_dir:-?}; $um_out"
um_check 'run refuses, exit 2, naming the unmerged path alone, dune not started'

# `start`: refused in the launcher's own exit, nothing left detached, and
# `wait last` answers with the refusal rather than the run before it.
um_before=$um_dir
um start --cap 30 ok; um_last
um_detail=
{ [ "$um_rc" = 2 ] && [ -n "$um_dir" ] && [ "$um_dir" != "$um_before" ] && ! um_ran &&
    ! grep -q '^started: ' <<<"$um_out" &&
    grep -Fqx 'unmerged refused: UU NOTES' <<<"$um_out"; } ||
  um_detail="start: exit $um_rc; run ${um_dir:-?}; $um_out"
um wait last --timeout 30
{ [ "$um_rc" = 2 ] && grep -q '^verdict: UNMERGED REFUSED ' <<<"$um_out"; } ||
  um_detail="$um_detail${um_detail:+; }wait last: exit $um_rc; $um_out"
um_check 'start refuses in its own exit, and wait last reads the refusal'

# `repeat`: no iteration ran.
um repeat --cap 30 2 ok; um_last
um_detail=
{ [ "$um_rc" = 2 ] && ! um_ran && [ ! -d "$um_dir/iteration-1" ] &&
    grep -Fqx 'test-run: repeat: UNMERGED REFUSED: UU NOTES; no iteration ran' <<<"$um_out"; } ||
  um_detail="exit $um_rc; run ${um_dir:-?}; $um_out"
um_check 'repeat refuses before its first iteration'

# `plan` runs nothing either way, and says what run would do.
um plan --cap 30 ok
um_detail=
{ [ "$um_rc" = 0 ] &&
    grep -Fqx 'unmerged: run would refuse this checkout (UNMERGED REFUSED, exit 2)' <<<"$um_out" &&
    grep -Fqx '  UU NOTES' <<<"$um_out"; } ||
  um_detail="exit $um_rc; $um_out"
um_check 'plan says run would refuse, naming the path'

# The override runs the tree as it is, and the digest says what it was.
OCANNL_TOOL_TEST_UNMERGED=allow um run --cap 30 ok; um_last
um_detail=
{ [ "$um_rc" = 0 ] && um_ran && [ ! -e "$um_dir/unmerged-refused" ] &&
    grep -Fqx "source:  $um_head + 2 uncommitted paths, 1 of them unmerged" <<<"$um_out"; } ||
  um_detail="exit $um_rc; run ${um_dir:-?}; $um_out"
um_check 'OCANNL_TOOL_TEST_UNMERGED=allow runs it, the source line counting the unmerged path'

# Resolved and staged, mid-merge still: an ordinary dirty tree, and it runs.
printf 'resolved\n' >"$um_root/NOTES"
um_git -C "$um_root" add NOTES >/dev/null 2>&1
um run --cap 30 ok; um_last
um_detail=
{ [ "$um_rc" = 0 ] && um_ran && um_git -C "$um_root" rev-parse -q --verify MERGE_HEAD >/dev/null &&
    grep -Fqx "source:  $um_head + 2 uncommitted paths" <<<"$um_out"; } ||
  um_detail="exit $um_rc; run ${um_dir:-?}; $um_out"
um_check 'a conflict resolved and staged runs, mid-merge, as any dirty tree'
lifecycle_cleanup
}
harness_leg_run '104: a checkout with unmerged paths is refused #source' leg_104

test_run_close
