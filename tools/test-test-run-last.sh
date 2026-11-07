#!/usr/bin/env bash
# Tests for tools/test-run.sh: `wait last` and `status last` beside launches still in flight
# and sibling worktrees, and --subject, which this member is run with.
#
#   tools/test-test-run-last.sh          # run every leg of this member
#   tools/test-test-run-last.sh --keep   # keep the temp dir for inspection
#   tools/test-test-run-last.sh --only '^99:'   # only the legs whose names match
#   tools/test-test-run-last.sh --subject FILE   # test FILE, not tools/test-run.sh
#
# One member of the harness of tools/test-run.sh (gh-ocannl-2002), whose
# others `tools/test-test-run.sh --members` lists; that script runs them all,
# or the ones holding the legs an --only names. The conventions every member
# shares -- --subject, how a leg is named, gated and abandoned, what is tested
# and the fixtures the legs run against -- are tools/test-test-run-common.sh's
# header. Leg numbers are the harness's, not this file's: they stayed as they
# were when it was one file, so a member's are not consecutive.
#
# Groups: #lifecycle (99, 105), #queries (100), #harness (102), #hermetic (70).
#
# Legs:
#  99. `wait last` and `status last` follow a launch still in flight
#      (gh-ocannl-1830): one queued behind a wrapper that waits before it runs
#      this script (the fleet slot's `--wait-window`, stood in for) is `queued:`
#      (status 3) and a wait for it times out (124), never the previous run's
#      green verdict, which `wait RUN` still reads by name; released, the wait
#      follows it to its own red verdict. One that ends without publishing (its
#      slot refused) is refused (`unpublished:`, wait 2, status 1) until a run
#      publishes, and so is one seen queued behind a live run. A run's own
#      chain alive after its verdict is no launch in flight, a later run
#      published or not; with its `launchers` record emptied it is (the
#      negative control). A launch published before its supervisor
#      recorded itself is no launch to come, and its verdict is not refused
#      afterwards. A launch queued behind a run that died without a verdict
#      is followed, never answered with that death. A launcher past its `cd`
#      to the root (`cd tools && bash ./test-run.sh run`) is still a launch. A
#      shell that execs a slot wrapper for its next launch after its run is a
#      launch to come, and a script path holding a space still names its
#      worktree. A queued prepush is a launch, a queued `prepush --list` is
#      not, and a launch behind `timeout -k 5 60`, `env -u X` or `nice -n 5`
#      is one. A detached wrapper and its guardian (both orphans of parent 1,
#      one argv) are one launch, a later one with that argv after the first
#      run's verdict is another, and a `--` inside a `-c` string is no
#      separator. The wrappers are the fleet's exec-argv shape (`q-slot.sh ... --
#      CMD`): a `-c` string is never parsed. The slot this script takes itself is taken after
#      publication: a run waiting for it is `last`, live, and its wait times
#      out on it.
# 100. every `last` lookup stays in its own worktree: beside a sibling's newer
#      red run in the same state root, `wait last`, `status last` and `paths
#      run last` name the worktree's own, `list` lists only its runs (`--all`
#      lists the sibling's too, the control), a launch queued in the sibling
#      holds no wait here, and one queued there with the very argv of a
#      detached one queued here hides nothing here.
# 105. the residuals of following a queued launch (gh-ocannl-1913): a
#      relative script word behind `env -C DIR` (`--chdir=DIR`, a relative
#      DIR, a nested env's `-C`, the last of two `-C` in one env, BSD env's
#      `-P` operand, a `-S STRING` holding the command) is DIR's worktree's,
#      not the wrapper's directory's,
#      and released, the moved launch is followed to its own verdict; a launch
#      a shell's own control flow still holds (`run A; <wait>; run B` in one
#      `-c` string) is seen once its own process runs it, `last` being A till
#      then (the stated boundary, pinned); a launch under another
#      OCANNL_TOOL_TEST_RUNS named by an `env` assignment in its argv (a `--`
#      or an interpreter's `--` after it; a symlink then `..` in it, read as
#      a logical cd does and as a physical one, a launch under either) is no
#      launch under this root and
#      one under that, an `env -i` (or a bare `-`) clears an exported root,
#      `-u` (spaced or attached, or in a cluster) with a HOME assignment is a
#      launch under that HOME's default root, `-u` alone one under every
#      root asked (its root is unknown), a `~` in a root is literal, and one
#      exported alone is the loud over-read on every host (environments are
#      never read).
# 102 sits last: this harness, run on leg 99 with --subject naming a copy of
#      the script whose `prepush --list` rule is taken out of proc_table_perl,
#      exits 1 with exactly the one check that rule answers failed, having
#      printed the copy's digest; the option outranks the variable, and a
#      subject that is no file, or no value, is refused (gh-ocannl-1925).

. "$(cd "$(dirname "$0")" && pwd)/test-test-run-common.sh"


shared_through renamed_dune
# ---------------------------------------------------------------------------
# Legs 99-100: `last` is this worktree's, and a launch still in flight is not
# yet `last` (gh-ocannl-1830)
# ---------------------------------------------------------------------------
# A launch is published only once this script runs it, so one queued behind a
# wrapper that waits first -- `fleet-worker.sh execution slot --wait-window --
# tools/test-run.sh run ...`, waiting out a measurement or a slot -- left
# `wait last` answering with the previous run's verdict: a false green. The
# wrapper here is a bash that carries the launch's words in its argv, as the
# slot's Python does, and waits for a release file before it execs them. The
# previous run is green and the queued one red, so a stale answer is told from
# the right one by its exit status alone. Two fixture checkouts share one
# state root; their dune passes, or fails on `@red`.
q_runs=$TMP/queued-runs
q_bin=$TMP/queued-bin
q_root=$TMP/queued-repo
q_sib=$TMP/queued-sibling
q_spaced="$TMP/queued spaced/repo"
q_wraps=()
mkdir -p "$q_runs" "$q_bin"
for q_r in "$q_root" "$q_sib" "$q_spaced"; do
  mkdir -p "$q_r/tools" "$q_r/test/config" "$q_r/arrayjit/test"
  cp "$SRC" "$q_r/tools/test-run.sh"
  stage_sourced "$q_r" || report 1 "queued: fixture" "$stage_error"
done
cat >"$q_bin/dune" <<'EOF'
#!/usr/bin/env bash
case " $* " in
  *" @red "*) echo 'built red'; exit 1 ;;
  *" @hold "*) while [ ! -e "$Q_HOLD" ]; do sleep .1; done ;;
esac
echo 'built ok'
EOF
chmod +x "$q_bin/dune"
q() { # root subcommand [argv...]: that checkout's script, against the shared state root
  local root=$1
  shift
  OCANNL_TOOL_TEST_RUNS=$q_runs PATH=${Q_PATH_PREFIX:+$Q_PATH_PREFIX:}$q_bin:$PATH "$root/tools/test-run.sh" "$@"
}
q_capture() { # root subcommand [argv...]
  q "$@" >"$TMP/queued.out" 2>&1
  q_rc=$?
  q_out=$(cat "$TMP/queued.out")
}
q_last() { q "$1" paths run last 2>/dev/null; }
# The stand-in for the fleet's slot wrapper, in its shape: `q-slot.sh RELEASE
# exec|refuse -- CMD...` waits for RELEASE (and takes it: one release, one wrapper), then runs CMD as its child and
# exits with its status -- or, for `refuse`, exits 75 having run nothing, as a
# refused slot does. Like the real one it first double-forks a guardian that
# lives as long as it does, and here a second longer: an orphan of parent 1
# carrying its very argv, alone once the wrapper is gone. And one that
# runs CMD first and then lives on, in a subshell, until RELEASE: the shape of
# a wrapper winding down after its batch (the subshell is its fork).
q_slot=$TMP/q-slot.sh q_linger_sh=$TMP/q-linger.sh
cat >"$q_slot" <<'EOF'
#!/usr/bin/env bash
release=$1 how=$2
shift 3
main=$$
( (while kill -0 "$main" 2>/dev/null; do sleep .1; done; sleep 1) </dev/null >/dev/null 2>&1 & )
while ! rm "$release" 2>/dev/null; do sleep .1; done
[ "$how" = exec ] || exit 75
"$@"
EOF
cat >"$q_linger_sh" <<'EOF'
#!/usr/bin/env bash
release=$1
shift 2
"$@" >/dev/null 2>&1
(while [ ! -e "$release" ]; do sleep .1; done)
EOF
chmod +x "$q_slot" "$q_linger_sh"
q_wrap() { # root release exec|refuse argv...
  local root=$1 release=$2 how=$3
  shift 3
  (cd "$root" && OCANNL_TOOL_TEST_RUNS=$q_runs PATH=$q_bin:$PATH exec bash "$q_slot" "$release" "$how" -- "$@") \
    >/dev/null 2>&1 &
  q_wrap_pid=$!
  q_wraps+=("$q_wrap_pid:$release")
}
# The leg's own processes, for the harness's interrupted exit (cleanup_fixture):
# a wrapper still waiting would poll a release file the cleanup deletes, forever.
q_kill_all() {
  local w
  for w in ${q_wraps[@]+"${q_wraps[@]}"}; do
    kill -KILL "${w%%:*}" 2>/dev/null
    wait "${w%%:*}" 2>/dev/null
  done
  q_wraps=()
  if [ -n "${q_follow:-}" ]; then
    kill -KILL "$q_follow" 2>/dev/null
    wait "$q_follow" 2>/dev/null
  fi
  # The runs this leg holds on a file: let them end on their own.
  touch "$TMP/queued.hold" "$TMP/queued.hold2" "$TMP/queued.slot" "$TMP/queued.sup" "$TMP/queued.preflight" 2>/dev/null
  return 0
}
q_release_all() {
  local w
  for w in ${q_wraps[@]+"${q_wraps[@]}"}; do
    touch "${w#*:}"
    wait "${w%%:*}" 2>/dev/null
  done
  q_wraps=()
}
q_until() { # seconds command...: 0 once the command succeeds
  local n=$(( $1 * 10 ))
  shift
  while [ "$n" -gt 0 ]; do "$@" && return 0; sleep .1; n=$((n - 1)); done
  return 1
}
q_finished() { local d; d=$(q_last "$1") && [ -f "$d/exit" ] && [ "$d" != "${2:-}" ]; }

leg_99() {
q_label="wait last and status last follow a launch queued before it publishes"
q_unpub_label="a launch seen in flight that ends unpublished is refused, not answered with the run before it"
q_own_label="a run's own launch chain alive after its verdict is no launch in flight"
q_own_nc_label="negative control: with its launchers record emptied, that chain is a launch in flight"
q_slot_label="the slot this script takes itself waits after the publication"
q_detail=
q_capture "$q_root" run build @ok
q_green=$(q_last "$q_root")
[ "$q_rc" = 0 ] && [ -n "$q_green" ] || { harness_abandon "the green run: exit $q_rc: $q_out"; return; }
q_wrap "$q_root" "$TMP/queued.release1" exec tools/test-run.sh run build @red
q_red_wrap=$q_wrap_pid
# The stale answer is green, by name: what `wait last` used to return.
q_capture "$q_root" wait "$q_green" --timeout 5
[ "$q_rc" = 0 ] || q_detail="wait on the green run by name: exit $q_rc (want 0): $q_out"
if [ -z "$q_detail" ]; then
  q_capture "$q_root" status last
  { [ "$q_rc" = 3 ] && grep -qF "queued: a launch in this worktree has not published its run yet: $q_red_wrap " <<<"$q_out" &&
    grep -qF "not that launch's: $q_green" <<<"$q_out" && ! grep -qF 'verdict:' <<<"$q_out"; } ||
    q_detail="status last while queued: exit $q_rc (want 3, queued, wrapper $q_red_wrap): $q_out"
fi
if [ -z "$q_detail" ]; then
  q_capture "$q_root" wait last --timeout 1
  { [ "$q_rc" = 124 ] && grep -qF 'wait timed out after 1s: a launch in this worktree has not published its run yet' <<<"$q_out" &&
    ! grep -qF 'verdict:' <<<"$q_out"; } ||
    q_detail="wait last while queued: exit $q_rc (want 124): $q_out"
fi
# One wait spans the queue, the publication and the run: started while the
# launch is queued, it ends with the queued run's red verdict.
if [ -z "$q_detail" ]; then
  q "$q_root" wait last --timeout 60 >"$TMP/queued-follow.out" 2>&1 &
  q_follow=$!
  sleep 1
  touch "$TMP/queued.release1"
  wait "$q_follow"
  q_rc=$?
  q_follow=
  q_out=$(cat "$TMP/queued-follow.out")
  q_red=$(q_last "$q_root")
  { [ "$q_rc" = 1 ] && [ -n "$q_red" ] && [ "$q_red" != "$q_green" ] && grep -qF "$q_red" <<<"$q_out" &&
    grep -qF '@red' "$q_red/cmd"; } ||
    q_detail="wait last across the release: exit $q_rc (want 1, the queued run's); last: ${q_red:-none}: $q_out"
fi
q_release_all
if [ -z "$q_detail" ]; then report 0 "$q_label"; else report 1 "$q_label" "$q_detail"; fi

# A launch refused before it ran -- its slot never came -- leaves `last` on the
# run before it, green here: once it is gone, that verdict is still not its own.
q_detail=
q_capture "$q_root" run build @ok
q_green=$(q_last "$q_root")
[ "$q_rc" = 0 ] || q_detail="the second green run: exit $q_rc: $q_out"
if [ -z "$q_detail" ]; then
  q_wrap "$q_root" "$TMP/queued.release2" refuse tools/test-run.sh run build @red
  q_refused=$q_wrap_pid
  q_capture "$q_root" wait last --timeout 1
  [ "$q_rc" = 124 ] || q_detail="wait last while the refused launch is queued: exit $q_rc (want 124): $q_out"
  touch "$TMP/queued.release2"
  wait "$q_refused"
fi
if [ -z "$q_detail" ]; then
  q_capture "$q_root" wait last --timeout 5
  { [ "$q_rc" = 2 ] && grep -qF "unpublished: a launch seen in flight in this worktree after $q_green ended without publishing a run" <<<"$q_out" &&
    ! grep -qF 'verdict:' <<<"$q_out"; } ||
    q_detail="wait last after the refused launch: exit $q_rc (want 2, unpublished): $q_out"
fi
if [ -z "$q_detail" ]; then
  q_capture "$q_root" status last
  { [ "$q_rc" = 1 ] && grep -qF 'unpublished:' <<<"$q_out"; } ||
    q_detail="status last after the refused launch: exit $q_rc (want 1, unpublished): $q_out"
fi
if [ -z "$q_detail" ]; then
  q_capture "$q_root" wait "$q_green" --timeout 5
  [ "$q_rc" = 0 ] || q_detail="the run before it, by name: exit $q_rc (want 0): $q_out"
fi
# A run published since makes the note inert.
if [ -z "$q_detail" ]; then
  q_capture "$q_root" run build @ok
  q_capture "$q_root" wait last --timeout 5
  { [ "$q_rc" = 0 ] && grep -qF 'verdict: pass' <<<"$q_out"; } ||
    q_detail="wait last after a later run: exit $q_rc (want 0): $q_out"
fi
# Queued behind a run still live, then refused (by the worktree lock that run
# holds, say): seen while the run is live, so its verdict is not the launch's
# either once both are over.
if [ -z "$q_detail" ]; then
  Q_HOLD=$TMP/queued.hold q_capture "$q_root" start build @hold
  q_held=$(q_last "$q_root")
  { [ "$q_rc" = 0 ] && [ -n "$q_held" ] && [ ! -f "$q_held/exit" ]; } ||
    q_detail="the held run: exit $q_rc; last ${q_held:-none}: $q_out"
fi
if [ -z "$q_detail" ]; then
  q_wrap "$q_root" "$TMP/queued.release5" refuse tools/test-run.sh run build @red
  q_refused=$q_wrap_pid
  q_capture "$q_root" wait last --timeout 1
  { [ "$q_rc" = 124 ] && grep -qxF "wait timed out after 1s: $q_held" <<<"$q_out"; } ||
    q_detail="wait last on the live run, a launch queued behind it: exit $q_rc (want 124 on $q_held): $q_out"
  touch "$TMP/queued.release5"
  wait "$q_refused"
  touch "$TMP/queued.hold"
  q_until 30 q_finished "$q_root" "" ||
    q_detail="${q_detail:-the held run never finished: $q_held}"
fi
if [ -z "$q_detail" ]; then
  q_capture "$q_root" wait last --timeout 10
  { [ "$q_rc" = 2 ] && grep -qF "unpublished: a launch seen in flight in this worktree after $q_held ended" <<<"$q_out"; } ||
    q_detail="wait last after the held run and the launch refused behind it: exit $q_rc (want 2, unpublished): $q_out"
fi
touch "$TMP/queued.hold"
q_release_all
if [ -z "$q_detail" ]; then report 0 "$q_unpub_label"; else report 1 "$q_unpub_label" "$q_detail"; fi

# The chain a run recorded at launch: a wrapper that ran it and lives on after
# its verdict (in a subshell, its fork) carries its launch's words, and is that
# run's, not one to come.
q_detail= q_nc_detail=
q_before=$(q_last "$q_root")
(cd "$q_root" && OCANNL_TOOL_TEST_RUNS=$q_runs PATH=$q_bin:$PATH \
  exec bash "$q_linger_sh" "$TMP/queued.release3" -- tools/test-run.sh run build @ok) &
q_wraps+=("$!:$TMP/queued.release3")
if ! q_until 30 q_finished "$q_root" "$q_before"; then
  q_detail="the lingering shell's run never finished: last: $(q_last "$q_root")"
else
  q_linger=$(q_last "$q_root")
  q_capture "$q_root" wait last --timeout 2
  { [ "$q_rc" = 0 ] && grep -qF "$q_linger" <<<"$q_out"; } ||
    q_detail="wait last with the run's own shell alive: exit $q_rc (want 0): $q_out"
  if [ -z "$q_detail" ]; then
    q_capture "$q_root" status last
    { [ "$q_rc" = 0 ] && grep -qF 'verdict: pass' <<<"$q_out"; } ||
      q_detail="status last with the run's own shell alive: exit $q_rc (want 0): $q_out"
  fi
  # Still that run's once a later run has published: every run's chain counts.
  if [ -z "$q_detail" ]; then
    q_capture "$q_root" run build @ok
    q_later=$(q_last "$q_root")
    q_capture "$q_root" wait last --timeout 2
    { [ "$q_rc" = 0 ] && [ "$q_later" != "$q_linger" ] && grep -qF "$q_later" <<<"$q_out"; } ||
      q_detail="wait last after a later run, the earlier run's shell alive: exit $q_rc (want 0 on ${q_later:-none}): $q_out"
  fi
  grep -q . "$q_linger/launchers" ||
    q_nc_detail="the run recorded no launch chain: $(cat "$q_linger/launchers" 2>&1)"
  if [ -z "$q_nc_detail" ]; then
    cp "$q_linger/launchers" "$TMP/queued-launchers"
    : >"$q_linger/launchers"
    q_capture "$q_root" wait last --timeout 1
    [ "$q_rc" = 124 ] || q_nc_detail="wait last with no chain on record: exit $q_rc (want 124): $q_out"
    cp "$TMP/queued-launchers" "$q_linger/launchers"
  fi
fi
q_release_all
# The control's own sighting noted the run; a run published since clears it.
q "$q_root" run build @ok >/dev/null 2>&1
if [ -z "$q_detail" ]; then report 0 "$q_own_label"; else report 1 "$q_own_label" "$q_detail"; fi
if [ -z "$q_nc_detail" ] && [ -z "$q_detail" ]; then report 0 "$q_own_nc_label"
else report 1 "$q_own_nc_label" "${q_nc_detail:-not reached: $q_detail}"; fi


# The milliseconds between a launch's publication and its supervisor's first
# record, held open: a perl on PATH that waits for a release file before it
# becomes the supervisor (the one perl started with OCANNL_TOOL_TESTRUN_BG).
# The launcher is `last`'s already, so it is no launch to come, and no note
# taken then may refuse the run's own verdict once it has one.
q_window_label="a launch published before its supervisor recorded itself is no launch to come"
q_detail=
q_perlbin=$TMP/queued-perl-bin
mkdir -p "$q_perlbin"
cat >"$q_perlbin/perl" <<EOF
#!/usr/bin/env bash
if [ -n "\${OCANNL_TOOL_TESTRUN_BG:-}" ] && [ -n "\${Q_SUP_DELAY:-}" ]; then
  while [ ! -e "\$Q_SUP_DELAY" ]; do sleep .1; done
fi
# The launcher's preflight, before anything is recorded (Q_PREFLIGHT_HOLD).
if [ -n "\${Q_PREFLIGHT_HOLD:-}" ] && [ "\$*" = '-e use Fcntl ":flock"; exit 0' ]; then
  while [ ! -e "\$Q_PREFLIGHT_HOLD" ]; do sleep .1; done
fi
exec "$(command -v perl)" "\$@"
EOF
chmod +x "$q_perlbin/perl"
q_before=$(q_last "$q_root")
Q_PATH_PREFIX=$q_perlbin Q_SUP_DELAY=$TMP/queued.sup q "$q_root" start build @ok >/dev/null 2>&1 &
q_follow=$!
q_moved() { local d; d=$(q_last "$q_root") && [ "$d" != "$q_before" ]; }
if ! q_until 30 q_moved; then
  q_detail="the delayed launch never published: last: $(q_last "$q_root")"
else
  q_window=$(q_last "$q_root")
  q_capture "$q_root" status last
  { [ "$q_rc" = 3 ] && grep -qF 'launch in progress' <<<"$q_out" && [ ! -f "$q_window/pid" ]; } ||
    q_detail="status last before the supervisor: exit $q_rc (want 3, launch in progress): $q_out"
fi
touch "$TMP/queued.sup"
wait "$q_follow"
q_follow=
if [ -z "$q_detail" ]; then
  q_capture "$q_root" wait last --timeout 30
  { [ "$q_rc" = 0 ] && grep -qF "$q_window" <<<"$q_out"; } ||
    q_detail="wait last once the run passed: exit $q_rc (want 0, not unpublished): $q_out"
fi
if [ -z "$q_detail" ]; then report 0 "$q_window_label"; else report 1 "$q_window_label" "$q_detail"; fi

# A run that died without a verdict, a launch queued behind it: the dead run's
# missing verdict is not that launch's answer, and the wait follows on.
q_dead_label="a launch queued behind a run that died without a verdict is followed"
q_detail=
Q_HOLD=$TMP/queued.hold2 q_capture "$q_root" start build @hold
q_dead=$(q_last "$q_root")
q_grouped() { [ -s "$q_dead/pgid" ] && [ -s "$q_dead/pid" ]; }
if [ "$q_rc" != 0 ] || ! q_until 30 q_grouped; then
  q_detail="the run to kill: exit $q_rc; last ${q_dead:-none}: $q_out"
else
  kill -KILL -- "-$(cat "$q_dead/pgid")" 2>/dev/null
  kill -KILL "$(cat "$q_dead/pid")" 2>/dev/null
  q_idle() { q "$q_root" idle >/dev/null 2>&1; }
  q_until 30 q_idle || q_detail="the killed run's lock never cleared"
fi
if [ -z "$q_detail" ]; then
  [ ! -f "$q_dead/exit" ] || q_detail="the killed run recorded a verdict: $(cat "$q_dead/exit")"
fi
if [ -z "$q_detail" ]; then
  q_wrap "$q_root" "$TMP/queued.release7" exec tools/test-run.sh run build @red
  q_capture "$q_root" status last
  { [ "$q_rc" = 3 ] && grep -qF 'queued:' <<<"$q_out"; } ||
    q_detail="status last, the dead run then a queued launch: exit $q_rc (want 3, queued): $q_out"
fi
if [ -z "$q_detail" ]; then
  q_capture "$q_root" wait last --timeout 3
  { [ "$q_rc" = 124 ] && grep -qF 'has not published its run yet' <<<"$q_out"; } ||
    q_detail="wait last, the dead run then a queued launch: exit $q_rc (want 124): $q_out"
fi
touch "$TMP/queued.release7"
if [ -z "$q_detail" ]; then
  q_capture "$q_root" wait last --timeout 30
  { [ "$q_rc" = 1 ] && ! grep -qF "$q_dead" <<<"$q_out" && grep -qF 'verdict:' <<<"$q_out"; } ||
    q_detail="wait last once the queued launch ran: exit $q_rc (want 1, its own verdict): $q_out"
fi
q_release_all
if [ -z "$q_detail" ]; then report 0 "$q_dead_label"; else report 1 "$q_dead_label" "$q_detail"; fi

# A launcher that has already made its `cd` to the root still names this
# worktree's script by its relative word.
q_fork_label="a launcher past its cd to the root is still a launch in flight"
q_detail=
if [ -z "$q_detail" ]; then
  q_before=$(q_last "$q_root")
  (cd "$q_root/tools" && Q_PATH_PREFIX=$q_perlbin Q_PREFLIGHT_HOLD=$TMP/queued.preflight \
    OCANNL_TOOL_TEST_RUNS=$q_runs PATH=$q_perlbin:$q_bin:$PATH exec bash ./test-run.sh run build @red) >/dev/null 2>&1 &
  q_wraps+=("$!:$TMP/queued.preflight")
  q_relaunch() { ps -axww -o args= 2>/dev/null | grep -qF 'bash ./test-run.sh run build @red'; }
  if ! q_until 30 q_relaunch; then
    q_detail="the launcher from tools/ never showed in ps"
  else
    sleep .5
    q_capture "$q_root" status last
    { [ "$q_rc" = 3 ] && grep -qF 'queued:' <<<"$q_out" && [ "$(q_last "$q_root")" = "$q_before" ]; } ||
      q_detail="status last, a launcher from tools/ held at its preflight: exit $q_rc (want 3, queued): $q_out"
  fi
fi
q_release_all
if [ -z "$q_detail" ]; then report 0 "$q_fork_label"; else report 1 "$q_fork_label" "$q_detail"; fi

# A run's credit is for the argv it was recorded under: a shell that ran one
# batch and then `exec`s a slot wrapper for the next keeps its pid and start token,
# and its new words are a launch no run has credited. And a script path that
# holds a space, flattened apart in the table, still names its worktree.
q_exec_label="a shell that execs a wrapper after its run is a launch to come; a spaced path still resolves"
q_detail=
q_before=$(q_last "$q_root")
(cd "$q_root" && OCANNL_TOOL_TEST_RUNS=$q_runs PATH=$q_bin:$PATH exec bash -c '
  tools/test-run.sh run build @ok >/dev/null 2>&1
  exec bash "$2" "$1" exec -- tools/test-run.sh run build @red' q-exec "$TMP/queued.release9" "$q_slot") >/dev/null 2>&1 &
q_wraps+=("$!:$TMP/queued.release9")
q_execed() { ps -axww -o args= 2>/dev/null | grep -qF "q-slot.sh $TMP/queued.release9"; }
if ! q_until 30 q_finished "$q_root" "$q_before" || ! q_until 30 q_execed; then
  q_detail="the exec'ing shell's first run never finished, or it never exec'd: last: $(q_last "$q_root")"
else
  q_capture "$q_root" wait last --timeout 1
  { [ "$q_rc" = 124 ] && grep -qF 'has not published its run yet' <<<"$q_out"; } ||
    q_detail="wait last after the shell exec'd its next wrapper: exit $q_rc (want 124): $q_out"
fi
touch "$TMP/queued.release9"
if [ -z "$q_detail" ]; then
  q_capture "$q_root" wait last --timeout 30
  [ "$q_rc" = 1 ] || q_detail="wait last once the exec'd launch ran: exit $q_rc (want 1): $q_out"
fi
q_release_all
if [ -z "$q_detail" ]; then
  (cd "$TMP" && OCANNL_TOOL_TEST_RUNS=$q_runs PATH=$q_bin:$PATH \
    exec bash "$q_slot" "$TMP/queued.release10" exec -- "$q_spaced/tools/test-run.sh" run build @red) >/dev/null 2>&1 &
  q_wraps+=("$!:$TMP/queued.release10")
  sleep .5
  q_capture "$q_spaced" status last
  { [ "$q_rc" = 3 ] && grep -qF 'queued:' <<<"$q_out"; } ||
    q_detail="status last in a spaced worktree, its launch queued elsewhere: exit $q_rc (want 3): $q_out"
fi
q_release_all
if [ -z "$q_detail" ]; then report 0 "$q_exec_label"; else report 1 "$q_exec_label" "$q_detail"; fi

# `prepush --list` runs nothing: queued behind a wrapper it is no launch, and
# a prepush that launches is one. The wrappers are never released, only
# killed, and a run published after them clears the sighting they leave.
q_pp_label="a queued prepush or prefixed launch is in flight; a queued prepush --list is not"
q_detail=
# Each: the status wanted, then the command queued behind the wrapper; the
# prefixed ones peel their options and those options' operands.
for q_pp_case in "3|tools/test-run.sh prepush --since base build" "0|tools/test-run.sh prepush --since base --list" \
  "3|timeout -k 5 60 tools/test-run.sh run build @red" "3|env -u X A=1 tools/test-run.sh run build @red" \
  "3|nice -n 5 bash tools/test-run.sh run build @red"; do
  IFS='|' read -r q_pp_want q_pp_arg <<<"$q_pp_case"
  # shellcheck disable=SC2086 # the queued command's words, split on purpose
  (cd "$q_root" && OCANNL_TOOL_TEST_RUNS=$q_runs PATH=$q_bin:$PATH \
    exec bash "$q_slot" "$TMP/queued.never" exec -- $q_pp_arg) >/dev/null 2>&1 &
  q_pp=$!
  sleep .5
  q_capture "$q_root" status last
  kill -KILL "$q_pp" 2>/dev/null
  wait "$q_pp" 2>/dev/null
  sleep 1.5 # its guardian, alone for a second once its wrapper is gone
  [ "$q_rc" = "$q_pp_want" ] ||
    { q_detail="status last beside a queued $q_pp_arg: exit $q_rc (want $q_pp_want): $q_out"; break; }
  q "$q_root" run build @ok >/dev/null 2>&1
done
if [ -z "$q_detail" ]; then report 0 "$q_pp_label"; else report 1 "$q_pp_label" "$q_detail"; fi

# A wrapper detached from its caller is an orphan of parent 1 just as its
# guardian is, the two carrying one argv: they are one launch, not none. And
# a `--` inside a `-c` string is script text, no wrapper's separator.
q_orphan_label="a detached wrapper and its guardian are one launch, a later identical one another; a -- in script text is none"
q_detail=
( (cd "$q_root" && OCANNL_TOOL_TEST_RUNS=$q_runs PATH=$q_bin:$PATH \
  exec bash "$q_slot" "$TMP/queued.release11" exec -- tools/test-run.sh run build @red) </dev/null >/dev/null 2>&1 &
  echo $! >"$TMP/queued.orphan")
q_orphan=$(cat "$TMP/queued.orphan")
q_wraps+=("$q_orphan:$TMP/queued.release11")
q_orphaned() { [ "$(ps -o ppid= -p "$q_orphan" 2>/dev/null | tr -d ' ')" = 1 ]; }
if ! q_until 10 q_orphaned; then
  q_detail="the detached wrapper is no orphan of parent 1: ppid $(ps -o ppid= -p "$q_orphan" 2>/dev/null)"
else
  sleep .5
  q_capture "$q_root" status last
  { [ "$q_rc" = 3 ] && grep -qF 'queued:' <<<"$q_out"; } ||
    q_detail="status last, a detached wrapper queued: exit $q_rc (want 3): $q_out"
fi
touch "$TMP/queued.release11"
if [ -z "$q_detail" ]; then
  q_capture "$q_root" wait last --timeout 30
  [ "$q_rc" = 1 ] || q_detail="wait last once the detached wrapper ran: exit $q_rc (want 1): $q_out"
fi
q_release_all
if [ -z "$q_detail" ]; then
  q "$q_root" run build @ok >/dev/null 2>&1
  (cd "$q_root" && exec bash -c ': -- tools/test-run.sh run build @red
    while [ ! -e "$1" ]; do sleep .1; done' q-text "$TMP/queued.release12") >/dev/null 2>&1 &
  q_wraps+=("$!:$TMP/queued.release12")
  sleep .5
  q_capture "$q_root" status last
  { [ "$q_rc" = 0 ] && grep -qF 'verdict: pass' <<<"$q_out"; } ||
    q_detail="status last beside a -- in script text: exit $q_rc (want 0): $q_out"
  q_release_all
  if [ -z "$q_detail" ]; then
    q_capture "$q_root" wait last --timeout 5
    [ "$q_rc" = 0 ] || q_detail="wait last once that shell ended: exit $q_rc (want 0, not unpublished): $q_out"
  fi
fi
# A later detached wrapper with the very argv of one whose run has its
# verdict is a new launch: the argv credit is bounded by that verdict.
if [ -z "$q_detail" ]; then
  q_detached() { # [orphaned]: starts the detached wrapper, its pid in q_orphan
    ( (cd "$q_root" && OCANNL_TOOL_TEST_RUNS=$q_runs PATH=$q_bin:$PATH \
      exec bash "$q_slot" "$TMP/queued.release13" exec -- tools/test-run.sh run build @ok) </dev/null >/dev/null 2>&1 &
      echo $! >"$TMP/queued.orphan")
    q_orphan=$(cat "$TMP/queued.orphan")
    q_wraps+=("$q_orphan:$TMP/queued.release13")
    [ -z "${1:-}" ] || q_until 10 q_orphaned
  }
  q_before=$(q_last "$q_root")
  touch "$TMP/queued.release13"
  q_detached
  # Gone, or a zombie a PID 1 that reaps nothing keeps (a container's).
  q_gone() { case $(ps -o state= -p "$q_orphan" 2>/dev/null | tr -d ' ') in '' | Z*) return 0 ;; esac; return 1; }
  if ! q_until 30 q_gone || ! q_finished "$q_root" "$q_before"; then
    q_detail="the first detached wrapper never finished its run: last: $(q_last "$q_root")"
  else
    # Start times read at a second's resolution at best (/proc's btime is
    # whole seconds): the relaunch comes well clear of the verdict.
    sleep 2.5
    if ! q_detached orphaned; then
      q_detail="the second detached wrapper is no orphan of parent 1: ppid $(ps -o ppid= -p "$q_orphan" 2>/dev/null)"
    else
      sleep .5
      q_capture "$q_root" status last
      { [ "$q_rc" = 3 ] && grep -qF 'queued:' <<<"$q_out"; } ||
        q_detail="status last, an identical detached wrapper queued after the first run's verdict: exit $q_rc (want 3): $q_out"
    fi
  fi
  # Released, it runs: no child of this shell, so it is waited out by pid.
  q_release_all
  q_until 30 q_gone || q_detail="${q_detail:-the second detached wrapper never ended}"
  q "$q_root" wait last --timeout 30 >/dev/null 2>&1
fi
if [ -z "$q_detail" ]; then report 0 "$q_orphan_label"; else report 1 "$q_orphan_label" "$q_detail"; fi

# The other wait: the fleet slot this script takes itself, through a fake
# fleet-worker.sh that holds the batch until a release file appears. The run
# is published before it asks for the slot, so `last` is that run, live.
q_detail=
q_fw=$TMP/queued-fleet-worker.sh
cat >"$q_fw" <<'FAKE'
#!/usr/bin/env bash
if [ "$1 $2 $3" = "execution slot --probe" ]; then echo "EXECUTION SLOT PROBE fakebox 4 4"; exit 0; fi
while [ "$1" != -- ]; do shift; done
shift
while [ ! -e "$Q_SLOT_RELEASE" ]; do sleep .1; done
exec "$@"
FAKE
chmod +x "$q_fw"
q_before=$(q_last "$q_root")
OCANNL_TOOL_FLEET_WORKER=$q_fw Q_SLOT_RELEASE=$TMP/queued.slot q_capture "$q_root" start build @red
q_slotted=$(q_last "$q_root")
{ [ "$q_rc" = 0 ] && [ -n "$q_slotted" ] && [ "$q_slotted" != "$q_before" ] && [ ! -f "$q_slotted/exit" ]; } ||
  q_detail="start under a held slot: exit $q_rc; last: ${q_slotted:-none} (before: $q_before): $q_out"
if [ -z "$q_detail" ]; then
  q_capture "$q_root" wait last --timeout 1
  { [ "$q_rc" = 124 ] && grep -qxF "wait timed out after 1s: $q_slotted" <<<"$q_out"; } ||
    q_detail="wait last on the run waiting for its slot: exit $q_rc (want 124 on $q_slotted): $q_out"
fi
touch "$TMP/queued.slot"
if [ -z "$q_detail" ]; then
  q_capture "$q_root" wait last --timeout 30
  { [ "$q_rc" = 1 ] && grep -qF "$q_slotted" <<<"$q_out"; } ||
    q_detail="wait last once the slot came: exit $q_rc (want 1): $q_out"
else
  q "$q_root" wait last --timeout 30 >/dev/null 2>&1
fi
if [ -z "$q_detail" ]; then report 0 "$q_slot_label"; else report 1 "$q_slot_label" "$q_detail"; fi
}
harness_leg_run '99: wait last and status last follow a launch still in flight #lifecycle' leg_99

# ---------------------------------------------------------------------------
# Leg 105: the residuals of following a queued launch (gh-ocannl-1913)
# ---------------------------------------------------------------------------
# The fixtures are leg 99's (q_root, q_sib, the wrappers). Three forms the
# reader missed or over-read: a relative script word behind `env -C DIR` is
# DIR's, not the wrapper's directory's; a launch a shell's own control flow
# still holds is seen once its own process runs it (the stated boundary,
# pinned); a launch under another OCANNL_TOOL_TEST_RUNS is no launch here
# where its argv says so (environments are never read).
leg_105() {
q_cd_label="a launch queued behind env -C DIR is DIR's worktree's, not the wrapper's directory's"
q_held_label="a launch a shell's own control flow holds is seen once its own process runs it, not before"
q_root_label="a launch under another OCANNL_TOOL_TEST_RUNS named in its argv is no launch here; one exported alone is the loud over-read"
q_detail=
q_other=$TMP/queued-other-runs
mkdir -p "$q_other"
q_under() { # runs root subcommand [argv...]: that checkout's script under the named state root
  local runs=$1 root=$2
  shift 2
  OCANNL_TOOL_TEST_RUNS=$runs PATH=$q_bin:$PATH "$root/tools/test-run.sh" "$@"
}
q "$q_root" run build @ok >/dev/null 2>&1 || { harness_abandon "the green run before the env -C cases"; return; }
q "$q_sib" run build @ok >/dev/null 2>&1 || { harness_abandon "the sibling's green run"; return; }
q_under "$q_other" "$q_root" run build @ok >/dev/null 2>&1 || { harness_abandon "the green run under the other root"; return; }
# One wrapper, killed and never released, read from both worktrees under this
# state root and, unless `-` is wanted, from this worktree under the other
# (q_other, or Q_OTHER_RUNS); the runs published afterwards clear the
# sightings it leaves. The wrapper exports Q_ENV_RUNS as its state root, the
# harness's by default.
q_case() { # want-root want-sib want-other dir argv...: the statuses wanted in q_root, in q_sib, in q_root under the other root
  local want_root=$1 want_sib=$2 want_other=$3 dir=$4 pid rc_root out_root rc_sib out_sib other=${Q_OTHER_RUNS:-$q_other}
  shift 4
  (cd "$dir" && OCANNL_TOOL_TEST_RUNS=${Q_ENV_RUNS:-$q_runs} PATH=$q_bin:$PATH \
    exec bash "$q_slot" "$TMP/queued.never" exec -- "$@") >/dev/null 2>&1 &
  pid=$!
  q_wraps+=("$pid:$TMP/queued.never") # for the harness's interrupted exit (q_kill_all)
  sleep .5
  q_capture "$q_root" status last
  rc_root=$q_rc out_root=$q_out
  q_capture "$q_sib" status last
  rc_sib=$q_rc out_sib=$q_out
  if [ "$want_other" = - ]; then
    q_rc=- q_out=
  else
    q_under "$other" "$q_root" status last >"$TMP/queued.out" 2>&1
    q_rc=$?
    q_out=$(cat "$TMP/queued.out")
  fi
  kill -KILL "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  local w kept=()
  for w in ${q_wraps[@]+"${q_wraps[@]}"}; do [ "${w%%:*}" = "$pid" ] || kept+=("$w"); done
  q_wraps=(${kept[@]+"${kept[@]}"})
  sleep 1.5 # its guardian, alone for a second once its wrapper is gone
  q "$q_root" run build @ok >/dev/null 2>&1
  q "$q_sib" run build @ok >/dev/null 2>&1
  q_under "$other" "$q_root" run build @ok >/dev/null 2>&1
  [ "$rc_root" = "$want_root" ] && [ "$rc_sib" = "$want_sib" ] && [ "$q_rc" = "$want_other" ] && return 0
  q_detail="status last beside a queued \`$*\` from $dir: $rc_root in the worktree, $rc_sib in the sibling, $q_rc under the other root (want $want_root, $want_sib, $want_other): $out_root | $out_sib | $q_out"
  return 1
}
# A wrapper outside any worktree, its command moved into this one; one in
# this worktree, its command moved into the sibling (over-read before: the
# relative word resolved against the wrapper's directory); a `--chdir=DIR`
# relative to the wrapper's directory; a nested env's `-C`, relative to the
# outer one's; two `-C` in one env, the last winning, relative to the
# directory env was entered in (as GNU env does); BSD env's `-P` operand; a
# `-S STRING` whose string is the command, spaced or attached.
q_case 3 0 - "$TMP" env -C "$q_root" tools/test-run.sh run build @red &&
  q_case 0 3 - "$q_root" env -C "$q_sib" tools/test-run.sh run build @red &&
  q_case 3 0 - "$TMP" env --chdir=queued-repo tools/test-run.sh run build @red &&
  q_case 3 0 - "$q_sib" env -C "$TMP" env -C queued-repo tools/test-run.sh run build @red &&
  q_case 3 0 - "$q_root" env -C "$TMP" -C . tools/test-run.sh run build @red &&
  q_case 3 0 - "$q_root" env -P /usr/bin tools/test-run.sh run build @red &&
  q_case 3 0 - "$q_root" env -S 'tools/test-run.sh run build @red' &&
  q_case 3 0 - "$q_root" env '-Stools/test-run.sh run build @red' || :
# Released, the moved launch publishes its run in this worktree, and the
# wait that followed it ends with that run's verdict.
if [ -z "$q_detail" ]; then
  q_before=$(q_last "$q_root")
  (cd "$TMP" && OCANNL_TOOL_TEST_RUNS=$q_runs PATH=$q_bin:$PATH \
    exec bash "$q_slot" "$TMP/queued.release14" exec -- env -C "$q_root" tools/test-run.sh run build @red) >/dev/null 2>&1 &
  q_wraps+=("$!:$TMP/queued.release14")
  sleep .5
  q "$q_root" wait last --timeout 60 >"$TMP/queued-follow.out" 2>&1 &
  q_follow=$!
  sleep 1
  touch "$TMP/queued.release14"
  wait "$q_follow"
  q_rc=$?
  q_follow=
  q_out=$(cat "$TMP/queued-follow.out")
  q_red=$(q_last "$q_root")
  { [ "$q_rc" = 1 ] && [ -n "$q_red" ] && [ "$q_red" != "$q_before" ] && grep -qF "$q_red" <<<"$q_out"; } ||
    q_detail="wait last across the release of an env -C launch: exit $q_rc (want 1, its own); last: ${q_red:-none}: $q_out"
  q_release_all
fi
if [ -z "$q_detail" ]; then report 0 "$q_cd_label"; else report 1 "$q_cd_label" "$q_detail"; fi

# A shell running `run A; <wait>; run B` from one `-c` string: once A has its
# verdict and B is still the shell's text, `last` is A and its verdict is the
# answer (a `-c` string is never parsed); once B's own process runs, `wait
# last` follows it to B's verdict.
q_detail=
q_before=$(q_last "$q_root")
(cd "$q_root" && OCANNL_TOOL_TEST_RUNS=$q_runs PATH=$q_bin:$PATH exec bash -c '
  tools/test-run.sh run build @ok >/dev/null 2>&1
  while [ ! -e "$1" ]; do sleep .1; done
  tools/test-run.sh run build @red' q-held "$TMP/queued.release15") >/dev/null 2>&1 &
q_wraps+=("$!:$TMP/queued.release15")
if ! q_until 30 q_finished "$q_root" "$q_before"; then
  q_detail="the held shell's first run never finished: last: $(q_last "$q_root")"
else
  q_capture "$q_root" status last
  { [ "$q_rc" = 0 ] && grep -qF 'verdict: pass' <<<"$q_out"; } ||
    q_detail="status last while the shell holds its second launch: exit $q_rc (want 0, the first run's): $q_out"
  if [ -z "$q_detail" ]; then
    q_capture "$q_root" wait last --timeout 1
    [ "$q_rc" = 0 ] || q_detail="wait last while the shell holds its second launch: exit $q_rc (want 0, the first run's): $q_out"
  fi
fi
q_green=$(q_last "$q_root")
q_release_all
if [ -z "$q_detail" ]; then
  q_capture "$q_root" wait last --timeout 30
  { [ "$q_rc" = 1 ] && [ "$(q_last "$q_root")" != "$q_green" ]; } ||
    q_detail="wait last once the held launch ran: exit $q_rc (want 1, its own): $q_out"
fi
if [ -z "$q_detail" ]; then report 0 "$q_held_label"; else report 1 "$q_held_label" "$q_detail"; fi

# A launch of this worktree under another state root: named by an `env`
# assignment in its argv (a trailing slash, an `-u` before it, a `--` after
# it, an interpreter and its `--` after it, a symlink then `..` in it, which
# the launch shell's logical cd resolves beside the link, or, the logical
# destination absent, beside the link's target: a launch is one under either
# reading) it is no launch under this root
# and one under that (the control); an `env -i` clears the exported root (the
# default root: none of these); the default root is under the launch's HOME,
# so `-u OCANNL_TOOL_TEST_RUNS HOME=DIR` is a launch under DIR's default
# root and none here, spaced or attached (`-uNAME`), and so is the cluster
# `-iC WORKTREE HOME=DIR` from outside the worktree and `env - HOME=DIR`
# (a bare `-` is `-i`); `-u` with no HOME named is a launch under an unknown
# root, so it is one everywhere it could be asked about; a root is read
# literally, so `~/queued-tilde` is the directory `~` of the worktree, as the
# script's own `cd "$RUNS"` reads it; the harness's own default root is never
# read, so the `-i` case is checked under the two roots that are not it.
# Exported alone, the root is over-read: environments are never read
# (/proc/PID/environ is the exec-time one, stale under a wrapper that exports
# before it runs its command), so the launch is in flight here, loud: a wait
# that times out and then `unpublished:`, as stated.
q_detail=
q_home=$TMP/queued-home
mkdir -p "$q_home" "$TMP/queued-actual/child"
ln -sfn "$TMP/queued-actual/child" "$TMP/queued-link"
q_case 0 0 3 "$q_root" env "OCANNL_TOOL_TEST_RUNS=$q_other" tools/test-run.sh run build @red &&
  q_case 0 0 3 "$q_root" env -u OCANNL_TOOL_TEST_RUNS "OCANNL_TOOL_TEST_RUNS=$q_other/" -- tools/test-run.sh run build @red &&
  q_case 0 0 3 "$q_root" env "OCANNL_TOOL_TEST_RUNS=$TMP/queued-link/../${q_other##*/}" tools/test-run.sh run build @red &&
  Q_OTHER_RUNS=$TMP/queued-actual/queued-absent q_case 0 0 3 "$q_root" env "OCANNL_TOOL_TEST_RUNS=$TMP/queued-link/../queued-absent" tools/test-run.sh run build @red &&
  q_case 0 0 3 "$q_root" env "OCANNL_TOOL_TEST_RUNS=$q_other" bash -- tools/test-run.sh run build @red &&
  q_case 0 0 0 "$q_root" env -i "HOME=$HOME" "PATH=$PATH" tools/test-run.sh run build @red &&
  Q_OTHER_RUNS=$q_home/.ocannl-test-runs q_case 0 0 3 "$q_root" env -u OCANNL_TOOL_TEST_RUNS "HOME=$q_home" tools/test-run.sh run build @red &&
  Q_OTHER_RUNS=$q_home/.ocannl-test-runs q_case 0 0 3 "$q_root" env -uOCANNL_TOOL_TEST_RUNS "HOME=$q_home" tools/test-run.sh run build @red &&
  Q_OTHER_RUNS=$q_home/.ocannl-test-runs q_case 0 0 3 "$TMP" env -iC "$q_root" "HOME=$q_home" "PATH=$PATH" tools/test-run.sh run build @red &&
  Q_OTHER_RUNS=$q_home/.ocannl-test-runs q_case 0 0 3 "$q_root" env - "HOME=$q_home" "PATH=$PATH" tools/test-run.sh run build @red &&
  Q_OTHER_RUNS=$q_home/.ocannl-test-runs q_case 3 0 3 "$q_root" env -u OCANNL_TOOL_TEST_RUNS tools/test-run.sh run build @red &&
  Q_OTHER_RUNS="$q_root/~/queued-tilde" q_case 0 0 3 "$q_root" env 'OCANNL_TOOL_TEST_RUNS=~/queued-tilde' tools/test-run.sh run build @red &&
  Q_ENV_RUNS=$q_other q_case 3 0 3 "$q_root" tools/test-run.sh run build @red || :
if [ -z "$q_detail" ]; then report 0 "$q_root_label"; else report 1 "$q_root_label" "$q_detail"; fi
}
harness_leg_run '105: the residuals of following a queued launch: env -C, a shell-held launch, another state root #lifecycle' leg_105

leg_100() {
q_sib_label="every last lookup stays in its own worktree beside a sibling's newer run"
q_detail=
q_capture "$q_root" run build @ok
q_own=$(q_last "$q_root")
sleep 1 # the sibling's run is NEWER, by name too
q_capture "$q_sib" run build @red
q_theirs=$(q_last "$q_sib")
{ [ "$q_rc" = 1 ] && [ -n "$q_own" ] && [ -n "$q_theirs" ] && [ "$q_own" != "$q_theirs" ]; } ||
  { harness_abandon "the two runs: sibling exit $q_rc, own ${q_own:-none}, theirs ${q_theirs:-none}: $q_out"; return; }
q_capture "$q_root" wait last --timeout 5
{ [ "$q_rc" = 0 ] && grep -qF "$q_own" <<<"$q_out"; } || q_detail="wait last: exit $q_rc (want 0, $q_own): $q_out"
if [ -z "$q_detail" ]; then
  q_capture "$q_root" status last
  { [ "$q_rc" = 0 ] && grep -qF "$q_own" <<<"$q_out" && ! grep -qF "$q_theirs" <<<"$q_out"; } ||
    q_detail="status last: exit $q_rc (want its own $q_own): $q_out"
fi
if [ -z "$q_detail" ]; then
  q_capture "$q_root" list
  q_list=$q_out
  q_capture "$q_root" list --all
  q_all=$q_out
  { grep -qF "$(basename "$q_own")" <<<"$q_list" && ! grep -qF "$(basename "$q_theirs")" <<<"$q_list" &&
    grep -qF "$(basename "$q_theirs")" <<<"$q_all" && grep -qF "$(basename "$q_own")" <<<"$q_all"; } ||
    q_detail="list: $q_list; list --all (the control: both): $q_all"
fi
if [ -z "$q_detail" ]; then
  q_capture "$q_sib" list
  { grep -qF "$(basename "$q_theirs")" <<<"$q_out" && ! grep -qF "$(basename "$q_own")" <<<"$q_out"; } ||
    q_detail="the sibling's list: $q_out"
fi
# A launch queued in the sibling is the sibling's, and holds no wait here.
if [ -z "$q_detail" ]; then
  q_wrap "$q_sib" "$TMP/queued.release4" exec tools/test-run.sh run build @ok
  q_capture "$q_root" wait last --timeout 2
  { [ "$q_rc" = 0 ] && grep -qF "$q_own" <<<"$q_out"; } ||
    q_detail="wait last here with the sibling's launch queued: exit $q_rc (want 0): $q_out"
  if [ -z "$q_detail" ]; then
    q_capture "$q_sib" status last
    { [ "$q_rc" = 3 ] && grep -qF 'queued:' <<<"$q_out"; } ||
      q_detail="the sibling's status last, its launch queued: exit $q_rc (want 3): $q_out"
  fi
fi
q_release_all
q "$q_sib" wait last --timeout 30 >/dev/null 2>&1
# One argv in two worktrees is two launches: a wrapper queued in the sibling
# with the very argv of a detached one queued here hides nothing here. Never
# released: killed, and a run published after clears the sightings.
if [ -z "$q_detail" ]; then
  ( (cd "$q_root" && OCANNL_TOOL_TEST_RUNS=$q_runs PATH=$q_bin:$PATH \
    exec bash "$q_slot" "$TMP/queued.never2" exec -- tools/test-run.sh run build @red) </dev/null >/dev/null 2>&1 &
    echo $! >"$TMP/queued.orphan2")
  q_here=$(cat "$TMP/queued.orphan2")
  q_wrap "$q_sib" "$TMP/queued.never2" exec tools/test-run.sh run build @red
  q_there=$q_wrap_pid
  q_orphan2() { [ "$(ps -o ppid= -p "$q_here" 2>/dev/null | tr -d ' ')" = 1 ]; }
  if ! q_until 10 q_orphan2; then
    q_detail="the detached wrapper here is no orphan of parent 1"
  else
    sleep .5
    q_capture "$q_root" status last
    { [ "$q_rc" = 3 ] && grep -qF 'queued:' <<<"$q_out"; } ||
      q_detail="status last here, the sibling queuing the very same argv: exit $q_rc (want 3): $q_out"
  fi
  kill -KILL "$q_here" "$q_there" 2>/dev/null
  wait "$q_there" 2>/dev/null
  sleep 1.5 # their guardians, alone for a second
  q "$q_root" run build @ok >/dev/null 2>&1
  q "$q_sib" run build @ok >/dev/null 2>&1
fi
if [ -z "$q_detail" ]; then report 0 "$q_sib_label"; else report 1 "$q_sib_label" "$q_detail"; fi
}
harness_leg_run '100: every last lookup stays in its own worktree #queries' leg_100

# ---------------------------------------------------------------------------
# Leg 102: --subject tests the named file in place of tools/test-run.sh
# (gh-ocannl-1925)
# ---------------------------------------------------------------------------
# The control every change to the script wants -- "this leg fails against the
# script without it" -- as one option. The subject here is the script with one
# rule of proc_table_perl taken out: the line that keeps a queued `prepush
# --list` (which launches nothing) from reading as a launch in flight. This
# harness, run on leg 99 against it, must exit 1 with exactly one FAIL, the
# check that rule answers, and every other check of the leg passed: a subject
# that reached no fixture root would pass them all, one staged without what it
# sources would fail them all. The run's header carries the mutant's digest,
# and its variable names a file that does not exist, which the option
# outranks. The refusals are the same harness stopped before any leg.
subj_label="--subject tests the named copy: without the prepush --list rule, leg 99 fails in that one check alone"
subj_refuse_label="a subject that is no file, or no value, is refused before any leg, by the option and by the variable"
# The check that rule answers, by the label leg 99 reports it under.
subj_check=$(sed -n 's/^q_pp_label="\(.*\)"$/\1/p' "$SELF")
subj_run() { # [NAME=VALUE...] -- [harness argv...]: sets subj_rc, subj_out
  local envs=()
  while [ "$1" != -- ]; do envs+=("$1"); shift; done
  shift
  subj_rc=0
  (
    unset OCANNL_TOOL_HARNESS_STRICT OCANNL_TOOL_HARNESS_EXPECTED_SKIPS \
      OCANNL_TOOL_HARNESS_NAME OCANNL_TOOL_HARNESS_SKIP_RECORD OCANNL_TOOL_HARNESS_DUNE_SKIP_RECORD \
      OCANNL_TOOL_HARNESS_SUBJECT
    mkdir -p "$TMP/subj/tmp"
    env ${envs[@]+"${envs[@]}"} TMPDIR="$TMP/subj/tmp" bash "$SELF" "$@"
  ) >"$TMP/subj/out" 2>&1 || subj_rc=$?
  subj_out=$(cat "$TMP/subj/out")
}
leg_102() {
[ -n "$subj_check" ] || { harness_abandon "no q_pp_label is assigned in $SELF"; return; }
local mutant=$TMP/subj/mutant.sh absent=$TMP/subj/no-such-script.sh digest fails detail=
mkdir -p "$TMP/subj"
awk 'index($0, "next if $w[$i + 1] eq \"prepush\" && grep") { n++; next }
     { print } END { exit n == 1 ? 0 : 1 }' "$SRC" >"$mutant" && bash -n "$mutant" ||
  { harness_abandon "the prepush --list rule is not in $SRC exactly once; the mutant is broken"; return; }
digest=$(cksum <"$mutant")
subj_run "OCANNL_TOOL_HARNESS_SUBJECT=$absent" -- --subject "$mutant" --only '^99:'
fails=$(grep '^FAIL  ' <<<"$subj_out")
if [ "$subj_rc" != 1 ]; then detail="exit $subj_rc (want 1)"
elif [ "$fails" != "FAIL  $subj_check" ]; then detail="the checks failed are not that one alone"
elif ! grep -q '^PASS  ' <<<"$subj_out"; then detail="no check passed"
elif ! grep -qxF "  digest $digest" <<<"$subj_out"; then detail="the header does not carry the mutant's digest $digest"
elif ! grep -qF "ran 1 leg(s):" <<<"$subj_out"; then detail="the run did not account for its one leg"
fi
if [ -z "$detail" ]; then report 0 "$subj_label"; else report 1 "$subj_label" "$detail: $subj_out"; fi
detail=
subj_run -- --subject "$absent" --only '^99:'
{ [ "$subj_rc" = 2 ] && [ "$subj_out" = "no $absent" ]; } ||
  detail="--subject on a missing file: exit $subj_rc (want 2): $subj_out"
if [ -z "$detail" ]; then
  subj_run "OCANNL_TOOL_HARNESS_SUBJECT=$absent" -- --only '^99:'
  { [ "$subj_rc" = 2 ] && [ "$subj_out" = "no $absent" ]; } ||
    detail="the variable on a missing file: exit $subj_rc (want 2): $subj_out"
fi
if [ -z "$detail" ]; then
  subj_run -- --only '^99:' --subject
  { [ "$subj_rc" = 2 ] && [ "$subj_out" = "${SELF##*/}: --subject needs a file" ]; } ||
    detail="--subject with no value: exit $subj_rc (want 2): $subj_out"
fi
if [ -z "$detail" ]; then
  subj_run "OCANNL_TOOL_HARNESS_SUBJECT=$mutant" -- --subject= --only '^99:'
  { [ "$subj_rc" = 2 ] && [ "$subj_out" = "${SELF##*/}: --subject needs a file" ]; } ||
    detail="--subject= with an empty value: exit $subj_rc (want 2): $subj_out"
fi
if [ -z "$detail" ]; then report 0 "$subj_refuse_label"; else report 1 "$subj_refuse_label" "$detail"; fi
}
harness_leg_run '102: --subject tests the named copy in place of the script #harness' leg_102

test_run_close
