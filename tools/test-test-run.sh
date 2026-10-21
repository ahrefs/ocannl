#!/usr/bin/env bash
# Tests for the shared `group_alive` used by tools/test-run.sh -- the predicate
# its two callers ask "is anything in this process group still running?" --
# and for the two sentences `stop` prints about a surviving
# process group, which are what that predicate is for (gh-ocannl-742).
#
#   tools/test-test-run.sh          # run every leg
#   tools/test-test-run.sh --keep   # keep the temp dir for inspection
#   tools/test-test-run.sh --only '^81:'   # only the legs whose names match
#   tools/test-test-run.sh --only '#lifecycle'   # one named group of legs
#
# Each leg's body is a function, run by `harness_leg_run 'N: what it pins
# #group' leg_N` (scripts/harness-support.sh), its name led by the leg numbers
# below and closed by the one group it belongs to (gh-ocannl-1202):
# #group-alive (1-5), #stop (6-8, 67), #queries (39), #repeat (9-25, 38, 83-85),
# #options (26-28, 37), #width (29-34, 48-52, 64, 70), #digest (35-36,
# 68-69, 71-73, 80, 87), #fresh (81), #lifecycle (40-47, 65, 74: the run/start
# lifecycle with its signal legs, and both fork-window negative controls),
# #source (59-63, 66), #slot (53-58, 78, 88), #device (75-77), #credentials (79),
# #bisect (82), #harness (86). A body that cannot go on (an extraction that
# matched nothing, a fixture that is not what its checks assume) ends with
# `harness_abandon DETAIL; return`, failing that leg alone: never `exit`,
# which would end the harness and hide every leg after it (gh-ocannl-1374).
# A shared fixture outside the legs that is not what they assume is a FAIL
# where it is built, and the run goes on; only a subject that cannot be staged
# at all (a missing script, the sourced-files scan) still stops the harness. A
# selector is an extended regular expression over those names; one matching no
# leg is refused (exit 2) as nothing tested. The shared fixtures stay outside
# the gates, so a filtered run sets up and cleans up as a full one does. A
# filtered run prints FILTERED for each leg it left out, then accounts for
# the legs it ran and every check they skipped with its reason, and never
# prints the full-pass footer. With no selector every leg runs, as before.
#
# It is the sibling of scripts/test-setup-ocaml-env.sh, whose leg 1 (f) tests
# the same predicate in the SessionStart hook. Neither is a dune test: they
# spawn, STOP and kill process groups, which is a poor fit for `dune runtest`.
# The Ubuntu CI leg runs both directly so Linux decides the kernel-dependent
# zombie-group control (gh-ocannl-795).
#
# It tests the WORKING-TREE copy: `group_alive` is extracted from the shared
# scripts/process-group.sh; `ps_token`, `proc_identity_matches`, `proc_alive`
# come from tools/test-run.sh; the dxg legs read the shipping tools/box-jobs.sh
# and tools/lab-map.sh for endpoint lookup. The `stop` legs drive that same tool as a
# subprocess. Each extraction is asserted structurally before use, so a sed
# that matched nothing cannot leave every leg passing without testing anything.
#
# Legs:
#   1. extraction -- the functions came out of the shipping script.
#   2. a genuinely live group reads ALIVE (the other side of leg 4: without
#      this, a group_alive that answered "dead" always would pass leg 4).
#   3. a group with no members at all reads DEAD, and so does the bare signal.
#   4. a group holding nothing but a ZOMBIE reads dead, where the bare
#      `kill -0 -- -PGID` this replaced reads it as alive. That misreading is
#      what made `stop` able to announce an orphaned group holding only corpses.
#   5. a pgid that is not a positive decimal integer is refused -- 0 and
#      negatives are kill specials (caller's own group, broadcast).
#   6. `stop` on a group whose leader IGNORES TERM says so and escalates --
#      and the escalation kills the whole group. Every fixture group holds two
#      processes, or a leader-only kill would pass for a group kill.
#   7. `stop` on a group whose leader exits on TERM says the TERM went out and
#      asks for a re-run, rather than claiming the group ignored it.
#   8. `stop` on a run recorded by the version that kept its lock BESIDE the
#      worktree (no `runs` root on record) still attributes a leftover holding
#      that in-tree lock to the run, reaps it, and says so.
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
#  26. an option written AFTER the dune arguments is refused, having run
#      nothing -- the misplacement dune would otherwise digest as a red run.
#  27. the same option in its documented place is consumed, not forwarded.
#  28. past dune's own `--` the same word is a program argument, untouched.
#  29. the dxg width cap is injected, announced on stderr, written into the run
#      log and recorded in the command, for a GPU run that named no width.
#  30. a width the caller named is honored in every spelling dune takes, and
#      says so; one past dune's own `--` is the program's, so it still caps.
#  31. a CPU backend, or a box with no bridge, is neither capped nor told
#      anything.
#  32. with OCANNL_BACKEND unset the backend is resolved (gh-ocannl-1066): a
#      batch the configurations pin to cc is silent; one whose configuration
#      names hip, one reaching a stanza naming cuda, and an unreadable one are
#      capped, saying why; a named width is only told.
#  33. hip caps too, and the width is spliced after dune's subcommand.
#  34. the sweep's `box_jobs_sweep_jobs` helper and the injected cap read one table, with
#      the sweep's own override still winning; the sweep's width follows the
#      unit's destination (a WSL boot's bridge cap, a native boot's own), and
#      every box_jobs_sweep_jobs call site passes that destination.
#  35. an invocation dune's own parser refuses -- unknown option, unknown
#      subcommand, malformed operand -- digests as INVOCATION REFUSED quoting
#      dune's complaint: `run`/`wait` exit 2 over a RECORDED exit 1, `status`
#      keeps its publication 0, and dune was invoked exactly once; the
#      launch's own prelude in the log (backends, width) does not hide it.
#  36. controls: a red run (`Error:`/`File` lines, exit 1) still digests as
#      FAIL with exit 1 -- including one whose output opens with a `dune:`
#      line, and one that prints a complete nested dune refusal and then
#      goes on (the shape must be alone in the log, not merely first).
#  37. the `--cap` guard still refuses BEFORE dune is spawned: the two exit-2
#      refusals are told apart by whether dune was ever invoked.
#  38. `repeat` stops after a refused first iteration and exits 2 under the
#      same verdict, while a merely red iteration is still repeated in full.
#  39. read-only paths and lock-status: absent state, invalid input, physical
#      paths, recorded and legacy paths, held/released and uninspectable locks.
#  40-47. run/start lifecycle: completion, signals (a group TERM landing inside
#      the attached launcher's own liveness probe included, with its
#      unsettled-probe control; gh-ocannl-1362), stop, competing launches,
#      orphan/background groups, publication window, last and legacy locks,
#      and byte-identical source-tree preservation. Each signal reaches a
#      launcher that inherited it ignored, as from a background `&`,
#      `bg-run.sh spawn` or `nohup`, and reset it, so the legs pass whatever
#      this harness inherited (gh-ocannl-1236); a fixture that never becomes
#      ready fails and ends its leg, never the harness.
#  48-52 sit beside the dxg legs, after leg 34: the native caps
#      (gh-ocannl-1033), against faked KFD topologies and a faked NVIDIA
#      device.
#  48. hip on a small SDMA pool (minix's recorded topology) and on a larger one
#      (tuf's) is capped at that box's per-slot width, announced and recorded;
#      cuda on a native NVIDIA boot is capped at its own, and so is a CPU
#      backend on the fleet's rog-nv-linux (named by a faked FLEET_LOCAL_BOX,
#      or by the hostname through FLEET_HOSTNAME_MAP; gh-ocannl-1065), with
#      the `execution slot --cpu` note; the sweep's native tuf/hip and
#      rog-nv/cuda units run at exactly the width injected on that box.
#  49. negative controls: no KFD topology, a GPU node that reports no pool, a
#      CPU backend on each AMD hazard, on an NVIDIA box that is not the
#      fleet's rog and on a WSL boot of rog, and each GPU backend on the other
#      one's hazard -- no cap, no word.
#  50. a width the caller named is honored on every native hazard, and a dxg
#      boot that also reports a small pool keeps its dxg cap.
#  51. with OCANNL_BACKEND unset on a native hazard the width follows the
#      batch's resolved backends: cc is uncapped on the AMD boxes and an
#      unfleeted NVIDIA one; a configuration naming hip, a reached hip stanza
#      or an unread answer gets hip's width; cc on the fleet's rog gets the CPU
#      width; hip and cc beside a small pool on rog take the tighter.
#  52. the native widths derive from one place: each hip cap is its measured
#      budget over its slot count, and the numbers are the measured ones.
#  74. fork TERM is deferred until the parent owns its child pid and the child
#      resets its handlers; no-mask and inherited-handler controls prove both.
#  75-77 sit just before leg 70: a wedged device cannot hold a batch
#      (gh-ocannl-1211).
#  75. a test past --test-cap ends the run as TIMEOUT (142) naming it, the test
#      killed and the worktree idle, blaming a device only on a GPU batch; a
#      test inside the cap, a lifted cap, a slow dune and the slot's long-lived
#      holder do not (skipped where a $0-renamed fixture dune does not read
#      back as `dune` in ps); a ps serving a skewed etime (a resume from
#      suspend, gh-ocannl-1319) neither ends a run nor spares a hung test, and
#      a test that execs under new names is still capped.
#  76. a GPU batch whose device query does not answer is DEVICE UNHEALTHY (69),
#      the probe being the slot's command ahead of dune and dune never started;
#      an answer or a fast failure starts dune once; an unread batch asks every
#      GPU backend, a CPU batch none; the real probe is bin/device_props, built
#      inside the slot from DUNE_BUILD_DIR's tree.
#  77. `plan` reports the probe and the per-test cap as `run` would apply them
#      (GPU batches only, not @slow, not where run resolves nothing), and asks
#      no device.
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
#      child then goes unseen.
#  85 sits after leg 83 (Linux only): under a stand-in for a PID 1 that reaps
#      no orphans -- a child subreaper that waits for its own child only --
#      the zombie-only groups `repeat` and `run` leave are read as gone, not
#      refused: a capped set keeps TIMEOUT, a GPU set passes through the
#      resolution's inert-zombie branch, and `run` keeps its verdict, while
#      the stand-in must have held a zombie and nothing live (gh-ocannl-1376).
#      The negative controls take out the iteration shell's reaping trap and
#      that branch: each set is then refused, exit 2.
#  64-65 sit between legs 51 and 52: `plan`, and the run's first phase, in
#      which the supervisor's child resolves the backends (gh-ocannl-1066,
#      gh-ocannl-1106).
#  64. `plan` prints the resolved backends, the width (a caller's too) and
#      the slot, runs no dune, leaves no run, and is refused under a held lock.
#  65. a `stop`, or a signal to the launcher alone, while the backends
#      resolve (the published run's first phase) cancels the run: exit 143,
#      dune never started and the log says so, the worktree idle; a hung
#      reader is cut off by the cap, a reader's leftover reaped with the run.
#  53-58 sit at the end: the fleet's run-time slot, taken by the runner
#      (gh-ocannl-1004), against a fake fleet-worker.sh and the fake readers
#      every leg runs with.
#  53. the kind: --cpu only when none of the batch's resolved backends holds a
#      GPU; a GPU, empty, unknown or unreadable configuration, `exec`, a
#      command-line GPU backend, a reachable GPU stanza, or a reachability
#      answer outside its grammar is --gpu; the cap bounds the wait.
#  54. a refused or unreachable slot is SLOT REFUSED, exit 75, and dune never runs.
#  55. a red suite under a held slot is still FAIL; `start` prints the caller's argv;
#      a batch admitted inside a measurement hold is admitted, its red still FAIL.
#  56. no probe, OCANNL_TOOL_FLEET_WORKER=none, and repeat all run dune directly;
#      a stale first skill tree does not hide a second that answers the probe.
#  57. every tracked ocannl_config naming a backend is one the backends are
#      read from.
#  58. the readers batch-backends.sh builds are test/config's: ocannl_read_config
#      answers --read=backend, ocannl_slot_kind asks Test_utils.Slot_kind and
#      prints the grammar batch_resolve parses, and Slot_kind's config_dirs is
#      BATCH_CONFIG_DIRS.
#  78. a test configuration's backend counts only where ocannl_slot_kind says
#      it does (gh-ocannl-1227: a batch off its configuration-free list or a
#      plain build): beside a hip configuration on a small SDMA pool, a batch
#      it answers with no `reads config` line -- naming nothing, or naming cc --
#      stays uncapped and --cpu; one it answers with one is -j 4, --gpu.
#  88. a GPU box's batch resolving to cc alone (gh-ocannl-1433) runs and WARNs
#      in the announcements, the log and the digest, and `--expect-backend`
#      refuses it before dune (BACKEND REFUSED, exit 2); the GPU backend held,
#      cc expected, a Mac, a batch holding no backend say nothing; a WSL bridge
#      names cuda or hip and recommends neither; the option
#      still refuses on a Mac and refuses an unread batch; `plan` reports both.
#  79. credentials never reach dune (gh-ocannl-1280): a run exporting GH_TOKEN
#      and FOO_API_KEY hands dune neither, while a plain variable and the
#      OCANNL_TOOL_* exports pass -- observed by a fixture dune, and where a
#      real dune is at hand in the run's own trace; a copy with the scrub
#      cut out hands dune both, the negative control for each observer. With
#      no dune on PATH, credentials the opam environment select_dune sources
#      sets are scrubbed too, against a copy without that second scrub; and
#      tools/dune-quiet.sh, run directly, scrubs on its own. The scrub text
#      itself fails closed, under sh, bash and dash where installed: an
#      unrunnable env and a deny-listed non-identifier refuse.
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
#  67 sits after leg 8 (Linux only): a lock whose acquirer exited, held
#      through the description it shared, is found by `stop`'s census and
#      reaped, though /proc/locks names only the dead acquirer, or nobody
#      (gh-ocannl-1107); the negative control, the census trusting that
#      name, leaves the holder running (skipped where nothing is named).
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
#  82 sits just before leg 70: `bisect` (gh-ocannl-1225), over fixture
#      repositories whose commits tell the fixture dune what to do. A red
#      build is a skip, never bad, and the first bad commit is found under
#      the runner frozen at the start -- the range's last commit carries one
#      that cannot run; a range left with only skipped commits lists them
#      (exit 1); SLOT REFUSED and a cap's TIMEOUT in a step abort the bisect
#      (75, 142; git stopped by the step's 128) judging nothing; ends that do
#      not hold here bisect nothing; no backend, an unrelated GOOD or a
#      misplaced option is refused before any step; first-parent by default
#      never steps into a merged branch, and refuses a GOOD merged in through
#      a second parent (--no-first-parent takes it); a TERM reaps the step's dune and
#      exits 143; a descendant a step's run leaves holding the tree is
#      waited for before git moves the tree; a relative runs directory keeps
#      every step's runs under it; a dune-workspace above the runs directory
#      is refused leaving no worktree behind; a TERM during setup, or before a
#      step has a group to signal, unwinds with 143 starting nothing. Every
#      dune call sees the pinned backend and neither of the
#      runner's own variables, and the invoking checkout and its worktree
#      list end as they began.
#  70 sits at the end, after every leg that fakes a device: a leg that
#      names no device probe still reads the harness's absent defaults, the
#      width-cap probes answer "no device" through the shipping readers, and
#      a hip run is neither capped nor told anything (gh-ocannl-1108).
#  86 sits after leg 70: a copy of this harness, its test-run.sh respelled so
#      leg 79's credential mutant cannot be cut, abandons leg 79 alone and
#      still runs leg 70 to the footer; the same copy with that check spelled
#      as the `exit 2` it was aborts inside leg 79, and leg 70 never reports
#      (gh-ocannl-1374).

set -u

. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"

# Hermetic against the fleet (gh-ocannl-1004): on a fleet box the shipping
# script takes a real run-time slot through the deployed fleet-worker.sh, and
# the fixture runs below must neither hold nor wait for one. The slot legs
# name their fake fleet-worker.sh explicitly.
export OCANNL_TOOL_FLEET_WORKER=none
# Hermetic against the caller's dune presentation: a DUNE_DIFF_COMMAND in the
# environment is recorded by every launch and changes what the digest may
# conclude from a log (gh-ocannl-1055); the legs that want one set it.
unset DUNE_DIFF_COMMAND
# Hermetic against the caller's backend from the first leg, not from
# dxg_probe's first `unset`: a worker's shell pinned to hip or cuda outranks
# the fixture configurations every backend leg resolves (gh-ocannl-1108); the
# legs that want one export it. The box's own devices are masked as soon as
# the scratch directory exists (hermetic_probes, below).
unset OCANNL_BACKEND

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/test-run.sh"
GROUP_SRC="$HERE/../scripts/process-group.sh"
HOOK_SRC="$HERE/../scripts/setup-ocaml-env.sh"
JOBS_SRC="$HERE/box-jobs.sh"
SWEEP_SRC="$HERE/sweep.sh"
[ -f "$SRC" ] || { echo "no $SRC" >&2; exit 2; }
[ -f "$GROUP_SRC" ] || { echo "no $GROUP_SRC" >&2; exit 2; }
[ -f "$HOOK_SRC" ] || { echo "no $HOOK_SRC" >&2; exit 2; }
[ -f "$JOBS_SRC" ] || { echo "no $JOBS_SRC" >&2; exit 2; }
[ -f "$SWEEP_SRC" ] || { echo "no $SWEEP_SRC" >&2; exit 2; }


# The harness needs the two facts about a process that `group_alive` needs, read
# INDEPENDENTLY of it: its state and its process group. Neither is available the
# same way everywhere -- a Git Bash/MSYS `ps` takes no `-o` at all, which the
# Windows CI job depends on and which would otherwise leave every state probe
# here reading "gone". So: /proc where it answers (MSYS has one), `ps` where it
# does not, and the prerequisite check below refuses to let a leg run on a
# system where neither does -- an unreadable state must skip a leg, never pass
# or fail one (Codex review round 1, P2).
pstate() { # <pid> -> its one-letter state, empty where this system will not say
  local line
  if [ -r "/proc/$1/stat" ]; then
    # Grouped, not `read ... 2>/dev/null`: the shell reports a failed
    # redirection before the command's own stderr redirection applies, and
    # these readers are asked about pids that are expected to be gone.
    { read -r line <"/proc/$1/stat"; } 2>/dev/null || return 0
    line=${line##*) }               # comm may itself hold ") "
    # shellcheck disable=SC2086
    set -- $line                    # `state ppid pgrp ...`
    printf '%s' "${1:-}"
    return 0
  fi
  ps -o state= -p "$1" 2>/dev/null | tr -d ' ' | cut -c1
}
ppgid() { # <pid> -> its process group, empty where this system will not say
  local line
  if [ -r "/proc/$1/stat" ]; then
    { read -r line <"/proc/$1/stat"; } 2>/dev/null || return 0
    line=${line##*) }
    # shellcheck disable=SC2086
    set -- $line
    printf '%s' "${3:-}"
    return 0
  fi
  ps -o pgid= -p "$1" 2>/dev/null | tr -d ' '
}
# Probed against a process known to be alive and to have a group -- this one.
# An empty answer here means the reader is absent, which is a different fact
# from a process being gone, and the two are indistinguishable at a leg.
have_state=1; [ -n "$(pstate $$)" ] || have_state=0
have_pgid=1;  [ -n "$(ppgid $$)" ]  || have_pgid=0

echo "testing $SRC and $GROUP_SRC"
printf '  digest %s\n' "$( (cksum <"$SRC") 2>/dev/null || echo '?')"
printf '  group digest %s\n' "$( (cksum <"$GROUP_SRC") 2>/dev/null || echo '?')"
printf '  state reader: %s; pgid reader: %s\n' \
  "$([ "$have_state" = 1 ] && echo present || echo ABSENT)" \
  "$([ "$have_pgid" = 1 ] && echo present || echo ABSENT)"
# What this harness inherited as ignored (gh-ocannl-1236): `bg-run.sh spawn`,
# `nohup` and `execution slot --bg` start it with HUP ignored, and a
# non-interactive bash cannot trap a signal ignored at its entry. The legs
# that deliver a signal reset it for their fixture, so this line explains a
# run rather than deciding one; perl reads the disposition the shell cannot.
printf '  inherited ignored signals: %s\n' \
  "$(perl -e 'my @i = grep { ($SIG{$_} // "") eq "IGNORE" } qw(HUP INT QUIT TERM);
              print @i ? "@i" : "none"' 2>/dev/null || echo '?')"

# Checked, not assumed: nothing here uses `set -e`, so a `mktemp` that fails
# would leave TMP empty and `rm -rf "$TMP"` would be handed the ROOT.
zparent=""   # leg 4's self-stopping zombie maker; cleanup must resume it
livepid=""   # leg 2's live group leader
leader=""    # the stop legs' current group leader
member=""    # and the second process it put in that group
member_token=""  # ... and its start token, since it is not this shell's child
repeat_pid="" # leg 14's active repeat coordinator
legacy_holder="" # leg 8's holder of the in-tree lock
inherit_holder="" # leg 67's holder of the inherited lock
escape_pid="" # leg 24's session-escaped descendant
escape_release=""
harness_scratch "test-test-run"
# Hermetic against the box's devices (gh-ocannl-1108), as
# tools/test-mutation-run.sh is: the three probes the width cap reads (the dxg
# bridge, the KFD topology, the NVIDIA control device) name paths that do not
# exist, so on a WSL, minix, tuf or rog boot every leg sees the Mac's answer --
# a leg that is not about the cap would otherwise gain one (a lifecycle run on
# minix is capped at -j 4), and a cap leg's negative control would read the
# box instead of its fixture. A leg opts in to a device by exporting a fixture
# path (dxg_probe, native_probe) or prefixing one, and goes back through
# hermetic_probes, never `unset`: an unset (or empty) probe falls back to the
# real device. Leg 70 pins this from the end of the file.
probe_dxg_default=$TMP/no-such-dxg
probe_kfd_default=$TMP/no-such-kfd
probe_nvidia_default=$TMP/no-such-nvidia
hermetic_probes() {
  export OCANNL_TOOL_DXG_DEVICE=$probe_dxg_default \
    OCANNL_TOOL_KFD_TOPOLOGY=$probe_kfd_default \
    OCANNL_TOOL_NVIDIA_DEVICE=$probe_nvidia_default
}
hermetic_probes
cleanup_fixture() {
  if declare -F lifecycle_cleanup >/dev/null; then lifecycle_cleanup; fi
  # Leg 4's zombie maker STOPS ITSELF and is resumed at the end of the leg.
  # Interrupted in between, nothing else would ever resume it: it would be
  # reparented to PID 1 still stopped, still holding its zombie child. Killing
  # it is not the answer either -- that orphans the zombie onto a PID 1 that, in
  # the very environment this leg is about, never reaps. Resume it so it reaps
  # its own child, and only insist if it will not go.
  local waited
  if [ -n "${zparent:-}" ] && kill -0 "$zparent" 2>/dev/null; then
    kill -CONT "$zparent" 2>/dev/null
    for waited in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$zparent" 2>/dev/null || break
      sleep 0.2
    done
    kill -KILL "$zparent" 2>/dev/null
    wait "$zparent" 2>/dev/null
  fi
  if [ -n "${livepid:-}" ] && kill -0 "$livepid" 2>/dev/null; then
    kill -KILL -- "-$livepid" 2>/dev/null
    kill -KILL "$livepid" 2>/dev/null
    wait "$livepid" 2>/dev/null
  fi
  # The stop legs' leader ignores TERM by design, so only KILL removes it --
  # and the group with it, since a leg interrupted before its `stop` leaves the
  # whole fixture group behind.
  if [ -n "${leader:-}" ] && kill -0 "$leader" 2>/dev/null; then
    kill -KILL -- "-$leader" 2>/dev/null
    kill -KILL "$leader" 2>/dev/null
    wait "$leader" 2>/dev/null
  fi
  # The group's second member is not this shell's child, so it is named
  # directly as well: a group kill that failed is exactly the case where it
  # would otherwise be left behind. Identity-checked -- see kill_member.
  if [ -n "${member:-}" ]; then kill_member; fi
  if [ -n "${repeat_pid:-}" ] && kill -0 "$repeat_pid" 2>/dev/null; then
    kill -TERM "$repeat_pid" 2>/dev/null
    wait "$repeat_pid" 2>/dev/null
  fi
  if [ -n "${legacy_holder:-}" ] && kill -0 "$legacy_holder" 2>/dev/null; then
    kill -KILL "$legacy_holder" 2>/dev/null
    wait "$legacy_holder" 2>/dev/null
  fi
  if declare -F inherit_kill >/dev/null; then inherit_kill; fi
  [ -z "${escape_release:-}" ] || touch "$escape_release"
  if [ -n "${escape_pid:-}" ] && kill -0 "$escape_pid" 2>/dev/null; then
    kill -KILL "$escape_pid" 2>/dev/null
  fi
  return 0
}
# Without these, a TERM or a Ctrl-C kills the shell outright and the EXIT trap
# never runs -- which is how an interrupted run could leave a stopped zombie
# maker behind, or a fixture process group outlive the harness that forked it.
# Exiting from the handler is what gets EXIT to fire.
#
# The INT arm covers the way this is actually interrupted, a Ctrl-C at a
# terminal. It is inert when the harness is itself a BACKGROUND job of a
# non-interactive shell -- such a child inherits SIGINT ignored, and a signal
# ignored on entry cannot be re-trapped -- so a scripted test of the cleanup
# path has to signal TERM to see anything happen.

# ---------------------------------------------------------------------------
# Leg 1: extraction
# ---------------------------------------------------------------------------
# Checked structurally -- opens with the header, closes with the brace, has a
# body -- rather than by grepping for one line of it: this guard must still hold
# while the function under test is being mutated to see a leg go red. The header
# is matched as a PREFIX, since it carries a trailing comment.
#
# `ps_token` is extracted alongside `group_alive` because the stop legs below
# have to FORGE a run directory's leader token, and a token spelled differently
# from the one the shipping script recomputes would leave every one of them
# falling through to "nothing left to signal" -- passing no leg, but testing
# neither sentence either.
for fn in group_alive ps_token proc_identity_matches proc_alive; do
  if [ "$fn" = group_alive ]; then fn_src="$GROUP_SRC"; else fn_src="$SRC"; fi
  g_min=10
  sed -n "/^$fn() {/,/^}/p" "$fn_src" >"$TMP/$fn.sh"
  g_lines="$(wc -l <"$TMP/$fn.sh" | tr -d ' ')"
  g_head="$(head -n1 "$TMP/$fn.sh")"
  case $g_head in "$fn() {"*) g_ok=1 ;; *) g_ok=0 ;; esac
  if [ "$g_ok" = 0 ] \
     || [ "$(tail -n1 "$TMP/$fn.sh")" != "}" ] || [ "$g_lines" -lt "$g_min" ]; then
    report 1 "$fn: extracted" "sed did not capture the function body from $fn_src"
  else
    report 0 "$fn: extracted ($g_lines lines)"
    # shellcheck disable=SC1090
    . "$TMP/$fn.sh"
  fi
done

# The decision behind this issue is one definition, not two copies tested in
# parallel. Count definitions across the shared helper and both production
# callers so either copy coming back fails the harness that CI runs.
leg_1() {
group_definition_count() { awk '/^group_alive\(\)/ { n++ } END { print n + 0 }' "$@"; }
one_group_definition() { [ "$(group_definition_count "$@")" = 1 ]; }
if one_group_definition "$GROUP_SRC" "$SRC" "$HOOK_SRC"; then
  report 0 "group_alive has one shared production definition"
else
  report 1 "group_alive has one shared production definition" \
    "found $(group_definition_count "$GROUP_SRC" "$SRC" "$HOOK_SRC") definitions across $GROUP_SRC, $SRC and $HOOK_SRC"
fi
# Synthetic duplicate: the clean production tree alone cannot distinguish a
# working uniqueness check from one that always answers yes.
cp "$GROUP_SRC" "$TMP/duplicate-process-group.sh"
if one_group_definition "$GROUP_SRC" "$SRC" "$HOOK_SRC" "$TMP/duplicate-process-group.sh"; then
  report 1 "the shared-definition check rejects a duplicate" \
    "the production definition plus a copied definition still read as unique"
else
  report 0 "the shared-definition check rejects a duplicate"
fi
}
harness_leg_run '1: group_alive has one shared production definition #group-alive' leg_1

# ---------------------------------------------------------------------------
# Leg 2: a live group reads alive
# ---------------------------------------------------------------------------
# `set -m` puts the child in a process group of its own, so its pid IS a pgid
# holding exactly one live process -- the shape all four call sites signal.
# That the child really leads its own group has to be CHECKED (a shell whose
# setpgrp does not take would otherwise have this leg quietly testing the
# harness's own group), so both legs need the pgid reader.
leg_2_3() {
live_label="a group holding a running process reads as alive"
empty_label="a group with no members left reads as dead"
if [ "$have_pgid" = 0 ]; then
  skip "$live_label" "no way to read a process's group on this system"
  skip "$empty_label" "no way to read a process's group on this system"
else
  set -m
  sleep 30 >/dev/null 2>&1 </dev/null &
  livepid=$!
  set +m
  lpgid="$(ppgid "$livepid")"
  if [ "$lpgid" != "$livepid" ]; then
    report 1 "$live_label" \
      "the child did not lead its own group (pgid '${lpgid:-gone}' vs pid $livepid)"
  elif ! kill -0 -- "-$livepid" 2>/dev/null; then
    report 1 "$live_label" "the group is not even signal-reachable; the leg tested nothing"
  elif group_alive "$livepid"; then
    report 0 "$live_label"
  else
    report 1 "$live_label" "group_alive said dead for a group with a live sleep in it"
  fi

  # -------------------------------------------------------------------------
  # Leg 3: an empty group reads dead
  # -------------------------------------------------------------------------
  kill -KILL -- "-$livepid" 2>/dev/null
  wait "$livepid" 2>/dev/null   # reaped here, so the group really is empty
  livepid=""
  if kill -0 -- "-$lpgid" 2>/dev/null; then
    report 1 "$empty_label" "pgid $lpgid is still signal-reachable after the kill and reap"
  elif group_alive "$lpgid"; then
    report 1 "$empty_label" "group_alive said alive for an empty group"
  else
    report 0 "$empty_label"
  fi
fi
}
harness_leg_run '2-3: a live group reads alive, an emptied one dead #group-alive' leg_2_3

# ---------------------------------------------------------------------------
# Leg 4: a zombie-only group reads dead
# ---------------------------------------------------------------------------
# The negative control for the whole change: `kill -0` must still say YES here,
# or the leg would pass without exercising the difference. Under an init that
# reaps, such a corpse is transient and the bare probe merely lost a race with
# it; under one that does not -- the ordinary container case -- it is PERMANENT,
# so waiting it out was never the fix.
#
# Making a zombie that is reliably still a zombie when looked at, and that leaves
# nothing behind afterwards, takes care. The parent puts the child in a process
# group of ITS OWN (so the group holds the zombie and nothing else), then STOPs
# itself before the child exits: a stopped shell runs no SIGCHLD handler, so it
# cannot reap, and the child stays a zombie for as long as the leg needs.
#
# All of that needs a state reader, and needs it INDEPENDENT of the function
# under test: without one, "is it a zombie yet" reads the same as "it is gone",
# the leg would sit out its whole retry budget and then judge `group_alive` on a
# premise it never established -- and the cleanup assertion at the end would
# read an empty state as "reaped" and pass on a system that cannot see the
# corpse at all. So the whole leg, zombie maker included, is skipped there.
leg_4() {
zlabel="a group holding nothing but a zombie reads as dead"
clabel="the state reader alone rejects a zombie-only group (signal probe forced to say alive)"
rlabel="the zombie leg reaps its own zombie, leaving no process-table entry"
ilabel="a zombie retains identity without being reported alive"
if [ "$have_state" = 0 ]; then
  skip "$zlabel" "no way to read a process's state on this system"
  skip "$clabel" "no way to read a process's state on this system"
  skip "$rlabel" "the zombie leg did not run, so it left nothing to reap"
  skip "$ilabel" "the zombie leg did not run, so identity was not testable"
else
zpidfile="$TMP/zombie.pid"; rm -f "$zpidfile"
bash -c 'set -m
         sleep 0.5 >/dev/null 2>&1 </dev/null &
         echo $! >"$1"
         set +m
         kill -STOP $$
         wait' _ "$zpidfile" >/dev/null 2>&1 </dev/null &
zparent=$!
zpid=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  [ -s "$zpidfile" ] && zpid="$(cat "$zpidfile")" && break
  sleep 0.1
done
# Wait for real zombiehood rather than guessing at it, through `pstate`, which
# is this harness's own reader and shares no code with `group_alive`.
zstate=""
if [ -n "$zpid" ]; then
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
    zstate="$(pstate "$zpid")"
    case "$zstate" in Z*) break ;; esac
    sleep 0.1
  done
fi
if [ -z "$zpid" ]; then
  report 1 "$zlabel" "could not start the zombie maker"
  report 1 "$clabel" "could not start the zombie maker"
else
  case "$zstate" in
    Z*)
      # Whether the BARE probe over-reports here is a property of the kernel:
      # Linux (and every container on it) counts a zombie as a group member and
      # says alive, which is the bug; Darwin's killpg answers ESRCH once the
      # group holds only corpses, so the bare probe happens to agree there. The
      # claim itself holds on both, and the control below runs on both.
      if kill -0 -- "-$zpid" 2>/dev/null; then
        zwhere="the bare group signal probe says ALIVE here -- the misreading this fixes"
      else
        zwhere="this kernel's killpg already answers dead for a zombie-only group"
      fi
      if group_alive "$zpid"; then
        report 1 "$zlabel" \
          "group_alive counted zombie group $zpid as work -- stop can report a phantom orphaned group"
      else
        report 0 "$zlabel ($zwhere)"
      fi
      printf '%s\n' "$zpid" >"$TMP/zombie-identity.pid"
      ps_token "$zpid" >"$TMP/zombie-identity.token"
      if proc_identity_matches "$TMP/zombie-identity.pid" "$TMP/zombie-identity.token" \
         && ! proc_alive "$TMP/zombie-identity.pid" "$TMP/zombie-identity.token"; then
        report 0 "$ilabel"
      else
        report 1 "$ilabel" \
          "the zombie either lost its recorded identity or was incorrectly reported live"
      fi
      # The negative control, run everywhere: shadow `kill` so the signal probe
      # inside group_alive succeeds, which is exactly what the bare check did on
      # the kernel where this was reproduced. The state reader must still refuse
      # the group -- otherwise this platform's pass above was the gate's doing
      # alone and the ladder itself is untested here.
      (
        kill() { case "$*" in -0*) return 0 ;; *) command kill "$@" ;; esac; }
        if ! kill -0 -- "-$zpid"; then
          exit 3   # the shadow did not take; the control tested nothing
        fi
        group_alive "$zpid" && exit 1
        exit 0
      )
      case $? in
        0) report 0 "$clabel" ;;
        3) report 1 "$clabel" 'the `kill` shadow did not take effect' ;;
        *) report 1 "$clabel" \
             "with the signal probe forced alive, group_alive called a zombie-only group alive" ;;
      esac
      ;;
    *)
      report 1 "$zlabel" \
        "the child never reached state Z (saw '${zstate:-gone}'), so the leg tested nothing"
      report 1 "$clabel" \
        "the child never reached state Z (saw '${zstate:-gone}'), so the leg tested nothing"
      report 1 "$ilabel" \
        "the child never reached state Z (saw '${zstate:-gone}'), so identity was not tested"
      ;;
  esac
fi
# Let the parent reap its own child rather than orphaning the zombie, then check
# we really left nothing behind.
kill -CONT "$zparent" 2>/dev/null
wait "$zparent" 2>/dev/null
zparent=""
zleft=""
if [ -n "$zpid" ]; then
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    zleft="$(pstate "$zpid")"
    [ -z "$zleft" ] && break
    sleep 0.1
  done
fi
if [ -z "$zleft" ]; then
  report 0 "$rlabel"
else
  report 1 "$rlabel" "pid $zpid is still present as '$zleft'"
fi
fi
}
harness_leg_run '4: a zombie-only group reads dead #group-alive' leg_4

# ---------------------------------------------------------------------------
# Leg 5: only a positive decimal integer is a pgid
# ---------------------------------------------------------------------------
# A corrupted or forged pgid file must never reach kill: `kill -0 -- -0` targets
# the CALLER's own group (which is alive, so the answer would be a confident
# yes) and a negative reading broadcasts.
leg_5() {
bad=""
for candidate in "" "0" "12x" "-1" "abc" " 7"; do
  if group_alive "$candidate"; then bad="$bad '$candidate'"; fi
done
if [ -z "$bad" ]; then
  report 0 "a pgid that is not a positive decimal integer is refused"
else
  report 1 "a pgid that is not a positive decimal integer is refused" \
    "accepted:$bad"
fi
}
harness_leg_run '5: only a positive decimal integer is a pgid #group-alive' leg_5

# ---------------------------------------------------------------------------
# Legs 6-7: the two sentences `stop` prints about a surviving process group
# ---------------------------------------------------------------------------
# They are driven for real: `tools/test-run.sh stop last` against a FORGED run
# directory -- the metadata a launch records (cmd, cap, wt, log, pgid, gtoken)
# written by hand around a process group this harness controls -- and the answer
# read from what stop actually printed.
#
# Nothing of the ambient run history is touched: OCANNL_TOOL_TEST_RUNS is
# pointed at this run's temp dir, so the `last` pointer these legs move is the
# fixture's own and dies with the temp dir. `last` rather than an explicit run
# directory, because that is the spelling an operator uses and it exercises the
# pointer too; the public `paths last` query supplies its filename without
# reconstructing the shipping script's worktree-key implementation.
#
# The fixture deliberately records no pid/ptoken and leaves no `exit` file: a
# run with either is owned or finished, and the group branch is the one reached
# by a run whose supervisor is gone while its process group is not.
l_ignored="stop: a group whose leader ignores TERM is reported with the unreaped-exits caveat"
l_killed="stop: that escalation kills the whole group, not just its leader"
l_took="stop: a group whose leader takes the TERM is reported as TERMed, not as ignoring it"

STOP_RUNS="$TMP/runs"
STOP_WT="$TMP/wt"
mkdir -p "$STOP_RUNS" "$STOP_WT"
STOP_RUNS=$(cd "$STOP_RUNS" && pwd -P)
STOP_WT=$(cd "$STOP_WT" && pwd -P)
stop_last=$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" paths last) || stop_last=

# A leg that cannot establish its premise must skip, not pass: without a pgid
# reader the fixture could name THIS shell's group and stop would signal the
# harness itself, and without a start token group_verified refuses the fixture
# and every leg reads the same "nothing left to signal".
stop_skip=""
if [ "$have_pgid" = 0 ]; then
  stop_skip="no way to read a process's group, so a fixture group cannot be told from this shell's own"
elif [ -z "$(ps_token $$)" ]; then
  stop_skip="this system records no start token, so a forged leader cannot be identity-verified"
fi

mk_fixture() { # <tag> <pgid> <last pointer path>; 0 iff the run is reachable as `last`
  local d="$STOP_RUNS/19700101-000000-$1"
  mkdir -p "$d" "$STOP_WT" || return 1
  # cmd and cap are what resolve_run demands before it will trust a directory
  # enough to signal anything named in it.
  printf 'runtest (test-test-run.sh fixture %s)\n' "$1" >"$d/cmd"
  printf '0\n' >"$d/cap"
  printf '%s\n' "$STOP_WT" >"$d/wt"
  # `runs` names the state root the lock would live under; without it the
  # script reads the run as one of the version that kept its lock in the tree.
  printf '%s\n' "$STOP_RUNS" >"$d/runs"
  : >"$d/log"
  printf '%s\n' "$2" >"$d/pgid"
  ps_token "$2" >"$d/gtoken"
  # group_verified refuses an empty token outright, so it is checked here
  # rather than left to reappear as a wording failure three legs later.
  [ -s "$d/gtoken" ] || return 1
  printf '%s\n' "$d" >"$3" || return 1
  return 0
}

start_leader() { # <marker> <bash -c body>; sets $leader (and its $member)
  local m=$1 body=$2 p g i
  leader=""; member=""; member_token=""
  rm -f "$m"
  # `set -m` puts the child in its own process group -- the shape stop signals,
  # and the only shape it is safe to hand a group kill.
  set -m
  bash -c "$body" _ "$m" >/dev/null 2>&1 </dev/null &
  p=$!
  set +m
  # Recorded BEFORE the checks below rather than after them: job control has
  # just put this child out of reach of any signal aimed at the harness, so an
  # INT arriving in the window between the fork and the recording would leave
  # the EXIT cleanup with nothing to kill and the fixture group running past
  # the run (Codex round 1, P2). Recording early is safe in the case the checks
  # are about to reject: a group kill aimed at a pid that never led a group
  # names a group that does not exist.
  leader=$p
  # The body writes the marker -- carrying the pid of the SECOND member it put
  # in the group -- after installing its TERM disposition, so this wait is what
  # makes the leg's premise true rather than merely likely.
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    [ -s "$m" ] && break
    sleep 0.1
  done
  g="$(ppgid "$p")"
  member="$(tr -dc '0-9' <"$m" 2>/dev/null)"
  member_token="$(ps_token "$member" 2>/dev/null)"
  # The second member is checked into the group, not assumed into it: a fixture
  # whose extra process ended up somewhere else would leave leg 6 judging the
  # escalation on the leader alone again, which is the hole it exists to close.
  if [ ! -s "$m" ] || [ "$g" != "$p" ] || [ -z "$member" ] ||
     [ "$(ppgid "$member")" != "$p" ]; then
    end_leader
    return 1
  fi
  return 0
}
kill_member() { # KILL the group's second member -- only if it is still IT
  # Unlike the leader, the member is a GRANDchild: nothing here holds it as a
  # zombie, so once `stop` has killed it, init reaps it and its pid is free to
  # be recycled. An unconditional numeric kill from the EXIT trap could then
  # land on an unrelated process on a busy host (Codex round 2, P2). The gate
  # is the same one the shipping script uses for every pid it did not just
  # fork: the recorded start token has to still be the one that pid answers
  # with. (lstart's one-second resolution is the known floor there, which is
  # why test-run.sh prefers /proc starttime where it exists.)
  [ -n "${member:-}" ] && [ -n "${member_token:-}" ] || return 0
  [ "$(ps_token "$member" 2>/dev/null)" = "$member_token" ] || return 0
  kill -KILL "$member" 2>/dev/null
  return 0
}
end_leader() { # whatever the leg concluded, the whole fixture group goes
  if [ -n "${leader:-}" ]; then
    kill -KILL -- "-$leader" 2>/dev/null
    kill -KILL "$leader" 2>/dev/null
  fi
  # Named separately as well as reached through the group: this is the
  # harness's backstop for the very defect leg 6 now tests for, an escalation
  # that reached only the leader. The leader needs no such check -- it is this
  # shell's own child, held as a zombie until the `wait` below, so its pid
  # cannot be recycled underneath us.
  kill_member
  if [ -n "${leader:-}" ]; then wait "$leader" 2>/dev/null; fi
  leader=""; member=""; member_token=""
  return 0
}
# Each fixture group holds TWO processes, because a group holding only its
# leader cannot tell a group kill from a leader kill: with one member, the
# incorrect `kill -KILL "$pg"` passes every leg here while real dune children
# would survive it (Codex round 1, P2). The member is a plain background
# `sleep` the leader forks before exec'ing its own, so it is in the group and
# inherits the leader's TERM disposition -- SIG_IGN survives both fork and
# exec, so an IGNORING fixture is TERM-proof as a whole, and a TAKING one dies
# as a whole. The marker doubles as the member's pid, so the harness can track
# a process that is not its own child.
body_ignores='trap "" TERM; sleep 600 & echo $! >"$1"; exec sleep 600'
body_takes='sleep 600 & echo $! >"$1"; exec sleep 600'
stop_out=""; stop_rc=""; stop_pg=""; stop_member=""; stop_err=""; stop_diag=""
stop_probe() { # <tag> <leader body> <script> <last pointer path>
  stop_out=""; stop_rc=""; stop_pg=""; stop_member=""; stop_err=""; stop_diag=""
  if ! start_leader "$TMP/$1.marker" "$2"; then
    stop_err="could not start a two-process group leader for the '$1' fixture"
    return 1
  fi
  # Kept past end_leader, which clears the live handles: the escalation claim
  # is about processes that are supposed to be gone by the time it is asked.
  stop_pg=$leader
  stop_member=$member
  if ! mk_fixture "$1" "$stop_pg" "$4"; then
    stop_err="could not build the '$1' fixture run directory under $STOP_RUNS"
    return 1
  fi
  # stdout and stderr kept APART. The sentence is stdout, and it is matched
  # whole; stderr carries diagnostics that are not part of the answer and can
  # appear for reasons that have nothing to do with the wording -- a /proc
  # entry vanishing under group_alive's scan being the one that actually bit
  # (Codex round 2, P2). Merging the two made a passing stop fail an exact
  # match. It is kept and shown on failure rather than discarded, since a
  # failing leg is exactly when it is worth reading.
  stop_out="$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$3" stop last 2>"$TMP/$1.stderr")"
  stop_rc=$?
  stop_diag="$(cat "$TMP/$1.stderr" 2>/dev/null)"
  return 0
}
said() { # <label> <the whole line stop must have printed>
  # The WHOLE output, not a substring of it, and a clean exit with it: these
  # two legs exist to keep the two sentences apart, and a containment test
  # would pass both for a stop that printed two of them at once (Codex
  # round 1, P2).
  if [ "${stop_rc:-1}" = 0 ] && [ "$stop_out" = "$2" ]; then
    report 0 "$1"
  else
    report 1 "$1" \
      "expected exactly \"$2\"; stop (exit ${stop_rc:-?}) printed: ${stop_out:-<nothing>}"
    [ -n "${stop_diag:-}" ] && printf '      on stderr: %s\n' "$stop_diag"
  fi
}

leg_6_8() {
[ -n "$stop_last" ] || { harness_abandon "stop: paths last named no pointer under $STOP_RUNS"; return; }
if [ -n "$stop_skip" ]; then
  skip "$l_ignored" "$stop_skip"
  skip "$l_killed" "$stop_skip"
  skip "$l_took" "$stop_skip"
else
  # -------------------------------------------------------------------------
  # Leg 6: the leader ignores TERM
  # -------------------------------------------------------------------------
  if stop_probe ignores "$body_ignores" "$SRC" "$stop_last"; then
    said "$l_ignored" \
      "orphaned process group $stop_pg survived TERM (possibly only as unreaped exited processes); escalated to KILL"
    # The sentence is a claim about what stop DID, so the doing is checked too:
    # an escalation that only announced itself would leave the group holding the
    # worktree lock, which is the whole reason stop reaches for KILL here.
    if [ "$have_state" = 0 ]; then
      skip "$l_killed" "no way to read a process's state on this system"
    else
      # Both members, so a KILL that reached only the leader fails here.
      k_left=""
      for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        k_left=""
        for k_pid in $stop_pg $stop_member; do
          case "$(pstate "$k_pid")" in '' | Z*) ;; *) k_left="$k_left $k_pid" ;; esac
        done
        [ -z "$k_left" ] && break
        sleep 0.1
      done
      if [ -z "$k_left" ]; then
        report 0 "$l_killed"
      else
        report 1 "$l_killed" \
          "still running two seconds after the escalation:$k_left (group $stop_pg)"
      fi
    fi
  else
    report 1 "$l_ignored" "$stop_err"
    report 1 "$l_killed" "$stop_err"
  fi
  end_leader

  # -------------------------------------------------------------------------
  # Leg 7: the leader takes the TERM
  # -------------------------------------------------------------------------
  # The difference from leg 6 is one `trap` in the leader and nothing else, so
  # a stop that reported both the same way would fail exactly one of them.
  if stop_probe takes "$body_takes" "$SRC" "$stop_last"; then
    said "$l_took" "sent TERM to the orphaned process group $stop_pg; re-run stop to confirm"
  else
    report 1 "$l_took" "$stop_err"
  fi
  end_leader

  # -------------------------------------------------------------------------
  # Leg 8: a legacy run's leftover is reaped through its in-tree lock
  # -------------------------------------------------------------------------
  # The version before gh-ocannl-606 kept the lock and its owner pointer
  # BESIDE the worktree, and a run directory it recorded has no `runs` root.
  # Its leftovers -- a descendant dune holding the inherited lock fd -- hold
  # THAT lock, so `stop` must read the run under those paths or it reports
  # nothing to reap while the lock stays held. The fixture: a run directory
  # with no `runs`, no pid and no pgid (only the lock can attribute
  # anything), the in-tree owner pointer naming it, and a perl holding an
  # flock on the in-tree lock file, which stop's census must find and TERM.
  l_legacy="stop: a legacy run's leftover holding the in-tree lock is attributed and reaped"
  legacy_dir=$STOP_RUNS/19700101-000000-legacy
  mkdir -p "$legacy_dir" "$STOP_WT"
  printf 'runtest (test-test-run.sh fixture legacy)\n' >"$legacy_dir/cmd"
  printf '0\n' >"$legacy_dir/cap"
  printf '%s\n' "$STOP_WT" >"$legacy_dir/wt"
  : >"$legacy_dir/log"
  printf '%s\n' "$legacy_dir" >"$STOP_WT/.test-run.lock.owner"
  printf '%s\n' "$legacy_dir" >"$stop_last"
  rm -f "$TMP/legacy.pid"
  perl -e 'use Fcntl ":flock";
           open(my $fh, ">>", $ARGV[0]) or exit 1;
           flock($fh, LOCK_EX | LOCK_NB) or exit 1;
           $| = 1; print "$$\n"; sleep 600' \
    "$STOP_WT/.test-run.lock" >"$TMP/legacy.pid" 2>/dev/null </dev/null &
  legacy_holder=$!
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    [ -s "$TMP/legacy.pid" ] && break
    sleep 0.1
  done
  # The premise, checked: the lock really is held before stop is asked.
  legacy_lock_out=$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" lock-status last)
  legacy_lock_rc=$?
  if [ ! -s "$TMP/legacy.pid" ] || [ "$legacy_lock_rc" != 3 ] || [ "$legacy_lock_out" != held ]; then
    report 1 "$l_legacy" "the fixture holder did not take the in-tree lock"
  else
    stop_out="$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" stop last 2>"$TMP/legacy.stderr")"
    stop_rc=$?
    stop_diag="$(cat "$TMP/legacy.stderr" 2>/dev/null)"
    said "$l_legacy" \
      "run is dead without a verdict, but its leftover processes held the worktree lock; reaped them"
    # The claim is about what stop DID: the holder must be gone and the lock free.
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
      kill -0 "$legacy_holder" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$legacy_holder" 2>/dev/null; then
      report 1 "$l_legacy (the holder is gone and the lock is free)" \
        "pid $legacy_holder still holds $STOP_WT/.test-run.lock after stop"
    elif OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" lock-status last >"$TMP/legacy-lock-after" \
         && [ "$(cat "$TMP/legacy-lock-after")" = idle ]; then
      report 0 "$l_legacy (the holder is gone and the lock is free)"
    else
      report 1 "$l_legacy (the holder is gone and the lock is free)" \
        "$STOP_WT/.test-run.lock is still locked after stop"
    fi
  fi
  if kill -0 "$legacy_holder" 2>/dev/null; then kill -KILL "$legacy_holder" 2>/dev/null; fi
  wait "$legacy_holder" 2>/dev/null
  legacy_holder=""
  rm -f "$STOP_WT/.test-run.lock" "$STOP_WT/.test-run.lock.owner"
fi
}
harness_leg_run '6-8: the sentences stop prints, and a legacy leftover reaped #stop' leg_6_8

# ---------------------------------------------------------------------------
# Everything the shipping script SOURCES has to exist in a fixture root, and the
# list is derived from the script rather than kept here: a `. tools/<new>.sh`
# added to test-run.sh otherwise makes every leg below die at startup, with the
# real message buried under forty failures (the Ubuntu leg of gh-ocannl-983's
# first CI round, where the mutation harness hit exactly this). The floor is the
# other half: a regex that stops matching must fail loudly, not stage nothing.
# A leg staging a root of its own abandons itself on a failure, as any leg's
# structural failure does (gh-ocannl-1374): `stage_sourced ROOT ||
# { harness_abandon "$stage_error"; return; }`. The two roots most of the legs
# share (repeat_root, life_root) are the exception, staged by stage_shared: the
# scan is the same for every root, so where it fails no later leg can stage its
# subject either, and the harness stops there with the reason (exit 2).
stage_error=
stage_sourced() { # <fixture root>; 1 with $stage_error saying why
  local dest=$1 rel staged=0
  stage_error=
  while IFS= read -r rel; do
    mkdir -p "$dest/$(dirname "$rel")" && cp "$HERE/../$rel" "$dest/$rel" ||
      { stage_error="cannot stage $rel into $dest"; return 1; }
    staged=$((staged + 1))
  done < <(sed -n 's/^\. \([A-Za-z0-9_./-]*\)$/\1/p' "$SRC")
  [ "$staged" -ge 3 ] ||
    { stage_error="only $staged sourced file(s) found in $SRC; the scan is broken"; return 1; }
}
stage_shared() { # <fixture root most later legs run against>
  stage_sourced "$1" || { echo "$stage_error" >&2; exit 2; }
}

# ---------------------------------------------------------------------------
# Leg 67: a lock its acquirer left behind, held through an inherited descriptor
# ---------------------------------------------------------------------------
# The shape every run's lock has (gh-ocannl-1107): the launcher opens fd 9,
# take_lock's perl -- a child sharing that description -- takes the flock and
# exits, and the lock lives on in the description the supervisor, dune and
# dune's leftovers hold. On Linux /proc/locks may keep naming the ACQUIRER --
# dead, or 0 from inside a pid namespace -- and a census that trusted any name
# it gave returned that pid, never swept fdinfo, and `stop` reaped none of the
# live holders. Other kernels drop the row once the acquirer exits; the census
# then sweeps anyway, so the reap is asserted everywhere and only the negative
# control needs a named acquirer. The fixture is that shape: a run directory
# with no pid, no pgid and no verdict (only the lock can attribute anything),
# the owner pointer naming it, and a perl HOLDER -- this shell's own child, so
# it stays reapable under a PID 1 that reaps nothing -- that opens the lock
# file, forks the ACQUIRER to flock the shared description, and reaps it. The
# negative control runs the census that trusted the name -- the shipping
# script with the liveness filter taken out -- and must leave the holder
# running.
leg_67() {
l_inherit="stop: a leftover holding the lock through an inherited descriptor, its acquirer gone, is reaped"
l_inherit_nc="negative control: a census that trusts /proc/locks' named acquirer misses that holder"
inherit_skip=
if [ ! -r /proc/locks ] || [ ! -d /proc/self/fdinfo ]; then
  inherit_skip="no /proc/locks and fdinfo here (Linux only): the census reads lsof, which lists openers"
elif [ "$have_state" = 0 ]; then
  inherit_skip="no way to read a process's state, so a reaped holder cannot be told from a zombie"
fi
inherit_dir=$STOP_RUNS/19700101-000000-inherited
mkdir -p "$inherit_dir" "$STOP_WT"
printf 'runtest (test-test-run.sh fixture inherited)\n' >"$inherit_dir/cmd"
printf '0\n' >"$inherit_dir/cap"
printf '%s\n' "$STOP_WT" >"$inherit_dir/wt"
printf '%s\n' "$STOP_RUNS" >"$inherit_dir/runs"
: >"$inherit_dir/log"
inherit_lock=$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" paths lock "$inherit_dir") || inherit_lock=
inherit_owner=$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" paths owner "$inherit_dir") || inherit_owner=

# The pids /proc/locks names for the fixture lock, matched by device AND inode
# as the shipping census matches them: inode numbers repeat across
# filesystems, and another mount's flock must not pass for the acquirer.
inherit_named() {
  perl -e '
    my @st = stat($ARGV[0]) or exit 1;
    my $dev = $st[0];
    my ($maj, $min) = (($dev >> 8) & 0xfff, ($dev & 0xff) | (($dev >> 12) & ~0xff));
    open my $fh, "<", "/proc/locks" or exit 1;
    while (<$fh>) {
      my @f = split;
      next unless defined $f[1] && $f[1] eq "FLOCK";
      my ($lmaj, $lmin, $ino) = split /:/, $f[5];
      print "$f[4]\n"
        if defined $ino && $ino == $st[1] &&
           hex($lmaj) == $maj && hex($lmin) == $min;
    }
  ' "$inherit_lock" 2>/dev/null
}
inherit_kill() { # the holder is this shell's child: unreaped, its pid cannot be recycled
  [ -n "${inherit_holder:-}" ] || return 0
  kill -KILL "$inherit_holder" 2>/dev/null
  wait "$inherit_holder" 2>/dev/null
  inherit_holder=
  return 0
}
inherit_err= inherit_acquirer= inherit_named_pids=
inherit_start() { # sets inherit_holder, inherit_acquirer, inherit_named_pids, or inherit_err
  local line
  inherit_holder= inherit_acquirer= inherit_named_pids= inherit_err=
  printf '%s\n' "$inherit_dir" >"$inherit_owner" || { inherit_err="cannot write $inherit_owner"; return 1; }
  rm -f "$TMP/inherited.pids"
  perl -e 'use Fcntl ":flock"; use POSIX ();
           open(my $fh, ">>", $ARGV[0]) or exit 1;
           defined(my $kid = fork) or exit 1;
           POSIX::_exit(flock($fh, LOCK_EX | LOCK_NB) ? 0 : 1) if !$kid;
           waitpid($kid, 0) == $kid && $? == 0 or exit 1;
           $| = 1; print "$$ $kid\n"; sleep 600' \
    "$inherit_lock" >"$TMP/inherited.pids" 2>/dev/null </dev/null &
  inherit_holder=$!
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    [ -s "$TMP/inherited.pids" ] && break
    sleep 0.1
  done
  read -r line inherit_acquirer <"$TMP/inherited.pids" 2>/dev/null
  if [ "${line:-}" != "$inherit_holder" ] || [ -z "${inherit_acquirer:-}" ]; then
    inherit_err="the holder did not report taking the lock through a reaped acquirer"; return 1
  fi
  # The premise, checked: the lock is held, its acquirer is gone, and
  # /proc/locks does not name the live holder -- whatever else it names.
  if [ "$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" lock-status "$inherit_dir")" != held ]; then
    inherit_err="the fixture lock is not held"; return 1
  fi
  kill -0 "$inherit_acquirer" 2>/dev/null && { inherit_err="the acquirer $inherit_acquirer is still alive"; return 1; }
  inherit_named_pids=$(inherit_named)
  if printf '%s\n' "$inherit_named_pids" | grep -qx "$inherit_holder"; then
    inherit_err="/proc/locks names the live holder $inherit_holder, not its acquirer"; return 1
  fi
  return 0
}
inherit_stop() { # <script>
  stop_out="$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$1" stop "$inherit_dir" 2>"$TMP/inherited.stderr")"
  stop_rc=$?
  stop_diag="$(cat "$TMP/inherited.stderr" 2>/dev/null)"
}
inherit_gone() { # 0 iff the holder is dead (a zombie: it is ours to reap) within two seconds
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    case "$(pstate "$inherit_holder")" in '' | Z*) return 0 ;; esac
    sleep 0.1
  done
  return 1
}

if [ -n "$inherit_skip" ]; then
  skip "$l_inherit" "$inherit_skip"
  skip "$l_inherit_nc" "$inherit_skip"
elif [ -z "$inherit_lock" ] || [ -z "$inherit_owner" ]; then
  report 1 "$l_inherit" "\`paths lock\`/\`paths owner\` did not name the fixture run's lock"
  report 1 "$l_inherit_nc" "\`paths lock\`/\`paths owner\` did not name the fixture run's lock"
else
  if ! inherit_start; then
    report 1 "$l_inherit" "$inherit_err"
  else
    inherit_stop "$SRC"
    said "$l_inherit" \
      "run is dead without a verdict, but its leftover processes held the worktree lock; reaped them"
    if ! inherit_gone; then
      report 1 "$l_inherit (the holder is gone and the lock is free)" \
        "pid $inherit_holder still holds $inherit_lock after stop"
    elif [ "$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" lock-status "$inherit_dir")" = idle ]; then
      report 0 "$l_inherit (the holder is gone and the lock is free)"
    else
      report 1 "$l_inherit (the holder is gone and the lock is free)" \
        "$inherit_lock is still locked after stop"
    fi
  fi
  inherit_kill

  # The census before gh-ocannl-1107: every named pid, live or not. The line
  # is replaced whole, and checked to be the only change, so a filter that
  # moved leaves this control failing rather than testing the shipping code.
  inherit_root=$TMP/inherit-repo
  mkdir -p "$inherit_root/tools"
  stage_sourced "$inherit_root" || { harness_abandon "$stage_error"; return; }
  awk '
    $0 == "        fd_holds_lock \"$p\" && live=\"$live$p" { print "        live=\"$live$p"; n++; next }
    { print }
    END { exit n == 1 ? 0 : 1 }
  ' "$SRC" >"$inherit_root/tools/test-run.sh" && chmod +x "$inherit_root/tools/test-run.sh"
  inherit_mut_rc=$?
  inherit_changed=$(diff "$SRC" "$inherit_root/tools/test-run.sh" | grep -c '^[<>]')
  if [ "$inherit_mut_rc" != 0 ] || [ "$inherit_changed" != 2 ]; then
    report 1 "$l_inherit_nc" "the liveness filter's line was not found once in $SRC"
  elif ! inherit_start; then
    report 1 "$l_inherit_nc" "$inherit_err"
  elif [ -z "$inherit_named_pids" ]; then
    # Nothing named, so the unfiltered census sweeps fdinfo too, and the
    # control could not tell it from the fix.
    skip "$l_inherit_nc" "this kernel's /proc/locks drops the row once its acquirer exits, so no census trusts a name here"
  else
    inherit_stop "$inherit_root/tools/test-run.sh"
    if inherit_gone; then
      report 1 "$l_inherit_nc" \
        "the unfiltered census reaped the holder too (stop printed: ${stop_out:-<nothing>}), so the leg cannot tell the fix"
    else
      report 0 "$l_inherit_nc"
    fi
  fi
  inherit_kill
fi
[ -z "$inherit_lock" ] || rm -f "$inherit_lock" "$inherit_owner"
}
harness_leg_run '67: a lock held through an inherited descriptor is reaped #stop' leg_67

# ---------------------------------------------------------------------------
# Legs 9-17: repeat mode's output, lifecycle and exit-code contract
# ---------------------------------------------------------------------------
# The fixture root is the shipping script plus what it sources (stage_sourced,
# defined before leg 67).
repeat_root=$TMP/repeat-repo
repeat_bin=$TMP/repeat-bin
mkdir -p "$repeat_root/tools" "$repeat_root/scripts" "$repeat_bin"
cp "$SRC" "$repeat_root/tools/test-run.sh"
stage_shared "$repeat_root"
chmod +x "$repeat_root/tools/test-run.sh"
# The two readers tools/batch-backends.sh would build, faked for every leg
# (gh-ocannl-1066): a run on a box where a backend meets a width cap, or under
# a fleet slot, resolves the batch's backends, and a build through the fixture
# dune would land in the calls every leg pins. The reader answers a test
# configuration's backend from the directory it runs in, the environment
# outranking the file as in the real resolution (FAKE_BACKEND_TEST and
# FAKE_BACKEND_ARRAYJIT stand for the files, cc by default as the tracked ones
# say; `fail` fails the read, `unset` resolves none). The reachability tool
# records the argv it was asked about and prints FAKE_REACH (by default: no
# stanza naming a backend, and one reading the configuration -- FAKE_READS --
# so the configurations count, as they do for any batch off the
# configuration-free list; leg 78 drives one that reads none, gh-ocannl-1227).
mkdir -p "$repeat_root/test/config" "$repeat_root/arrayjit/test"
# A build directory, as any worktree that has built has: without one the run's
# last phase records an empty promotion list without asking dune (gh-ocannl-1087).
mkdir -p "$repeat_root/_build"
batch_reader=$TMP/fake-read-config.sh
cat >"$batch_reader" <<'FAKE'
#!/usr/bin/env bash
case $PWD in
  */test/config) v=${FAKE_BACKEND_TEST:-cc} ;;
  */arrayjit/test) v=${FAKE_BACKEND_ARRAYJIT:-cc} ;;
  *) exit 3 ;;
esac
[ "$v" = fail ] && exit 1
[ -z "${OCANNL_BACKEND:-}" ] || v=$OCANNL_BACKEND
[ "$v" = unset ] && v=
printf '%s\n' "$v"
FAKE
batch_reach=$TMP/fake-slot-kind.sh
export FAKE_READS='reads config: it reaches t_cfg in x, which reads the configuration'
cat >"$batch_reach" <<'FAKE'
#!/usr/bin/env bash
[ -z "${FAKE_REACH_CALLS:-}" ] || printf '%s\n' "$*" >>"$FAKE_REACH_CALLS"
[ -z "${FAKE_REACH_IGNORE_TERM:-}" ] || trap '' TERM INT HUP
[ -z "${FAKE_REACH_SLEEP:-}" ] || sleep "$FAKE_REACH_SLEEP"
[ -z "${FAKE_REACH_LEAVE:-}" ] || sleep "$FAKE_REACH_LEAVE" &
printf '%b\n' "${FAKE_REACH:-$FAKE_READS\nend}"
exit "${FAKE_REACH_EXIT:-0}"
FAKE
chmod +x "$batch_reader" "$batch_reach"
export OCANNL_TOOL_READ_CONFIG=$batch_reader OCANNL_TOOL_SLOT_KIND=$batch_reach
# Nor does any leg probe a device unless it says so (gh-ocannl-1211): the probe
# of a GPU batch would build bin/device_props through the fixture dune, and its
# `_probe` would stand between the slot and dune in every call the slot legs
# pin. Legs 75-77 name a stand-in. And no caller's bounds leak in: a raised
# per-test cap (an agent running @slow exports one) would change every run's
# watcher here and the values legs 75-77 read back.
export OCANNL_TOOL_DEVICE_PROBE=none
unset OCANNL_TOOL_PER_TEST_CAP OCANNL_TOOL_DEVICE_PROBE_CAP
# Read-only query contract: no state store, no first run, and no lock file yet.
# The fixture root (not the caller's cwd) owns all omitted-RUN queries.
leg_39() {
query_runs=$TMP/query-state/runs
query_expected=$(cd "$TMP" && pwd -P)/query-state/runs
query() {
  OCANNL_TOOL_TEST_RUNS=$query_runs "$repeat_root/tools/test-run.sh" "$@"
}
query_rc=0
query_out=$(query paths runs) || query_rc=$?
query_wt=$(query paths worktree)
query_lock=$(query paths lock)
query_owner=$(query paths owner)
query_last=$(query paths last)
query_idle=$(query lock-status) || query_rc=$?
if [ "$query_rc" = 0 ] && [ "$query_out" = "$query_expected" ] \
   && [ "$query_wt" = "$(cd "$repeat_root" && pwd -P)" ] \
   && [ "$query_idle" = idle ] && [ ! -e "$TMP/query-state" ] \
   && [ -n "$query_lock" ] && [ -n "$query_owner" ] && [ -n "$query_last" ]; then
  report 0 "queries: absent state resolves physically without creating files"
else
  report 1 "queries: absent state resolves physically without creating files" \
    "rc=$query_rc runs=$query_out worktree=$query_wt lock=$query_lock idle=$query_idle"
fi
query_bad=
for args in 'paths' 'paths bogus' 'paths lock last extra' 'paths run' \
            'paths run missing' 'paths lock missing' 'lock-status last' \
            'lock-status last extra'; do
  query_out=$(query $args 2>"$TMP/query-error")
  query_rc=$?
  if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
    query_bad="$args: rc=$query_rc output=$query_out"; break
  fi
done
if [ -z "$query_bad" ] && [ ! -e "$TMP/query-state" ]; then
  report 0 "queries: malformed arguments and missing runs fail without state mutation"
else
  report 1 "queries: malformed arguments and missing runs fail without state mutation" "$query_bad"
fi

# Three or more leading slashes are a local root, not a UNC host/share.
query_slash_root=ocannl-query-slashes-${TMP##*/}
query_out=$(OCANNL_TOOL_TEST_RUNS="///$query_slash_root/runs" \
  "$repeat_root/tools/test-run.sh" paths runs)
query_rc=$?
query_root=$(cd / && pwd -P)
if [ "$query_rc" = 0 ] && [ "$query_out" = "${query_root%/}/$query_slash_root/runs" ] \
   && [ ! -e "/$query_slash_root" ]; then
  report 0 "queries: redundant leading slashes preserve the local root without writes"
else
  report 1 "queries: redundant leading slashes preserve the local root without writes" \
    "rc=$query_rc runs=$query_out root=$query_root"
fi

case $(uname -s) in
  MSYS* | MINGW*) skip "queries: double slashes preserve the POSIX local root without writes" "Windows reserves UNC roots" ;;
  *)
    query_out=$(OCANNL_TOOL_TEST_RUNS="//$query_slash_root/runs" \
      "$repeat_root/tools/test-run.sh" paths runs)
    query_rc=$?
    if [ "$query_rc" = 0 ] && [ "$query_out" = "${query_root%/}/$query_slash_root/runs" ] \
       && [ ! -e "/$query_slash_root" ]; then
      report 0 "queries: double slashes preserve the POSIX local root without writes"
    else
      report 1 "queries: double slashes preserve the POSIX local root without writes" "rc=$query_rc runs=$query_out"
    fi ;;
esac

query_bad=
for query_break in $'\n' $'\r'; do
  query_break_root=$TMP/query-break${query_break}root
  for query_command in 'paths runs' 'lock-status'; do
    query_out=$(OCANNL_TOOL_TEST_RUNS="$query_break_root" \
      "$repeat_root/tools/test-run.sh" $query_command 2>"$TMP/query-error")
    query_rc=$?
    if [ "$query_rc" != 2 ] || [ -n "$query_out" ] \
       || ! grep -q 'single-line paths' "$TMP/query-error" \
       || [ -e "$query_break_root" ]; then
      query_bad="$query_command: rc=$query_rc output=$query_out"; break
    fi
  done
  [ -z "$query_bad" ] || break
done
if [ -z "$query_bad" ]; then
  report 0 "queries: CR and LF roots fail loudly without state writes"
else
  report 1 "queries: CR and LF roots fail loudly without state writes" "$query_bad"
fi

# A competing CDPATH location must neither redirect the identity nor leak cd's
# announcement into the one-path stdout contract.
mkdir -p "$repeat_root/query-relative" "$TMP/query-cdpath/query-relative"
query_out=$(CDPATH="$TMP/query-cdpath" OCANNL_TOOL_TEST_RUNS=query-relative/missing \
  "$repeat_root/tools/test-run.sh" paths runs)
query_rc=$?
if [ "$query_rc" = 0 ] && [ "$query_out" = "$query_wt/query-relative/missing" ] \
   && [ ! -e "$repeat_root/query-relative/missing" ] \
   && [ ! -e "$TMP/query-cdpath/query-relative/missing" ]; then
  report 0 "queries: CDPATH cannot redirect relative state roots or add stdout"
else
  report 1 "queries: CDPATH cannot redirect relative state roots or add stdout" \
    "rc=$query_rc runs=$query_out"
fi

# Physical normalization must agree with launch even when the caller uses a
# symlinked script or state root. MSYS may copy rather than create a symlink.
ln -s "$repeat_root" "$TMP/query-repo-link" 2>/dev/null
ln -s "$repeat_root" "$TMP/query-state-link" 2>/dev/null
if [ -L "$TMP/query-repo-link" ] && [ -L "$TMP/query-state-link" ]; then
  query_out=$(OCANNL_TOOL_TEST_RUNS="$TMP/query-state-link/nonexistent/runs" \
    "$TMP/query-repo-link/tools/test-run.sh" paths runs)
  query_rc=$?
  if [ "$query_rc" = 0 ] && [ "$query_out" = "$query_wt/nonexistent/runs" ] \
     && [ ! -e "$repeat_root/nonexistent" ]; then
    report 0 "queries: symlink spellings identify the same physical worktree and state root"
  else
    report 1 "queries: symlink spellings identify the same physical worktree and state root" \
      "rc=$query_rc runs=$query_out expected=$query_wt/nonexistent/runs"
  fi
else
  skip "queries: symlink spellings identify the same physical worktree and state root" \
    "native symlinks unavailable"
fi

# Missing children of a symlink/.. prefix can differ between logical cd and
# mkdir's physical traversal. Refuse that prediction, but accept a resolved root.
mkdir -p "$TMP/query-parent-outside/child"
ln -s "$TMP/query-parent-outside/child" "$repeat_root/query-parent-link" 2>/dev/null
if [ -L "$repeat_root/query-parent-link" ]; then
  query_bad=
  for query_command in 'paths runs' 'lock-status'; do
    query_out=$(OCANNL_TOOL_TEST_RUNS=query-parent-link/../fresh \
      "$repeat_root/tools/test-run.sh" $query_command 2>"$TMP/query-error")
    query_rc=$?
    if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
      query_bad="$query_command: rc=$query_rc output=$query_out"; break
    fi
  done
  [ ! -e "$TMP/query-parent-outside/fresh" ] && [ ! -e "$repeat_root/fresh" ] || query_bad="query created state"
  mkdir "$TMP/query-parent-outside/fresh"
  query_out=$(OCANNL_TOOL_TEST_RUNS=query-parent-link/../fresh \
    "$repeat_root/tools/test-run.sh" paths runs)
  query_rc=$?
  query_want=$(cd "$repeat_root" && cd query-parent-link/../fresh && pwd -P)
  if [ -z "$query_bad" ] && [ "$query_rc" = 0 ] && [ "$query_out" = "$query_want" ]; then
    report 0 "queries: ambiguous missing symlink parents refuse while resolved paths match launch"
  else
    report 1 "queries: ambiguous missing symlink parents refuse while resolved paths match launch" "$query_bad rc=$query_rc output=$query_out expected=$query_want"
  fi
else
  skip "queries: ambiguous missing symlink parents refuse while resolved paths match launch" "native symlinks unavailable"
fi

# A plain symlink name can hide a physical path ending in LF. Refuse it
# before command substitution trims that character into a different path.
case $(uname -s) in MSYS* | MINGW*) query_line_paths=0 ;; *) query_line_paths=1 ;; esac
if [ "$query_line_paths" = 1 ] && [ -L "$TMP/query-state-link" ]; then
  query_line_target=$TMP/query-physical$'\n'
  mkdir "$query_line_target"
  ln -s "$query_line_target" "$TMP/query-line-link"
  query_out=$(OCANNL_TOOL_TEST_RUNS="$TMP/query-line-link" \
    "$repeat_root/tools/test-run.sh" paths runs 2>"$TMP/query-error")
  query_rc=$?
  if [ "$query_rc" = 2 ] && [ -z "$query_out" ] \
     && grep -q 'single-line paths' "$TMP/query-error"; then
    report 0 "queries: physical line breaks behind symlinks are refused before capture"
  else
    report 1 "queries: physical line breaks behind symlinks are refused before capture" "rc=$query_rc runs=$query_out"
  fi
else
  skip "queries: physical line breaks behind symlinks are refused before capture" "requires POSIX symlink filenames"
fi

# Exercise UNC preservation portably: the launch shell's physical-prefix
# reader reports a network root. The query must retain that exact prefix,
# not run it through a separate filesystem grammar that collapses // to /.
query_unc_root=$TMP/query-unc
mkdir "$query_unc_root"
query_unc_root=$(cd "$query_unc_root" && pwd -P)
cat >"$TMP/query-unc-env" <<'EOF'
pwd() {
  if [ "$PWD" = "$QUERY_TEST_UNC_ROOT" ]; then printf '//query-host/share\n';
  else builtin pwd "$@"; fi
}
EOF
query_out=$(BASH_ENV="$TMP/query-unc-env" QUERY_TEST_UNC_ROOT="$query_unc_root" \
  OCANNL_TOOL_TEST_RUNS="$query_unc_root/missing/runs" \
  "$repeat_root/tools/test-run.sh" paths runs)
query_rc=$?
if [ "$query_rc" = 0 ] && [ "$query_out" = //query-host/share/missing/runs ] \
   && [ ! -e "$query_unc_root/missing" ]; then
  report 0 "queries: the launch shell's UNC prefix survives missing-suffix normalization"
else
  report 1 "queries: the launch shell's UNC prefix survives missing-suffix normalization" \
    "rc=$query_rc runs=$query_out"
fi

# Parent traversal is resolved only by the filesystem, never by a second
# path grammar. Test both sides for local and shell-reported UNC prefixes.
mkdir "$query_unc_root/existing"
for query_style in local unc; do
  query_env=$TMP/query-empty-env
  : >"$query_env"
  [ "$query_style" != unc ] || query_env=$TMP/query-unc-env
  query_want=$query_unc_root/runs
  [ "$query_style" != unc ] || query_want=//query-host/share/runs
  query_good=$(BASH_ENV="$query_env" QUERY_TEST_UNC_ROOT="$query_unc_root" \
    OCANNL_TOOL_TEST_RUNS="$query_unc_root/existing/../runs" \
    "$repeat_root/tools/test-run.sh" paths runs)
  query_good_rc=$?
  query_bad=
  for query_command in 'paths runs' 'lock-status'; do
    query_out=$(BASH_ENV="$query_env" QUERY_TEST_UNC_ROOT="$query_unc_root" \
      OCANNL_TOOL_TEST_RUNS="$query_unc_root/missing/../../target" \
      "$repeat_root/tools/test-run.sh" $query_command 2>"$TMP/query-error")
    query_rc=$?
    if [ "$query_rc" != 2 ] || [ -n "$query_out" ] \
       || ! grep -q 'missing state path containing ..' "$TMP/query-error"; then
      query_bad="$query_command: rc=$query_rc runs=$query_out"; break
    fi
  done
  if [ "$query_good_rc" = 0 ] && [ "$query_good" = "$query_want" ] \
     && [ -z "$query_bad" ] && [ ! -e "$query_unc_root/missing" ] \
     && [ ! -e "$query_unc_root/runs" ] && [ ! -e "$TMP/target" ]; then
    report 0 "queries: $query_style existing parent traversal resolves; missing traversal refuses without writes"
  else
    report 1 "queries: $query_style existing parent traversal resolves; missing traversal refuses without writes" \
      "resolved=$query_good_rc:$query_good expected=$query_want $query_bad"
  fi
done

# MSYS accepts both /c/... and C:/... at launch. Its Perl File::Spec does
# not necessarily share that interpretation, so compare both against Bash.
case $(uname -s) in
  MSYS* | MINGW*)
    # Reach the drive root itself, not just an existing temporary prefix.
    query_drive=$(cygpath -m "$TMP")
    query_drive=${query_drive%%/*}/
    query_drive_physical=$(cd "$query_drive" && pwd -P)
    query_drive_runs=${query_drive}ocannl-query-root-${TMP##*/}
    query_out=$(OCANNL_TOOL_TEST_RUNS="$query_drive_runs" "$repeat_root/tools/test-run.sh" paths runs)
    query_rc=$?
    if [ "$query_rc" = 0 ] && [ "$query_out" = "$query_drive_physical/ocannl-query-root-${TMP##*/}" ] \
       && [ ! -e "$query_drive_runs" ]; then
      report 0 "queries: missing state directly below a drive root matches the launch shell without writes"
    else
      report 1 "queries: missing state directly below a drive root matches the launch shell without writes" \
        "rc=$query_rc runs=$query_out root=$query_drive_physical"
    fi
    query_native=$(cygpath -m "$query_expected")
    query_out=$(OCANNL_TOOL_TEST_RUNS="$query_native" "$repeat_root/tools/test-run.sh" paths runs)
    query_rc=$?
    if [ "$query_rc" = 0 ] && [ "$query_out" = "$query_expected" ] \
       && [ ! -e "$TMP/query-state" ]; then
      report 0 "queries: absent Windows drive spelling matches the launch shell"
    else
      report 1 "queries: absent Windows drive spelling matches the launch shell" \
        "rc=$query_rc runs=$query_out expected=$query_expected"
    fi
    query_native=$(cygpath -m "$query_unc_root")
    query_good=$(OCANNL_TOOL_TEST_RUNS="$query_native/existing/../runs" \
      "$repeat_root/tools/test-run.sh" paths runs)
    query_good_rc=$?
    # Native C:/ paths resolve missing/.. through Windows before Bash cd.
    # Match that actual launch-shell identity, not POSIX lexical semantics.
    query_native_parent=$(cd "$query_native/missing/.." && pwd -P)
    query_native_parent_rc=$?
    query_out=$(OCANNL_TOOL_TEST_RUNS="$query_native/missing/../runs" \
      "$repeat_root/tools/test-run.sh" paths runs)
    query_rc=$?
    if [ "$query_good_rc" = 0 ] && [ "$query_good" = "$query_unc_root/runs" ] \
       && [ "$query_native_parent_rc" = 0 ] && [ "$query_rc" = 0 ] \
       && [ "$query_out" = "$query_native_parent/runs" ] \
       && [ ! -e "$query_unc_root/missing" ] && [ ! -e "$query_unc_root/runs" ]; then
      report 0 "queries: drive parent traversal matches the native launch shell without writes"
    else
      report 1 "queries: drive parent traversal matches the native launch shell without writes" \
        "resolved=$query_good_rc:$query_good parent=$query_native_parent_rc:$query_native_parent runs=$query_rc:$query_out"
    fi
    ;;
  *) skip "queries: missing state directly below a drive root matches the launch shell without writes" "requires Windows drive paths"
     skip "queries: absent Windows drive spelling matches the launch shell" "requires Windows drive paths"
     skip "queries: drive parent traversal matches the native launch shell without writes" "requires Windows drive paths" ;;
esac

# The pointer returned by paths last must feed the SAME resolver as status and
# stop; the published fixture has deliberately no live pid and no verdict.
mkdir -p "$query_expected/fixture"
query_runs=$query_expected
query_dir=$query_expected/fixture
printf 'fixture\n' >"$query_dir/cmd"
printf '0\n' >"$query_dir/cap"
printf '%s\n' "$query_wt" >"$query_dir/wt"
printf '%s\n' "$query_runs" >"$query_dir/runs"
printf '%s\n' "$query_dir" >"$query_last"
case $(uname -s) in
  MSYS* | MINGW*)
    query_native=$(cygpath -m "$query_runs")
    query_out=$(OCANNL_TOOL_TEST_RUNS="$query_native" "$repeat_root/tools/test-run.sh" paths runs)
    query_rc=$?
    if [ "$query_rc" = 0 ] && [ "$query_out" = "$(cd "$query_native" && pwd -P)" ]; then
      report 0 "queries: existing Windows drive spelling matches the launch shell"
    else
      report 1 "queries: existing Windows drive spelling matches the launch shell" \
        "rc=$query_rc runs=$query_out"
    fi
    ;;
  *) skip "queries: existing Windows drive spelling matches the launch shell" "requires Windows drive paths" ;;
esac
query_bad=
for ref in last fixture "$query_dir"; do
  [ "$(query paths run "$ref")" = "$query_dir" ] \
    && [ "$(query paths lock "$ref")" = "$query_lock" ] \
    && [ "$(query paths owner "$ref")" = "$query_owner" ] \
    && [ "$(query paths worktree "$ref")" = "$query_wt" ] \
    && [ "$(query paths runs "$ref")" = "$query_runs" ] \
    && [ "$(query paths last "$ref")" = "$query_last" ] \
    || query_bad="$ref did not resolve the published fixture"
done
query_out=$(query lock-status last); query_rc=$?
if [ -z "$query_bad" ] && [ "$query_rc" = 0 ] && [ "$query_out" = idle ] \
   && [ ! -e "$query_lock" ] && [ ! -e "$query_owner" ]; then
  report 0 "queries: last, bare id and absolute run share recorded paths without creating locks"
else
  report 1 "queries: last, bare id and absolute run share recorded paths without creating locks" "$query_bad"
fi

# Absolute references remain useful after the caller's ambient root changes.
# Omitted and bare-id references still require that current root to be valid.
printf 'not a directory\n' >"$TMP/query-ambient-file"
query_bad=
for query_ambient in "$TMP/query-ambient-file" "$TMP/query-ambient-missing/../runs"; do
  for field in run worktree runs lock owner last; do
    case $field in
      run) query_want=$query_dir ;; worktree) query_want=$query_wt ;;
      runs) query_want=$query_runs ;; lock) query_want=$query_lock ;;
      owner) query_want=$query_owner ;; last) query_want=$query_last ;;
    esac
    query_out=$(OCANNL_TOOL_TEST_RUNS="$query_ambient" \
      "$repeat_root/tools/test-run.sh" paths "$field" "$query_dir" 2>"$TMP/query-error")
    query_rc=$?
    if [ "$query_rc" != 0 ] || [ "$query_out" != "$query_want" ]; then
      query_bad="$field: rc=$query_rc output=$query_out expected=$query_want"; break
    fi
  done
  query_out=$(OCANNL_TOOL_TEST_RUNS="$query_ambient" \
    "$repeat_root/tools/test-run.sh" lock-status "$query_dir" 2>"$TMP/query-error")
  query_rc=$?
  [ "$query_rc" = 0 ] && [ "$query_out" = idle ] || query_bad="lock-status: $query_rc:$query_out"
  for query_command in 'paths runs' 'paths runs fixture' 'lock-status'; do
    query_out=$(OCANNL_TOOL_TEST_RUNS="$query_ambient" \
      "$repeat_root/tools/test-run.sh" $query_command 2>"$TMP/query-error")
    query_rc=$?
    [ "$query_rc" = 2 ] && [ -z "$query_out" ] || query_bad="$query_command: $query_rc:$query_out"
  done
  [ -z "$query_bad" ] || break
done
if [ -z "$query_bad" ] && [ -f "$TMP/query-ambient-file" ] \
   && [ ! -e "$TMP/query-ambient-missing" ] && [ ! -e "$query_lock" ]; then
  report 0 "queries: absolute runs ignore unrelated ambient state while current references validate it"
else
  report 1 "queries: absolute runs ignore unrelated ambient state while current references validate it" "$query_bad"
fi

# A missing absolute reference must not select a matching state-root suffix.
query_missing_absolute=/ocannl-query-missing-$$
mkdir -p "$query_runs/$query_missing_absolute"
cp "$query_dir/cmd" "$query_dir/cap" "$query_dir/wt" "$query_dir/runs" "$query_runs/$query_missing_absolute/"
query_bad=
for query_command in 'paths run' 'paths lock' 'lock-status'; do
  query_out=$(query $query_command "$query_missing_absolute" 2>"$TMP/query-error")
  query_rc=$?
  if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
    query_bad="$query_command: rc=$query_rc output=$query_out"; break
  fi
done
if [ -z "$query_bad" ] && [ ! -e "$query_missing_absolute" ] && [ ! -e "$query_lock" ]; then
  report 0 "queries: missing absolute references never fall back under the state root"
else
  report 1 "queries: missing absolute references never fall back under the state root" "$query_bad"
fi

# Explicit references use the same host-absolute grammar as recorded paths.
case $(uname -s) in
  MSYS* | MINGW*) query_drive_ref=$(cygpath -m "$query_dir"); query_drive_ref_rc=0 ;;
  *)
    query_drive_ref=C:/fixture
    mkdir -p "$repeat_root/$query_drive_ref"
    cp "$query_dir/cmd" "$query_dir/cap" "$query_dir/wt" "$query_dir/runs" "$repeat_root/$query_drive_ref/"
    query_drive_ref_rc=2 ;;
esac
query_out=$(env -u HOME -u OCANNL_TOOL_TEST_RUNS \
  "$repeat_root/tools/test-run.sh" paths run "$query_drive_ref" 2>"$TMP/query-error")
query_rc=$?
query_drive_lock=$(env -u HOME -u OCANNL_TOOL_TEST_RUNS \
  "$repeat_root/tools/test-run.sh" lock-status "$query_drive_ref" 2>"$TMP/query-error")
query_drive_lock_rc=$?
query_bad=
[ "$query_rc" = "$query_drive_ref_rc" ] && [ "$query_drive_lock_rc" = "$query_drive_ref_rc" ] || query_bad="rc=$query_rc lock=$query_drive_lock_rc"
if [ "$query_drive_ref_rc" = 0 ]; then
  [ "$query_out" = "$query_dir" ] && [ "$query_drive_lock" = idle ] || query_bad="native reference changed identity"
else
  [ -z "$query_out" ] && [ -z "$query_drive_lock" ] || query_bad="relative reference bypassed current state"
  [ "$(query paths run "$query_drive_ref")" = "$query_wt/$query_drive_ref" ] || query_bad="relative reference no longer resolves with valid state"
fi
if [ -z "$query_bad" ] && [ ! -e "$query_lock" ]; then
  report 0 "queries: explicit drive references follow host absolute-path semantics"
else
  report 1 "queries: explicit drive references follow host absolute-path semantics" "$query_bad"
fi

query_out=$(env -u HOME -u OCANNL_TOOL_TEST_RUNS \
  "$repeat_root/tools/test-run.sh" paths runs "$query_dir" 2>"$TMP/query-error")
query_rc=$?
query_home_lock=$(env -u HOME -u OCANNL_TOOL_TEST_RUNS \
  "$repeat_root/tools/test-run.sh" lock-status "$query_dir" 2>"$TMP/query-error")
query_home_lock_rc=$?
query_home_current=$(env -u HOME -u OCANNL_TOOL_TEST_RUNS \
  "$repeat_root/tools/test-run.sh" paths runs 2>"$TMP/query-error")
query_home_current_rc=$?
if [ "$query_rc" = 0 ] && [ "$query_out" = "$query_runs" ] \
   && [ "$query_home_lock_rc" = 0 ] && [ "$query_home_lock" = idle ] \
   && [ "$query_home_current_rc" = 2 ] && [ -z "$query_home_current" ] \
   && [ ! -e "$query_lock" ]; then
  report 0 "queries: HOME is required only when its current-root default is used"
else
  report 1 "queries: HOME is required only when its current-root default is used" \
    "runs=$query_rc:$query_out lock=$query_home_lock_rc:$query_home_lock current=$query_home_current_rc:$query_home_current"
fi

# A held lock needs no process metadata to be reported held. Run the holder
# synchronously; Perl closes its nonstandard descriptors when execing the query.
query_out=$(perl -MFcntl=:flock -e '
  open my $fh, ">>", $ARGV[0] or die $!;
  flock($fh, LOCK_EX | LOCK_NB) or die $!;
  system @ARGV[1..$#ARGV];
  exit($? >> 8);
' "$query_lock" env OCANNL_TOOL_TEST_RUNS="$query_runs" \
  "$repeat_root/tools/test-run.sh" lock-status last)
query_rc=$?
query_after=$(query lock-status); query_after_rc=$?
if [ "$query_rc" = 3 ] && [ "$query_out" = held ] \
   && [ "$query_after_rc" = 0 ] && [ "$query_after" = idle ] \
   && [ ! -s "$query_lock" ]; then
  report 0 "queries: occupied and released locks have distinct statuses without writes"
else
  report 1 "queries: occupied and released locks have distinct statuses without writes" \
    "held=$query_rc:$query_out released=$query_after_rc:$query_after"
fi
rm "$query_lock"
mkdir "$query_lock"
query_out=$(query lock-status 2>"$TMP/query-error"); query_rc=$?
if [ "$query_rc" = 2 ] && [ -z "$query_out" ] && [ -d "$query_lock" ]; then
  report 0 "queries: an uninspectable lock is an error, never idle"
else
  report 1 "queries: an uninspectable lock is an error, never idle" "rc=$query_rc output=$query_out"
fi
rmdir "$query_lock"
# Drive-form records are absolute only on Windows. Both recorded path fields
# follow host semantics rather than accepting a foreign relative spelling.
query_bad=
for query_record_field in wt runs; do
  case $query_record_field in wt) query_record_path=$query_wt; query_field=worktree ;; runs) query_record_path=$query_runs; query_field=runs ;; esac
  case $(uname -s) in
    MSYS* | MINGW*) query_drive_record=$(cygpath -m "$query_record_path"); query_drive_rc=0 ;;
    *) query_drive_record=C:/foreign-state; query_drive_rc=2 ;;
  esac
  printf '%s\n' "$query_drive_record" >"$query_dir/$query_record_field"
  query_out=$(query paths "$query_field" "$query_dir" 2>"$TMP/query-error")
  query_rc=$?
  query_drive_lock=$(query lock-status "$query_dir" 2>"$TMP/query-error")
  query_drive_lock_rc=$?
  if [ "$query_rc" != "$query_drive_rc" ] || [ "$query_drive_lock_rc" != "$query_drive_rc" ]; then
    query_bad="$query_record_field: path=$query_rc:$query_out lock=$query_drive_lock_rc:$query_drive_lock"
  elif [ "$query_drive_rc" = 0 ]; then
    [ "$query_out" = "$query_drive_record" ] && [ "$query_drive_lock" = idle ] || query_bad="native record changed identity"
  else
    [ -z "$query_out" ] && [ -z "$query_drive_lock" ] || query_bad="foreign record produced output"
  fi
  printf '%s\n' "$query_record_path" >"$query_dir/$query_record_field"
  [ -z "$query_bad" ] || break
done
if [ -z "$query_bad" ] && [ ! -e "$query_lock" ]; then
  report 0 "queries: recorded drive paths follow host absolute-path semantics"
else
  report 1 "queries: recorded drive paths follow host absolute-path semantics" "$query_bad"
fi

# Missing runs metadata is legacy; present broken metadata is an error.
query_bad=
for query_metadata in empty relative multiline directory trailing_blank unterminated nul; do
  rm -f "$query_dir/runs"
  case $query_metadata in
    empty) : >"$query_dir/runs" ;;
    relative) printf 'relative/runs\n' >"$query_dir/runs" ;;
    multiline) printf '%s\nextra\n' "$query_runs" >"$query_dir/runs" ;;
    directory) mkdir "$query_dir/runs" ;;
    trailing_blank) printf '%s\n\n' "$query_runs" >"$query_dir/runs" ;;
    unterminated) printf '%s' "$query_runs" >"$query_dir/runs" ;;
    nul) printf '%s\000\n' "$query_runs" >"$query_dir/runs" ;;
  esac
  for query_ref in last "$query_dir"; do
    for query_command in 'paths runs' 'paths lock' 'lock-status'; do
      query_out=$(query $query_command "$query_ref" 2>"$TMP/query-error")
      query_rc=$?
      if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
        query_bad="$query_metadata $query_command $query_ref: rc=$query_rc output=$query_out"; break
      fi
    done
    [ -z "$query_bad" ] || break
  done
  [ ! -d "$query_dir/runs" ] || rmdir "$query_dir/runs"
  [ -z "$query_bad" ] || break
done
if [ -z "$query_bad" ] && [ ! -e "$query_lock" ] && [ ! -e "$query_owner" ]; then
  report 0 "queries: present malformed state metadata is never mistaken for a legacy run"
else
  report 1 "queries: present malformed state metadata is never mistaken for a legacy run" "$query_bad"
fi
printf '%s\n' "$query_runs" >"$query_dir/runs"

# The regular last pointer is a record too; shell capture must not repair it.
query_bad=
for query_pointer_kind in trailing_blank unterminated; do
  case $query_pointer_kind in
    trailing_blank) printf '%s\n\n' "$query_dir" >"$query_last" ;;
    unterminated) printf '%s' "$query_dir" >"$query_last" ;;
  esac
  for query_command in 'paths run' 'paths runs' 'lock-status'; do
    query_out=$(query $query_command last 2>"$TMP/query-error")
    query_rc=$?
    if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
      query_bad="$query_pointer_kind $query_command: rc=$query_rc output=$query_out"; break
    fi
  done
  [ -z "$query_bad" ] || break
done
printf '%s\n' "$query_dir" >"$query_last"
if [ -z "$query_bad" ] && [ "$(query paths run last)" = "$query_dir" ] \
   && [ "$(query lock-status last)" = idle ] && [ ! -e "$query_lock" ]; then
  report 0 "queries: last pointer records are validated before capture"
else
  report 1 "queries: last pointer records are validated before capture" "$query_bad"
fi

# A copied directory at LAST is not a legacy symlink or a regular pointer.
rm "$query_last"
mkdir "$query_last"
cp "$query_dir/cmd" "$query_dir/cap" "$query_dir/wt" "$query_dir/runs" "$query_last/"
query_bad=
for query_command in 'paths run' 'paths lock' 'lock-status'; do
  query_out=$(query $query_command last 2>"$TMP/query-error")
  query_rc=$?
  if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
    query_bad="$query_command: rc=$query_rc output=$query_out"; break
  fi
done
if [ -z "$query_bad" ] && [ -d "$query_last" ] && [ ! -e "$query_lock" ]; then
  report 0 "queries: a directory at the last pointer is refused without mutation"
else
  report 1 "queries: a directory at the last pointer is refused without mutation" "$query_bad"
fi
rm "$query_last/cmd" "$query_last/cap" "$query_last/wt" "$query_last/runs"
rmdir "$query_last"
printf '%s\n' "$query_dir" >"$query_last"

# Legacy pointer symlinks must preserve physical identity before capture too.
if [ "$query_line_paths" = 1 ] && [ -L "$TMP/query-state-link" ]; then
  query_bad=
  query_pointer_target=$query_dir$'\n'
  mkdir "$query_pointer_target"
  rm "$query_last"
  ln -s "$query_pointer_target" "$query_last"
  for query_command in 'paths run' 'paths lock' 'lock-status'; do
    query_out=$(query $query_command last 2>"$TMP/query-error")
    query_rc=$?
    if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
      query_bad="$query_command: rc=$query_rc output=$query_out"; break
    fi
  done
  [ -L "$query_last" ] || query_bad="query replaced the legacy pointer"
  rm "$query_last"
  ln -s "$query_dir" "$query_last"
  if [ -z "$query_bad" ] && [ "$(query paths run last)" = "$query_dir" ] \
     && [ "$(query lock-status last)" = idle ] && [ -L "$query_last" ] \
     && [ ! -e "$query_lock" ]; then
    report 0 "queries: legacy pointer symlinks retain physical identity before capture"
  else
    report 1 "queries: legacy pointer symlinks retain physical identity before capture" "$query_bad"
  fi
  rm "$query_last"
  printf '%s\n' "$query_dir" >"$query_last"
else
  skip "queries: legacy pointer symlinks retain physical identity before capture" "requires POSIX symlink filenames"
fi

# Legacy paths come from the recorded worktree, not the querying script's root.
rm "$query_dir/runs"
printf '%s\n' "$STOP_WT" >"$query_dir/wt"
if [ "$(query paths lock last)" = "$STOP_WT/.test-run.lock" ] \
   && [ "$(query paths owner last)" = "$STOP_WT/.test-run.lock.owner" ]; then
  report 0 "queries: legacy runs retain their recorded in-tree lock paths"
else
  report 1 "queries: legacy runs retain their recorded in-tree lock paths"
fi
}
harness_leg_run '39: read-only paths and lock-status queries #queries' leg_39

cat >"$repeat_bin/dune" <<'EOF'
#!/usr/bin/env bash
set -u
# Close the inherited lock descriptor before probing through a fresh open.
# Acquiring here would prove repeat released its one set-wide lock too early.
# Ask the production read-only API; a held answer must come from the actual
# lock, not a glob over implementation-private filenames. A separate absent
# lock control below prevents a constant "held" query from blessing this leg.
lock_probe() {
  local answer rc
  answer=$(tools/test-run.sh lock-status 9>&-)
  rc=$?
  if [ "$rc" != 3 ] || [ "$answer" != held ]; then
    echo "repeat fixture expected held lock, got $rc: $answer" >&2
    exit 91
  fi
}
lock_probe
# The run's last phase asks dune for its version and its promotion list
# (gh-ocannl-1087), still under the lock. The version is
# REPEAT_TEST_DUNE_VERSION (3.24.2 by default; `fail` fails). The list is the
# lines of the file REPEAT_TEST_PROMOTIONS names, printed on stdout from 3.22
# and on stderr before, as dune does, with REPEAT_TEST_PROMOTION_NOISE on the
# other stream; unset, the list fails, so every leg that sets none digests
# from the log. Neither query is counted or recorded among the calls every leg
# pins; the list's argv goes to REPEAT_TEST_PROMOTION_CALLS when that is set.
if [ "${1:-}" = --version ]; then
  [ "${REPEAT_TEST_DUNE_VERSION:-}" != fail ] || exit 1
  printf '%s\n' "${REPEAT_TEST_DUNE_VERSION:-3.24.2}"
  exit 0
fi
# These list/digest fixtures have no corrected payload; the separate real-Dune
# test-promotion-record harness tests capture and replay.
if [ "${1:-}" = promotion ] && [ "${2:-}" = diff ]; then exit 1; fi
if [ "${1:-}" = promotion ]; then
  [ -z "${REPEAT_TEST_PROMOTION_CALLS:-}" ] || printf '%s\n' "$*" >>"$REPEAT_TEST_PROMOTION_CALLS"
  [ -n "${REPEAT_TEST_PROMOTIONS:-}" ] || exit 1
  case ${REPEAT_TEST_DUNE_VERSION:-3.24.2} in
    3.2[01].* | 3.1[0-9].*)
      cat "$REPEAT_TEST_PROMOTIONS" >&2
      [ -z "${REPEAT_TEST_PROMOTION_NOISE:-}" ] || printf '%s\n' "$REPEAT_TEST_PROMOTION_NOISE"
      ;;
    *)
      cat "$REPEAT_TEST_PROMOTIONS"
      [ -z "${REPEAT_TEST_PROMOTION_NOISE:-}" ] || printf '%s\n' "$REPEAT_TEST_PROMOTION_NOISE" >&2
      ;;
  esac
  exit 0
fi
# Repeat establishes a fresh context with `dune clean` before each measured
# invocation. The fixture keeps setup out of the iteration count and streams.
if [ "${1:-}" = clean ]; then
  printf 'clean %s\n' "$*" >>"$REPEAT_TEST_CALLS"
  exit 0
fi
n=0
[ ! -f "$REPEAT_TEST_COUNTER" ] || n=$(cat "$REPEAT_TEST_COUNTER")
n=$((n + 1))
printf '%s\n' "$n" >"$REPEAT_TEST_COUNTER"
# The trace options dune was handed, each observed on its own (leg 80): one
# line per `--trace-file` word to REPEAT_TEST_TRACE_ARGS when set -- its
# 1-based position, `before` or `after` dune's own `--`, and its value. Only
# the runner's own is kept out of the calls every leg pins: exactly
# `--trace-file=<the run's directory>/trace.csexp`, as the second word, right
# after the subcommand, where the launch puts it. Every other word -- a
# second copy of that one, a caller's own, a program's argument past `--` --
# stays in the calls verbatim. The synthetic trace REPEAT_TEST_TRACE names,
# when set, is copied to where the runner's own points.
pos=0 side=before
for arg in "$@"; do
  pos=$((pos + 1))
  if [ "$side" = before ] && [ "$arg" = -- ]; then side=after; continue; fi
  case $arg in
    --trace-file | --trace-file=*)
      [ -z "${REPEAT_TEST_TRACE_ARGS:-}" ] ||
        printf '%s %s %s\n' "$pos" "$side" "${arg#--trace-file=}" >>"$REPEAT_TEST_TRACE_ARGS" ;;
  esac
done
if [ -n "${OCANNL_TOOL_TESTRUN_OWN:-}" ] && [ $# -ge 2 ] &&
   [ "$2" = "--trace-file=$OCANNL_TOOL_TESTRUN_OWN/trace.csexp" ]; then
  trace_to=${2#--trace-file=}
  trace_sub=$1
  shift 2
  set -- "$trace_sub" "$@"
  [ -z "${REPEAT_TEST_TRACE:-}" ] || cp "$REPEAT_TEST_TRACE" "$trace_to"
fi
# A build directory the argv names is made, as dune would (leg 81: a `fresh`
# run's own goes with the run).
if [ -n "${REPEAT_TEST_MKBUILD:-}" ]; then
  for arg in "$@"; do
    case $arg in --build-dir=*) mkdir -p "${arg#--build-dir=}/default" ;; esac
  done
fi
printf '%s\n' "$*" >>"$REPEAT_TEST_CALLS"
if [ -n "${REPEAT_TEST_WAIT_PREFIX:-}" ] \
   && { [ -z "${REPEAT_TEST_WAIT_AT:-}" ] || [ "$REPEAT_TEST_WAIT_AT" = "$n" ]; }; then
  : >"$REPEAT_TEST_WAIT_PREFIX.ready"
  while [ ! -e "$REPEAT_TEST_WAIT_PREFIX.release" ]; do sleep 0.05; done
fi
case $REPEAT_TEST_MODE in
  stable) printf 'stable stdout\n'; printf 'stable stderr\n' >&2 ;;
  stdout) printf 'stdout %s\n' "$n"; printf 'stable stderr\n' >&2 ;;
  stderr) printf 'stable stdout\n'; printf 'stderr %s\n' "$n" >&2 ;;
  fail) printf 'stable stdout\n'; printf 'stable failure\n' >&2; exit 7 ;;
  fail_first)
    if [ "$n" = 1 ]; then printf 'first failure\n' >&2; exit 7; fi
    printf 'later stdout\n'; printf 'later stderr\n' >&2
    ;;
  orphan_first)
    if [ "$n" = 1 ]; then
      printf '%s\n' "$$" >"$REPEAT_TEST_ORPHAN_PID"
      trap ': >"$REPEAT_TEST_ORPHAN_REAPED"; exit 0' TERM
      kill -KILL "$PPID"
      while :; do sleep 1; done
    fi
    [ -e "$REPEAT_TEST_ORPHAN_REAPED" ] || exit 93
    printf 'post-reap stdout\n'; printf 'post-reap stderr\n' >&2
    ;;
  session_escape)
    for arg in "$@"; do
      case $arg in
        --build-dir=*) mkdir -p "${arg#--build-dir=}" ;;
      esac
    done
    perl -MPOSIX -e '
      $SIG{HUP} = "IGNORE";
      POSIX::setsid() >= 0 or die "setsid: $!";
      open my $pf, ">", $ENV{REPEAT_TEST_ESCAPE_PID} or die;
      print $pf "$$\n"; close $pf;
      select undef, undef, undef, 0.05 until -e $ENV{REPEAT_TEST_ESCAPE_RELEASE};
    ' </dev/null >/dev/null 2>&1 &
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
      [ -s "$REPEAT_TEST_ESCAPE_PID" ] && break
      sleep 0.05
    done
    printf 'escaped parent stdout\n'; printf 'escaped parent stderr\n' >&2
    ;;
  # dune's own command-line refusals, verbatim from dune 3.24 (cmdliner): the
  # `dune: <complaint>` / `Usage: dune ...` pair on stderr, exit 1, nothing
  # else -- the operand one wraps its complaint onto an indented continuation
  # line, which the digest must quote whole.
  usage_option)
    printf "dune: unknown option '--frobnicate'.\nUsage: dune build [OPTION]… [TARGET]…\nTry 'dune build --help' or 'dune --help' for more information.\n" >&2
    exit 1 ;;
  # A program that ran, printing a `test-run: ` line of its own and then a
  # nested refusal: output, not the launcher's prelude (Codex review round 2
  # on PR #832).
  prefixed_refusal)
    printf "test-run: a program's own line\n" >&2
    printf "dune: unknown option '--frobnicate'.\nUsage: dune build [OPTION]… [TARGET]…\n" >&2
    exit 1 ;;
  usage_command)
    printf "dune: unknown command 'frobnicate', must be one of 'build', 'clean', 'exec', 'runtest' or 'test'.\nUsage: dune COMMAND …\nTry 'dune --help' for more information.\n" >&2
    exit 1 ;;
  usage_operand)
    printf "dune: option '-j': invalid concurrency value, must be 'auto' or a positive\n      number\nUsage: dune build [OPTION]… [TARGET]…\nTry 'dune build --help' or 'dune --help' for more information.\n" >&2
    exit 1 ;;
  # gh-ocannl-1281: a test whose GPU device initialization outlived the
  # in-process deadline -- its line, at a line's start, among an ordinary red
  # run's -- and a test quoting such a line, indented, which is no death.
  wedged_init)
    printf 'File "test/operations/dune", lines 3-9, characters 0-120:\n' >&2
    printf 'OCANNL device_init_deadline: hip device initialization did not finish within 120s; the device or driver is likely wedged. Dying by SIGALRM.\n' >&2
    exit 1 ;;
  # ...and the same line followed by more output than the digest's 10MB scan
  # window: parallel actions can bury it.
  wedged_long)
    printf 'OCANNL device_init_deadline: hip device initialization did not finish within 120s; the device or driver is likely wedged. Dying by SIGALRM.\n' >&2
    perl -e 'my $l = ("y" x 99) . "\n"; print STDERR $l for 1 .. 110000'
    exit 1 ;;
  wedged_quoted)
    printf 'File "test/operations/dune", lines 3-9, characters 0-120:\n' >&2
    printf '  child | OCANNL device_init_deadline: hip device initialization did not finish within 1s\n' >&2
    exit 1 ;;
  # An ordinary red run: a located error, exit 1 -- and the same with a test's
  # own output opening on a `dune:` line, which no `Usage:` line follows.
  red)
    printf 'File "test/operations/fixture.ml", line 3, characters 4-9:\nError: This expression has type int but an expression was expected of type float\n' >&2
    exit 1 ;;
  dune_prefixed_red)
    printf 'dune: is what this test prints first\n'
    printf 'File "test/operations/fixture.ml", line 3, characters 4-9:\nError: This expression has type int but an expression was expected of type float\n' >&2
    exit 1 ;;
  # A program dune DID run (`dune exec`) that shells out to dune with bad
  # arguments, prints that nested refusal first, then fails on its own: the
  # complete refusal shape, followed by evidence that something ran -- a
  # located error in one, plain output in the other.
  nested_usage_red)
    printf "dune: unknown option '--frobnicate'.\nUsage: dune build [OPTION]… [TARGET]…\nTry 'dune build --help' or 'dune --help' for more information.\n" >&2
    printf 'File "test/operations/fixture.ml", line 3, characters 4-9:\nError: This expression has type int but an expression was expected of type float\n' >&2
    exit 1 ;;
  nested_usage_output)
    printf "dune: unknown option '--frobnicate'.\nUsage: dune build [OPTION]… [TARGET]…\nTry 'dune build --help' or 'dune --help' for more information.\n" >&2
    printf 'the program went on to print this and then failed\n'
    exit 1 ;;
  # gh-ocannl-1055. A rule whose action failed before its diff ran: dune
  # quotes the stanza -- naming the `.corrected` file its `diff?` would read --
  # over the runner's own error, verbatim from the reported run (plus an
  # `.expected` stanza line and a test's `--- ` banner that no `+++ ` follows).
  action_no_diff)
    printf 'File "test/einsum/dune", lines 197-217, characters 0-458:\n197 | (rule\n198 |  (alias runtest-surjectivity)\n.....\n215 |     -diff-cmd\n216 |     -)\n217 |    (diff? surjectivity.ml surjectivity.ml.corrected))))\n' >&2
    printf 'File "test/operations/fixture.expected", line 1, characters 0-0:\n--- stage 2 ---\nnot a diff\n' >&2
    printf 'ppx_inline_test error: the following -only-test flags matched nothing: surjectivty.ml.\n' >&2
    exit 1 ;;
  # A golden mismatch as dune prints it (git diff), plain and colored.
  diff_hunk | diff_hunk_color)
    if [ "$REPEAT_TEST_MODE" = diff_hunk_color ]; then b=$(printf '\033[1m') r=$(printf '\033[0m')
    else b= r=; fi
    printf 'File "test/operations/fixture.expected", line 1, characters 0-0:\n' >&2
    printf '%sdiff --git a/_build/default/test/operations/fixture.expected b/_build/default/test/operations/fixture.exe.output%s\n' "$b" "$r" >&2
    printf '%s--- a/_build/default/test/operations/fixture.expected%s\n' "$b" "$r" >&2
    printf '%s+++ b/_build/default/test/operations/fixture.exe.output%s\n' "$b" "$r" >&2
    printf '@@ -1 +1 @@\n-old line\n+new line\n' >&2
    exit 1 ;;
  # The same mismatch through `--diff-command patdiff` (its doubled headers),
  # and through `--diff-command -`, which prints the location and no diff.
  patdiff_hunk)
    printf 'File "test/operations/fixture.expected", line 1, characters 0-0:\n' >&2
    printf -- '------ _build/default/test/operations/fixture.expected\n' >&2
    printf '++++++ _build/default/test/operations/fixture.exe.output\n' >&2
    printf '@|-1,1 +1,1 ====\n-|old line\n+|new line\n' >&2
    exit 1 ;;
  diff_suppressed)
    printf 'File "test/operations/fixture.expected", line 1, characters 0-0:\n' >&2
    exit 1 ;;
  # An inline-expect rule's mismatch under `--diff-command -`: the remaining
  # location names the `.ml` source, no `.expected` or `.corrected` at all.
  inline_suppressed)
    printf 'File "test/einsum/surjectivity.ml", line 12, characters 0-1:\n' >&2
    exit 1 ;;
  # A real hunk followed by more output than the digest's 10MB scan window.
  diff_then_noise)
    printf -- '--- a/_build/default/test/operations/fixture.expected\n' >&2
    printf '+++ b/_build/default/test/operations/fixture.exe.output\n' >&2
    printf '@@ -1 +1 @@\n-old line\n+new line\n' >&2
    yes 'noise from a parallel action' | head -c 10500000
    exit 1 ;;
  # A dune killed by a signal, not by its caller: the run's last phase must
  # exit with the status an exec'd dune would have had (gh-ocannl-1087).
  self_kill) kill -KILL $$ ;;
  # The per-test cap (gh-ocannl-1211). An action as dune starts one: a child
  # leading its own process group, which this dune kills on TERM as dune
  # does. It runs REPEAT_TEST_ACTION_SECS (60: a hung test) and records its
  # pid in REPEAT_TEST_ACTION_PID. The parent is perl renamed `dune` -- a
  # shebang script's process name is its interpreter's, and the watcher
  # finds tests as the children of a process named dune ($0 sets the name ps
  # reads on Linux and macOS; where it does not, leg 75 skips). With
  # REPEAT_TEST_ACTION_HOPS it runs that perl script instead, from hop 1: an
  # action that keeps its pid while exec renames it (gh-ocannl-1319).
  action)
    exec perl -e '$0 = "dune"; my ($pidf, $secs) = @ARGV;
      my $c = fork(); die "fork: $!" unless defined $c;
      if (!$c) { setpgrp(0, 0); open my $f, ">", "$pidf.tmp" or die; print $f "$$\n"; close $f;
                 rename "$pidf.tmp", $pidf;
                 my $hops = $ENV{REPEAT_TEST_ACTION_HOPS} // "";
                 my @cmd = $hops ne "" ? ($^X, $hops, 1) : ("sleep", $secs);
                 exec @cmd; exit 127 }
      $SIG{TERM} = sub { kill "KILL", -$c; exit 143 };
      waitpid($c, 0); print "action done\n"; exit 0' \
      "$REPEAT_TEST_ACTION_PID" "${REPEAT_TEST_ACTION_SECS:-60}"
    ;;
  # A dune that is itself slow, starting no action: dune is not a test.
  slow_dune)
    exec perl -e '$0 = "dune"; select undef, undef, undef, $ARGV[0]; print "slow dune done\n"' \
      "${REPEAT_TEST_ACTION_SECS:-3}"
    ;;
  *) echo "unknown repeat fixture mode: $REPEAT_TEST_MODE" >&2; exit 92 ;;
esac
EOF
chmod +x "$repeat_bin/dune"
cat >"$repeat_bin/diff" <<'EOF'
#!/usr/bin/env bash
set -u
if [ -n "${REPEAT_TEST_DIFF_WAIT_PREFIX:-}" ]; then
  : >"$REPEAT_TEST_DIFF_WAIT_PREFIX.ready"
  while [ ! -e "$REPEAT_TEST_DIFF_WAIT_PREFIX.release" ]; do /bin/sleep 0.05; done
fi
exec "$REPEAT_TEST_REAL_DIFF" "$@"
EOF
chmod +x "$repeat_bin/diff"

repeat_out= repeat_rc= repeat_dir=
await_fixture_ready() { # MARKER [ATTEMPTS] -- never assume the fixture arrived
  local marker=$1 attempts=${2:-300} attempt=0
  while [ "$attempt" -lt "$attempts" ]; do
    [ -e "$marker" ] && return 0
    sleep 0.1
    attempt=$((attempt + 1))
  done
  return 1
}
# Negative control for the timeout half of the helper's contract. The live
# fixtures below prove its success half; this absent marker must be refused.
# A shared fixture that is not what the legs assume is a FAIL here, outside any
# leg, and the run goes on (gh-ocannl-1374): the legs reading it are named in
# the label, and whatever they conclude, the run is red.
if await_fixture_ready "$TMP/repeat-never-ready" 1; then
  report 1 "fixture: await_fixture_ready accepted an absent marker (every leg awaiting a fixture reads it)"
fi
# The three width-cap probes read what the leg exported, which is the
# harness-wide absent defaults (hermetic_probes) unless the leg opted in. A leg
# that sets the array fixture_launcher runs the script under it (leg 85: a
# parent that reaps no orphans); empty, the script runs directly.
fixture_launcher=()
fixture_probe() { # tag mode runs subcommand [argv...]
  local tag=$1 mode=$2 runs=$3
  shift 3
  mkdir -p "$runs"
  : >"$TMP/$tag.counter"
  : >"$TMP/$tag.calls"
  REPEAT_TEST_MODE=$mode \
  REPEAT_TEST_COUNTER=$TMP/$tag.counter \
  REPEAT_TEST_CALLS=$TMP/$tag.calls \
  REPEAT_TEST_WAIT_PREFIX= \
  REPEAT_TEST_WAIT_AT= \
  REPEAT_TEST_ORPHAN_PID= \
  REPEAT_TEST_ORPHAN_REAPED= \
  REPEAT_TEST_DIFF_WAIT_PREFIX= \
  REPEAT_TEST_REAL_DIFF="$(command -v diff)" \
  OCANNL_TOOL_TEST_RUNS=$runs \
  PATH=$repeat_bin:$PATH \
    ${fixture_launcher[@]+"${fixture_launcher[@]}"} \
    "$repeat_root/tools/test-run.sh" "$@" >"$TMP/$tag.out" 2>"$TMP/$tag.err"
  fixture_rc=$?
  fixture_out=$(cat "$TMP/$tag.out")
  fixture_err=$(cat "$TMP/$tag.err")
  fixture_calls=$(cat "$TMP/$tag.calls")
  fixture_dir=$(OCANNL_TOOL_TEST_RUNS="$runs" "$repeat_root/tools/test-run.sh" paths run last 2>/dev/null)
}
repeat_probe() {
  local tag=$1 mode=$2
  shift 2
  fixture_probe "$tag" "$mode" "$TMP/repeat-runs-$tag" repeat "$@"
  repeat_rc=$fixture_rc repeat_out=$fixture_out repeat_dir=$fixture_dir
}
# A background set whose fixture never became ready ends its leg (gh-ocannl-1374), released and
# cancelled first: its lock and its group must not reach the legs that run after it.
repeat_abandon() { # wait prefix of the background set in $repeat_pid
  touch "$1.release"
  [ -n "$repeat_pid" ] || return 0
  kill -TERM -- "-$repeat_pid" 2>/dev/null || kill -TERM "$repeat_pid" 2>/dev/null
  wait "$repeat_pid" 2>/dev/null
  repeat_pid=
}

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

# ---------------------------------------------------------------------------
# Legs 26-28: this script's own options are refused on the wrong side of dune
# ---------------------------------------------------------------------------
# `run build @alias --cap 900` used to forward `--cap 900` to dune, which exits
# 1 on the unknown option having built nothing; the digest then reads `FAIL
# (exit 1)` with no matching error lines -- a verdict indistinguishable from a
# failing test, and a session acting on it debugs code that never ran. So the
# refusal is asserted together with the CALLS file being empty: "runs nothing"
# is the half that makes the exit code trustworthy.
argv_probe() { # tag subcommand [argv...]
  local tag=$1 runs=${argv_runs:-$TMP/argv-runs-$1}
  shift
  fixture_probe "$tag" "${argv_mode:-stable}" "$runs" "$@"
  argv_rc=$fixture_rc argv_out=$fixture_out argv_err=$fixture_err
  argv_calls=$fixture_calls argv_dir=$fixture_dir
}
argv_rc= argv_out= argv_err= argv_calls= argv_dir= argv_mode= argv_runs=

# Every subcommand that takes options, and both spellings of a value option:
# the guard sits on one code path, but a future refactor that splits it must
# not be able to leave one entry point forwarding.
leg_26_28() {
misplaced_label="an option after the dune arguments is refused before dune runs"
misplaced_detail=
for probe in "run:run build @cheap --cap 900" \
             "run-test-cap:run build @cheap --test-cap 9" \
             "start:start build @cheap --cap=900" \
             "repeat:repeat 2 build @cheap --alone"; do
  argv_probe "misplaced-${probe%%:*}" ${probe#*:}
  # Exit 2 is the usage code; the message must name the option AND the order,
  # or the reader is left where the unknown-option error left them.
  case $argv_rc in
    2) ;;
    *) misplaced_detail="${probe#*:}: exit $argv_rc (want 2): $argv_err" ; break ;;
  esac
  case $argv_err in
    *"belongs before the dune arguments"*"tools/test-run.sh"*) ;;
    *) misplaced_detail="${probe#*:}: unhelpful refusal: $argv_err" ; break ;;
  esac
  if [ -n "$argv_calls" ]; then
    misplaced_detail="${probe#*:}: dune was invoked anyway: $argv_calls"
    break
  fi
done
if [ -z "$misplaced_detail" ]; then
  report 0 "$misplaced_label"
else
  report 1 "$misplaced_label" "$misplaced_detail"
fi

# The control for the leg above: the refusal must be about PLACEMENT, not about
# the word. A guard that also rejected the documented form would pass leg 26.
argv_probe cap-placed run --cap 900 build @cheap
if [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ]; then
  report 0 "a correctly placed --cap is consumed and never reaches dune"
else
  report 1 "a correctly placed --cap is consumed and never reaches dune" \
    "exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err"
fi

# Past dune's own separator the word is an executable's argument -- the caller's
# business, and the one place `--cap` after the target can be meant.
argv_probe cap-after-sep run exec ./prog.exe -- --cap 900
if [ "$argv_rc" = 0 ] && [ "$argv_calls" = "exec ./prog.exe -- --cap 900" ]; then
  report 0 "an option past dune's -- reaches the program untouched"
else
  report 1 "an option past dune's -- reaches the program untouched" \
    "exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err"
fi
}
harness_leg_run '26-28: options on the wrong side of the dune arguments #options' leg_26_28

# ---------------------------------------------------------------------------
# Legs 29-34: the dxg width cap (gh-ocannl-983)
# ---------------------------------------------------------------------------
# A box that reaches its GPU through WSL2's `/dev/dxg` bridge overflows the
# bridge's VM-bus ring at dune's default width, and the runtime reports the lost
# messages as device/binary/stream-creation refusals: a red suite in the very
# stanzas a real backend regression lands in. tools/sweep.sh capped its unit
# there; a manual run did not, and an hour of box time went into rediscovering
# the cap by bisection. So `run`/`start` injects it -- but ONLY where the caller
# expressed no width at all, which is what keeps "runs dune as given" true.
#
# Every leg drives the real launch path against the fixture dune and reads the
# recorded argv, so what is pinned is the command dune was actually given, not a
# decision variable. The bridge is faked through OCANNL_TOOL_DXG_DEVICE (the
# device path the table probes), which is the only reason these run on a Mac.
dxg_present=$TMP/fake-dxg-device
dxg_absent=$TMP/fake-dxg-device-absent
: >"$dxg_present"
rm -f "$dxg_absent"
# The fixture is the whole experiment: a present path that is absent, or an
# absent one that exists, would make every leg below agree with itself.
if [ ! -e "$dxg_present" ] || [ -e "$dxg_absent" ]; then
  report 1 "fixture: the dxg device paths are not what the legs assume (every leg faking a dxg bridge reads them)"
fi
dxg_probe() { # tag device backend subcommand [argv...]
  local tag=$1 device=$2 backend=$3
  shift 3
  # Exported rather than written as a prefix: bash's rules for assignments
  # before a FUNCTION call differ between modes, and a leg that silently ran
  # without its backend would pass by testing the no-op case.
  export OCANNL_TOOL_DXG_DEVICE=$device
  if [ -n "$backend" ]; then export OCANNL_BACKEND=$backend; else unset OCANNL_BACKEND; fi
  argv_probe "$tag" "$@"
  unset OCANNL_BACKEND
  hermetic_probes
}
# The sourced lab-map.sh and the lab's endpoint map as a fixture, which this
# section's last leg and leg 48 resolve sweep destinations through (see there).
LAB_MAP_SRC=$(dirname "$SWEEP_SRC")/lab-map.sh
LAB_MAP='rog rog-nv-linux rog-nv-win rog-nv-wsl rog-lan
minix minix-amd-linux minix-amd-win minix-amd-wsl minix-lan
tuf tuf-amd-linux tuf-amd-win tuf-amd-wsl'

# The injection, in all four places it has to appear: dune's argv, the RECORDED
# command (so a later digest shows the width too), stderr, and the run's log --
# the artifact whoever triages the red suite reads.
leg_29_34() {
dxg_probe cap-inject "$dxg_present" cuda run build @cheap
dxg_detail=
[ "$argv_rc" = 0 ] || dxg_detail="exit $argv_rc"
[ -n "$dxg_detail" ] || [ "$argv_calls" = "build -j 2 @cheap" ] ||
  dxg_detail="calls: ${argv_calls:-<none>}"
[ -n "$dxg_detail" ] || grep -q -- 'capping dune at -j 2' <<<"$argv_err" ||
  dxg_detail="stderr did not announce the cap: $argv_err"
[ -n "$dxg_detail" ] || grep -q -- 'box-jobs.sh' <<<"$argv_err" ||
  dxg_detail="stderr did not name the cap's source: $argv_err"
[ -n "$dxg_detail" ] || grep -q -- 'agent-notes/build-and-test.md' <<<"$argv_err" ||
  dxg_detail="stderr did not name the agent note: $argv_err"
[ -n "$dxg_detail" ] || { [ -n "$argv_dir" ] && grep -q -- 'capping dune at -j 2' "$argv_dir/log"; } ||
  dxg_detail="the run log does not carry the cap: ${argv_dir:-<no run>}"
[ -n "$dxg_detail" ] || { [ -n "$argv_dir" ] && grep -q -- '-j 2' "$argv_dir/cmd"; } ||
  dxg_detail="the recorded command does not carry the cap: $(cat "$argv_dir/cmd" 2>/dev/null)"
if [ -z "$dxg_detail" ]; then
  report 0 "dxg: a GPU run that named no width is capped, announced and recorded"
else
  report 1 "dxg: a GPU run that named no width is capped, announced and recorded" "$dxg_detail"
fi

# A width the caller named is honored, in every spelling dune takes -- and
# `-j` PAST dune's own separator is the program's argument, not a width, so
# that one is still capped. The negative control matters: a scan that stopped
# at the wrong place would either forward two widths or cap nothing.
dxg_detail=
for probe in "sp-space:-j 8:build -j 8 @cheap" \
             "sp-glued:-j8:build -j8 @cheap" \
             "sp-long:--jobs 8:build --jobs 8 @cheap" \
             "sp-eq:--jobs=8:build --jobs=8 @cheap" \
             "sp-auto:-jauto:build -jauto @cheap"; do
  tag=${probe%%:*}; rest=${probe#*:}; want=${rest#*:}
  # shellcheck disable=SC2086
  dxg_probe "cap-$tag" "$dxg_present" cuda run build ${rest%%:*} @cheap
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "$want" ]; then
    dxg_detail="${rest%%:*}: exit $argv_rc; calls: ${argv_calls:-<none>} (want $want)"; break
  fi
  case $argv_err in
    *"names its own dune width"*"NOT injected"*) ;;
    *) dxg_detail="${rest%%:*}: the honored width was not reported: ${argv_err:-<nothing>}"; break ;;
  esac
done
if [ -z "$dxg_detail" ]; then
  dxg_probe cap-past-sep "$dxg_present" cuda run exec ./prog.exe -- -j 8
  [ "$argv_rc" = 0 ] && [ "$argv_calls" = "exec -j 2 ./prog.exe -- -j 8" ] ||
    dxg_detail="past the separator: exit $argv_rc; calls: ${argv_calls:-<none>}"
fi
if [ -z "$dxg_detail" ]; then
  report 0 "dxg: a width the caller named is honored; one past dune's -- is not a width"
else
  report 1 "dxg: a width the caller named is honored; one past dune's -- is not a width" "$dxg_detail"
fi

# The two halves of the condition, each denied on its own. Silence is part of
# the contract here: an announcement on an ordinary box or a CPU suite is noise
# that trains its reader to skip the one line that matters.
dxg_detail=
for probe in "cpu:$dxg_present:cc" "cpu-multidev:$dxg_present:multidev_cc" \
             "offbox-cuda:$dxg_absent:cuda" "offbox-hip:$dxg_absent:hip"; do
  tag=${probe%%:*}; rest=${probe#*:}
  dxg_probe "cap-$tag" "${rest%%:*}" "${rest#*:}" run build @cheap
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "build @cheap" ] ||
     grep -qi 'cap\|dxg' <<<"$argv_err"; then
    dxg_detail="$tag: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"; break
  fi
done
if [ -z "$dxg_detail" ]; then
  report 0 "dxg: no cap and no announcement for a CPU backend, or off a dxg host"
else
  report 1 "dxg: no cap and no announcement for a CPU backend, or off a dxg host" "$dxg_detail"
fi

# The backend no longer has to be exported to be read (gh-ocannl-1066): it is
# resolved, from the test configurations, the stanzas the argv reaches and the
# argv. With OCANNL_BACKEND unset, a batch the configurations pin to cc and
# that reaches no stanza naming a GPU is a CPU batch -- no cap, and silence;
# one whose configuration names hip, one reaching a stanza that names cuda, and
# one whose backends cannot be read are capped, with the announcement saying
# which backend and why. A caller who named a width is only told.
dxg_detail=
dxg_probe cap-unset "$dxg_present" "" run build @cheap
{ [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ] && ! grep -qi 'cap\|dxg' <<<"$argv_err"; } ||
  dxg_detail="a cc batch: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"
[ -n "$dxg_detail" ] || { [ -n "$argv_dir" ] && grep -q '^test-run: batch: holds cc: test/config resolves backend=cc$' "$argv_dir/log"; } ||
  dxg_detail="a cc batch: the log does not record what it holds: $(cat "$argv_dir/log" 2>/dev/null)"
for probe in "config|hip|$FAKE_READS\nend|test/config resolves backend=hip" \
             "reach|cc|names cuda: it reaches t_cuda in x, which names cuda\nend|it reaches t_cuda in x, which names cuda" \
             "unread|fail|$FAKE_READS\nend|its backends are unread"; do
  [ -z "$dxg_detail" ] || break
  IFS='|' read -r tag bt reach what <<<"$probe"
  export FAKE_BACKEND_TEST=$bt FAKE_REACH=$reach
  dxg_probe "cap-unset-$tag" "$dxg_present" "" run build @cheap
  unset FAKE_BACKEND_TEST FAKE_REACH
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "build -j 2 @cheap" ]; then
    dxg_detail="$tag: exit $argv_rc; calls: ${argv_calls:-<none>} (want build -j 2 @cheap)"; break
  fi
  case $(tr '\n' ' ' <<<"$argv_err") in
    *"capping dune at -j 2"*"$what"*) ;;
    *) dxg_detail="$tag: the announcement does not say why: ${argv_err:-<nothing>}"; break ;;
  esac
done
if [ -z "$dxg_detail" ]; then
  export FAKE_BACKEND_TEST=hip
  dxg_probe cap-unset-explicit "$dxg_present" "" run build -j 8 @cheap
  unset FAKE_BACKEND_TEST
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build -j 8 @cheap" ] &&
    grep -q 'names its own dune width' <<<"$argv_err"; } ||
    dxg_detail="with a width named: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"
fi
if [ -z "$dxg_detail" ]; then
  report 0 "dxg: with OCANNL_BACKEND unset the backend is resolved: a cc batch is silent, a GPU or unread one capped"
else
  report 1 "dxg: with OCANNL_BACKEND unset the backend is resolved: a cc batch is silent, a GPU or unread one capped" "$dxg_detail"
fi

# The other GPU backend, and the position the width is spliced at: immediately
# after dune's subcommand, which is where dune accepts it for every target --
# and always before dune's own `--`, past which it would be an argument to the
# program dune runs.
dxg_probe cap-hip "$dxg_present" hip run exec ./prog.exe -- alpha
dxg_detail=
[ "$argv_rc" = 0 ] && [ "$argv_calls" = "exec -j 2 ./prog.exe -- alpha" ] ||
  dxg_detail="exit $argv_rc; calls: ${argv_calls:-<none>}"
if [ -z "$dxg_detail" ]; then
  report 0 "dxg: hip is capped too, and the width precedes dune's own separator"
else
  report 1 "dxg: hip is capped too, and the width precedes dune's own separator" "$dxg_detail"
fi

# The anti-drift half calls the shared shipping sweep helper directly. The
# sweep and this runner must source that table, and the sweep must pass the
# resolved destination to every width lookup.
dxg_detail=
grep -q '^\. "\$SWEEP_TOOLS/box-jobs.sh"$' "$SWEEP_SRC" ||
  dxg_detail="sweep.sh does not source tools/box-jobs.sh"
[ -n "$dxg_detail" ] || grep -q '^\. tools/box-jobs\.sh$' "$SRC" ||
  dxg_detail="test-run.sh does not source tools/box-jobs.sh"
# The width depends on how the unit reached its box (gh-ocannl-1029): the two
# lab boxes dual-boot, and only a WSL boot crosses the bridge. Every call site
# must hand box_jobs_sweep_jobs the unit's RESOLVED destination -- the `@<box>`
# placeholder is replaced in UNITS before any unit runs -- since a call that
# dropped it would silently give a native boot the WSL cap. And the
# destinations are resolved by the sourced lab-map.sh's lab_dest_of (with the lab_row it
# reads), over the lab's endpoint map as `wake-lab.sh endpoint-map` printed it
# on 2026-09-27 -- a FIXTURE, never the live map, which is site data outside the
# repository and the sweep's only table of aliases (gh-ocannl-1121). So what is
# pinned is the relationship: whatever alias the sweep picks for a boot,
# box-jobs.sh classifies it as that boot. (LAB_MAP_SRC and the LAB_MAP fixture
# are set beside dxg_probe, outside this leg's gate: leg 48 reads them too.)
if [ -z "$dxg_detail" ]; then
  unit_jobs_calls=$(grep -c 'box_jobs_sweep_jobs "' "$SWEEP_SRC")
  unit_jobs_dest_calls=$(grep -c 'box_jobs_sweep_jobs "$machine" "$backend" "$host"' "$SWEEP_SRC")
  if [ "$unit_jobs_calls" -eq 0 ] || [ "$unit_jobs_calls" != "$unit_jobs_dest_calls" ]; then
    dxg_detail="sweep.sh calls box_jobs_sweep_jobs $unit_jobs_calls times, $unit_jobs_dest_calls with the unit's destination"
  fi
fi
[ -n "$dxg_detail" ] || [ -r "$LAB_MAP_SRC" ] ||
  dxg_detail="cannot read shared endpoint lookups: $LAB_MAP_SRC"
if [ -z "$dxg_detail" ]; then
  # The values themselves, from the shipping table, in one shell: the sweep's
  # hip unit over each of minix's boots (and with no destination, or one the
  # table does not know), rog's cuda unit over each of its boots (uncapped over
  # the bridge, its slot width natively), a CPU unit on a native boot, tuf's
  # native hip unit at its slot width (leg 48 pins both slot widths to the ones
  # test-run.sh injects on such a box), the override that must still win over
  # both transports, the local probe, and the two constants.
  sweep_cap=$(
    . "$JOBS_SRC"
    . "$LAB_MAP_SRC"
    printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s' \
      "$(box_jobs_sweep_jobs minix hip)" \
      "$(box_jobs_sweep_jobs minix hip "$(lab_dest_of minix wsl)")" \
      "$(box_jobs_sweep_jobs minix hip "$(lab_dest_of minix linux)")" \
      "$(box_jobs_sweep_jobs minix hip 10.0.0.7)" \
      "$(box_jobs_sweep_jobs rog-nv cuda "$(lab_dest_of rog wsl)")" \
      "$(box_jobs_sweep_jobs rog-nv cuda "$(lab_dest_of rog linux)")" \
      "$(box_jobs_sweep_jobs minix multidev_cc "$(lab_dest_of minix linux)")" \
      "$(box_jobs_sweep_jobs tuf hip "$(lab_dest_of tuf linux)")" \
      "$(OCANNL_TOOL_SWEEP_JOBS=7 box_jobs_sweep_jobs minix hip)" \
      "$(OCANNL_TOOL_SWEEP_JOBS=7 box_jobs_sweep_jobs minix hip "$(lab_dest_of minix linux)")" \
      "$(box_jobs_sweep_jobs m4-max metal)" \
      "$(OCANNL_TOOL_DXG_DEVICE=$dxg_present box_jobs_local_cap cuda)" \
      "$BOX_JOBS_DXG_CAP/$BOX_JOBS_SDMA_CAP"
  )
  case $sweep_cap in
    "2|2|8|2||8||8|7|7||2|2/8") ;;
    *) dxg_detail="shared table disagrees: $sweep_cap (want 2|2|8|2||8||8|7|7||2|2/8)" ;;
  esac
fi
if [ -z "$dxg_detail" ]; then
  # Every alias the sweep can resolve to is one box-jobs.sh classifies, and
  # classifies as the boot lab_dest_of says it is.
  for box in rog minix; do
    for kind in wsl linux; do
      got=$(. "$JOBS_SRC"; . "$LAB_MAP_SRC"; box_jobs_dest_transport "$(lab_dest_of "$box" "$kind")")
      case $kind:$got in
        wsl:dxg | linux:native) ;;
        *) dxg_detail="$box's $kind alias classifies as '${got}'" ;;
      esac
    done
  done
fi
if [ -z "$dxg_detail" ]; then
  report 0 "dxg: the sweep's box_jobs_sweep_jobs and the injected cap come from one table"
else
  report 1 "dxg: the sweep's box_jobs_sweep_jobs and the injected cap come from one table" "$dxg_detail"
fi
}
harness_leg_run '29-34: the dxg width cap #width' leg_29_34

# ---------------------------------------------------------------------------
# Legs 48-52: the native GPU width caps (gh-ocannl-1033)
# ---------------------------------------------------------------------------
# The fleet runs several correctness batches at once on each native GPU box
# (lukstafi/ludics-lite#316 and ludics-lite#344), measured at a width per batch: -j 4 on
# minix, whose device-wide SDMA pool default-width hip batches can drain
# between them (gh-ocannl-1029), -j 8 on tuf and -j 8 on rog-nv. The widths used to hold only while
# every worker remembered them; now `run`/`start` injects them the way it
# injects the dxg cap. The pool comes from the KFD topology, faked here through
# OCANNL_TOOL_KFD_TOPOLOGY as directories of `properties` files copied from
# what the boxes report (only the keys the probe reads, plus a neighbour
# either side, so a parser keyed on line position would miss); the NVIDIA boot
# through OCANNL_TOOL_NVIDIA_DEVICE, the way the dxg legs fake the bridge.
kfd_node() { # dir simd engines queues-per-engine
  mkdir -p "$1"
  printf 'cpu_cores_count 0\nsimd_count %s\nhive_id 0\nnum_sdma_engines %s\nnum_sdma_xgmi_engines 0\nnum_sdma_queues_per_engine %s\nnum_cp_queues 8\n' \
    "$2" "$3" "$4" >"$1/properties"
}
kfd_cpu_node() { # dir
  mkdir -p "$1"
  printf 'cpu_cores_count 32\nsimd_count 0\nnum_sdma_engines 0\nnum_sdma_queues_per_engine 0\nnum_cp_queues 0\n' >"$1/properties"
}
kfd_small=$TMP/fake-kfd-minix   # gfx1151: 1 engine x 6, as minix-amd-linux reports it
kfd_large=$TMP/fake-kfd-tuf     # gfx1102: 2 engines x 6, as tuf-amd-linux reports it
kfd_nopool=$TMP/fake-kfd-nopool # a GPU node that reports no SDMA engine
kfd_absent=$TMP/fake-kfd-absent
kfd_cpu_node "$kfd_small/0"; kfd_node "$kfd_small/1" 80 1 6
kfd_cpu_node "$kfd_large/0"; kfd_node "$kfd_large/1" 64 2 6
kfd_cpu_node "$kfd_nopool/0"; kfd_node "$kfd_nopool/1" 80 0 6
rm -rf "$kfd_absent"
nv_present=$TMP/fake-nvidiactl
nv_absent=$TMP/fake-nvidiactl-absent
# The fleet's name for this box (lukstafi/ludics-lite's FLEET_LOCAL_BOX): rog,
# or another fleet box -- an NVIDIA machine that is not the fleet's rog. Every
# native probe pins one, so the harness never reads the name of the box it
# runs on; an empty one leaves FLEET_LOCAL_BOX unset, for the hostname legs,
# which pin FLEET_HOSTNAME_MAP (native_map) instead.
fleet_rog=rog-nv-linux
fleet_none=mac-studio
native_fleet=$fleet_none
native_map=
: >"$nv_present"
rm -f "$nv_absent"
if [ ! -f "$kfd_small/1/properties" ] || [ -e "$kfd_absent" ] ||
   [ ! -e "$nv_present" ] || [ -e "$nv_absent" ]; then
  report 1 "fixture: the native-cap paths are not what the legs assume (every leg faking a native device reads them)"
fi
# The pool the fixtures must read as, from the shipping probe itself: a fixture
# the parser misread would leave legs 48 and 49 agreeing with a wrong number.
native_pools=$(
  . "$JOBS_SRC"
  for t in "$kfd_small" "$kfd_large" "$kfd_nopool" "$kfd_absent"; do
    printf '%s,' "$(OCANNL_TOOL_KFD_TOPOLOGY=$t box_jobs_sdma_pool)"
  done
)
if [ "$native_pools" != "6,12,,," ]; then
  report 1 "fixture: the KFD fixtures read as pools '$native_pools', not 6,12,,, -- fix the fixture or the probe (every leg faking a native device reads them)"
fi
native_probe() { # tag dxg kfd nvidia backend subcommand [argv...]
  local tag=$1 dxg=$2 kfd=$3 nv=$4 backend=$5
  shift 5
  export OCANNL_TOOL_DXG_DEVICE=$dxg OCANNL_TOOL_KFD_TOPOLOGY=$kfd OCANNL_TOOL_NVIDIA_DEVICE=$nv
  if [ -n "$native_fleet" ]; then export FLEET_LOCAL_BOX=$native_fleet; else unset FLEET_LOCAL_BOX; fi
  if [ -n "$native_map" ]; then export FLEET_HOSTNAME_MAP=$native_map; else unset FLEET_HOSTNAME_MAP; fi
  if [ -n "$backend" ]; then export OCANNL_BACKEND=$backend; else unset OCANNL_BACKEND; fi
  argv_probe "$tag" "$@"
  unset OCANNL_BACKEND FLEET_LOCAL_BOX FLEET_HOSTNAME_MAP
  hermetic_probes
}

# Leg 48: the injection on each native hazard, in the four places the dxg leg
# pins it -- argv, the recorded command, stderr (with its source and the note)
# and the run's log -- plus what the announcement says the hazard was. The
# fleet's state records rog's slots here, which the CPU rows need.
leg_48() {
native_detail=
native_fleet=$fleet_rog
# The last field is the sweep unit the same box runs, as `<machine>/<lab box>`:
# a sweep unit takes no correctness slot, so on tuf and rog it must run at the
# slot width injected here, and its width is compared with the injected one
# rather than with a number (gh-ocannl-1211's 2026-10-14 sweep ran tuf's unit 16
# wide). minix's unit is the deliberate exception: one unit's SDMA cap, not its
# slot width (tools/box-jobs.sh, box_jobs_sweep_cap).
for probe in "sdma:$kfd_small:$nv_absent:hip:4:SDMA:-" \
             "wide-sdma:$kfd_large:$nv_absent:hip:8:AMD GPU reports 12 allocatable SDMA:tuf/tuf" \
"nvidia:$kfd_absent:$nv_present:cuda:8:native NVIDIA boot:rog-nv/rog" \
             "nvidia-cc:$kfd_absent:$nv_present:cc:8:rog-nv-linux, natively booted:-" \
             "nvidia-multidev:$kfd_absent:$nv_present:multidev_cc:8:execution slot --cpu:-"; do
  IFS=: read -r tag kfd nv backend want what sweep_unit <<<"$probe"
  native_probe "native-$tag" "$dxg_absent" "$kfd" "$nv" "$backend" run build @cheap
  [ "$argv_rc" = 0 ] || { native_detail="$tag: exit $argv_rc"; break; }
  [ "$argv_calls" = "build -j $want @cheap" ] ||
    { native_detail="$tag: calls: ${argv_calls:-<none>} (want build -j $want @cheap)"; break; }
  case $argv_err in
    *"capping dune at -j $want"*"$what"*"box-jobs.sh"*"agent-notes/build-and-test.md"*) ;;
    *) native_detail="$tag: the announcement is not the native one: ${argv_err:-<nothing>}"; break ;;
  esac
  { [ -n "$argv_dir" ] && grep -q -- "capping dune at -j $want" "$argv_dir/log"; } ||
    { native_detail="$tag: the run log does not carry the cap: ${argv_dir:-<no run>}"; break; }
  { [ -n "$argv_dir" ] && grep -q -- "-j $want" "$argv_dir/cmd"; } ||
    { native_detail="$tag: the recorded command does not carry the cap: $(cat "$argv_dir/cmd" 2>/dev/null)"; break; }
  if [ "$sweep_unit" != - ]; then
    sweep_width=$(unset OCANNL_TOOL_SWEEP_JOBS; . "$JOBS_SRC"; . "$LAB_MAP_SRC"
      box_jobs_sweep_jobs "${sweep_unit%/*}" "$backend" "$(lab_dest_of "${sweep_unit#*/}" linux)")
    [ "build -j $sweep_width @cheap" = "$argv_calls" ] ||
      { native_detail="$tag: the sweep's native ${sweep_unit%/*}/$backend unit runs at '${sweep_width}', not the injected width ($argv_calls)"; break; }
  fi
done
# The hostname path of the fleet's identity rule: with FLEET_LOCAL_BOX unset, a
# map whose glob matches any host names it rog, and one matching none does not.
native_fleet=
for probe in "*=rog-nv-linux:build -j 8 @cheap" "no-such-host-*=rog-nv-linux:build @cheap"; do
  [ -z "$native_detail" ] || break
  native_map=${probe%%:*}
  native_probe native-hostname "$dxg_absent" "$kfd_absent" "$nv_present" cc run build @cheap
  [ "$argv_rc" = 0 ] && [ "$argv_calls" = "${probe#*:}" ] ||
    native_detail="hostname map '$native_map': exit $argv_rc; calls: ${argv_calls:-<none>} (want ${probe#*:})"
done
native_map=
native_fleet=$fleet_none
if [ -z "$native_detail" ]; then
  report 0 "native: hip on a small or a larger SDMA pool, and cuda and cc on a native NVIDIA boot, are capped, announced and recorded"
else
  report 1 "native: hip on a small or a larger SDMA pool, and cuda and cc on a native NVIDIA boot, are capped, announced and recorded" "$native_detail"
fi
}
harness_leg_run '48: native GPU width caps injected #width' leg_48

# Leg 49: the negative controls. Each denies one half of a condition leg 48
# met, so a probe that capped every Linux GPU batch -- or keyed on the backend
# alone -- fails here. Silence is part of the contract, as for the dxg legs.
leg_49() {
native_detail=
for probe in "cpu-on-wide-sdma:$kfd_large:$nv_absent:cc" \
             "cuda-on-wide-sdma:$kfd_large:$nv_absent:cuda" \
             "no-kfd:$kfd_absent:$nv_absent:hip" \
             "no-pool:$kfd_nopool:$nv_absent:hip" \
             "cpu-on-sdma:$kfd_small:$nv_absent:cc" \
             "cpu-on-unfleeted-nvidia:$kfd_absent:$nv_present:cc" \
             "cuda-on-sdma:$kfd_small:$nv_absent:cuda" \
             "hip-on-nvidia:$kfd_absent:$nv_present:hip" \
             "cuda-off-nvidia:$kfd_absent:$nv_absent:cuda"; do
  IFS=: read -r tag kfd nv backend <<<"$probe"
  native_probe "native-$tag" "$dxg_absent" "$kfd" "$nv" "$backend" run build @cheap
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "build @cheap" ] ||
     grep -qi 'cap\|sdma\|nvidia\|dxg' <<<"$argv_err"; then
    native_detail="$tag: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"; break
  fi
done
# A WSL boot of the fleet's rog: the bridge carries only the GPU, so a CPU
# batch there is not the native slots' and meets no cap.
if [ -z "$native_detail" ]; then
  native_fleet=$fleet_rog
  native_probe native-cpu-on-nvidia-dxg "$dxg_present" "$kfd_absent" "$nv_present" cc run build @cheap
  native_fleet=$fleet_none
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "build @cheap" ] ||
     grep -qi 'cap\|sdma\|nvidia\|dxg' <<<"$argv_err"; then
    native_detail="cpu-on-nvidia-dxg: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"
  fi
fi
if [ -z "$native_detail" ]; then
  report 0 "native: no cap and no announcement off the measured hazards, or for the other backend"
else
  report 1 "native: no cap and no announcement off the measured hazards, or for the other backend" "$native_detail"
fi
}
harness_leg_run '49: native width-cap negative controls #width' leg_49

# Leg 50: an explicit width still wins on a native hazard, and says so; and a
# WSL boot keeps the bridge's cap even where its topology would read as a
# small pool (the bridge is the tighter limit, and the one measured there).
leg_50() {
native_detail=
for probe in "sdma:$kfd_small:$nv_absent:hip:small-SDMA-pool host" \
             "wide-sdma:$kfd_large:$nv_absent:hip:native AMD GPU host" \
             "nvidia:$kfd_absent:$nv_present:cuda:native NVIDIA host" \
             "nvidia-cpu:$kfd_absent:$nv_present:cc:rog-nv-linux"; do
  IFS=: read -r tag kfd nv backend what <<<"$probe"
  native_fleet=$fleet_rog
  native_probe "native-explicit-$tag" "$dxg_absent" "$kfd" "$nv" "$backend" run build -j 16 @cheap
  native_fleet=$fleet_none
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "build -j 16 @cheap" ]; then
    native_detail="$tag: exit $argv_rc; calls: ${argv_calls:-<none>} (want build -j 16 @cheap)"; break
  fi
  case $argv_err in
    *"$what"*"names its own dune width"*"NOT injected"*) ;;
    *) native_detail="$tag: the honored width was not reported: ${argv_err:-<nothing>}"; break ;;
  esac
done
if [ -z "$native_detail" ]; then
  native_probe native-dxg-wins "$dxg_present" "$kfd_small" "$nv_present" hip run build @cheap
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build -j 2 @cheap" ] &&
    grep -q 'WSL2' <<<"$argv_err"; } ||
    native_detail="dxg with a small pool: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"
fi
if [ -z "$native_detail" ]; then
  report 0 "native: a named width is honored on each native hazard; a dxg boot keeps its own cap"
else
  report 1 "native: a named width is honored on each native hazard; a dxg boot keeps its own cap" "$native_detail"
fi
}
harness_leg_run '50: a named width wins on a native hazard #width' leg_50

# Leg 51: the backend resolved on a native hazard (gh-ocannl-1066). With
# OCANNL_BACKEND unset the width follows what the batch can hold: a batch the
# configurations pin to cc is uncapped on the AMD boxes and on an NVIDIA box
# that is not the fleet's rog; one whose configuration names hip -- the minix
# shape the issue names, reached through the configuration rather than the
# environment -- gets hip's width on either pool, as does one reaching a
# stanza that names hip, and one whose backends cannot be read; on the fleet's
# rog a cc batch gets the CPU width; and beside a small AMD pool there, a batch
# that can hold both hip and cc takes the tighter of the two. The recorded
# command carries the width, and a named width is honored and only told.
leg_51() {
native_detail=
for probe in "cc-on-sdma|$kfd_small|$nv_absent|$fleet_none|cc|$FAKE_READS\nend|build @cheap|" \
             "cc-on-wide-sdma|$kfd_large|$nv_absent|$fleet_none|cc|$FAKE_READS\nend|build @cheap|" \
             "cc-on-unfleeted-nvidia|$kfd_absent|$nv_present|$fleet_none|cc|$FAKE_READS\nend|build @cheap|" \
             "hip-config-sdma|$kfd_small|$nv_absent|$fleet_none|hip|$FAKE_READS\nend|build -j 4 @cheap|SDMA*test/config resolves backend=hip" \
             "hip-config-wide-sdma|$kfd_large|$nv_absent|$fleet_none|hip|$FAKE_READS\nend|build -j 8 @cheap|12 allocatable SDMA*test/config resolves backend=hip" \
             "hip-reach-sdma|$kfd_small|$nv_absent|$fleet_none|cc|names hip: it reaches t_hip in x, which names hip\nend|build -j 4 @cheap|SDMA*it reaches t_hip in x" \
             "unread-sdma|$kfd_small|$nv_absent|$fleet_none|fail|$FAKE_READS\nend|build -j 4 @cheap|SDMA*its backends are unread" \
             "cc-on-rog|$kfd_absent|$nv_present|$fleet_rog|cc|$FAKE_READS\nend|build -j 8 @cheap|rog-nv-linux, natively booted*test/config resolves backend=cc*execution slot --cpu" \
             "tightest-on-rog|$kfd_small|$nv_present|$fleet_rog|cc|names hip: it reaches t_hip in x, which names hip\n$FAKE_READS\nend|build -j 4 @cheap|SDMA*it reaches t_hip in x"; do
  IFS='|' read -r tag kfd nv native_fleet bt reach want what <<<"$probe"
  export FAKE_BACKEND_TEST=$bt FAKE_REACH=$reach
  native_probe "native-unset-$tag" "$dxg_absent" "$kfd" "$nv" "" run build @cheap
  unset FAKE_BACKEND_TEST FAKE_REACH
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "$want" ]; then
    native_detail="$tag: exit $argv_rc; calls: ${argv_calls:-<none>} (want $want)"; break
  fi
  if [ -z "$what" ]; then
    ! grep -qi 'cap\|sdma\|nvidia' <<<"$argv_err" ||
      { native_detail="$tag: an uncapped batch was told something: $argv_err"; break; }
  else
    # shellcheck disable=SC2254  # the expected phrases are a glob by design
    case $(tr '\n' ' ' <<<"$argv_err") in
      *"capping dune at -j "*$what*) ;;
      *) native_detail="$tag: the announcement does not say why: ${argv_err:-<nothing>}"; break ;;
    esac
    { [ -n "$argv_dir" ] && grep -q -- "${want% @cheap}" "$argv_dir/cmd"; } ||
      { native_detail="$tag: the recorded command does not carry the cap: $(cat "$argv_dir/cmd" 2>/dev/null)"; break; }
  fi
done
for probe in "hip-config-sdma|hip|names its own dune width" "cc-on-sdma|cc|"; do
  [ -z "$native_detail" ] || break
  IFS='|' read -r tag bt what <<<"$probe"
  native_fleet=$fleet_none
  export FAKE_BACKEND_TEST=$bt
  native_probe "native-unset-explicit-$tag" "$dxg_absent" "$kfd_small" "$nv_absent" "" run build -j 16 @cheap
  unset FAKE_BACKEND_TEST
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "build -j 16 @cheap" ]; then
    native_detail="$tag with a width named: exit $argv_rc; calls: ${argv_calls:-<none>}"; break
  fi
  if [ -n "$what" ]; then
    grep -q "$what" <<<"$argv_err" || { native_detail="$tag with a width named: not told: ${argv_err:-<nothing>}"; break; }
  else
    ! grep -qi 'cap\|sdma\|nvidia' <<<"$argv_err" ||
      { native_detail="$tag with a width named: told something: $argv_err"; break; }
  fi
done
native_fleet=$fleet_none
if [ -z "$native_detail" ]; then
  report 0 "native: with OCANNL_BACKEND unset the width follows the batch's resolved backends, the tightest of them"
else
  report 1 "native: with OCANNL_BACKEND unset the width follows the batch's resolved backends, the tightest of them" "$native_detail"
fi
}
harness_leg_run '51: the native width follows the resolved backends #width' leg_51

# Leg 64: `plan` prints what `run` would do with an argv -- the resolved
# backends and why, the width, the slot -- and runs nothing: no dune call, no
# run recorded. Driven on the minix shape of leg 51 (a configuration naming
# hip on a small SDMA pool).
leg_64() {
native_detail=
export FAKE_BACKEND_TEST=hip
native_probe native-plan "$dxg_absent" "$kfd_small" "$nv_absent" "" plan build @cheap
unset FAKE_BACKEND_TEST
{ [ "$argv_rc" = 0 ] && [ -z "$argv_calls" ] && [ -z "$argv_dir" ]; } ||
  native_detail="exit $argv_rc; dune calls: ${argv_calls:-<none>}; run: ${argv_dir:-<none>}; stderr: $argv_err"
for line in "command: dune build -j 4 @cheap" "backends: hip, cc" "holds hip: test/config resolves backend=hip" \
            "width: -j 4, injected (sdma hazard, for hip)" "slot: none"; do
  [ -z "$native_detail" ] || break
  grep -qF -- "$line" <<<"$argv_out" || native_detail="no '$line' in: $argv_out"
done
# A width the caller named is reported as the caller's, with or without a cap
# on the box (Codex review round 1 on PR #832).
for probe in "$kfd_small|hip|width: the caller's (a -j 4 cap applies here, for hip)" \
             "$kfd_absent|cc|width: the caller's (no backend of the batch meets a cap on this box)"; do
  [ -z "$native_detail" ] || break
  IFS='|' read -r kfd bt want <<<"$probe"
  export FAKE_BACKEND_TEST=$bt
  native_probe native-plan-explicit "$dxg_absent" "$kfd" "$nv_absent" "" plan build -j 16 @cheap
  unset FAKE_BACKEND_TEST
  { [ "$argv_rc" = 0 ] && [ -z "$argv_calls" ] && grep -qF -- "$want" <<<"$argv_out" &&
    grep -qF -- "command: dune build -j 16 @cheap" <<<"$argv_out"; } ||
    native_detail="explicit width ($bt): exit $argv_rc; stdout: $argv_out"
done
# It holds the worktree lock like a launch, under a run directory it never
# publishes and removes: none is left behind, and a plan while a run holds
# the worktree is refused.
if [ -z "$native_detail" ]; then
  ls -d "$TMP/argv-runs-native-plan"/2*Z-* >/dev/null 2>&1 &&
    native_detail="plan left a run directory: $(ls -d "$TMP/argv-runs-native-plan"/2*Z-*)"
fi
if [ -z "$native_detail" ]; then
  plan_lock_runs=$TMP/argv-runs-plan-locked
  mkdir -p "$plan_lock_runs"
  plan_lock=$(OCANNL_TOOL_TEST_RUNS=$plan_lock_runs "$repeat_root/tools/test-run.sh" paths lock)
  perl -e 'use Fcntl ":flock"; open(my $fh, ">>", $ARGV[0]) or exit 1; flock($fh, LOCK_EX) or exit 1;
           open(my $r, ">", $ARGV[1]); close $r; sleep 20' "$plan_lock" "$TMP/plan-lock.ready" &
  plan_lock_pid=$!
  for _ in $(seq 1 50); do [ -e "$TMP/plan-lock.ready" ] && break; sleep 0.1; done
  argv_runs=$plan_lock_runs native_probe native-plan-locked "$dxg_absent" "$kfd_small" "$nv_absent" "" plan build @cheap
  argv_runs=
  kill "$plan_lock_pid" 2>/dev/null
  wait "$plan_lock_pid" 2>/dev/null
  { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -q "another test-run is active" <<<"$argv_err"; } ||
    native_detail="plan under a held lock: exit $argv_rc; stderr: $argv_err; stdout: $argv_out"
fi
# A resolution that spends the plan's cap is forecast as the cap's verdict,
# the way run records it (Codex review round 4 on PR #832).
if [ -z "$native_detail" ]; then
  export FAKE_REACH_SLEEP=30
  plan_spent_start=$SECONDS
  native_probe native-plan-spent "$dxg_absent" "$kfd_small" "$nv_absent" "" plan --cap 2 build @cheap
  unset FAKE_REACH_SLEEP
  { [ "$argv_rc" = 0 ] && [ -z "$argv_calls" ] && [ $((SECONDS - plan_spent_start)) -lt 15 ] &&
    grep -q "used the whole cap" <<<"$argv_out"; } ||
    native_detail="plan with its cap spent: exit $argv_rc; stdout: $argv_out"
fi
# A plan cancelled while a TERM-ignoring reader runs waits for that reader's
# grace and KILL before it exits, so the worktree is idle the moment it is
# gone (Codex review round 4 on PR #832).
if [ -z "$native_detail" ]; then
  plan_term_runs=$TMP/argv-runs-plan-term
  mkdir -p "$plan_term_runs"
  : >"$TMP/plan-term.reach"
  FAKE_REACH_CALLS=$TMP/plan-term.reach FAKE_REACH_SLEEP=30 FAKE_REACH_IGNORE_TERM=1 FAKE_BACKEND_TEST=hip \
  OCANNL_TOOL_TEST_RUNS=$plan_term_runs OCANNL_TOOL_DXG_DEVICE=$dxg_absent \
  OCANNL_TOOL_KFD_TOPOLOGY=$kfd_small OCANNL_TOOL_NVIDIA_DEVICE=$nv_absent PATH=$repeat_bin:$PATH \
    "$repeat_root/tools/test-run.sh" plan build @cheap >"$TMP/plan-term.out" 2>"$TMP/plan-term.err" &
  plan_term_pid=$!
  plan_term_marker=
  for _ in $(seq 1 100); do
    plan_term_marker=$(ls "$plan_term_runs"/2*/pid 2>/dev/null | head -n 1)
    [ -n "$plan_term_marker" ] && [ -s "$TMP/plan-term.reach" ] && break
    sleep 0.1
  done
  kill -TERM "$plan_term_pid" 2>/dev/null
  wait "$plan_term_pid"
  plan_term_rc=$?
  { [ -s "$TMP/plan-term.reach" ] && [ -n "$plan_term_marker" ] && [ "$plan_term_rc" = 143 ] && [ ! -e "$(dirname "$plan_term_marker")" ] &&
    OCANNL_TOOL_TEST_RUNS=$plan_term_runs "$repeat_root/tools/test-run.sh" idle 2>/dev/null; } ||
    native_detail="a cancelled plan: marker ${plan_term_marker:-<none>}; exit $plan_term_rc (want 143); or not idle once it exited: $(cat "$TMP/plan-term.err")"
fi
if [ -z "$native_detail" ]; then
  report 0 "plan: prints the resolved backends, the width and the slot, and runs nothing"
else
  report 1 "plan: prints the resolved backends, the width and the slot, and runs nothing" "$native_detail"
fi
}
harness_leg_run '64: plan prints what run would do #width' leg_64

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

# Leg 52: one place for the numbers. A hip cap is not a literal of its own
# but its measured budget over its box's slot count, so a change to either
# moves it; and the values are the measured ones (lukstafi/ludics-lite#316,
# #344, #391), pinned here so an edit to box-jobs.sh has to say what it measured.
leg_52() {
native_consts=$(
  . "$JOBS_SRC"
  printf '%s/%s/%s/%s/%s/%s/%s/%s/%s/%s/%s/%s' "$BOX_JOBS_SDMA_CAP" \
    "$BOX_JOBS_SDMA_BUDGET" "$BOX_JOBS_SDMA_SLOTS" "$BOX_JOBS_SDMA_SLOT_CAP" \
    "$BOX_JOBS_WIDE_SDMA_BUDGET" "$BOX_JOBS_WIDE_SDMA_SLOTS" "$BOX_JOBS_WIDE_SDMA_SLOT_CAP" \
    "$BOX_JOBS_NATIVE_NVIDIA_SLOTS" "$BOX_JOBS_NATIVE_CUDA_TOKENS" "$BOX_JOBS_NATIVE_CUDA_CAP" \
    "$BOX_JOBS_NATIVE_CPU_CAP" "$BOX_JOBS_SDMA_MEASURED_POOL"
)
native_detail=
[ "$native_consts" = "8/16/4/4/24/3/8/4/2/8/8/6" ] ||
  native_detail="box-jobs.sh's native numbers are $native_consts (want 8/16/4/4/24/3/8/4/2/8/8/6)"
[ -n "$native_detail" ] ||
  grep -q '^BOX_JOBS_SDMA_SLOT_CAP=\$((BOX_JOBS_SDMA_BUDGET / BOX_JOBS_SDMA_SLOTS))$' "$JOBS_SRC" ||
  native_detail="BOX_JOBS_SDMA_SLOT_CAP is not derived from the SDMA budget and the slot count"
[ -n "$native_detail" ] ||
  grep -q '^BOX_JOBS_WIDE_SDMA_SLOT_CAP=\$((BOX_JOBS_WIDE_SDMA_BUDGET / BOX_JOBS_WIDE_SDMA_SLOTS))$' "$JOBS_SRC" ||
  native_detail="BOX_JOBS_WIDE_SDMA_SLOT_CAP is not derived from the wide-pool budget and the slot count"
if [ -z "$native_detail" ]; then
  report 0 "native: the per-slot widths derive from box-jobs.sh's measured numbers"
else
  report 1 "native: the per-slot widths derive from box-jobs.sh's measured numbers" "$native_detail"
fi
}
harness_leg_run '52: the native widths come from one place #width' leg_52

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
for mode in red dune_prefixed_red nested_usage_red nested_usage_output; do
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
  # it included), dune is handed none: the caller's option is the only one,
  # verbatim, and nothing is recorded. Past dune's own `--` the word is a
  # program's argument: the runner's own still goes before the separator, once,
  # and the program's reaches it untouched.
  if [ -z "$slow_detail" ]; then
    for probe in "build --trace-file $TMP/t.csexp @cheap|2 before --trace-file" \
                 "build --trace-file=$TMP/t.csexp @cheap|2 before $TMP/t.csexp" \
                 "describe @cheap|" "--display=short build @cheap|" \
                 "exec ./prog.exe -- --trace-file=$TMP/p.csexp|OWN
5 after $TMP/p.csexp"; do
      want=${probe#*|} probe=${probe%%|*}
      : >"$slow_args"
      # shellcheck disable=SC2086
      REPEAT_TEST_TRACE_ARGS=$slow_args argv_probe slow-elsewhere run $probe
      want=${want/OWN/2 before $(slow_physical "$argv_dir/trace.csexp")}
      if [ "$(cat "$slow_args")" != "$want" ] || [ "$argv_calls" != "$probe" ] || [ -e "$argv_dir/slowest" ]; then
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
live_real=$( (command -v dune >/dev/null 2>&1 || . "$HERE/opam-env.sh" >/dev/null 2>&1
  command -v dune) 2>/dev/null) || live_real=
if [ -z "$live_real" ] || ! command -v python3 >/dev/null 2>&1; then
  skip "$live_label" "no dune on PATH, nor through tools/opam-env.sh, or no python3"
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

# The guard from leg 26 is not made redundant by the digest: it knows the
# correct order and refuses without spawning dune. With a fixture that WOULD
# refuse, the guard's refusal is the one that arrives, and the calls file --
# empty here, one line for the digest's refusal -- is what separates the two
# exit-2 outcomes.
leg_37() {
argv_mode=usage_option argv_probe guard-before-dune run build @cheap --cap 900
argv_mode=
if [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] \
   && case $argv_err in *"belongs before the dune arguments"*) true ;; *) false ;; esac \
   && case $argv_out in *"INVOCATION REFUSED"*) false ;; *) true ;; esac; then
  report 0 "the --cap guard still refuses before dune is spawned"
else
  report 1 "the --cap guard still refuses before dune is spawned" \
    "exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err; stdout: $argv_out"
fi
}
harness_leg_run '37: the --cap guard refuses before dune is spawned #options' leg_37

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
# is contained even if it never reached the ordinary pgid publication.
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
      /\$SIG\{\$_\} = "DEFAULT" for qw\(ALRM INT TERM HUP\)/ && control == "inherited-handler" { next }
      /^  \$pid = fork\(\);$/ {
        found++
        print "  $pid = do {"
        print "    my $forked = fork();"
        print "    if (defined $forked && !$forked) {"
        print "      $write->(\"$ENV{FORK_PREFIX}.child-token\", $self_token->() . \"\\n\");"
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
      END { if (found != 1) exit 1 }
    ' "$SRC" >"$life_root/tools/test-run.sh" || {
      cp "$SRC" "$life_root/tools/test-run.sh"
      harness_abandon "fork: the seam '  \$pid = fork();' is not in $SRC exactly once"
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
      probe_n=$(grep -c 'while sup_alive_settled "\$run_dir"; do wait "\$sup"' "$SRC")
      sed 's/while sup_alive_settled "\$run_dir"; do wait "\$sup"/while sup_alive "$run_dir"; do wait "$sup"/' \
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
# Legs 53-58: the fleet's run-time slot, taken by the runner (gh-ocannl-1004)
# ---------------------------------------------------------------------------
# A fake fleet-worker.sh stands in for lukstafi/ludics-lite's: it answers the
# probe (or, in `noprobe` mode, refuses it as a version without nested slots
# would), records the slot call, and then refuses, cannot reach its registry,
# or holds the slot and execs the command as the real one does. The fake
# readers every leg runs with (batch_reader, batch_reach) answer the batch's
# backends, so the kind decision is driven without a build.
slot_fake=$TMP/fake-fleet-worker.sh
cat >"$slot_fake" <<'FAKE'
#!/usr/bin/env bash
# The runner's own trace option (leg 80) is kept out of the slot calls these
# legs pin, by the fixture dune's rule: exactly `--trace-file=<the run's
# directory>/trace.csexp`, once, right after the subcommand that follows the
# dune it runs. Every other word stays verbatim.
shown=() prev2= prev1= dropped=
for arg in "$@"; do
  if [ -z "$dropped" ] && [ -n "${OCANNL_TOOL_TESTRUN_OWN:-}" ] &&
     [ "$arg" = "--trace-file=$OCANNL_TOOL_TESTRUN_OWN/trace.csexp" ] && [ "${prev2##*/}" = dune ]; then
    case $prev1 in build | runtest | test | exec) dropped=1; prev2=$prev1 prev1=$arg; continue ;; esac
  fi
  shown+=("$arg")
  prev2=$prev1 prev1=$arg
done
printf '%s\n' "${shown[*]}" >>"$FAKE_FW_CALLS"
if [ "$1 $2 $3" = "execution slot --probe" ]; then
  [ "${FAKE_FW_MODE:-}" = noprobe ] && { echo "fleet-worker.sh: execution slot [--wait <seconds>] ..." >&2; exit 2; }
  # Inside a live measurement hold the probe's line carries the hold's id.
  if [ "${FAKE_FW_MODE:-}" = measurement ]; then echo "EXECUTION SLOT PROBE fakebox 4 2 measurement M-7"
  else echo "EXECUTION SLOT PROBE fakebox 4 2"; fi
  exit 0
fi
shift 2
while [ "$1" != -- ]; do shift; done
shift
case ${FAKE_FW_MODE:-} in
  refuse) echo "EXECUTION SLOT REFUSED fakebox: all 2 GPU tokens (slots 1-2 of 4) busy after 0s; a batch that holds no GPU declares --cpu"; exit 1 ;;
  unreachable) echo "EXECUTION SLOT UNREACHABLE anchor: registry unread, no slot taken"; exit 4 ;;
  # The batch runs under the caller's measurement hold (lukstafi/ludics-lite#480),
  # and something inside it -- a nested runner, say -- logs a refusal of its own.
  measurement)
    echo "EXECUTION SLOT fakebox: inside measurement M-7, running under the hold: $*" >&2
    echo "EXECUTION SLOT REFUSED fakebox: a nested batch's own refusal"
    exec "$@" ;;
  # The real slot's sleep inhibitor: a process of its own group, beside the
  # command for the whole batch, and no child of dune (gh-ocannl-1211).
  holder)
    echo "EXECUTION SLOT fakebox: slot 1 of 4, GPU token 1 of 2 held for: $*" >&2
    perl -e 'setpgrp(0, 0); sleep 30' 9>&- &
    holder=$!
    "$@"
    rc=$?
    kill -KILL -- "-$holder" 2>/dev/null
    exit "$rc" ;;
esac
echo "EXECUTION SLOT fakebox: slot 1 of 4, GPU token 1 of 2 held for: $*" >&2
exec "$@"
FAKE
chmod +x "$slot_fake"
slot_probe() { # tag mode test-backend arrayjit-backend subcommand [argv...]
  local tag=$1 mode=$2 bt=$3 ba=$4
  shift 4
  : >"$TMP/$tag.fw"
  : >"$TMP/$tag.reach"
  FAKE_FW_CALLS=$TMP/$tag.fw FAKE_FW_MODE=$mode FAKE_BACKEND_TEST=$bt FAKE_BACKEND_ARRAYJIT=$ba \
  FAKE_REACH_CALLS=$TMP/$tag.reach \
  OCANNL_TOOL_FLEET_WORKER=$slot_fake argv_probe "$tag" "$@"
  slot_calls=$(grep -v -- '--probe' "$TMP/$tag.fw")
}
slot_calls=

# Leg 53: the kind. --cpu only when none of the batch's resolved backends
# holds a GPU: a GPU backend in a configuration, none (Context.auto tries GPUs
# first), an unreadable or unknown one, a reached stanza naming a GPU, an
# answer the reachability tool cut short, and a dune argv that can pick a
# backend the configurations do not show are each --gpu -- the slot's
# fail-closed default. The fake's recorded call is the argv the runner handed
# the real slot, so it pins the wait (the cap bounds it) and that dune's own
# argv passes through intact.
leg_53() {
slot_detail=
for probe in "cpu:cc:sync_cc:--cpu" "unset:unset:cc:--gpu" "multidev:multidev_cc:cc:--cpu" \
             "gpu-test:cuda:cc:--gpu" "gpu-arrayjit:cc:hip:--gpu" "metal:cc:metal:--gpu" \
             "unknown:cc:vulkan:--gpu" "unreadable:cc:fail:--gpu"; do
  IFS=: read -r tag bt ba want <<<"$probe"
  slot_probe "slot-$tag" hold "$bt" "$ba" run build @cheap
  [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ] ||
    { slot_detail="$tag: exit $argv_rc; dune calls: ${argv_calls:-<none>}"; break; }
  [ "$slot_calls" = "execution slot --wait 600 $want -- dune build @cheap" ] ||
    { slot_detail="$tag: slot call: ${slot_calls:-<none>} (want $want)"; break; }
  grep -q "fleet correctness slots.*as $want" <<<"$(tr '\n' ' ' <<<"$argv_err")" ||
    { slot_detail="$tag: the slot and its kind were not announced: $argv_err"; break; }
  { [ -n "$argv_dir" ] && grep -q -- "as $want" "$argv_dir/log" &&
      grep -q "^test-run: batch: " "$argv_dir/log" &&
      grep -q "^EXECUTION SLOT fakebox: slot 1 of 4" "$argv_dir/log"; } ||
    { slot_detail="$tag: the log does not record the backends, the decision and the slot: $(cat "$argv_dir/log" 2>/dev/null)"; break; }
done
for probe in "exec:exec ./prog.exe" "flag-gpu:exec ./prog.exe -- --ocannl_backend=cuda" \
             "flag-build:build @cheap --ocannl-backend=hip" "flag-bare:build @cheap --ocannl-backend"; do
  [ -z "$slot_detail" ] || break
  IFS=: read -r tag argv <<<"$probe"
  # shellcheck disable=SC2086  # the argv is word-split on purpose
  slot_probe "slot-argv-$tag" hold cc cc run $argv
  [ "$slot_calls" = "execution slot --wait 600 --gpu -- dune $argv" ] ||
    slot_detail="argv $tag: slot call: ${slot_calls:-<none>} (want --gpu)"
done
# A run that reaches a stanza naming a GPU backend is --gpu however the
# configurations resolve, and the reachability question is asked with the
# caller's own dune argv; one reaching only a stanza that names a CPU backend
# stays --cpu.
if [ -z "$slot_detail" ]; then
  FAKE_REACH="names cuda: it reaches test_cuda_pool_offset in test/operations, which names cuda\nend" \
    slot_probe slot-reach-gpu hold cc cc run build @cheap
  [ "$slot_calls" = "execution slot --wait 600 --gpu -- dune build @cheap" ] ||
    slot_detail="reach: slot call: ${slot_calls:-<none>} (want --gpu)"
  [ -n "$slot_detail" ] || [ "$(cat "$TMP/slot-reach-gpu.reach")" = "build @cheap" ] ||
    slot_detail="reach: asked about: $(cat "$TMP/slot-reach-gpu.reach") (want build @cheap)"
  [ -n "$slot_detail" ] || grep -q "batch: holds cuda: it reaches test_cuda_pool_offset" "$argv_dir/log" ||
    slot_detail="reach: the log does not say why: $(cat "$argv_dir/log")"
fi
if [ -z "$slot_detail" ]; then
  FAKE_REACH="names cc: it reaches cc_backend_trace_name in test/operations, which names cc\nend" \
    slot_probe slot-reach-cpu hold cc cc run build @cheap
  [ "$slot_calls" = "execution slot --wait 600 --cpu -- dune build @cheap" ] ||
    slot_detail="reach of a CPU-named stanza: slot call: ${slot_calls:-<none>} (want --cpu)"
fi
# Every answer the grammar does not accept is every backend: garbled, cut
# short of its `end`, running past it, naming what is not a backend, or the
# tool's own unknown.
for probe in "garbled:garbled" "cut-short:names cc: it reaches x in y, which names cc" \
             "past-end:end\nnames cuda: it reaches x in y, which names cuda" \
             "unknown-name:names vulkan: it reaches x in y, which names vulkan\nend" \
             "unmodelled:unknown: it carries a word this does not model"; do
  [ -z "$slot_detail" ] || break
  FAKE_REACH=${probe#*:} slot_probe "slot-reach-${probe%%:*}" hold cc cc run build @cheap
  [ "$slot_calls" = "execution slot --wait 600 --gpu -- dune build @cheap" ] ||
    slot_detail="an unreadable reachability answer (${probe%%:*}): slot call: ${slot_calls:-<none>} (want --gpu)"
done
# A reachability read that fails is unread whatever it printed -- even a
# complete, CPU-safe `end` (Codex review round 5 on PR #832).
if [ -z "$slot_detail" ]; then
  FAKE_REACH_EXIT=1 slot_probe slot-reach-failed hold cc cc run build @cheap
  [ "$slot_calls" = "execution slot --wait 600 --gpu -- dune build @cheap" ] ||
    slot_detail="a failed reachability read: slot call: ${slot_calls:-<none>} (want --gpu)"
fi
if [ -z "$slot_detail" ]; then
  slot_probe slot-argv-cpu-flag hold cc cc run build @cheap --ocannl_backend=cc
  [ "$slot_calls" = "execution slot --wait 600 --cpu -- dune build @cheap --ocannl_backend=cc" ] ||
    slot_detail="a command-line CPU backend: slot call: ${slot_calls:-<none>} (want --cpu)"
  # ...and the reachability tool is asked without the flag the argv scan
  # already read (Codex review round 6 on PR #832).
  [ -n "$slot_detail" ] || [ "$(cat "$TMP/slot-argv-cpu-flag.reach")" = "build @cheap" ] ||
    slot_detail="a command-line CPU backend: reachability asked about: $(cat "$TMP/slot-argv-cpu-flag.reach")"
fi
if [ -z "$slot_detail" ]; then
  # At most half the cap, so a refusal returns before the cap's alarm.
  # Half of what the resolution left of the cap: 45, or 44 when resolving
  # crossed a second (Codex review round 3 on PR #832).
  slot_probe slot-capped hold cc cc run --cap 90 build @cheap
  case $slot_calls in
    "execution slot --wait 45 --cpu -- dune build @cheap" | "execution slot --wait 44 --cpu -- dune build @cheap") ;;
    *) slot_detail="capped: the slot's wait is not half the cap: ${slot_calls:-<none>}" ;;
  esac
fi
if [ -z "$slot_detail" ]; then
  report 0 "slot: a fleet box's run takes its slot, --cpu only when none of the batch's resolved backends holds a GPU"
else
  report 1 "slot: a fleet box's run takes its slot, --cpu only when none of the batch's resolved backends holds a GPU" "$slot_detail"
fi
}
harness_leg_run '53: the slot kind #slot' leg_53

# Leg 54: a refused or unreachable slot is its own verdict, never a test one:
# exit 75, dune never invoked, and the fleet's own line quoted in the log.
leg_54() {
slot_detail=
for mode in refuse unreachable; do
  slot_probe "slot-$mode" "$mode" cc cc run build @cheap
  [ "$argv_rc" = 75 ] || { slot_detail="$mode: exit $argv_rc (want 75): $argv_out"; break; }
  [ -z "$argv_calls" ] || { slot_detail="$mode: dune ran: $argv_calls"; break; }
  case $argv_out in
    *"verdict: SLOT REFUSED"*) ;;
    *) slot_detail="$mode: verdict: $argv_out"; break ;;
  esac
done
if [ -z "$slot_detail" ]; then
  report 0 "slot: a refused or unreachable slot reports SLOT REFUSED (exit 75), and dune never runs"
else
  report 1 "slot: a refused or unreachable slot reports SLOT REFUSED (exit 75), and dune never runs" "$slot_detail"
fi
}
harness_leg_run '54: a refused or unreachable slot is SLOT REFUSED #slot' leg_54

# Leg 55a: `start` prints the caller's dune command, not the supervisor's.
leg_55a() {
slot_probe slot-start hold cc cc start build @cheap
case $argv_out in
  *"execution slot"* | *"fake-fleet-worker"*) report 1 "slot: start prints the caller's command" "$argv_out" ;;
  *"command: dune build @cheap"*) report 0 "slot: start prints the caller's command" ;;
  *) report 1 "slot: start prints the caller's command" "${argv_out:-<nothing>}" ;;
esac
OCANNL_TOOL_TEST_RUNS=$TMP/argv-runs-slot-start "$repeat_root/tools/test-run.sh" wait last >/dev/null 2>&1
}
harness_leg_run '55a: start prints the caller'\''s command #slot' leg_55a

# Leg 55: a red suite under a slot is still FAIL -- the slot's admission line
# is what tells the two apart, not the status.
leg_55() {
argv_mode=red slot_probe slot-red hold cc cc run build @cheap
case "$argv_rc:$argv_out" in
  1:*"verdict: FAIL"*) report 0 "slot: a red suite under a held slot is still FAIL" ;;
  *) report 1 "slot: a red suite under a held slot is still FAIL" "exit $argv_rc: $argv_out" ;;
esac
}
harness_leg_run '55: a red suite under a held slot is still FAIL #slot' leg_55

# Leg 55b: inside a live measurement hold the slot admits the batch under the
# hold (`inside measurement <id>`, lukstafi/ludics-lite#480) -- an admission
# like a held slot's, so a refusal line the batch itself logs leaves a red
# suite FAIL, never SLOT REFUSED. The probe's trailing `measurement <id>` still
# reads as a fleet box.
leg_55b() {
argv_mode=red slot_probe slot-measurement measurement cc cc run build @cheap
case "$argv_rc:$argv_out" in
  1:*"verdict: FAIL"*)
    if [ "$slot_calls" = "execution slot --wait 600 --cpu -- dune build @cheap" ] &&
       grep -q "^EXECUTION SLOT fakebox: inside measurement M-7" "$argv_dir/log"; then
      report 0 "slot: a batch admitted inside a measurement hold is admitted, not SLOT REFUSED"
    else
      report 1 "slot: a batch admitted inside a measurement hold is admitted, not SLOT REFUSED" \
        "slot call: ${slot_calls:-<none>}; log: $(cat "$argv_dir/log" 2>/dev/null)"
    fi ;;
  *) report 1 "slot: a batch admitted inside a measurement hold is admitted, not SLOT REFUSED" \
       "exit $argv_rc: $argv_out" ;;
esac
}
harness_leg_run '55b: a batch inside a measurement hold is admitted #slot' leg_55b

# Leg 56: no slot where the fleet cannot give one safely -- a fleet-worker.sh
# without the probe (a version from before nested slots, around which a
# worker's own wrapper would cost two slots), the slot turned off, and repeat
# -- each runs dune directly and says nothing about a slot.
leg_56() {
slot_detail=
slot_probe slot-noprobe noprobe cc cc run build @cheap
{ [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ] && [ -z "$slot_calls" ] &&
    ! grep -qi 'slot' <<<"$argv_err"; } ||
  slot_detail="noprobe: exit $argv_rc; slot calls: ${slot_calls:-<none>}; stderr: $argv_err"
if [ -z "$slot_detail" ]; then
  : >"$TMP/slot-off.fw"
  FAKE_FW_CALLS=$TMP/slot-off.fw OCANNL_TOOL_FLEET_WORKER=none argv_probe slot-off run build @cheap
  { [ "$argv_rc" = 0 ] && [ ! -s "$TMP/slot-off.fw" ]; } ||
    slot_detail="none: exit $argv_rc; fleet-worker called: $(cat "$TMP/slot-off.fw")"
fi
# Two skill trees, the first without the probe: the second one answers, and
# its slot is taken -- a stale first candidate must not turn the slot off.
if [ -z "$slot_detail" ]; then
  slot_home=$TMP/slot-home
  mkdir -p "$slot_home/.claude/skills/issue-wave/scripts" "$slot_home/.codex/skills/issue-wave/scripts"
  printf '#!/usr/bin/env bash\necho "usage" >&2; exit 2\n' >"$slot_home/.claude/skills/issue-wave/scripts/fleet-worker.sh"
  cp "$slot_fake" "$slot_home/.codex/skills/issue-wave/scripts/fleet-worker.sh"
  chmod +x "$slot_home/.claude/skills/issue-wave/scripts/fleet-worker.sh" "$slot_home/.codex/skills/issue-wave/scripts/fleet-worker.sh"
  : >"$TMP/slot-second.fw"
  HOME=$slot_home FAKE_FW_CALLS=$TMP/slot-second.fw FAKE_FW_MODE=hold FAKE_BACKEND_TEST=cc FAKE_BACKEND_ARRAYJIT=cc \
  FAKE_REACH_CALLS=$TMP/slot-second.reach \
  OCANNL_TOOL_FLEET_WORKER= argv_probe slot-second run build @cheap
  grep -q -- "execution slot --wait 600 --cpu -- dune build @cheap" "$TMP/slot-second.fw" ||
    slot_detail="second candidate: its slot was not taken: $(cat "$TMP/slot-second.fw")"
fi
if [ -z "$slot_detail" ]; then
  slot_probe slot-repeat hold cc cc repeat 2 build @cheap
  { [ "$argv_rc" = 0 ] && [ -z "$slot_calls" ]; } ||
    slot_detail="repeat: exit $argv_rc; slot calls: ${slot_calls:-<none>}"
fi
if [ -z "$slot_detail" ]; then
  report 0 "slot: no probe, the slot turned off, and repeat all run dune directly"
else
  report 1 "slot: no probe, the slot turned off, and repeat all run dune directly" "$slot_detail"
fi
}
harness_leg_run '56: no slot where the fleet cannot give one safely #slot' leg_56

# Leg 57: the configurations the backends are read from are the ones that set
# a backend. Every tracked `ocannl_config` naming one must sit in one of
# batch-backends.sh's BATCH_CONFIG_DIRS (the test directories copy theirs from
# test/config), so a new directory with a backend of its own fails here until
# the runner reads it too.
# BATCH_CONFIG_DIRS as batch-backends.sh declares it, which legs 57 and 58 read.
slot_dirs=$(sed -n 's/^BATCH_CONFIG_DIRS="\(.*\)"$/\1/p' "$HERE/batch-backends.sh")
leg_57() {
slot_detail=
[ -n "$slot_dirs" ] || slot_detail="BATCH_CONFIG_DIRS not found in batch-backends.sh"
if [ -z "$slot_detail" ] && git -C "$HERE/.." rev-parse --git-dir >/dev/null 2>&1; then
  while IFS= read -r cfg; do
    grep -Eq '^[[:space:]]*backend[[:space:]]*=' "$HERE/../$cfg" || continue
    d=$(dirname "$cfg")
    case " $slot_dirs " in *" $d "*) continue ;; esac
    slot_detail="$cfg names a backend but $d is not in BATCH_CONFIG_DIRS ($slot_dirs)"
    break
  done < <(git -C "$HERE/.." ls-files -- '*ocannl_config' 'ocannl_config')
else
  [ -n "$slot_detail" ] || { skip "batch: the backend-bearing configurations are the ones the backends are read from" "not a git checkout"; slot_detail=skipped; }
fi
if [ -z "$slot_detail" ]; then
  report 0 "batch: the backend-bearing configurations are the ones the backends are read from"
elif [ "$slot_detail" != skipped ]; then
  report 1 "batch: the backend-bearing configurations are the ones the backends are read from" "$slot_detail"
fi
}
harness_leg_run '57: backend-bearing configurations are the ones read #slot' leg_57

# Leg 58: the two readers the backends come from are the ones test/config
# builds. The legs above drive fakes, so a renamed executable, a dropped
# `--read=backend`, or a reachability tool no longer asking Slot_kind would
# pass them while every real batch fell back to every backend -- a narrower
# width and a GPU token, silently. Pinned against the stanzas and the readers'
# own sources instead.
leg_58() {
slot_detail=
slot_src=$HERE/batch-backends.sh
slot_built=$(sed -n 's|.*"\$dune" build \(\./test/config/[A-Za-z0-9_.]* \./test/config/[A-Za-z0-9_.]*\) .*|\1|p' "$slot_src")
slot_reader_exe=$(sed -n 's|^ *reader=\$build_dir/default/\(test/config/[A-Za-z0-9_]*\)\.exe$|\1|p' "$slot_src")
slot_reach_exe=$(sed -n 's|^ *reach=\$build_dir/default/\(test/config/[A-Za-z0-9_]*\)\.exe$|\1|p' "$slot_src")
# ...and from the build tree dune was pointed at, not a fixed _build.
grep -q '^ *local build_dir=\${DUNE_BUILD_DIR:-_build}$' "$slot_src" ||
  slot_detail="batch-backends.sh does not run the readers from DUNE_BUILD_DIR's tree"
if [ -n "$slot_detail" ]; then :
elif [ -z "$slot_reader_exe" ] || [ -z "$slot_reach_exe" ]; then
  slot_detail="batch-backends.sh names no reader (${slot_reader_exe:-?}) or reachability tool (${slot_reach_exe:-?})"
else
  for exe in "$slot_reader_exe" "$slot_reach_exe"; do
    case " $slot_built " in
      *" ./$exe.exe "*) ;;
      *) slot_detail="batch-backends.sh runs $exe but builds only: ${slot_built:-nothing}"; break ;;
    esac
    grep -q "^ (name $(basename "$exe"))$" "$HERE/../test/config/dune" ||
      { slot_detail="test/config/dune has no executable named $(basename "$exe")"; break; }
  done
fi
[ -n "$slot_detail" ] || grep -q '| Some "backend" ->' "$HERE/../$slot_reader_exe.ml" ||
  slot_detail="$slot_reader_exe.ml does not answer --read=backend"
[ -n "$slot_detail" ] || grep -q 'Test_utils.Slot_kind.answer' "$HERE/../$slot_reach_exe.ml" ||
  slot_detail="$slot_reach_exe.ml does not ask Test_utils.Slot_kind"
# The reachability tool's grammar is the one batch_resolve parses: a `names
# <backend>: <why>` line per backend, a `reads config: <why>` line when a
# configuration counts, then `end`; or one `unknown: <why>`.
for form in 'printf "names %s: %s\n"' 'printf "reads config: %s\n"' 'printf "end\n"' \
            'printf "unknown: %s\n"'; do
  [ -z "$slot_detail" ] || break
  grep -qF "$form" "$HERE/../$slot_reach_exe.ml" ||
    slot_detail="$slot_reach_exe.ml has no $form"
done
# ...and Slot_kind's own list of the directories whose configuration the runner
# reads -- slot_kind_live refuses a reader reading any other's
# (gh-ocannl-1249) -- is BATCH_CONFIG_DIRS.
if [ -z "$slot_detail" ]; then
  slot_kind_dirs=$(sed -n 's/^let config_dirs = \[ \(.*\) \]$/\1/p' "$HERE/../test/support/slot_kind.ml" | tr -d '"' | sed 's/ *; */ /g')
  [ "$slot_kind_dirs" = "$slot_dirs" ] ||
    slot_detail="Slot_kind.config_dirs ($slot_kind_dirs) is not BATCH_CONFIG_DIRS ($slot_dirs)"
fi
if [ -z "$slot_detail" ]; then
  report 0 "batch: the readers batch-backends.sh builds are test/config's, answering --read=backend and Slot_kind in its grammar"
else
  report 1 "batch: the readers batch-backends.sh builds are test/config's, answering --read=backend and Slot_kind in its grammar" "$slot_detail"
fi
}
harness_leg_run '58: the backend readers are test/config'\''s #slot' leg_58

# Leg 78: a test configuration's backend counts only where ocannl_slot_kind
# says it does -- every batch but one of its configuration-free aliases or a
# plain build (gh-ocannl-1227). A test configuration naming hip -- AGENTS.md's
# one-off route -- under a fleet slot beside a small SDMA pool, minix's shape:
# a batch answered with no `reads config` line and nothing named (as
# @test/operations/scans is) holds no backend, so it runs at dune's own width
# and --cpu, its log saying why; one naming cc holds cc alone, the
# configuration unread. The control: the same batch answered with a `reads
# config` line is hip's -- -j 4 and --gpu.
leg_78() {
reads_label="batch: a configuration's backend counts only where the reachability tool says it does"
reads_detail=
native_fleet=$fleet_none
for probe in "none-only|end|build @cheap|--cpu|holds no backend: it reaches no stanza" \
             "named-cc|names cc: it reaches t_cc in x, which names cc\nend|build @cheap|--cpu|reads no configuration" \
             "reads|$FAKE_READS\nend|build -j 4 @cheap|--gpu|holds hip: test/config resolves backend=hip"; do
  IFS='|' read -r tag reach want kind logline <<<"$probe"
  : >"$TMP/reads-$tag.fw"
  export FAKE_BACKEND_TEST=hip FAKE_REACH=$reach FAKE_FW_CALLS=$TMP/reads-$tag.fw FAKE_FW_MODE=hold \
    OCANNL_TOOL_FLEET_WORKER=$slot_fake
  native_probe "reads-$tag" "$dxg_absent" "$kfd_small" "$nv_absent" "" run build @cheap
  # Back to the harness-wide `none`, never unset: unset, the next legs would
  # find this box's own fleet-worker.sh.
  unset FAKE_BACKEND_TEST FAKE_REACH FAKE_FW_CALLS FAKE_FW_MODE
  export OCANNL_TOOL_FLEET_WORKER=none
  slot_calls=$(grep -v -- '--probe' "$TMP/reads-$tag.fw")
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "$want" ]; then
    reads_detail="$tag: exit $argv_rc; dune calls: ${argv_calls:-<none>} (want $want)"; break
  fi
  [ "$slot_calls" = "execution slot --wait 600 $kind -- dune $want" ] ||
    { reads_detail="$tag: slot call: ${slot_calls:-<none>} (want $kind)"; break; }
  { [ -n "$argv_dir" ] && grep -qF -- "test-run: batch: $logline" "$argv_dir/log"; } ||
    { reads_detail="$tag: the log does not say '$logline': $(cat "$argv_dir/log" 2>/dev/null)"; break; }
  if [ "$kind" = --cpu ]; then
    ! grep -qi 'capping\|sdma' <<<"$argv_err" ||
      { reads_detail="$tag: an uncapped batch was told about a cap: $argv_err"; break; }
  fi
done
if [ -z "$reads_detail" ]; then
  report 0 "$reads_label"
else
  report 1 "$reads_label" "$reads_detail"
fi
}
harness_leg_run '78: a configuration'\''s backend counts only where the reachability tool says it does #slot' leg_78

# Leg 88 (gh-ocannl-1433): what the batch's backends say about the GPU it was
# meant to test. On a GPU box (cuda on a native NVIDIA boot, hip beside a KFD
# pool), a batch resolving to cc alone runs, green, and WARNs in the
# announcements, the log and the digest; `--expect-backend cuda` refuses it
# before dune (BACKEND REFUSED, exit 2, no dune call, the digest naming what
# it held). The controls: the same batch on cuda, or expecting cc, says
# nothing; a box with no GPU hazard (every Mac) says nothing for cc, yet the
# option still resolves and refuses there; an unread batch (`exec`) proves no
# backend and is refused; a batch holding no backend is not warned about;
# `plan` reports both; a non-backend or a misplaced option is a usage error.
leg_88() {
local d= label="backend check: a GPU box's cc-only batch is warned about, and --expect-backend refuses it"
native_fleet=$fleet_none
for probe in "nv-cc|$nv_present|$kfd_absent|cuda|" "kfd-cc|$nv_absent|$kfd_small|hip|"; do
  IFS='|' read -r tag nv kfd gpu _ <<<"$probe"
  native_probe "be-$tag" "$dxg_absent" "$kfd" "$nv" "" run build @cheap
  [ "$argv_rc" = 0 ] && [ -n "$argv_calls" ] ||
    { d="$tag: the warned batch did not run: exit $argv_rc; calls ${argv_calls:-<none>}"; break; }
  want="WARNING: this box has a GPU backend ($gpu), but this batch holds only cc: it tests nothing on the GPU."
  grep -qF -- "$want" <<<"$argv_err" ||
    { d="$tag: stderr lacks the warning: $argv_err"; break; }
  grep -qF -- "$want" <<<"$argv_out" ||
    { d="$tag: the digest lacks the warning: $argv_out"; break; }
  { [ -n "$argv_dir" ] && grep -qF -- "test-run: $want" "$argv_dir/log"; } ||
    { d="$tag: the log lacks the warning"; break; }
  native_probe "be-$tag-expect" "$dxg_absent" "$kfd" "$nv" "" run --expect-backend "$gpu" build @cheap
  if [ "$argv_rc" != 2 ] || [ -n "$argv_calls" ]; then
    d="$tag --expect-backend $gpu: exit $argv_rc (want 2); dune calls: ${argv_calls:-<none>} (want none)"; break
  fi
  case $argv_out in
    *"verdict: BACKEND REFUSED"*"backend refused: --expect-backend $gpu: this batch holds cc, not $gpu."*) ;;
    *) d="$tag --expect-backend $gpu: the digest: $argv_out"; break ;;
  esac
  # Controls: the GPU backend selected, with and without the option; cc expected outright.
  for ctl in "selected|$gpu|" "selected-expected|$gpu|--expect-backend $gpu" "cc-expected||--expect-backend cc"; do
    IFS='|' read -r ctag backend opt <<<"$ctl"
    # shellcheck disable=SC2086  # the option is word-split on purpose
    native_probe "be-$tag-$ctag" "$dxg_absent" "$kfd" "$nv" "$backend" run $opt build @cheap
    if [ "$argv_rc" != 0 ] || [ -z "$argv_calls" ] || grep -q 'WARNING:\|BACKEND REFUSED\|backend refused' <<<"$argv_err$argv_out"; then
      d="$tag control $ctag: exit $argv_rc; calls ${argv_calls:-<none>}; stderr $argv_err; digest $argv_out"; break 2
    fi
  done
done
if [ -z "$d" ]; then
  # A WSL guest's bridge is a hazard for cuda and hip alike: the warning names both and
  # recommends neither, since minix-amd-wsl's GPU is not cuda (Codex review round 1).
  native_probe be-wsl "$dxg_present" "$kfd_absent" "$nv_absent" "" run build @cheap
  grep -qF "WARNING: this box has a GPU backend (cuda or hip), but this batch holds only cc" <<<"$argv_err" &&
    grep -qF "(OCANNL_BACKEND=<this box's GPU backend>)" <<<"$argv_err" ||
    d="a WSL bridge: stderr $argv_err"
fi
if [ -z "$d" ]; then
  # No GPU hazard: no warning for cc, but the option still resolves, and refuses.
  native_probe be-mac "$dxg_absent" "$kfd_absent" "$nv_absent" "" run build @cheap
  [ "$argv_rc" = 0 ] && ! grep -q 'WARNING' <<<"$argv_err$argv_out" ||
    d="no hazard: exit $argv_rc; stderr $argv_err; digest $argv_out"
fi
if [ -z "$d" ]; then
  native_probe be-mac-expect "$dxg_absent" "$kfd_absent" "$nv_absent" "" run --expect-backend metal build @cheap
  [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -q 'verdict: BACKEND REFUSED' <<<"$argv_out" ||
    d="no hazard, --expect-backend metal: exit $argv_rc; calls ${argv_calls:-<none>}; digest $argv_out"
fi
if [ -z "$d" ]; then
  native_probe be-mac-held "$dxg_absent" "$kfd_absent" "$nv_absent" metal run --expect-backend metal build @cheap
  [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ] ||
    d="no hazard, metal expected and held: exit $argv_rc; calls ${argv_calls:-<none>}; digest $argv_out"
fi
if [ -z "$d" ]; then
  native_probe be-unread "$dxg_absent" "$kfd_absent" "$nv_present" cuda run --expect-backend cuda exec ./prog.exe
  [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -q 'backends are unread' <<<"$argv_out" ||
    d="an unread batch: exit $argv_rc; calls ${argv_calls:-<none>}; digest $argv_out"
fi
if [ -z "$d" ]; then
  FAKE_REACH='end' native_probe be-none "$dxg_absent" "$kfd_absent" "$nv_present" "" run build @cheap
  [ "$argv_rc" = 0 ] && ! grep -q 'WARNING' <<<"$argv_err$argv_out" ||
    d="a batch holding no backend: exit $argv_rc; stderr $argv_err"
fi
if [ -z "$d" ]; then
  native_probe be-plan "$dxg_absent" "$kfd_absent" "$nv_present" "" plan --expect-backend cuda build @cheap
  case $argv_out in
    *"expect-backend: run would refuse this batch (BACKEND REFUSED, exit 2)"*"this batch holds cc, not cuda"*) ;;
    *) d="plan --expect-backend cuda: exit $argv_rc; output $argv_out" ;;
  esac
  [ -n "$d" ] || [ -z "$argv_calls" ] || d="plan ran dune: $argv_calls"
fi
if [ -z "$d" ]; then
  native_probe be-plan-warn "$dxg_absent" "$kfd_absent" "$nv_present" "" plan build @cheap
  grep -qF 'WARNING: this box has a GPU backend (cuda), but this batch holds only cc' <<<"$argv_out" ||
    d="plan on a GPU box's cc batch: output $argv_out"
fi
if [ -z "$d" ]; then
  argv_probe be-bad-name run --expect-backend vulkan build @cheap
  [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -q 'not a backend' <<<"$argv_err" ||
    d="--expect-backend vulkan: exit $argv_rc; stderr $argv_err"
fi
if [ -z "$d" ]; then
  argv_probe be-misplaced run build @cheap --expect-backend cuda
  [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -q 'belongs before the dune arguments' <<<"$argv_err" ||
    d="a misplaced --expect-backend: exit $argv_rc; stderr $argv_err"
fi
if [ -z "$d" ]; then report 0 "$label"; else report 1 "$label" "$d"; fi
}
harness_leg_run '88: a GPU box'\''s cc-only batch is warned about, and --expect-backend refuses it #slot' leg_88

# ---------------------------------------------------------------------------
# Legs 75-77: a wedged device cannot hold a batch (gh-ocannl-1211)
# ---------------------------------------------------------------------------
# A GPU fault on tuf left every later test of a HIP batch spinning in device
# init, eight at once, for 32 minutes, under a run cap with half an hour to
# go. Two bounds answer it: the per-test cap ends the run once a test outlives
# it (75), and a GPU batch's device probe refuses to start dune when a fresh
# device query does not answer (76); `plan` reports both (77).
#
# Leg 75: the per-test cap. A test past it ends the run as the run's cap
# would -- TIMEOUT, 142, the test named -- long before the test would end, the
# test killed and the worktree idle. Controls: a test inside the cap passes;
# OCANNL_TOOL_PER_TEST_CAP is the default, and --test-cap 0 lifts it; and
# neither a slow dune nor the slot's own long-lived holder, which is no child
# of dune, is taken for a test (a watcher counting every process group of the
# run would end both runs). The message blames a device only on a GPU batch,
# and always names the remedy. Tests are timed on the watcher's own clock:
# a stub ps serving every etime as 1.2 million years, as minix's procps did
# after a resume (gh-ocannl-1319), neither ends a run nor spares a hung test,
# and the stub's log shows it served the watcher. A bad value is refused
# before dune runs. The fixture's dune is perl renamed through $0, which Linux
# (prctl) and macOS (the argv area BSD ps reads) both show as the process
# name; the leg skips only where a renamed process does not read back as
# `dune` (Git Bash, say).
leg_75() {
test_cap_label="test cap: a test past --test-cap ends the run as TIMEOUT naming it; a test inside it, a skewed ps etime, a slow dune and the slot's holder do not"
perl -e '$0 = "dune"; sleep 10' </dev/null >/dev/null 2>&1 &
test_cap_named=$!
sleep 0.3
test_cap_named_ok=
ps -A -o pid= -o comm= 2>/dev/null | grep -Eq "^ *$test_cap_named +(.*/)?dune$" && test_cap_named_ok=1
kill "$test_cap_named" 2>/dev/null
wait "$test_cap_named" 2>/dev/null
if [ -z "${test_cap_named_ok:-}" ]; then
  skip "$test_cap_label" "a process renamed through \$0 does not read back as dune in ps here"
else
  test_cap_detail=
  test_cap_n=0
  test_cap_start=$SECONDS
  test_cap_n=$((test_cap_n + 1))
  export REPEAT_TEST_ACTION_PID=$TMP/test-cap-action-$test_cap_n.pid
  argv_mode=action argv_probe test-cap-hung run --test-cap 2 build @cheap
  test_cap_took=$((SECONDS - test_cap_start))
  test_cap_action=$(cat "$REPEAT_TEST_ACTION_PID" 2>/dev/null)
  case $argv_out in
    *"verdict: TIMEOUT (a test outlived the per-test cap; run was killed, not judged) (exit 142)"*"hung test: sleep 60"*"past the 2s per-test cap"*) ;;
    *) test_cap_detail="hung: digest: $argv_out" ;;
  esac
  [ -n "$test_cap_detail" ] || [ "$argv_rc" = 142 ] || test_cap_detail="hung: exit $argv_rc (want 142)"
  [ -n "$test_cap_detail" ] || [ "$test_cap_took" -lt 20 ] ||
    test_cap_detail="hung: the run took ${test_cap_took}s under a 2s test cap"
  [ -n "$test_cap_detail" ] || [ -n "$test_cap_action" ] || test_cap_detail="hung: the test never started"
  if [ -z "$test_cap_detail" ]; then
    case $(pstate "$test_cap_action") in
      '' | Z) ;;
      *) test_cap_detail="hung: the test (pid $test_cap_action) survived the run"
         kill -KILL "$test_cap_action" 2>/dev/null ;;
    esac
  fi
  [ -n "$test_cap_detail" ] || grep -q "^test-run: per-test cap: pid $test_cap_action has run" "$argv_dir/log" ||
    test_cap_detail="hung: the log does not name the test: $(cat "$argv_dir/log")"
  # A CPU batch is told the remedy and never sent looking for a GPU fault.
  [ -n "$test_cap_detail" ] || grep -qF "rerun with --test-cap 0" <<<"$argv_out" ||
    test_cap_detail="hung: the digest names no remedy: $argv_out"
  if [ -z "$test_cap_detail" ] &&
     { cat "$argv_dir/log"; printf '%s\n' "$argv_out"; } | grep -qi "wedged device\|kernel journal"; then
    test_cap_detail="hung: a CPU batch's run blames a device: $argv_out $(cat "$argv_dir/log")"
  fi
  [ -n "$test_cap_detail" ] ||
    OCANNL_TOOL_TEST_RUNS=$TMP/argv-runs-test-cap-hung "$repeat_root/tools/test-run.sh" idle >/dev/null 2>&1 ||
    test_cap_detail="hung: the worktree is not idle after the run"
  for probe in "inside:3:1:0" "env-default:1:4:142" "lifted:1:3:0"; do
    [ -z "$test_cap_detail" ] || break
    IFS=: read -r tag cap_s secs want <<<"$probe"
    test_cap_n=$((test_cap_n + 1))
    REPEAT_TEST_ACTION_PID=$TMP/test-cap-action-$test_cap_n.pid
    case $tag in
      inside) REPEAT_TEST_ACTION_SECS=$secs argv_mode=action \
                argv_probe "test-cap-$tag" run --test-cap "$cap_s" build @cheap ;;
      lifted) OCANNL_TOOL_PER_TEST_CAP=$cap_s REPEAT_TEST_ACTION_SECS=$secs argv_mode=action \
                argv_probe "test-cap-$tag" run --test-cap 0 build @cheap ;;
      *) OCANNL_TOOL_PER_TEST_CAP=$cap_s REPEAT_TEST_ACTION_SECS=$secs argv_mode=action \
           argv_probe "test-cap-$tag" run build @cheap ;;
    esac
    [ "$argv_rc" = "$want" ] || { test_cap_detail="$tag: exit $argv_rc (want $want): $argv_out"; break; }
    if [ "$want" = 0 ] && [ -e "$argv_dir/test-cap" ]; then
      test_cap_detail="$tag: a passing run left a test-cap record"
    fi
  done
  # A skewed ps clock (gh-ocannl-1319): after a resume from suspend, minix's
  # procps reported a test a minute old as 38109073018720s old, and the cap
  # ended the run. A ps on PATH that serves every etime as that age: a test
  # inside the cap still passes, a hung one is still capped, and the stub's
  # log proves the watcher was served the skewed etime.
  if [ -z "$test_cap_detail" ]; then
    skew_bin=$TMP/test-cap-skew-bin
    mkdir -p "$skew_bin"
    : >"$TMP/test-cap-skew.served"
    cat >"$skew_bin/ps" <<EOF
#!/usr/bin/env perl
# The real ps, with every etime column replaced by 441077234-07:58:40.
my @cols;
for (my \$i = 0; \$i < \$#ARGV; \$i++) { push @cols, \$ARGV[\$i + 1] if \$ARGV[\$i] eq "-o" }
my (\$at) = grep { \$cols[\$_] =~ /^etime=?\$/ } 0 .. \$#cols;
open my \$ps, "-|", "$(command -v ps)", @ARGV or exit 1;
while (my \$line = <\$ps>) {
  if (defined \$at) {
    chomp \$line;
    my @f = split " ", \$line, scalar @cols;
    if (@f == @cols) {
      \$f[\$at] = "441077234-07:58:40";
      \$line = join(" ", @f);
      if (open my \$log, ">>", "$TMP/test-cap-skew.served") { print \$log "\$line\n"; close \$log }
    }
    \$line .= "\n";
  }
  print \$line;
}
close \$ps;
exit(\$? >> 8);
EOF
    chmod +x "$skew_bin/ps"
    for probe in "skew-inside:4:2:0" "skew-hung:2:60:142"; do
      [ -z "$test_cap_detail" ] || break
      IFS=: read -r tag cap_s secs want <<<"$probe"
      test_cap_n=$((test_cap_n + 1))
      REPEAT_TEST_ACTION_PID=$TMP/test-cap-action-$test_cap_n.pid
      : >"$TMP/test-cap-skew.served"
      PATH=$skew_bin:$PATH REPEAT_TEST_ACTION_SECS=$secs argv_mode=action \
        argv_probe "test-cap-$tag" run --test-cap "$cap_s" build @cheap
      test_cap_action=$(cat "$REPEAT_TEST_ACTION_PID" 2>/dev/null)
      [ "$argv_rc" = "$want" ] || { test_cap_detail="$tag: exit $argv_rc (want $want): $argv_out $(cat "$argv_dir/log")"; break; }
      grep -q "^ *$test_cap_action .*441077234-07:58:40" "$TMP/test-cap-skew.served" ||
        { test_cap_detail="$tag: the watcher was never served the test's skewed etime: $(head -c 2000 "$TMP/test-cap-skew.served")"; break; }
      if [ "$want" = 0 ]; then
        [ ! -e "$argv_dir/test-cap" ] || test_cap_detail="$tag: a passing run left a test-cap record"
      else
        grep -Eq "^test-run: per-test cap: pid $test_cap_action has run [0-9]{1,2}s, past the 2s cap" "$argv_dir/log" ||
          test_cap_detail="$tag: the log does not name the test at its own age: $(cat "$argv_dir/log")"
        case $(pstate "$test_cap_action") in
          '' | Z) ;;
          *) test_cap_detail="$tag: the test (pid $test_cap_action) survived the run"
             kill -KILL "$test_cap_action" 2>/dev/null ;;
        esac
      fi
    done
  fi
  # A test that execs keeps its pid and changes its name: a wrapper handing
  # over to the real program, here a chain of ten 1.2s hops, each renamed. Its
  # age runs on across the renames, so a 4s cap ends the run long before the
  # chain would; a watcher restarting the age at each rename never would.
  if [ -z "$test_cap_detail" ]; then
    cat >"$TMP/test-cap-hops.pl" <<'EOF'
my $n = shift;
$0 = "hop$n";
select undef, undef, undef, 1.2;
exit 0 if $n >= 10;
exec $^X, __FILE__, $n + 1;
EOF
    test_cap_n=$((test_cap_n + 1))
    REPEAT_TEST_ACTION_PID=$TMP/test-cap-action-$test_cap_n.pid
    REPEAT_TEST_ACTION_HOPS=$TMP/test-cap-hops.pl argv_mode=action \
      argv_probe test-cap-renamed run --test-cap 4 build @cheap
    test_cap_action=$(cat "$REPEAT_TEST_ACTION_PID" 2>/dev/null)
    { [ "$argv_rc" = 142 ] && [ -n "$test_cap_action" ] &&
      grep -q "^test-run: per-test cap: pid $test_cap_action has run" "$argv_dir/log"; } ||
      test_cap_detail="renamed: exit $argv_rc (want 142, the test pid ${test_cap_action:-unknown} named): $argv_out $(cat "$argv_dir/log")"
    case $(pstate "$test_cap_action") in
      '' | Z) ;;
      *) [ -n "$test_cap_detail" ] || test_cap_detail="renamed: the test (pid $test_cap_action) survived the run"
         kill -KILL "$test_cap_action" 2>/dev/null ;;
    esac
  fi
  if [ -z "$test_cap_detail" ]; then
    : >"$TMP/test-cap-holder.fw"
    FAKE_FW_CALLS=$TMP/test-cap-holder.fw FAKE_FW_MODE=holder OCANNL_TOOL_FLEET_WORKER=$slot_fake \
    REPEAT_TEST_ACTION_SECS=4 argv_mode=slow_dune argv_probe test-cap-holder run --test-cap 1 build @cheap
    { [ "$argv_rc" = 0 ] && [ "$(grep -c -- '--probe' "$TMP/test-cap-holder.fw")" = 1 ] &&
      [ "$(grep -vc -- '--probe' "$TMP/test-cap-holder.fw")" = 1 ]; } ||
      test_cap_detail="slow dune beside the slot's holder: exit $argv_rc (want 0; slot calls: $(cat "$TMP/test-cap-holder.fw")): $argv_out"
  fi
  # ...and a GPU batch's hung test is pointed at the kernel journal.
  if [ -z "$test_cap_detail" ]; then
    test_cap_n=$((test_cap_n + 1))
    REPEAT_TEST_ACTION_PID=$TMP/test-cap-action-$test_cap_n.pid
    argv_mode=action slot_probe test-cap-gpu hold hip cc run --test-cap 2 build @cheap
    { [ "$argv_rc" = 142 ] && grep -qF "read the kernel journal" <<<"$argv_out" &&
      grep -q "a wedged device hangs every test after it" "$argv_dir/log"; } ||
      test_cap_detail="a GPU batch's hung test: exit $argv_rc; digest: $argv_out"
  fi
  if [ -z "$test_cap_detail" ]; then
    argv_probe test-cap-bad run --test-cap 2x build @cheap
    { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -q -- "--test-cap must be a nonnegative integer" <<<"$argv_err"; } ||
      test_cap_detail="a bad value: exit $argv_rc; dune calls: ${argv_calls:-<none>}; stderr: $argv_err"
  fi
  unset REPEAT_TEST_ACTION_PID
  if [ -z "$test_cap_detail" ]; then report 0 "$test_cap_label"; else report 1 "$test_cap_label" "$test_cap_detail"; fi
fi
}
harness_leg_run '75: the per-test cap #device' leg_75

# Leg 76: the device probe, against a stand-in (OCANNL_TOOL_DEVICE_PROBE) that
# answers, fails fast, or never answers, per backend. A GPU batch's probe runs
# inside the slot, ahead of dune; a query that does not answer within the
# bound refuses the batch as DEVICE UNHEALTHY (69) with dune never started and
# the query killed, while one that fails fast is recorded and passed over. An
# unread batch probes every GPU backend; a CPU batch probes nothing and keeps
# the slot call the other legs pin. The real probe is bin/device_props. A hang
# is judged only in a run the stand-in started in (probe_hang_run, below).
probe_fake=$TMP/fake-device-probe.sh
cat >"$probe_fake" <<'FAKE'
#!/usr/bin/env bash
b=${1#--ocannl_backend=}
# A stand-in the bound KILLs before it records anything, as a loaded box can make one
# (gh-ocannl-1366): once, where the leg left the marker.
if [ -e "$FAKE_PROBE_CALLS.unstarted" ]; then rm -f "$FAKE_PROBE_CALLS.unstarted"; exec sleep 30; fi
printf '%s %s\n' "$PWD" "$*" >>"$FAKE_PROBE_CALLS"
# A stand-in that forks its query rather than exec'ing it leaves the query behind its own exit
# unless the probe's bound reaches the whole tree (FAKE_PROBE_FORK; gh-ocannl-1252).
if [ -n "${FAKE_PROBE_FORK:-}" ]; then
  case " ${FAKE_PROBE_HANG:-} " in
    *" $b "*) sleep 30 & printf '%s\n' "$!" >"$FAKE_PROBE_CALLS.child"; wait; exit 0 ;;
  esac
fi
case " ${FAKE_PROBE_HANG:-} " in *" $b "*) printf '%s\n' "$$" >"$FAKE_PROBE_CALLS.pid"; exec sleep 30 ;; esac
case " ${FAKE_PROBE_ABSENT:-} " in *" $b "*) echo "no $b here" >&2; exit 2 ;; esac
echo "static.device[0].device_name = fake $b"
FAKE
chmod +x "$probe_fake"
# One bound, two questions: whether the query answered, and -- for a stand-in that never answers
# -- whether it had started, recording its call and pid, before the bound KILLed it. A loaded box
# can delay the stand-in's start past a second (gh-ocannl-1366), so the two are kept apart. A query
# that answers runs under probe_answer_cap, which it never comes near and so never waits out. A
# hang is judged by probe_hang_run, whose bound starts at 1s and is tripled, up to
# probe_hang_cap_max, for as long as the stand-in has not recorded its start: a run it never
# started in proves nothing about the query the leg asked for, so it is repeated rather than
# judged, with a note (none where the leg withholds the start itself: probe_hang_late). probe_cap is
# the bound the last run had, which the checks quote.
probe_answer_cap=30 probe_hang_cap_max=27 probe_cap=
probe_run() { # tag test-backend arrayjit-backend hang absent subcommand [argv...]
  local tag=$1 bt=$2 ba=$3 hang=$4 absent=$5
  shift 5
  probe_cap=${probe_run_cap:-$probe_answer_cap}
  : >"$TMP/$tag.probe"
  rm -f "$TMP/$tag.probe.pid" "$TMP/$tag.probe.child"
  FAKE_PROBE_CALLS=$TMP/$tag.probe FAKE_PROBE_HANG=$hang FAKE_PROBE_ABSENT=$absent \
  OCANNL_TOOL_DEVICE_PROBE=$probe_fake OCANNL_TOOL_DEVICE_PROBE_CAP=$probe_cap \
    slot_probe "$tag" hold "$bt" "$ba" "$@"
  probe_calls=$(cat "$TMP/$tag.probe")
}
probe_hang_run() { # start-evidence (pid|child) tag test-backend arrayjit-backend hang absent subcommand [argv...]
  local evidence=$1 cap=1
  shift
  while :; do
    probe_hang_start=$SECONDS
    probe_run_cap=$cap probe_run "$@"
    probe_hang_took=$((SECONDS - probe_hang_start))
    [ ! -s "$TMP/$1.probe.$evidence" ] || return 0
    [ "$cap" -lt "$probe_hang_cap_max" ] || return 0
    [ -n "${probe_hang_late:-}" ] ||
      printf '      note: %s: the stand-in had not started within %ss; judging a run bounded at %ss\n' \
        "$1" "$cap" "$((cap * 3))"
    cap=$((cap * 3))
  done
}
probe_calls= probe_hang_start= probe_hang_took=
leg_76() {
probe_label="device probe: a GPU batch whose fresh device query does not answer is DEVICE UNHEALTHY, dune never started"
probe_detail=
probe_root=$(cd "$repeat_root" && pwd -P)
probe_hang_run pid probe-hang hip cc hip "" run build @cheap
case $argv_out in
  *"verdict: DEVICE UNHEALTHY (a fresh device query did not answer; nothing ran) (exit 69)"*"hip: NO ANSWER within ${probe_cap}s"*) ;;
  *) probe_detail="hang: digest: $argv_out" ;;
esac
[ -n "$probe_detail" ] || { [ "$argv_rc" = 69 ] && [ -z "$argv_calls" ]; } ||
  probe_detail="hang: exit $argv_rc (want 69); dune calls: ${argv_calls:-<none>}"
[ -n "$probe_detail" ] || [ "$probe_hang_took" -lt $((probe_cap + 14)) ] ||
  probe_detail="hang: the refusal took ${probe_hang_took}s under a ${probe_cap}s bound"
case $slot_calls in
  "execution slot --wait 600 --gpu -- "*" $probe_root/tools/test-run.sh _probe dune $probe_fake $probe_cap hip -- dune build @cheap") ;;
  *) [ -n "$probe_detail" ] || probe_detail="hang: the probe is not the slot's command, ahead of dune: $slot_calls" ;;
esac
[ -n "$probe_detail" ] || [ "$probe_calls" = "$probe_root/test/config --ocannl_backend=hip" ] ||
  probe_detail="hang: the query was not asked once, for hip, from test/config: $probe_calls"
[ -n "$probe_detail" ] || grep -q "did not answer within ${probe_cap}s" "$argv_dir/log" ||
  probe_detail="hang: the log does not say why dune never started: $(cat "$argv_dir/log")"
if [ -z "$probe_detail" ]; then
  probe_pid=$(cat "$TMP/probe-hang.probe.pid" 2>/dev/null)
  if [ -z "$probe_pid" ]; then
    probe_detail="hang: the stand-in recorded its call but no pid within ${probe_cap}s"
  else
    case $(pstate "$probe_pid") in
      '' | Z) ;;
      *) probe_detail="hang: the query (pid $probe_pid) survived its bound"; kill -KILL "$probe_pid" 2>/dev/null ;;
    esac
  fi
fi
# A stand-in KILLed before it recorded its start -- as one was on a loaded box (gh-ocannl-1366) --
# does not fail the leg: its run is repeated under a wider bound, and that run is judged. The
# first run's stand-in records nothing by construction (the marker), not by a race with the
# bound, so the judged run is exactly the second, bounded at 3s.
if [ -z "$probe_detail" ]; then
  : >"$TMP/probe-late.probe.unstarted"
  probe_hang_late=1 probe_hang_run pid probe-late hip cc hip "" run build @cheap
  probe_pid=$(cat "$TMP/probe-late.probe.pid" 2>/dev/null)
  { [ "$probe_cap" = 3 ] && [ ! -e "$TMP/probe-late.probe.unstarted" ] && [ "$argv_rc" = 69 ] &&
    [ -z "$argv_calls" ] && [ -n "$probe_pid" ] &&
    [ "$probe_calls" = "$probe_root/test/config --ocannl_backend=hip" ] &&
    grep -qF "hip: NO ANSWER within 3s" "$argv_dir/device-probe"; } ||
    probe_detail="unstarted: judged under ${probe_cap}s (want 3s); exit $argv_rc (want 69); dune calls: ${argv_calls:-<none>}; probes: ${probe_calls:-<none>}"
  case ${probe_pid:+$(pstate "$probe_pid")} in
    '' | Z) ;;
    *) kill -KILL "$probe_pid" 2>/dev/null
       [ -n "$probe_detail" ] || probe_detail="unstarted: the query (pid $probe_pid) survived its bound" ;;
  esac
fi
# A query a forking stand-in left behind is killed with it: the bound KILLs the query's whole
# process group (Codex review round 2 on PR #1172).
if [ -z "$probe_detail" ]; then
  FAKE_PROBE_FORK=1 probe_hang_run child probe-fork hip cc hip "" run build @cheap
  probe_child=$(cat "$TMP/probe-fork.probe.child" 2>/dev/null)
  if [ "$argv_rc" != 69 ] || [ -z "$probe_child" ]; then
    probe_detail="fork: exit $argv_rc (want 69); forked query: ${probe_child:-none recorded}"
  else
    case $(pstate "$probe_child") in
      '' | Z) ;;
      *) probe_detail="fork: the forked query (pid $probe_child) outlived the bound"
         kill -KILL "$probe_child" 2>/dev/null ;;
    esac
  fi
fi
# An answer, and a fast failure, start dune once under the same slot call.
for probe in "answer::hip: answered in" "absent:hip:hip: failed fast (exit 2"; do
  [ -z "$probe_detail" ] || break
  IFS=: read -r tag absent want <<<"$probe"
  probe_run "probe-$tag" hip cc "" "$absent" run build @cheap
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ] && grep -qF -- "$want" <<<"$argv_out" &&
    [ "$probe_calls" = "$probe_root/test/config --ocannl_backend=hip" ]; } ||
    probe_detail="$tag: exit $argv_rc; dune calls: ${argv_calls:-<none>}; probes: $probe_calls; digest: $argv_out"
done
# An unread batch holds any backend: every GPU one is asked, in order.
if [ -z "$probe_detail" ]; then
  probe_run probe-unread cc fail "" "cuda metal" run build @cheap
  probe_asked=$(sed 's/.*--ocannl_backend=//' <<<"$probe_calls" | paste -s -d ' ' -)
  { [ "$argv_rc" = 0 ] && [ "$probe_asked" = "cuda hip metal" ]; } ||
    probe_detail="unread: exit $argv_rc; asked: $probe_asked (want cuda hip metal)"
fi
# A CPU batch asks nothing, and its slot call is the bare one.
if [ -z "$probe_detail" ]; then
  probe_run probe-cpu cc cc hip "" run build @cheap
  { [ "$argv_rc" = 0 ] && [ -z "$probe_calls" ] &&
    [ "$slot_calls" = "execution slot --wait 600 --cpu -- dune build @cheap" ]; } ||
    probe_detail="cpu: exit $argv_rc; probes: ${probe_calls:-<none>}; slot call: $slot_calls"
fi
# Without a stand-in the probe is bin/device_props, built by `_probe` inside
# the slot -- after the slot call, before the batch's own dune -- from the
# tree dune is pointed at. The fixture dune builds nothing, so the batch is
# recorded as not probed and still starts.
if [ -z "$probe_detail" ] && [ "$(uname -s)" != Darwin ]; then
  for build_dir in "" elsewhere; do
    : >"$TMP/probe-build.probe"
    FAKE_PROBE_CALLS=$TMP/probe-build.probe OCANNL_TOOL_DEVICE_PROBE= DUNE_BUILD_DIR=$build_dir \
      slot_probe "probe-build${build_dir:+-$build_dir}" hold hip cc run build @cheap
    want_exe=$probe_root/${build_dir:-_build}/default/bin/device_props.exe
    { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build ./bin/device_props.exe
build @cheap" ] && grep -qF "not probed: $want_exe did not build" "$argv_dir/device-probe" &&
      [ ! -s "$TMP/probe-build.probe" ]; } ||
      { probe_detail="the real probe (DUNE_BUILD_DIR=${build_dir:-unset}): exit $argv_rc; dune calls: $argv_calls; record: $(cat "$argv_dir/device-probe" 2>/dev/null)"; break; }
  done
fi
if [ -z "$probe_detail" ]; then report 0 "$probe_label"; else report 1 "$probe_label" "$probe_detail"; fi
}
harness_leg_run '76: the device probe #device' leg_76

# Leg 77: `plan` reports the probe and the per-test cap, and asks no device.
leg_77() {
plan_probe_detail=
probe_run probe-plan hip cc "" "" plan build @cheap
{ [ "$argv_rc" = 0 ] && [ -z "$argv_calls" ] && [ -z "$probe_calls" ] &&
  grep -qF "device probe: hip, bounded at ${probe_answer_cap}s" <<<"$argv_out" &&
  grep -qF "test cap: 1500s for each process dune starts, the default for a GPU batch" <<<"$argv_out"; } ||
  plan_probe_detail="gpu: exit $argv_rc; probes: ${probe_calls:-<none>}; plan: $argv_out"
# The default's scope: no CPU batch, no @slow target; an explicit cap is kept
# on either, and 0 lifts it.
for probe in "cpu|cc|build @cheap|test cap: none (no default: a CPU batch)" \
             "slow|hip|build @test/training/slow-cifar_conv|test cap: none (no default: the targets reach the slow tests (@test/training/slow-cifar_conv))" \
             "slow-asked|hip|--test-cap 9000 build @slow|test cap: 9000s for each process dune starts, asked for" \
             "cpu-lifted|cc|--test-cap 0 build @cheap|test cap: none (--test-cap 0)"; do
  [ -z "$plan_probe_detail" ] || break
  IFS='|' read -r tag bt argv want <<<"$probe"
  # shellcheck disable=SC2086  # the argv is word-split on purpose
  probe_run "probe-plan-$tag" "$bt" cc "" "" plan $argv
  { [ "$argv_rc" = 0 ] && grep -qF -- "$want" <<<"$argv_out"; } ||
    plan_probe_detail="$tag: exit $argv_rc; want '$want'; plan: $argv_out"
done
# Where run resolves no backends (no fleet slot, no width hazard), plan
# reports neither bound, however its own resolution reads the batch.
if [ -z "$plan_probe_detail" ]; then
  FAKE_BACKEND_TEST=hip OCANNL_TOOL_DEVICE_PROBE=$probe_fake argv_probe probe-plan-unread plan build @cheap
  { [ "$argv_rc" = 0 ] && grep -qF "device probe: none" <<<"$argv_out" &&
    grep -qF "test cap: none (no default: run does not resolve" <<<"$argv_out"; } ||
    plan_probe_detail="no slot, no hazard: exit $argv_rc; plan: $argv_out"
fi
if [ -z "$plan_probe_detail" ]; then
  report 0 "plan: reports the device probe and the per-test cap, asking no device"
else
  report 1 "plan: reports the device probe and the per-test cap, asking no device" "$plan_probe_detail"
fi
}
harness_leg_run '77: plan reports the probe and the per-test cap #device' leg_77

# Leg 83: `repeat` has both bounds (gh-ocannl-1252). A repeat samples timing-dependent goldens,
# often GPU tests, and ran with neither: a wedged device held it for N times the iteration cap.
# Now a test past the per-test cap ends its iteration as TIMEOUT (142) naming the test, the test
# killed, and ENDS the set, since every later iteration would hang the same way; one inside it does
# not. The default follows `run`'s decision from the same resolution of the batch (a GPU batch:
# 1500s, announced in the log; a CPU batch: none), and an explicit cap applies to any batch. A GPU
# batch's device probe runs before each iteration, inside it: a query that does not answer ends
# the set as DEVICE UNHEALTHY (69) with dune never started -- not even its clean -- and an answer
# lets every iteration run, the device asked once each.
# Whether the fixture's renamed dune reads back as `dune` in ps, as leg 75 asks
# it: asked once, by the first of the later legs to run that needs it.
renamed_dune_readable() { # sets test_cap_named_ok=1 where it does
  local named
  [ -z "${test_cap_named_ok:-}" ] || return 0
  perl -e '$0 = "dune"; sleep 10' </dev/null >/dev/null 2>&1 &
  named=$!
  sleep 0.3
  ps -A -o pid= -o comm= 2>/dev/null | grep -Eq "^ *$named +(.*/)?dune$" && test_cap_named_ok=1
  kill "$named" 2>/dev/null
  wait "$named" 2>/dev/null
  return 0
}
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
renamed_dune_readable
if [ "$(uname -s)" != Linux ] || ! command -v python3 >/dev/null 2>&1 ||
   ! python3 "$reap_stub" >/dev/null 2>&1; then
  why="needs a Linux child subreaper (prctl PR_SET_CHILD_SUBREAPER, through python3)"
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
direct_real=$( (command -v dune >/dev/null 2>&1 || . "$HERE/opam-env.sh" >/dev/null 2>&1
  command -v dune) 2>/dev/null) || direct_real=
if [ -z "$direct_real" ] || ! command -v python3 >/dev/null 2>&1; then
  skip "$direct_label" "no dune on PATH, nor through tools/opam-env.sh, or no python3"
  skip "$direct_nc_label" "no dune on PATH, nor through tools/opam-env.sh, or no python3"
  skip "$direct_fd_label" "no dune on PATH, nor through tools/opam-env.sh, or no python3"
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
    (* Each step's supervisor names its own directory, and dune gives each action a TMPDIR and
       a trace directory of its own: everything else is what the replay owes. *)
    let mine k = List.mem k [ "TMPDIR"; "DUNE_ACTION_TRACE_DIR"; "OCANNL_TOOL_TESTRUN_RD" ] in
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

# ---------------------------------------------------------------------------
# Leg 79: credentials never reach dune (gh-ocannl-1280)
# ---------------------------------------------------------------------------
# Dune records the environment of every process it spawns in the worktree's
# `_build/trace.csexp`, so a token the session exports would sit inside the
# checkout after every build. The run below carries GH_TOKEN (a deny-listed
# name) and FOO_API_KEY (matched by a pattern), both with obviously fake
# values, beside CRED_TEST_PLAIN and the harness's own OCANNL_TOOL_* exports,
# which must still pass. Two observers: a fixture dune that writes the names of
# its environment -- decided on every host -- and, where a real dune is at hand,
# that dune's own trace. The negative control is the shipping script with its
# scrub cut out, which must hand dune both credentials: an observer that cannot
# see them proves nothing about the scrub.
leg_79() {
cred_label="credentials: dune runs without GH_TOKEN and FOO_API_KEY, a plain variable and OCANNL_TOOL_* still pass"
cred_nc_label="negative control: with the scrub cut out, dune sees both credentials"
cred_trace_label="credentials: the run's real dune trace names neither GH_TOKEN nor FOO_API_KEY, and does name the plain variable"
cred_trace_nc_label="negative control: with the scrub cut out, the real trace names both credentials"
cred_mutant=$(mutant no-credential-scrub \
  '/^  eval "\$\(credential_env_scrub_text\)" \|\|$/ { getline; print "  :"; next } { print }')
if [ -z "$cred_mutant" ] || cmp -s "$SRC" "$cred_mutant"; then
  harness_abandon "the credential scrub could not be cut out of $SRC; the mutant is broken"
  return
fi
cred_bin=$TMP/cred-bin
mkdir -p "$cred_bin"
cat >"$cred_bin/dune" <<'EOF'
#!/usr/bin/env bash
# The run's last phase asks for the version and the promotion list: an empty one.
case ${1:-} in
  --version) echo 3.24.2; exit 0 ;;
  promotion) exit 0 ;;
esac
env | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' >"$CRED_TEST_NAMES"
EOF
chmod +x "$cred_bin/dune"
cred_root() { # name script -> a fixture root running SCRIPT as tools/test-run.sh
  local root=$TMP/$1
  mkdir -p "$root/tools" "$root/scripts" "$root/_build"
  cp "$2" "$root/tools/test-run.sh"
  stage_sourced "$root" || return 1
  chmod +x "$root/tools/test-run.sh"
  printf '%s' "$root"
}
cred_run() { # root runs-dir names-file subcommand argv...: the launch, with the two credentials exported
  local root=$1 runs=$2 names=$3
  shift 3
  GH_TOKEN=fixture-not-a-token FOO_API_KEY=fixture-not-a-key CRED_TEST_PLAIN=kept \
    CRED_TEST_NAMES=$names OCANNL_TOOL_TEST_RUNS=$runs \
    "$root/tools/test-run.sh" "$@" >"$names.out" 2>&1
}
cred_has() { grep -qx -e "$1" "$2"; }
cred_ship=$(cred_root cred-ship "$SRC") && cred_cut=$(cred_root cred-cut "$cred_mutant") ||
  { harness_abandon "cannot stage the credential fixture roots (what test-run.sh sources)"; return; }
cred_rc=0
PATH=$cred_bin:$PATH cred_run "$cred_ship" "$TMP/cred-runs" "$TMP/cred-ship.names" run build @fixture ||
  cred_rc=$?
cred_detail=
if [ "$cred_rc" != 0 ]; then
  cred_detail="the run exited $cred_rc: $(tail -5 "$TMP/cred-ship.names.out")"
elif [ ! -s "$TMP/cred-ship.names" ]; then
  cred_detail="the fixture dune recorded no environment"
elif cred_has GH_TOKEN "$TMP/cred-ship.names"; then
  cred_detail="dune saw GH_TOKEN"
elif cred_has FOO_API_KEY "$TMP/cred-ship.names"; then
  cred_detail="dune saw FOO_API_KEY"
elif ! cred_has CRED_TEST_PLAIN "$TMP/cred-ship.names"; then
  cred_detail="dune did not see CRED_TEST_PLAIN"
elif ! cred_has OCANNL_TOOL_DEVICE_PROBE "$TMP/cred-ship.names"; then
  cred_detail="dune did not see the harness's OCANNL_TOOL_DEVICE_PROBE"
fi
report "$([ -z "$cred_detail" ] && echo 0 || echo 1)" "$cred_label" "$cred_detail"
cred_rc=0
PATH=$cred_bin:$PATH cred_run "$cred_cut" "$TMP/cred-cut-runs" "$TMP/cred-cut.names" run build @fixture ||
  cred_rc=$?
if [ "$cred_rc" = 0 ] && cred_has GH_TOKEN "$TMP/cred-cut.names" &&
  cred_has FOO_API_KEY "$TMP/cred-cut.names"; then
  report 0 "$cred_nc_label"
else
  report 1 "$cred_nc_label" "exit $cred_rc; names: $(tr '\n' ' ' <"$TMP/cred-cut.names" 2>/dev/null)"
fi
# And when dune is not on PATH, select_dune sources tools/opam-env.sh, whose switch environment
# can set a credential the startup scrub never saw: a fake opam answers `opam env` with both
# credentials and a PATH reaching the fixture dune. Through `repeat`, whose dune the launching
# process starts itself (a `run`'s dune is started by a fresh `_resolve`, whose own startup scrub
# also catches it). The PATH holds only that opam and the system directories; a host whose system
# directories hold a dune skips (the opam environment would never be asked for). The negative
# control cuts out select_dune's second scrub alone.
cred_opam_label="credentials: the opam environment select_dune sources is scrubbed too"
cred_opam_nc_label="negative control: without select_dune's scrub, dune sees the switch's credentials"
cred_opam_bin=$TMP/cred-opam-bin
mkdir -p "$cred_opam_bin"
cat >"$cred_opam_bin/opam" <<'EOF'
#!/usr/bin/env bash
[ "${1:-}" = env ] || exit 9
: >>"$CRED_TEST_OPAM_ASKED"
printf "GH_TOKEN='fixture-switch-token'; export GH_TOKEN; FOO_API_KEY='fixture-switch-key'; export FOO_API_KEY; PATH='%s':\"\$PATH\"; export PATH;\n" "$CRED_TEST_DUNE_DIR"
EOF
chmod +x "$cred_opam_bin/opam"
cred_opam_mutant=$(mutant no-select-dune-scrub '/^  scrub_credentials$/ { next } { print }')
cred_opam_run() { # tag script -> names file; 0 when the opam environment was asked for
  local root
  root=$(cred_root "$1" "$2")
  cp "$HERE/opam-env.sh" "$root/tools/opam-env.sh"
  rm -f "$TMP/$1.asked"
  PATH=$cred_opam_bin:/usr/bin:/bin:/usr/sbin:/sbin CRED_TEST_DUNE_DIR=$cred_bin \
    CRED_TEST_OPAM_ASKED=$TMP/$1.asked cred_run "$root" "$TMP/$1-runs" "$TMP/$1.names" repeat 2 build @fixture
  [ -e "$TMP/$1.asked" ]
}
if [ -z "$cred_opam_mutant" ] || cmp -s "$SRC" "$cred_opam_mutant"; then
  harness_abandon "select_dune's scrub could not be cut out of $SRC; the mutant is broken"
  return
elif PATH=/usr/bin:/bin:/usr/sbin:/sbin command -v dune >/dev/null 2>&1; then
  skip "$cred_opam_label" "a dune in the system directories: select_dune never sources the opam environment"
  skip "$cred_opam_nc_label" "a dune in the system directories: select_dune never sources the opam environment"
else
  if ! cred_opam_run cred-opam "$SRC"; then
    report 1 "$cred_opam_label" "the opam environment was not asked for: $(tail -3 "$TMP/cred-opam.names.out")"
  elif cred_has GH_TOKEN "$TMP/cred-opam.names" || cred_has FOO_API_KEY "$TMP/cred-opam.names" ||
    ! cred_has CRED_TEST_PLAIN "$TMP/cred-opam.names"; then
    report 1 "$cred_opam_label" "names: $(tr '\n' ' ' <"$TMP/cred-opam.names" 2>/dev/null)"
  else
    report 0 "$cred_opam_label"
  fi
  if cred_opam_run cred-opam-cut "$cred_opam_mutant" && cred_has GH_TOKEN "$TMP/cred-opam-cut.names" &&
    cred_has FOO_API_KEY "$TMP/cred-opam-cut.names"; then
    report 0 "$cred_opam_nc_label"
  else
    report 1 "$cred_opam_nc_label" "names: $(tr '\n' ' ' <"$TMP/cred-opam-cut.names" 2>/dev/null)"
  fi
fi
# The scrub text itself fails CLOSED, under every shell at hand (dash where installed): an `env`
# that cannot run is a refusal, never an empty environment that passes; a deny-listed name a parent
# passed that is not a shell identifier (`odd-name_TOKEN`) cannot be unset, so it refuses too,
# naming it. With PATH holding nothing but `env`, valid credentials are removed and a plain variable
# stays -- the text needs no other external program, so no parser stage can fail open -- and a
# value spanning lines with a glob character in it neither breaks the walk nor leaks the caller's
# IFS or `set -f`. A newline-named entry refuses through the second listing, and inherited
# functions named env, unset or printf are bypassed.
cred_text_label="credentials: the scrub text refuses when env cannot run, on a deny-listed non-identifier or newline-named entry, and bypasses inherited functions"
cred_text_ok_label="credentials: the scrub text, with only env on PATH, removes valid names and restores IFS and set -f"
cred_text_detail= cred_text_ok_detail=
cred_nl='
'
cred_env_only=$TMP/cred-env-only
mkdir -p "$cred_env_only"
ln -s "$(command -v env)" "$cred_env_only/env"
cred_helper=$HERE/credential-env.sh
for cred_shell in /bin/sh "$(command -v bash)" "$(command -v dash 2>/dev/null)"; do
  [ -n "$cred_shell" ] || continue
  cred_rc=0
  cred_out=$(env -i PATH=/nonexistent GH_TOKEN=fixture-not-a-token "$cred_shell" -c \
    '. "$1"; eval "$(credential_env_scrub_text)"' cred "$cred_helper" 2>&1) || cred_rc=$?
  [ "$cred_rc" != 0 ] || cred_text_detail="$cred_text_detail; $cred_shell passed with env unrunnable"
  cred_rc=0
  # Refused by name -- or, where the shell itself drops such an entry at startup (dash), gone from
  # what its children inherit, which the trailing `env` shows.
  cred_out=$(env -i "PATH=$cred_env_only" 'odd-name_TOKEN=fixture-not-a-token' "$cred_shell" -c \
    '. "$1"; eval "$(credential_env_scrub_text)" || { printf "%s\n" "$credential_env_left"; exit 1; }; env' \
    cred "$cred_helper" 2>&1) || cred_rc=$?
  case $cred_rc:$cred_out in
    1:*"odd-name_TOKEN (not a shell identifier)"*) ;;
    0:*odd-name_TOKEN*) cred_text_detail="$cred_text_detail; $cred_shell passed odd-name_TOKEN to its children" ;;
    0:*) ;;
    *) cred_text_detail="$cred_text_detail; $cred_shell on odd-name_TOKEN: exit $cred_rc, said: $cred_out" ;;
  esac
  cred_rc=0
  cred_out=$(env -i "PATH=$cred_env_only" GH_TOKEN=fixture-not-a-token "FOO_API_KEY=fixture
BAR_TOKEN=* not-a-key" CRED_TEST_PLAIN=kept "$cred_shell" -c \
    'set -u; IFS=" "; . "$1"; eval "$(credential_env_scrub_text)" || exit 3
     printf "ifs=[%s] opts=%s\n" "$IFS" "$-"; env' cred "$cred_helper" 2>&1) || cred_rc=$?
  case $cred_rc in 0) ;; *) cred_text_ok_detail="$cred_text_ok_detail; $cred_shell exited $cred_rc: $cred_out" ;; esac
  case $cred_out in *GH_TOKEN=* | *FOO_API_KEY=*) cred_text_ok_detail="$cred_text_ok_detail; $cred_shell left a credential" ;; esac
  case $cred_out in *CRED_TEST_PLAIN=kept*) ;; *) cred_text_ok_detail="$cred_text_ok_detail; $cred_shell lost the plain variable" ;; esac
  cred_first=${cred_out%%"$cred_nl"*}
  case $cred_first in "ifs=[ ] opts="*f*) cred_text_ok_detail="$cred_text_ok_detail; $cred_shell left set -f on" ;; "ifs=[ ] opts="*) ;;
    *) cred_text_ok_detail="$cred_text_ok_detail; $cred_shell did not restore IFS: $cred_first" ;; esac
  # A name holding a newline reads, line by line, as a plain `GH_TOKEN`: the second listing
  # still sees it, so it refuses -- or the shell dropped it at startup (dash) and its children
  # never see it.
  cred_rc=0
  cred_out=$(env -i "PATH=$cred_env_only" "ODD${cred_nl}GH_TOKEN=fixture-not-a-token" "$cred_shell" -c \
    '. "$1"; eval "$(credential_env_scrub_text)" || { printf "%s\n" "$credential_env_left"; exit 1; }; env' \
    cred "$cred_helper" 2>&1) || cred_rc=$?
  case $cred_rc:$cred_out in
    1:*"GH_TOKEN (still in the environment)"*) ;;
    0:*"${cred_nl}GH_TOKEN="* | 0:"GH_TOKEN="*) cred_text_detail="$cred_text_detail; $cred_shell passed a newline-named GH_TOKEN to its children" ;;
    0:*) ;;
    *) cred_text_detail="$cred_text_detail; $cred_shell on a newline-named GH_TOKEN: exit $cred_rc, said: $cred_out" ;;
  esac
done
# Inherited functions named env, unset and printf (bash imports exported ones) are bypassed: the
# scrub still removes the credential, rather than asking a function that answers nothing.
cred_bash=$(command -v bash)
cred_rc=0
cred_out=$(env -i "PATH=$cred_env_only" GH_TOKEN=fixture-not-a-token CRED_HELPER="$cred_helper" \
  CRED_BASH="$cred_bash" "$cred_bash" -c \
  'env() { :; }; unset() { :; }; printf() { :; }; export -f env unset printf
   exec "$CRED_BASH" -c '\''. "$CRED_HELPER"; eval "$(credential_env_scrub_text)" || exit 3; command env'\''' \
  2>&1) || cred_rc=$?
case $cred_rc:$cred_out in
  0:*GH_TOKEN=*) cred_text_detail="$cred_text_detail; inherited env/unset/printf functions let GH_TOKEN through" ;;
  0:*PATH=*) ;;
  *) cred_text_detail="$cred_text_detail; under inherited functions: exit $cred_rc, said: $cred_out" ;;
esac
report "$([ -z "$cred_text_detail" ] && echo 0 || echo 1)" "$cred_text_label" "${cred_text_detail#; }"
report "$([ -z "$cred_text_ok_detail" ] && echo 0 || echo 1)" "$cred_text_ok_label" "${cred_text_ok_detail#; }"
# The Windows wrapper test-run.sh starts dune through, which AGENTS.md also has a Git Bash session
# run by hand, scrubs on its own: a copy beside the helper, the fixture dune on PATH, and a copy
# with the scrub cut out as the negative control.
cred_quiet_label="credentials: tools/dune-quiet.sh, run directly, starts dune without them"
cred_quiet_nc_label="negative control: dune-quiet.sh without its scrub hands dune both credentials"
mkdir -p "$TMP/cred-quiet" "$TMP/cred-quiet-cut"
cp "$HERE/dune-quiet.sh" "$HERE/credential-env.sh" "$TMP/cred-quiet/"
cp "$HERE/credential-env.sh" "$TMP/cred-quiet-cut/"
awk '/^eval "\$\(credential_env_scrub_text\)" \|\| \{$/ { skip = 1 } skip { if ($0 == "}") skip = 0; next } { print }' \
  "$HERE/dune-quiet.sh" >"$TMP/cred-quiet-cut/dune-quiet.sh"
cred_quiet_run() { # dir
  rm -f "$1.names"
  GH_TOKEN=fixture-not-a-token FOO_API_KEY=fixture-not-a-key CRED_TEST_PLAIN=kept \
    CRED_TEST_NAMES=$1.names PATH=$cred_bin:$PATH bash "$1/dune-quiet.sh" build @fixture >"$1.out" 2>&1
}
if cmp -s "$HERE/dune-quiet.sh" "$TMP/cred-quiet-cut/dune-quiet.sh"; then
  harness_abandon "the scrub could not be cut out of tools/dune-quiet.sh; the mutant is broken"
  return
fi
cred_rc=0
cred_quiet_run "$TMP/cred-quiet" || cred_rc=$?
if [ "$cred_rc" = 0 ] && [ -s "$TMP/cred-quiet.names" ] && ! cred_has GH_TOKEN "$TMP/cred-quiet.names" &&
  ! cred_has FOO_API_KEY "$TMP/cred-quiet.names" && cred_has CRED_TEST_PLAIN "$TMP/cred-quiet.names"; then
  report 0 "$cred_quiet_label"
else
  report 1 "$cred_quiet_label" "exit $cred_rc; names: $(tr '\n' ' ' <"$TMP/cred-quiet.names" 2>/dev/null)"
fi
cred_rc=0
cred_quiet_run "$TMP/cred-quiet-cut" || cred_rc=$?
if [ "$cred_rc" = 0 ] && cred_has GH_TOKEN "$TMP/cred-quiet-cut.names" &&
  cred_has FOO_API_KEY "$TMP/cred-quiet-cut.names"; then
  report 0 "$cred_quiet_nc_label"
else
  report 1 "$cred_quiet_nc_label" "exit $cred_rc; see $TMP/cred-quiet-cut.out"
fi
# The trace itself, which is what the issue found tokens in. A host with no
# dune skips it (the toolchain-free CI job); a dune that writes no trace there
# skips it too, since there is then nothing to leak into.
cred_real=$( (command -v dune >/dev/null 2>&1 || . "$HERE/opam-env.sh" >/dev/null 2>&1
  command -v dune) 2>/dev/null) || cred_real=
if [ -z "$cred_real" ]; then
  skip "$cred_trace_label" "no dune on PATH, nor through tools/opam-env.sh"
  skip "$cred_trace_nc_label" "no dune on PATH, nor through tools/opam-env.sh"
else
  cred_project() { # root: a one-rule dune project whose build spawns a process
    printf '(lang dune 3.20)\n' >"$1/dune-project"
    printf '(rule\n (with-stdout-to spawned.txt (run sh -c "echo spawned")))\n' >"$1/dune"
    rm -rf "$1/_build"
  }
  cred_project "$cred_ship"
  cred_project "$cred_cut"
  # The run hands dune a trace file of its own and deletes it once its
  # slowest actions are read (leg 80), so the trace is observed through the
  # reader seam: a stand-in reader that copies the run's trace to the root's
  # `_build/trace.csexp`, where the default trace would have been.
  cred_copier=$TMP/cred-trace-copier.sh
  printf '#!/usr/bin/env bash\ncp "$3" "$CRED_TRACE_COPY"\n' >"$cred_copier"
  cred_trace() { # root -> 0 when its trace exists, the three names' counts in cred_gh/foo/plain
    local trace=$1/_build/trace.csexp
    [ -f "$trace" ] || return 1
    cred_gh=$(grep -a -c GH_TOKEN "$trace")
    cred_foo=$(grep -a -c FOO_API_KEY "$trace")
    cred_plain=$(grep -a -c CRED_TEST_PLAIN "$trace")
    cred_counts="GH_TOKEN: $cred_gh, FOO_API_KEY: $cred_foo, CRED_TEST_PLAIN: $cred_plain"
  }
  cred_real_run() { # root tag
    (cd "$1" && PATH=$(dirname "$cred_real"):$PATH OCANNL_TOOL_SLOWEST_READER=$cred_copier \
      CRED_TRACE_COPY=$1/_build/trace.csexp cred_run "$1" "$TMP/$2-runs" "$TMP/$2.names" \
      run build --cache=disabled ./spawned.txt)
  }
  cred_rc=0
  cred_real_run "$cred_ship" cred-real || cred_rc=$?
  if [ "$cred_rc" != 0 ]; then
    report 1 "$cred_trace_label" "the run exited $cred_rc: $(tail -5 "$TMP/cred-real.names.out")"
  elif ! cred_trace "$cred_ship"; then
    skip "$cred_trace_label" "$("$cred_real" --version) wrote no trace for the run"
  elif [ "$cred_gh" = 0 ] && [ "$cred_foo" = 0 ] && [ "$cred_plain" != 0 ]; then
    report 0 "$cred_trace_label"
  else
    report 1 "$cred_trace_label" "$cred_counts"
  fi
  cred_rc=0
  cred_real_run "$cred_cut" cred-real-cut || cred_rc=$?
  if [ "$cred_rc" != 0 ]; then
    report 1 "$cred_trace_nc_label" "the run exited $cred_rc: $(tail -5 "$TMP/cred-real-cut.names.out")"
  elif ! cred_trace "$cred_cut"; then
    skip "$cred_trace_nc_label" "$("$cred_real" --version) wrote no trace for the run"
  elif [ "$cred_gh" != 0 ] && [ "$cred_foo" != 0 ]; then
    report 0 "$cred_trace_nc_label"
  else
    report 1 "$cred_trace_nc_label" "$cred_counts"
  fi
fi
}
harness_leg_run '79: credentials never reach dune #credentials' leg_79

# ---------------------------------------------------------------------------
# Leg 82: bisect (gh-ocannl-1225)
# ---------------------------------------------------------------------------
# A fixture repository whose commits each carry a STATE the fixture dune
# reads from the tree it runs in: `pass`, `fail` (the test red), `nobuild`
# (`@check` red), `refuse` (the fake slot refuses), `hang` (the test outlives
# the cap) and `hold` (the test waits to be cancelled). The fixture dune
# records, per call, the STATE, the OCANNL_BACKEND it saw and whether the
# runner's own variables reached it. The invoking checkout sits on `master`
# at the base commit; the range is the `range` branch, whose LAST commit
# replaces tools/test-run.sh with one that exits 3 -- so a bisect that ran a
# tested commit's own runner, rather than its frozen copy, aborts at the bad
# end instead of finding anything.
leg_82() {
if ! command -v git >/dev/null 2>&1; then
  skip 'bisect: exit mapping, backend, isolation (leg 82)' 'no git on PATH'
else
  bi_git() { GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null git "$@"; }
  bi_bin=$TMP/bisect-bin
  mkdir -p "$bi_bin"
  cat >"$bi_bin/dune" <<'EOF'
#!/usr/bin/env bash
st=$(cat STATE 2>/dev/null)
printf '%s %s tree=%s runner=%s :: %s\n' "$st" "${OCANNL_BACKEND:-<unset>}" \
  "${OCANNL_TOOL_TESTRUN_TREE-<unset>}" "${OCANNL_TOOL_TESTRUN_RUNNER-<unset>}" "$*" >>"$BISECT_CALLS"
case " $* " in
  *" @check "*) [ "$st" != nobuild ] || { echo 'File "x.ml", line 1: Error: broken'; exit 1; }; exit 0 ;;
esac
case $st in
  fail) echo 'Error: the test failed'; exit 1 ;;
  hang) exec sleep 60 ;;
  escape*)
    # A descendant that leaves dune's group, keeping the run's lock (fd 9),
    # and reports the tree's STATE as it found it two seconds later.
    perl -MPOSIX -e 'POSIX::setsid(); sleep 2; open my $f, "<", "STATE"; my $now = <$f>;
      chomp $now; open my $o, ">>", $ENV{BISECT_ESCAPE}; print $o "$ARGV[0] $now\n"' "$st" \
      </dev/null >/dev/null 2>&1 &
    # Out of the group before dune exits: the supervisor reaps what is left in it.
    sleep 0.5
    ;;
  hold)
    printf '%s\n' "$$" >"$BISECT_HOLD.pid"
    exec perl -e 'sleep 60' ;;
esac
echo ok
EOF
  # The slot: refused where the tree's STATE says so, held otherwise.
  cat >"$bi_bin/fleet-worker.sh" <<'EOF'
#!/usr/bin/env bash
if [ "$1 $2 $3" = "execution slot --probe" ]; then echo "EXECUTION SLOT PROBE fakebox 4 2"; exit 0; fi
while [ "$1" != -- ]; do shift; done
shift
if [ "$(cat STATE 2>/dev/null)" = refuse ]; then
  echo "EXECUTION SLOT REFUSED fakebox: all 2 GPU tokens (slots 1-2 of 4) busy after 0s"
  exit 1
fi
exec "$@"
EOF
  chmod +x "$bi_bin/dune" "$bi_bin/fleet-worker.sh"
  bi_n=0
  bi_repo() { # <states of the range...>; a fresh fixture, sets bi_root, bi_base, bi_shas
    local s
    bi_n=$((bi_n + 1))
    bi_root=$TMP/bisect-repo-$bi_n
    mkdir -p "$bi_root/tools" "$bi_root/scripts" && cp "$SRC" "$bi_root/tools/test-run.sh" || return 1
    stage_sourced "$bi_root" || return 1
    printf 'pass\n' >"$bi_root/STATE"
    { bi_git init -q "$bi_root" && bi_git -C "$bi_root" config user.name harness &&
      bi_git -C "$bi_root" config user.email harness@invalid &&
      bi_git -C "$bi_root" config commit.gpgsign false &&
      bi_git -C "$bi_root" add -A && bi_git -C "$bi_root" commit -q --no-verify -m base &&
      bi_git -C "$bi_root" branch -M master && bi_git -C "$bi_root" checkout -q -b range; } >/dev/null 2>&1 ||
      { report 1 'bisect: fixture repository'; return 1; }
    bi_base=$(bi_git -C "$bi_root" rev-parse HEAD)
    bi_shas=()
    for s in "$@"; do
      printf '%s\n' "$s" >"$bi_root/STATE"
      bi_git -C "$bi_root" commit -q --no-verify --allow-empty -am "$s" >/dev/null 2>&1 || return 1
      bi_shas+=("$(bi_git -C "$bi_root" rev-parse HEAD)")
    done
    # The range's last commit carries a runner that cannot run anything.
    printf '#!/usr/bin/env bash\nexit 3\n' >"$bi_root/tools/test-run.sh"
    bi_git -C "$bi_root" commit -q --no-verify --amend -a --no-edit >/dev/null 2>&1 || return 1
    bi_shas[${#bi_shas[@]}-1]=$(bi_git -C "$bi_root" rev-parse HEAD)
    bi_git -C "$bi_root" checkout -q master >/dev/null 2>&1
  }
  bi_runs=$TMP/bisect-runs
  bi_run() { # [VAR=value...] -- bisect args...; sets bi_rc, bi_out, bi_err, bi_dir
    local envs=()
    while [ "$1" != -- ]; do envs+=("$1"); shift; done
    shift
    : >"$TMP/bisect.calls"
    env GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null OCANNL_TOOL_TEST_RUNS=$bi_runs \
      BISECT_CALLS=$TMP/bisect.calls BISECT_HOLD=$TMP/bisect-hold BISECT_ESCAPE=$TMP/bisect.escape \
      PATH=$bi_bin:$PATH ${envs[@]+"${envs[@]}"} "$bi_root/tools/test-run.sh" bisect "$@" \
      >"$TMP/bisect.out" 2>"$TMP/bisect.err"
    bi_rc=$?
    bi_out=$(cat "$TMP/bisect.out") bi_err=$(cat "$TMP/bisect.err")
    bi_dir=$(sed -n 's/^bisect: \(\/.*\)$/\1/p' "$TMP/bisect.out" | head -n 1)
  }
  bi_clean() { # the invoking checkout as the fixture left it, and no tree behind
    [ "$(bi_git -C "$bi_root" rev-parse HEAD)" = "$bi_base" ] &&
      [ -z "$(bi_git -C "$bi_root" status --porcelain)" ] &&
      [ "$(bi_git -C "$bi_root" worktree list | grep -c '')" = 1 ] &&
      [ -n "$bi_dir" ] && [ ! -e "$bi_dir/tree" ]
  }
  bi_verdicts() { cut -f3 "$bi_dir/steps.tsv" 2>/dev/null | tr '\n' ' '; }

  # (a) the mapping: a red build is a skip, never bad; the first bad commit is
  # found; every dune call saw the pinned backend, none the runner's own
  # variables; the bad end ran under the frozen runner; the records stay.
  l_found="bisect: a red build is skipped, never bad, and the first bad commit is found under the frozen runner"
  bi_detail=
  bi_repo pass nobuild pass fail fail || bi_detail="fixture"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=hip OCANNL_TOOL_FLEET_WORKER=none -- --cap 60 "$bi_base" range @t
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 0 ]; then bi_detail="exit $bi_rc: $bi_out $bi_err"
  elif ! grep -q "^result: FIRST BAD COMMIT ${bi_shas[3]} fail\$" <<<"$bi_out"; then
    bi_detail="result: $(grep '^result:' <<<"$bi_out") (want ${bi_shas[3]})"
  elif ! grep -q "^bad-end	${bi_shas[4]}	bad	" "$bi_dir/steps.tsv" ||
       ! grep -q "^good-end	$bi_base	good	" "$bi_dir/steps.tsv"; then
    bi_detail="the ends were not run first: $(cat "$bi_dir/steps.tsv")"
  elif ! grep -q "	${bi_shas[1]}	skip	" "$bi_dir/steps.tsv"; then
    bi_detail="the commit whose build is red was not a skip: $(cat "$bi_dir/steps.tsv")"
  elif ! grep -q '^nobuild hip ' "$TMP/bisect.calls" ||
       grep -v '^[a-z]* hip tree=<unset> runner=<unset> :: ' "$TMP/bisect.calls" | grep -q .; then
    bi_detail="a dune call missed the pinned backend, or saw the runner's variables: $(cat "$TMP/bisect.calls")"
  elif grep -q '^nobuild .* @t$' "$TMP/bisect.calls"; then
    bi_detail="the test ran on a commit whose build is red: $(cat "$TMP/bisect.calls")"
  elif [ ! -s "$bi_dir/bisect-log" ] || [ ! -s "$bi_dir/steps/bad-end-test.log" ] ||
       [ ! -s "$bi_dir/steps/bad-end-test.out" ]; then
    bi_detail="the records are missing: $(ls "$bi_dir" "$bi_dir/steps" 2>&1 | tr '\n' ' ')"
  elif ! bi_clean; then
    bi_detail="the invoking checkout changed, or the tree was left: $(bi_git -C "$bi_root" worktree list)"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_found"; else report 1 "$l_found" "$bi_detail"; fi

  # (b) a range whose only untested commits did not build ends without a
  # first bad commit (exit 1, the candidates listed) -- a skip read as bad
  # would have named the red build.
  l_skipped="bisect: a range left with only skipped commits names the candidates, exit 1"
  bi_detail=
  bi_repo nobuild fail || bi_detail="fixture"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- "$bi_base" range @t
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 1 ] || grep -q 'FIRST BAD COMMIT' <<<"$bi_out" ||
       ! grep -q "only skipped commits are left" <<<"$bi_out" ||
       ! grep -q "^  ${bi_shas[0]} nobuild" <<<"$bi_out"; then
    bi_detail="exit $bi_rc: $bi_out"
  elif ! bi_clean; then bi_detail="the tree was left, or the checkout changed"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_skipped"; else report 1 "$l_skipped" "$bi_detail"; fi

  # (c) a refused slot mid-bisect ABORTS it: git bisect stops (the step's 128),
  # the bisect exits 75, nothing is called bad, and the tree goes.
  l_slot="bisect: SLOT REFUSED in a step aborts the bisect with exit 75, judging nothing"
  bi_detail=
  bi_repo refuse fail || bi_detail="fixture"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=$bi_bin/fleet-worker.sh -- "$bi_base" range @t
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 75 ] || ! grep -q '^result: ABORTED: the build run of .*SLOT REFUSED' <<<"$bi_out"; then
    bi_detail="exit $bi_rc: $bi_out $bi_err"
  elif [ "$(bi_verdicts)" != "bad good ABORT " ]; then
    bi_detail="steps: $(cat "$bi_dir/steps.tsv")"
  elif ! grep -q 'exit code 128' "$bi_dir/git-bisect.out" 2>/dev/null &&
       ! grep -q 'exit code 128' <<<"$bi_err"; then
    bi_detail="git bisect run was not stopped by the step's 128: $(cat "$bi_dir/git-bisect.out") $bi_err"
  elif grep -q '^refuse ' "$TMP/bisect.calls"; then
    bi_detail="dune ran under a refused slot: $(cat "$TMP/bisect.calls")"
  elif ! bi_clean; then bi_detail="the tree was left, or the checkout changed"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_slot"; else report 1 "$l_slot" "$bi_detail"; fi

  # (d) a test that outlives the cap is not a verdict either: exit 142, abort.
  l_cap="bisect: a cap's TIMEOUT in a step aborts the bisect with exit 142"
  bi_detail=
  bi_repo hang fail || bi_detail="fixture"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- --cap 3 "$bi_base" range @t
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 142 ] || ! grep -q '^result: ABORTED: the test run of .*TIMEOUT' <<<"$bi_out" ||
       [ "$(bi_verdicts)" != "bad good ABORT " ]; then
    bi_detail="exit $bi_rc: $bi_out $bi_err"
  elif ! grep -q 'exit code 128' "$bi_dir/git-bisect.out" 2>/dev/null &&
       ! grep -q 'exit code 128' <<<"$bi_err"; then
    bi_detail="git bisect run was not stopped by the step's 128: $(cat "$bi_dir/git-bisect.out") $bi_err"
  elif ! bi_clean; then bi_detail="the tree was left, or the checkout changed"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_cap"; else report 1 "$l_cap" "$bi_detail"; fi

  # (e) ends that do not hold here bisect nothing: exit 1, said why.
  l_ends="bisect: a GOOD end that fails here, or a BAD one that passes, bisects nothing (exit 1)"
  bi_detail=
  bi_repo pass pass || bi_detail="fixture"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- "$bi_base" range @t
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 1 ] || ! grep -q "^result: the BAD end ${bi_shas[1]} passes here" <<<"$bi_out" ||
       [ "$(bi_verdicts)" != "good " ] || [ -s "$bi_dir/bisect-log" ]; then
    bi_detail="BAD passing: exit $bi_rc: $bi_out"
  fi
  if [ -z "$bi_detail" ]; then
    bi_repo fail fail || bi_detail="fixture"
    bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- "${bi_shas[0]}" range @t
    if [ "$bi_rc" != 1 ] || ! grep -q "^result: the GOOD end ${bi_shas[0]} fails here" <<<"$bi_out" ||
       [ "$(bi_verdicts)" != "bad bad " ]; then
      bi_detail="GOOD failing: exit $bi_rc: $bi_out"
    fi
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_ends"; else report 1 "$l_ends" "$bi_detail"; fi

  # (f) refusals before any step: no backend, an unrelated GOOD, a misplaced
  # option -- exit 2, no dune, no bisect directory; and the tree variable is
  # refused by a subcommand that does not take it.
  l_usage="bisect: no backend, a GOOD that is not BAD's ancestor or a misplaced option is refused (exit 2) before any step"
  bi_detail=
  bi_repo fail || bi_detail="fixture"
  bi_before=$(ls "$bi_runs" 2>/dev/null | grep -c '^bisect-')
  for probe in "nobackend:OCANNL_BACKEND=:$bi_base range @t:passes OCANNL_BACKEND" \
                "badbackend:OCANNL_BACKEND=vulkan:$bi_base range @t:names no backend" \
                "unrelated:OCANNL_BACKEND=cc:range $bi_base @t:not an ancestor" \
                "misplaced:OCANNL_BACKEND=cc:$bi_base range @t --cap 9:belongs before"; do
    [ -z "$bi_detail" ] || break
    IFS=: read -r tag benv args want <<<"$probe"
    # shellcheck disable=SC2086  # the arguments are word-split on purpose
    bi_run "$benv" OCANNL_TOOL_FLEET_WORKER=none -- $args
    if [ "$bi_rc" != 2 ] || ! grep -q "$want" <<<"$bi_err" || [ -s "$TMP/bisect.calls" ]; then
      bi_detail="$tag: exit $bi_rc; stderr: $bi_err; calls: $(cat "$TMP/bisect.calls")"
    fi
  done
  if [ -z "$bi_detail" ] && [ "$(ls "$bi_runs" 2>/dev/null | grep -c '^bisect-')" != "$bi_before" ]; then
    bi_detail="a refused bisect left a bisect directory"
  fi
  if [ -z "$bi_detail" ]; then
    bi_tree_out=$(OCANNL_TOOL_TESTRUN_TREE=$TMP OCANNL_TOOL_TEST_RUNS=$bi_runs "$bi_root/tools/test-run.sh" list 2>&1)
    bi_tree_rc=$?
    [ "$bi_tree_rc" = 2 ] && grep -q 'does not take it' <<<"$bi_tree_out" ||
      bi_detail="list under OCANNL_TOOL_TESTRUN_TREE: exit $bi_tree_rc: $bi_tree_out"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_usage"; else report 1 "$l_usage" "$bi_detail"; fi

  # (g) first-parent by default: the commits of a merged branch are never
  # stepped on, though one of them is red.
  l_fp="bisect: first-parent by default never steps into a merged branch, and refuses a GOOD only a second parent reaches"
  bi_detail=
  bi_repo pass fail || bi_detail="fixture"
  if [ -z "$bi_detail" ]; then
    # range: base - pass - M(side: fail, pass) - fail(+broken runner)
    bi_git -C "$bi_root" checkout -q -b side master >/dev/null 2>&1
    printf 'fail\n' >"$bi_root/STATE"; bi_git -C "$bi_root" commit -q --no-verify -am side-fail >/dev/null 2>&1
    bi_side1=$(bi_git -C "$bi_root" rev-parse HEAD)
    printf 'pass\n' >"$bi_root/STATE"; bi_git -C "$bi_root" commit -q --no-verify -am side-pass >/dev/null 2>&1
    bi_side2=$(bi_git -C "$bi_root" rev-parse HEAD)
    bi_git -C "$bi_root" checkout -q -b fp "${bi_shas[0]}" >/dev/null 2>&1
    bi_git -C "$bi_root" merge -q --no-ff --no-edit side >/dev/null 2>&1
    printf 'pass\n' >"$bi_root/STATE"; bi_git -C "$bi_root" commit -q --no-verify --allow-empty -am merged >/dev/null 2>&1
    bi_merge=$(bi_git -C "$bi_root" rev-parse HEAD)
    printf 'fail\n' >"$bi_root/STATE"; bi_git -C "$bi_root" commit -q --no-verify -am fp-fail >/dev/null 2>&1
    bi_fpbad=$(bi_git -C "$bi_root" rev-parse HEAD)
    bi_git -C "$bi_root" checkout -q master >/dev/null 2>&1
    bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- "$bi_base" fp @t
    if [ "$bi_rc" != 0 ] || ! grep -q "^result: FIRST BAD COMMIT $bi_fpbad " <<<"$bi_out"; then
      bi_detail="exit $bi_rc: $bi_out $bi_err"
    elif grep -q "$bi_side1\|$bi_side2" "$bi_dir/steps.tsv"; then
      bi_detail="a merged branch's commit was stepped on: $(cat "$bi_dir/steps.tsv")"
    elif [ -z "$bi_merge" ]; then
      bi_detail="fixture: no merge"
    fi
    # A GOOD merged in through a second parent bounds nothing on the
    # first-parent chain: refused, unless the walk takes every parent.
    if [ -z "$bi_detail" ]; then
      bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- "$bi_side2" fp @t
      if [ "$bi_rc" != 2 ] || ! grep -q "not on BAD's first-parent chain" <<<"$bi_err" ||
         [ -s "$TMP/bisect.calls" ]; then
        bi_detail="a second-parent GOOD: exit $bi_rc: $bi_err"
      else
        bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- --no-first-parent "$bi_side2" fp @t
        [ "$bi_rc" = 0 ] && grep -q "^result: FIRST BAD COMMIT $bi_fpbad " <<<"$bi_out" ||
          bi_detail="a second-parent GOOD under --no-first-parent: exit $bi_rc: $bi_out $bi_err"
      fi
    fi
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_fp"; else report 1 "$l_fp" "$bi_detail"; fi

  # (i) a descendant a step's run leaves behind, holding the tree's lock,
  # finishes before git checks out the next commit: it reads its own commit.
  l_settle="bisect: a step's verdict waits for a descendant still holding the tree, so it never sees the next commit"
  bi_detail=
  rm -f "$TMP/bisect.escape"
  bi_repo escape1 escape2 fail fail || bi_detail="fixture"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- "$bi_base" range @t
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 0 ] || ! grep -q "^result: FIRST BAD COMMIT ${bi_shas[2]} fail\$" <<<"$bi_out"; then
    bi_detail="exit $bi_rc: $bi_out $bi_err"
  elif [ ! -s "$TMP/bisect.escape" ] || awk '$1 != $2 { bad = 1 } END { exit !bad }' "$TMP/bisect.escape"; then
    bi_detail="a descendant saw another commit's tree (its STATE, then what it read): $(cat "$TMP/bisect.escape" 2>&1)"
  elif ! bi_clean; then bi_detail="the tree was left, or the checkout changed"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_settle"; else report 1 "$l_settle" "$bi_detail"; fi

  # (j) a relative runs directory is resolved once, by the bisect: every
  # step's runs land beside the bisect's records and outlive the tree.
  l_rel="bisect: a relative OCANNL_TOOL_TEST_RUNS keeps every step's runs under the one directory it names"
  bi_detail=
  bi_repo pass fail || bi_detail="fixture"
  rm -rf "${TMP:?}/bisect-relruns"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none OCANNL_TOOL_TEST_RUNS=../bisect-relruns \
    -- "$bi_base" range @t
  bi_relruns=$(cd -P "$TMP/bisect-relruns" 2>/dev/null && pwd -P)
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 0 ] || [ -z "$bi_relruns" ] || [ "$(dirname "$bi_dir")" != "$bi_relruns" ]; then
    bi_detail="exit $bi_rc; bisect directory ${bi_dir:-?} (want one under $TMP/bisect-relruns): $bi_out $bi_err"
  else
    while IFS=$'\t' read -r _ _ _ b t; do
      for r in "$b" "$t"; do
        [ "$r" = - ] && continue
        [ "$(dirname "$r")" = "$bi_relruns" ] && [ -s "$r/exit" ] ||
          bi_detail="a step's run is not under $bi_relruns, or did not outlive the tree: $r"
      done
    done <"$bi_dir/steps.tsv"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_rel"; else report 1 "$l_rel" "$bi_detail"; fi

  # (k) a setup refusal leaves nothing behind: a dune-workspace above the
  # runs directory would root dune above the tree, and is refused before
  # any worktree is registered or bisect directory made.
  l_setup="bisect: a dune-workspace above the runs directory is refused (exit 2) leaving no worktree or bisect directory"
  bi_detail=
  bi_repo pass fail || bi_detail="fixture"
  mkdir -p "$TMP/bisect-wsanc/runs" && printf '(lang dune 3.18)\n' >"$TMP/bisect-wsanc/dune-workspace"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none \
    OCANNL_TOOL_TEST_RUNS=$TMP/bisect-wsanc/runs -- "$bi_base" range @t
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 2 ] || ! grep -q 'would root dune above' <<<"$bi_err" || [ -s "$TMP/bisect.calls" ]; then
    bi_detail="exit $bi_rc: $bi_out $bi_err"
  elif [ "$(bi_git -C "$bi_root" worktree list | grep -c '')" != 1 ] ||
       ls "$TMP/bisect-wsanc/runs" | grep -q '^bisect-'; then
    bi_detail="the refusal left a worktree or bisect directory: $(bi_git -C "$bi_root" worktree list; ls "$TMP/bisect-wsanc/runs")"
  fi
  rm -f "$TMP/bisect-wsanc/dune-workspace"
  if [ -z "$bi_detail" ]; then report 0 "$l_setup"; else report 1 "$l_setup" "$bi_detail"; fi

  # (l) a TERM that lands before a step has a process group to signal: during
  # setup (`git worktree add` slowed by a git wrapper), the bisect unwinds
  # leaving nothing; during an end's checkout, no step is started at all.
  # Either way the exit is 143.
  l_early="bisect: a TERM during setup or before a step starts unwinds with 143, starting nothing"
  bi_detail=
  bi_gitbin=$TMP/bisect-gitbin
  mkdir -p "$bi_gitbin"
  bi_realgit=$(command -v git)
  cat >"$bi_gitbin/git" <<'EOF'
#!/usr/bin/env bash
case " $* " in
  *" worktree add "*) [ "$BISECT_GIT_SLOW" != add ] || { : >"$BISECT_GIT_READY"; sleep 3; } ;;
  *" checkout --quiet --detach "*) [ "$BISECT_GIT_SLOW" != checkout ] || { : >"$BISECT_GIT_READY"; sleep 3; } ;;
esac
exec "$BISECT_REAL_GIT" "$@"
EOF
  chmod +x "$bi_gitbin/git"
  for slow in add checkout; do
    [ -z "$bi_detail" ] || break
    bi_repo pass fail || { bi_detail="fixture"; break; }
    rm -f "$TMP/bisect-git.ready"
    : >"$TMP/bisect.calls"
    bi_early_runs=$TMP/bisect-early-$slow
    env GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null OCANNL_TOOL_TEST_RUNS=$bi_early_runs \
      BISECT_CALLS=$TMP/bisect.calls BISECT_GIT_SLOW=$slow BISECT_GIT_READY=$TMP/bisect-git.ready \
      BISECT_REAL_GIT=$bi_realgit PATH=$bi_gitbin:$bi_bin:$PATH OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none \
      "$bi_root/tools/test-run.sh" bisect "$bi_base" range @t >"$TMP/bisect.out" 2>"$TMP/bisect.err" &
    bi_pid=$!
    for _ in $(seq 1 300); do [ -e "$TMP/bisect-git.ready" ] && break; sleep 0.1; done
    kill -TERM "$bi_pid" 2>/dev/null
    for _ in $(seq 1 300); do kill -0 "$bi_pid" 2>/dev/null || break; sleep 0.1; done
    if kill -0 "$bi_pid" 2>/dev/null; then
      kill -KILL "$bi_pid" 2>/dev/null
      bi_detail="$slow: the bisect outlived its TERM by 30s"
    fi
    wait "$bi_pid" 2>/dev/null
    bi_rc=$?
    [ -z "$bi_detail" ] || break
    if [ ! -e "$TMP/bisect-git.ready" ]; then
      bi_detail="$slow: the slowed git call never came: $(cat "$TMP/bisect.out" "$TMP/bisect.err")"
    elif [ "$bi_rc" != 143 ]; then
      bi_detail="$slow: exit $bi_rc: $(cat "$TMP/bisect.out" "$TMP/bisect.err")"
    elif [ -s "$TMP/bisect.calls" ]; then
      bi_detail="$slow: a step started after the TERM: $(cat "$TMP/bisect.calls")"
    elif [ "$(bi_git -C "$bi_root" worktree list | grep -c '')" != 1 ]; then
      bi_detail="$slow: a worktree was left registered: $(bi_git -C "$bi_root" worktree list)"
    elif [ "$slow" = add ] && ls "$bi_early_runs" 2>/dev/null | grep -q '^bisect-'; then
      bi_detail="add: the cancelled setup left its bisect directory: $(ls "$bi_early_runs")"
    elif [ "$slow" = checkout ] && ! grep -q '^result: CANCELLED (TERM)' "$TMP/bisect.out"; then
      bi_detail="checkout: $(cat "$TMP/bisect.out" "$TMP/bisect.err")"
    fi
  done
  if [ -z "$bi_detail" ]; then report 0 "$l_early"; else report 1 "$l_early" "$bi_detail"; fi

  # (h) a TERM to the bisect stops the step's run (its dune reaped), exits
  # 143 and removes the tree.
  l_cancel="bisect: TERM stops the running step, reaps its dune, exits 143 and removes the tree"
  bi_detail=
  bi_repo hold fail || bi_detail="fixture"
  rm -f "$TMP/bisect-hold.pid"
  if [ -z "$bi_detail" ]; then
    : >"$TMP/bisect.calls"
    env GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null OCANNL_TOOL_TEST_RUNS=$bi_runs \
      BISECT_CALLS=$TMP/bisect.calls BISECT_HOLD=$TMP/bisect-hold PATH=$bi_bin:$PATH \
      OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none \
      "$bi_root/tools/test-run.sh" bisect "$bi_base" range @t >"$TMP/bisect.out" 2>"$TMP/bisect.err" &
    bi_pid=$!
    for _ in $(seq 1 300); do [ -s "$TMP/bisect-hold.pid" ] && break; sleep 0.1; done
    bi_hold=$(cat "$TMP/bisect-hold.pid" 2>/dev/null)
    if [ -z "$bi_hold" ]; then
      bi_detail="the held step never started: $(cat "$TMP/bisect.out" "$TMP/bisect.err")"
      kill -TERM "$bi_pid" 2>/dev/null
    else
      kill -TERM "$bi_pid"
    fi
    for _ in $(seq 1 300); do kill -0 "$bi_pid" 2>/dev/null || break; sleep 0.1; done
    if kill -0 "$bi_pid" 2>/dev/null; then
      bi_detail="the bisect outlived its TERM by 30s"
      kill -KILL "$bi_pid" 2>/dev/null
    fi
    wait "$bi_pid" 2>/dev/null
    bi_rc=$?
    bi_out=$(cat "$TMP/bisect.out")
    bi_dir=$(sed -n 's/^bisect: \(\/.*\)$/\1/p' "$TMP/bisect.out" | head -n 1)
    if [ -n "$bi_detail" ]; then :
    elif [ "$bi_rc" != 143 ] || ! grep -q '^result: CANCELLED (TERM)' <<<"$bi_out"; then
      bi_detail="exit $bi_rc: $bi_out $(cat "$TMP/bisect.err")"
    elif case "$(pstate "$bi_hold")" in '' | Z*) false ;; esac; then
      bi_detail="the held dune $bi_hold survived the cancellation"
      kill -KILL "$bi_hold" 2>/dev/null
    elif ! bi_clean; then bi_detail="the tree was left, or the checkout changed"
    fi
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_cancel"; else report 1 "$l_cancel" "$bi_detail"; fi
fi
}
harness_leg_run '82: bisect #bisect' leg_82

# ---------------------------------------------------------------------------
# Leg 70: a leg that names no device still reads the fake (gh-ocannl-1108)
# ---------------------------------------------------------------------------
# Every leg above that faked a device went back through hermetic_probes. One
# that `unset` a probe instead would hand every later leg the box's real
# /dev/dxg, KFD topology or /dev/nvidiactl -- a leak no Mac or CI runner can
# show, having none of them, and on minix a lifecycle run took -j 4. So
# this leg runs last and names no probe. The shipping readers, sourced fresh,
# must name the absent defaults (the path, not merely its absence, so the
# check means the same on a box without the device) and answer "no device";
# and a GPU run through the real launch path must be neither capped nor told.
# The controls: the path check refuses a probe unset in a subshell, which is
# the leak; and an exported fixture pool caps the same hip run, so a probe
# that never reached the width code cannot pass the uncapped half.
# Outside the body: leg 86 reads it, to tell that this leg reported.
herm_label="a leg that names no device probe reads the harness's absent defaults: no device, no cap, no word"
leg_70() {
herm_detail=
herm_view() { # the shipping readers' answers: paths, then dxg:nvidia:pool
  (
    . "$JOBS_SRC"
    printf '%s|%s|%s|' "$(box_jobs_dxg_device)" "$(box_jobs_kfd_topology)" "$(box_jobs_nvidia_device)"
    box_jobs_dxg_host && printf 'dxg' || printf 'no-dxg'
    box_jobs_native_nvidia_host && printf ':nvidia' || printf ':no-nvidia'
    printf ':pool=%s\n' "$(box_jobs_sdma_pool)"
  )
}
herm_want="$probe_dxg_default|$probe_kfd_default|$probe_nvidia_default|no-dxg:no-nvidia:pool="
for p in "$probe_dxg_default" "$probe_kfd_default" "$probe_nvidia_default"; do
  [ ! -e "$p" ] || { herm_detail="the absent default $p exists"; break; }
done
if [ -z "$herm_detail" ]; then
  herm_seen=$(herm_view)
  [ "$herm_seen" = "$herm_want" ] ||
    herm_detail="the shipping readers answer '$herm_seen', not '$herm_want'"
fi
for backend in hip cuda; do
  [ -z "$herm_detail" ] || break
  export OCANNL_BACKEND=$backend
  argv_probe "herm-$backend" run build @cheap
  unset OCANNL_BACKEND
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "build @cheap" ] ||
     grep -qi 'cap\|sdma\|nvidia\|dxg' <<<"$argv_err"; then
    herm_detail="$backend: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"
  fi
done
# Control: the leak itself -- a probe unset -- is what the path check refuses.
if [ -z "$herm_detail" ]; then
  herm_leak=$(unset OCANNL_TOOL_KFD_TOPOLOGY; herm_view)
  [ "$herm_leak" != "$herm_want" ] ||
    herm_detail="control: with the KFD probe unset the readers still answer '$herm_leak', so the path check cannot see a leak"
fi
# Control: the probe the uncapped half relied on does reach the width -- a
# fixture pool exported the way a leg opts in caps the same hip run.
if [ -z "$herm_detail" ]; then
  export OCANNL_TOOL_KFD_TOPOLOGY=$kfd_small OCANNL_BACKEND=hip
  argv_probe herm-optin run build @cheap
  unset OCANNL_BACKEND
  hermetic_probes
  herm_cap=$(. "$JOBS_SRC"; printf '%s' "$BOX_JOBS_SDMA_SLOT_CAP")
  [ -n "$herm_cap" ] && [ "$argv_calls" = "build -j $herm_cap @cheap" ] ||
    herm_detail="control: hip on an exported small pool was not capped at -j ${herm_cap:-?}: calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"
fi
if [ -z "$herm_detail" ]; then
  report 0 "$herm_label"
else
  report 1 "$herm_label" "$herm_detail"
fi
}
harness_leg_run '70: a leg that names no device reads the absent defaults #width' leg_70

# ---------------------------------------------------------------------------
# Leg 86: a structural failure ends only its own leg (gh-ocannl-1374)
# ---------------------------------------------------------------------------
# Each leg's body is a function, and a check a leg cannot get past -- here,
# leg 79's mutant that must cut the credential scrub out of test-run.sh -- ends
# it through harness_abandon, not `exit`: the legs after it still run and are
# reported. A copy of this harness runs against a copy of test-run.sh whose
# scrub line is spelled so the mutant's pattern no longer matches it (the drift
# the check exists for), with leg 79 and the file's last leg, 70, selected. The
# copy must report leg 79 abandoned, report leg 70 and reach its footer. The
# negative control is the same copy with that check's `return` put back as the
# `exit 2` it was: the copy then aborts inside leg 79, saying so, and leg 70
# never reports, so the oracle tells the two apart.
self_label="a leg's structural failure ends that leg alone: leg 79's broken mutant is abandoned, and leg 70 after it still runs and reports"
self_nc_label="negative control: the same check spelled as an exit aborts the harness inside leg 79, and leg 70 never reports"
# The name leg 79 runs under, read from its own call below rather than restated.
self_leg79=$(sed -n "s/^harness_leg_run '\\(79: [^']*\\)' leg_79\$/\\1/p" "$HERE/test-test-run.sh")
self_run() { # tag: the copy under $TMP/self-<tag>, run on legs 79 and 70; sets self_rc, self_out
  local root=$TMP/self-$1
  self_rc=0
  (
    unset OCANNL_TOOL_HARNESS_STRICT OCANNL_TOOL_HARNESS_EXPECTED_SKIPS \
      OCANNL_TOOL_HARNESS_NAME OCANNL_TOOL_HARNESS_SKIP_RECORD
    mkdir -p "$root/tmp"
    TMPDIR=$root/tmp bash "$root/tools/test-test-run.sh" --only '^(79|70):'
  ) >"$root.out" 2>&1 || self_rc=$?
  self_out=$(cat "$root.out")
}
self_stage() { # tag: a copy of tools/ and scripts/ with the scrub line respelled
  local root=$TMP/self-$1
  mkdir -p "$root" && cp -R "$HERE" "$root/tools" && cp -R "$HERE/../scripts" "$root/scripts" &&
    awk '$0 == "  eval \"$(credential_env_scrub_text)\" ||" { n++; print "  eval \"$(credential_env_scrub_text)\"  ||"; next }
         { print } END { exit n == 1 ? 0 : 1 }' "$SRC" >"$root/tools/test-run.sh" &&
    bash -n "$root/tools/test-run.sh"
}
leg_86() {
[ -n "$self_leg79" ] || { harness_abandon "no call runs leg 79 as leg_79 in $HERE/test-test-run.sh"; return; }
local abandoned="FAIL  $self_leg79: abandoned, its later checks did not run"
self_stage ship || { harness_abandon "the harness copy, or its respelled scrub line, could not be staged"; return; }
self_run ship
if [ "$self_rc" = 1 ] && grep -qxF "$abandoned" <<<"$self_out" &&
   { grep -qxF "PASS  $herm_label" <<<"$self_out" || grep -qxF "FAIL  $herm_label" <<<"$self_out"; } &&
   grep -qF "ran 2 leg(s):" <<<"$self_out" && ! grep -qF "aborted before finish" <<<"$self_out"; then
  report 0 "$self_label"
else
  report 1 "$self_label" "exit $self_rc (want 1): $self_out"
fi
self_stage exit || { harness_abandon "the negative control's copy could not be staged"; return; }
awk '$0 == "  harness_abandon \"the credential scrub could not be cut out of $SRC; the mutant is broken\"" {
       print; getline; if ($0 == "  return") { n++; print "  exit 2"; next } }
     { print } END { exit n == 1 ? 0 : 1 }' "$HERE/test-test-run.sh" >"$TMP/self-exit/tools/test-test-run.sh" ||
  { harness_abandon "the negative control's exit could not be spliced into the harness copy"; return; }
self_run exit
if [ "$self_rc" = 2 ] && grep -qF "aborted before finish (rc=2) inside leg: $self_leg79" <<<"$self_out" &&
   ! grep -qF "$herm_label" <<<"$self_out"; then
  report 0 "$self_nc_label"
else
  report 1 "$self_nc_label" "exit $self_rc (want 2): $self_out"
fi
}
harness_leg_run '86: a structural failure ends only its own leg #harness' leg_86

finish
