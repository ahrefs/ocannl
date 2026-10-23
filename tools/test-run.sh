#!/usr/bin/env bash
# Run dune test targets (`dune runtest ...`, `dune build @slow`, directory
# aliases) with a correct, compact, machine-readable verdict -- so that nobody,
# and especially no coding agent, hand-rolls shell around dune again.
#
# Each ingredient answers a failure mode that has actually cost a session:
#   - Unpiped status: `dune runtest 2>&1 | tail` reports TAIL's status (no
#     pipefail), so a promotion diff reads as a green run.
#   - No `status` variable anywhere: in zsh it is a read-only alias for `$?`,
#     so ad-hoc wrappers that use it die before printing their sentinel and a
#     green suite looks failed.
#   - A wall-clock cap: a hung run (macOS XProtect stalling a fresh exe, a
#     wedged backend) otherwise strands whatever is waiting on it. And one per
#     TEST, with a device query before a GPU batch starts: a faulted GPU makes
#     every later test spin, which the run's cap alone let run for 32 minutes
#     (gh-ocannl-1211; see tools/per-test-cap.sh and tools/device-probe.sh).
#   - A verdict FILE, not a process probe: waiter loops on `pgrep -x dune`
#     match the editor's immortal `dune ocaml-merlin` daemons and spin forever
#     (one PR review accumulated ten such stranded shells); `kill -0 $pid` can
#     latch onto a recycled pid, and answers yes for a ZOMBIE -- which is why
#     every liveness question here reads process state too (proc_alive,
#     group_alive). `wait` here polls for the verdict file with a hard timeout,
#     so it cannot strand.
#   - No credential in dune's environment: dune records every spawned
#     process's environment in the worktree's `_build/trace.csexp`, so a
#     session's GH_TOKEN sat in every checkout an agent greps (gh-ocannl-1280).
#     The deny-list is tools/credential-env.sh, shared with the sweep and
#     machine-verify; everything else (opam, PATH, OCANNL_*) passes through.
#
# Usage:
#   tools/test-run.sh run   [--cap N] [--test-cap N] [--expect-backend B]... [DUNE ARGS...]
#                                                      # foreground; digest; dune's status (2: refused)
#   tools/test-run.sh start [--cap N] [--test-cap N] [--expect-backend B]... [DUNE ARGS...]
#                                                      # detached; survives the session
#   tools/test-run.sh plan  [--cap N] [--test-cap N] [--expect-backend B]... [DUNE ARGS...]
#                                                      # what `run` would inject and take; runs nothing
#   tools/test-run.sh repeat [--cap N] [--test-cap N] [--alone] N [DUNE ARGS...]
#                                                      # compare N isolated runs
#   tools/test-run.sh repeat [--cap N] [--test-cap N] [--alone] --direct N @DIR/KIND-NAME... [-j N]
#                                                      # build once, replay the tests N times
#   tools/test-run.sh fresh [--cap N] [--test-cap N] [--expect-backend B]... @DIR/KIND-NAME... [-j N]
#                                                      # one fresh, timed run of named tests
#   tools/test-run.sh bisect [--cap N] [--test-cap N] [--build TARGET]... [--no-first-parent]
#                            GOOD BAD TARGET...      # the first commit that turns TARGETs red
#                                                    # (tools/bisect-steps.sh says how)
#   tools/test-run.sh status [RUN|last]                # one-shot, never blocks
#   tools/test-run.sh wait   [RUN|last] [--timeout N]  # bounded; exits with the run's status
#   tools/test-run.sh stop   [RUN|last]                # TERM the run's process group
#   tools/test-run.sh list                             # recent runs and their states
#   tools/test-run.sh paths FIELD [RUN|last]          # one absolute path (see below)
#   tools/test-run.sh lock-status [RUN|last]          # idle/held; exits 0/3 (2 error)
#   tools/test-run.sh idle                             # 0 idle, 3 locked, 2 unreadable
#                                                     # snapshot, not a reservation
#   tools/test-run.sh affected [--referenced-only] --since BASE|REV|REV..REV|--files PATH...
#                                                      # the per-test aliases a change reaches,
#                                                      # link-only ones last; runs none
#                                                      # (tools/affected-tests.sh)
#
# Read-only queries: `paths run [RUN|last]` resolves a recorded run (default last).
# `paths worktree|runs|lock|owner|last [RUN|last]` without a run names THIS
# script's worktree and state root, even before its first launch; with a run it
# names that run's recorded worktree/state (including legacy in-tree locks).
# Absolute run references do not inspect the unrelated current state root;
# current relative paths are interpreted at this worktree, ignoring CDPATH.
# `paths last` names the pointer FILE, not the run it points to. Output is one
# raw absolute path and a newline, never shell code. Query paths must contain
# no CR or LF (refused with exit 2); the launch metadata format is unchanged.
# Exit 2 also means invalid input
# or unavailable metadata. `lock-status` without a run probes THIS worktree's
# current and legacy locks; with a run, only its recorded lock. It prints idle
# (0) or held (3); an unreadable lock is an error (2), never idle. These are
# snapshots, not reservations or evidence that a particular process is alive.
# Neither query creates state, publishes pointers, or launches a toolchain.
# Missing state directories are supported. Parent traversal must resolve through
# the launch shell; unresolved missing/.. is refused (2). Native drive paths may
# resolve missing/.. even where POSIX spellings do not. Queries follow that shell.
# A missing suffix is refused if symlink-parent traversal gives logical cd and
# physical mkdir different existing prefixes. Already resolved roots still work.
#
# `repeat` runs each iteration through dune in a freshly cleaned, cache-disabled
# build context, keeps its separate stdout/stderr and exit status, and compares
# every pair. `--alone` adds `-j 1`,
# so no sibling dune action overlaps the selected target. Unlike `run`/`start`,
# `repeat` never injects the GPU `-j` cap (see width_cap below): an isolation tool
# runs at the width it is given, so on a capped box pass `--alone` or `-j` yourself. Its cap is per
# iteration; N must be at least 2. An stdout/status difference is red (exit 1
# when dune itself stayed green); stderr-only drift is reported distinctly but
# is not red. Any red dune iteration keeps a nonzero dune status. It has `run`'s
# two bounds (gh-ocannl-1252), decided once by the same resolution of the batch:
# the per-test cap beside each iteration's dune, and on a GPU batch the device
# probe before each iteration; a test past the cap, or a device that does not
# answer, ends the whole set (TIMEOUT 142, DEVICE UNHEALTHY 69), since every
# later iteration would hang on the same wedged device.
#
# The clean context per iteration is the price of re-executing through dune
# (gh-ocannl-1252 asked whether it is needed): dune 3.24 re-runs a
# target-producing rule only when its dependencies change, and the rebuild half
# of the price cannot come from dune's shared cache. Measured on mac-studio
# (dune 3.24.2): under the default `--cache=enabled-except-user-rules` dune
# stores nothing for this workspace's rules, compiles included (a fresh
# DUNE_CACHE_ROOT held 0 files after building arrayjit/lib), and a
# `runtest-hello_world_op` iteration took 21-23s at -j 4 under it and under
# `--cache=disabled` alike; `--cache=enabled` stores everything, the test runs
# too, which restores a test's output instead of re-running it. So `repeat`
# keeps `--cache=disabled`, which says what it means.
#
# `repeat --direct` (gh-ocannl-1377) pays the rebuild once. Its operands are
# `fresh`'s (per-test aliases and -j N). A build phase -- a clean iteration
# into the set's own build directory, with a trace -- builds the tests and runs
# them once; the plan then reads from that trace which processes are each named
# test's own action (the rule `fresh` times a test by), asks dune for their
# rules (`dune rules`: the expanded action, its directory, programs, arguments,
# redirections and variables) and keeps a snapshot of their targets as dune's
# own run left them. Each iteration then replays those rules in the built tree
# (tools/replay-rules.sh: in the environment `dune exec --no-build` reports for
# the build context, plus the per-action TMPDIR and BUILD_PATH_PREFIX_MAP dune
# adds) and runs dune's `--force` over the aliases, which re-runs their own
# actions -- the golden diffs, a run-and-diff rule, an inline-test runner --
# and rebuilds nothing. A pair compares dune's streams and status as above,
# plus the replayed rules' outputs, statuses and printed stdout (red) and
# stderr (diagnostic). The first replay is also compared with the snapshot:
# replays that agree with each other but not with dune's own run are REPLAY
# DIVERGED (exit 1), a replay that did not reproduce the rules' conditions,
# never a stable test. Refused before any iteration (exit 2, DIRECT REFUSED or
# NOT RUN): a rule dune ran in a sandbox, an action outside the replayed forms,
# a named test none of whose processes ran; a build phase that failed before
# running a named test is BUILD FAILED with its status. The per-test cap
# times the replay's processes as it times dune's, and a GPU batch's device
# probe precedes the build phase and every iteration. What a direct iteration
# does NOT reproduce, beside a sandbox: dune's own scheduling of the named
# tests beside each other (a replay runs them one after another, so `--alone`
# narrows only the build phase and dune's --force), and a fresh build tree (a
# file a test leaves beside its targets is there for the next iteration, as it
# would be for a second dune run).
#
# `fresh` runs the named per-test aliases once, as `run` does (cap, slot, width,
# digest), but fresh: dune is given a build directory of the run's own
# (`<run>/build`, removed when the run ends), `--force` and `--cache=disabled`,
# so every action the aliases reach executes -- and none of it lands in the
# worktree's `_build`, so nothing there is cleaned, invalidated or rebuilt.
# (The runner's own helpers -- the backend readers on a fleet box, a GPU
# batch's device probe -- still build there, as they do for `run`, which only
# brings those helpers up to date.)
# The price is compiling the tests' closure in that directory (some ten
# seconds at -j 4 on mac-studio). There is no in-place alternative: dune 3.24 re-runs a
# target-producing rule -- a `(test)` stanza's run, a `(target x.actual)`
# producer -- only when its rule or dependencies change. `--force` re-runs
# alias actions alone, and deleting the target from `_build` does NOT re-run
# its rule: dune trusts its database, and the alias's `diff` then reads a
# missing file and queues the deletion of the `.expected` as a promotion.
# The operands are per-test aliases only, `@DIR/runtest-NAME`, `@DIR/slow-NAME`
# or `@DIR/train-NAME` (`@@` for DIR alone), each named once, plus `-j N`;
# anything else is refused before dune runs. The digest adds each named test's
# wall time from the run's dune trace (`timings` in the run directory): the
# seconds of the processes that are its action -- the alias's own action
# (`@DIR/KIND-NAME`, or one in a subdirectory for a recursive `@`) or the
# producer of an output in DIR named after the test, `NAME.<EXT>` with any
# extension but a compile artifact's (`NAME.exe.output` for a `(test)`
# stanza, `NAME.actual`, `NAME.ll.actual`, `NAME.coverage`; any target of the
# process counts), a final `-VARIANT` of NAME read as `.VARIANT` too
# (`runtest-x-extension` writes `x.extension.actual`); failing both, the runs
# of `NAME.exe` in DIR that no other named test claimed -- summed,
# with their count, exits and which rule matched (`action` or `program`). A
# named test with no such process is NOT RUN: no process of the run is its
# action, so there is no fresh time to report, and a run
# that was otherwise green exits 2 under that verdict, naming it; one whose
# times could not be read is UNTIMED, exit 1. Times are wall-clock under the
# box's load and the batch's own parallelism -- a test overlapping the
# closure's compiles or another named test shares the machine with them;
# name one test, or pass -j 1, to time it alone. A red fresh run cannot be
# promoted from the worktree (its build directory is gone): rerun it with `run`.
#
# Exit codes: `run` and `wait` exit with dune's status, with TWO substitutions.
# A run the fleet's run-time slot refused (see plan_slot below) exits 75 under
# the verdict `SLOT REFUSED`: nothing ran, and the fleet's own line says why.
# And an invocation dune's own CLI refused -- an unknown option or subcommand, a
# missing or malformed operand -- exits 2, this script's usage code, under the
# verdict `INVOCATION REFUSED`. dune prints `dune: <complaint>` then `Usage:
# dune ...`, exits 1 and runs nothing, and that 1 is not a test result: read as
# one it sends a session debugging code that was never compiled (staging#652).
# The recorded status (the `exit:` sentinel, `status`, `list`) stays dune's
# own 1; the substitution is the caller-facing one, so a caller branching on
# the status needs no log parsing to tell "fix the command line" from "read
# the failures". `repeat` preserves the
# first nonzero dune status, or exits 1 when otherwise-green stdout/statuses
# differ, and exits 2 with the same verdict when its first iteration is refused
# (there is nothing to repeat) (142 = the cap expired -- for `run`/`start`
# the run's or a test's, for `repeat` an iteration's or a test's -- 69 = a GPU
# batch's device probe found the device wedged and dune never started, 143/130 = cancelled, 137 = SIGKILLed, 124 = `wait` itself
# timed out; dune never reaches those on its own). `status` exits 0 finished, 3 still running
# (or verdict publication in flight), 1 died without a verdict. Usage and lock
# refusals exit 2 -- every one of them, including EITHER misplacement of this
# script's own options: before the subcommand (`--cap 5400 run ...`) and after
# the dune arguments (`run build @alias --cap 900`). Options go in between:
# `run --cap 5400 build @alias`. The second misplacement is also refused BEFORE
# dune is spawned, by the scan below, because for those two words the correct
# order is known and the message can name it; every other refusal is dune's
# to make and is recognised afterwards in the digest, quoting dune's complaint.
# But a launcher that swallows the status -- e.g. an agent harness's background
# mode reporting its own wrapper's 0 -- turns that refusal into a false green,
# so read the message, not only the code: a usage error runs no tests at all.
# A red run whose log carries, at a line's start, the line a GPU backend's
# device-init deadline writes before its process dies by SIGALRM
# (gh-ocannl-1281) keeps dune's status under the verdict `WEDGED DEVICE`, and
# the digest quotes that line: the device, not the tests, is what to read.
# And a run whose batch does not hold a backend its `--expect-backend` names
# exits 2 under the verdict `BACKEND REFUSED`, dune never started (see
# plan_backend_check: a GPU box's batch that resolved to cc only says so even
# without the option). A `dune build` naming an alias dune does not know
# (`@test/operations/runtest-misspelt`) exits 2 under the verdict `ALIAS
# REFUSED`, asked of dune before the batch's backends, the slot or the device
# probe, and dune's build never started (see plan_aliases).
#
# Everything after the options is dune's argv, verbatim (default: `runtest`):
#   tools/test-run.sh run runtest test/operations
#   tools/test-run.sh run build @slow
#   OCANNL_BACKEND=cuda tools/test-run.sh run runtest
#
# Prefer `run`. In an agent harness, launch `run` through the harness's own
# background mode and let the harness notify on exit -- that already removes
# every reason to write a waiter. `start`/`wait` exist only for a run that must
# outlive the launching session. The cap defaults to $OCANNL_TOOL_TEST_CAP or 3600s;
# `--cap 0` disables it (then supply your own bound). `--expect-backend B`
# (repeatable) refuses a batch that does not hold backend B, before dune starts
# (BACKEND REFUSED, exit 2): pass the GPU backend a GPU-box leg is for, since a
# batch there that resolves to cc alone is green and tests nothing on the GPU
# (plan_backend_check, gh-ocannl-1433). `--test-cap N` bounds each
# process dune starts (each test, in a test batch) and ENDS the run when one
# outlives it, as the run's own cap would; N (or $OCANNL_TOOL_PER_TEST_CAP)
# applies to any batch, 0 lifts it, and by default only a GPU batch not
# reaching `@slow` gets 1500s (see plan_test_cap, test_cap_watch). A GPU batch
# also first asks each of its GPU backends for a fresh device query, and
# refuses to start dune when one does not answer (DEVICE UNHEALTHY, exit 69;
# see tools/device-probe.sh). `repeat` takes both bounds too (see above).
#
# One run at a time per worktree, enforced with an flock: a second `run`/`start`/`repeat`
# refuses loudly, pointing at the active run, instead of queueing behind dune's
# own lock -- "I lost track of a run so I started another" is exactly the spiral
# this script exists to prevent. `stop` the active run if it is truly stale.
# The lock, its owner pointer and the `last` pointer live under the runs
# directory ($OCANNL_TOOL_TEST_RUNS, default ~/.ocannl-test-runs), keyed by the
# worktree's path -- never in the worktree, which a run leaves exactly as it
# found it. A run is two processes: this launching shell, which takes the lock
# and publishes the run, and a perl supervisor that inherits the lock, caps
# and signals its child -- this script's `_resolve`, which decides the width
# and the fleet slot, runs dune, then records dune's promotion list -- and
# records the verdict (see supervisor_perl).
#
# A run directory also records WHICH source ran, as of the launch: `head` (the
# checkout's HEAD commit) and `dirty` (its `git status --porcelain`, empty when
# clean). Both are optional -- absent outside a Git checkout -- and neither
# says anything about an edit made after the launch, or about ignored files and
# the environment, which are configuration rather than source (see
# record_checkout). The digest prints them as its `source:` line: the commit,
# then `(clean)` or `+ N uncommitted paths`. At the run's end it records
# `promotions`: dune's own promotion list, which the digest reports in place
# of reading the log for diffs. `promotion-files` keeps corrected contents and
# their originals for `tools/promote.sh --from-run RUN`, even after a later build
# replaces dune's list (see record_promotions). And `slowest`, which the digest
# also prints: the five longest processes in the batch's own dune trace (a
# trace file of the run's, never the worktree's `_build/trace.csexp`), with
# their seconds, recorded on every exit path, the caps' included (see
# explicit_trace_file).
#
# Windows: run it from Git Bash, whose MSYS perl carries the flock and the cap.
# Best-effort even there -- process-group kills may only reach dune itself, not
# its compiler children. A Cygwin bash ships no perl by default, so the
# preflight below refuses it outright rather than letting the lock misreport
# what happened (gh-ocannl-662).

set -u

die() { echo "test-run: $*" >&2; exit 2; }

normalize_cap() { # [VARIABLE OPTION]: `cap` and `--cap` by default
  local var=${1:-cap} opt=${2:---cap} v
  v=${!var}
  # A mistyped cap would reach perl's numeric compare as 0 and silently
  # disable the alarm -- the one property this script must never lose.
  case $v in '' | *[!0-9]*) die "$opt must be a nonnegative integer of seconds (0 disables)" ;; esac
  # Bounded BEFORE arithmetic: an oversized value would wrap in bash's signed
  # arithmetic, and perl's alarm range is signed too. Nine digits is ~31 years.
  [ ${#v} -le 9 ] || die "$opt too large (max 9 digits)"
  # Leading zeroes are accepted as decimal rather than reaching bash as octal.
  printf -v "$var" '%s' "$(( 10#$v ))"
}

reject_misplaced_options() {
  # dune has no `--cap`, `--test-cap`, `--alone` or `--direct`, so any of them among the arguments
  # FORWARDED to dune is this script's options written on the wrong side of the
  # target -- never something a caller could mean. Left to reach dune it exits 1
  # with `dune: unknown option`, having built nothing; the digest would now
  # report that as INVOCATION REFUSED, but only this guard knows the correct
  # order, so it refuses first and names it, without spawning dune at all.
  # The scan stops at dune's own `--`, past which the words belong to an
  # executable (`run exec foo.exe -- --cap 900` passes its cap to foo.exe).
  for arg do
    case $arg in
      --) return 0 ;;
      --cap | --cap=* | --alone | --alone=* | --test-cap | --test-cap=* | --direct | --direct=* | \
        --expect-backend | --expect-backend=*)
        case $sub in
          repeat) form='repeat [--cap N] [--test-cap N] [--alone] [--direct] N [DUNE ARGS...]' ;;
          fresh) form='fresh [--cap N] [--test-cap N] [--expect-backend B]... @DIR/KIND-NAME... [-j N]' ;;
          bisect) form=$bisect_usage ;;
          *) form="$sub [--cap N] [--test-cap N] [--expect-backend B]... [DUNE ARGS...]" ;;
        esac
        die "$arg belongs before the dune arguments, not after them:
  tools/test-run.sh $form
  dune has no such option, so this would have exited 1 having run nothing." ;;
    esac
  done
}

# The GPU width cap (gh-ocannl-983, gh-ocannl-1033). A box that reaches its GPU
# through WSL2's `/dev/dxg` bridge overflows the bridge's VM-bus ring when the
# suite's test executables hold the device at once, and the runtime reports the
# lost messages as device/binary/stream-creation refusals -- a red suite in
# exactly the stanzas a real backend regression lands in. tools/sweep.sh has
# capped its unit there since 2026-09-05; every other way in ran at dune's
# default width, and an hour of box time plus a misleading bisect went into
# rediscovering the cap.
#
# A native boot has limits of its own, and the fleet runs several correctness
# batches at once on each native GPU box (lukstafi/ludics-lite#316 and ludics-lite#344),
# measured at a width per batch: minix's small SDMA queue pool is device-wide,
# so hip batches at dune's default can drain it between them, and tuf's and
# rog-nv's slots were measured at -j 8, not at their 16 and 24 cores. rog-nv's
# four slots admit only two batches to its GPU (the fleet's GPU tokens), so the
# other two are CPU batches', and those were only ever measured at -j 8 too
# (gh-ocannl-1065). A slot count that holds only while every caller remembers
# the width is a hazard, not a limit, so the native widths are injected the
# same way. tools/box-jobs.sh decides
# which hazard, if any, this box and backend meet (box_jobs_local_hazard) and
# owns every number.
#
# So a `run`/`start` that expressed NO width at all, on such a box, gets the
# tightest cap any backend the batch can hold meets here injected, and is told
# so -- loudly, because the alternative reading of what follows is a backend
# regression. The "runs dune as given" contract is untouched wherever the
# caller named a width: an explicit `-j`/`--jobs` (in any of dune's spellings,
# and any abbreviation of the long one) is always honored, and only says that
# the cap exists.
#
# The backends are RESOLVED, by tools/batch-backends.sh (gh-ocannl-1066): the
# test configurations through the same Utils resolution a test run makes, the
# stanzas the argv reaches that name a backend by marker, and the argv itself --
# one resolution, which the fleet slot's --cpu/--gpu reads too. It used to be
# OCANNL_BACKEND alone, on the ground that resolving a config file in shell
# could silently halve a legitimate run's width; the resolution is OCaml now,
# and an unreadable answer is every backend, which costs minutes of width, not
# a red suite. Resolving means building the two readers, so it happens only
# where something depends on it: a box where some backend meets a cap, or a
# fleet box's slot (plan_batch). It is the run's first phase (`_resolve`),
# run by the supervisor as the child that later runs dune.
explicit_jobs() { # dune argv; 0 iff it names a width before dune's own `--`
  for arg do
    case $arg in
      # Past the separator the words belong to an executable dune runs.
      --) return 1 ;;
      # `-j`, `-j4`, `-jauto`, `--jobs 4`, `--jobs=4` -- and any unambiguous
      # abbreviation cmdliner accepts for the long form. Erring towards "the
      # caller named a width" is the safe direction: it only declines to inject.
      -j | -j?* | --j?*) return 0 ;;
    esac
  done
  return 1
}

# 0 iff the dune argv chooses its own diff presentation before dune's own `--`
# (the digest's reading of a log depends on it; see `digest`): `--diff-command`
# in either spelling, or an abbreviation cmdliner accepts -- `--dif` already
# names no other option. Erring towards "chosen" is the safe direction: it only
# makes the digest say a promotion is possible rather than absent.
explicit_diff_command() { # dune argv
  for arg do
    case $arg in
      --) return 1 ;;
      --dif*) return 0 ;;
    esac
  done
  return 1
}

# What each hazard is, for the announcements: the condition that was found,
# and why dune's default width is wrong under it.
hazard_name() { # <hazard>; a noun phrase for the host, and the issue behind its cap
  case $1 in
    dxg) printf 'dxg host (gh-ocannl-983)' ;;
    sdma) printf 'small-SDMA-pool host (gh-ocannl-1033)' ;;
    wide-sdma) printf 'native AMD GPU host (lukstafi/ludics-lite#344)' ;;
    nvidia) printf 'native NVIDIA host (gh-ocannl-1033)' ;;
    nvidia-cpu) printf 'the fleet'"'"'s rog-nv-linux (gh-ocannl-1065)' ;;
  esac
}
hazard_found() { # <hazard> <backend> <why the batch holds it>
  case $1 in
    dxg) printf 'This box reaches its GPU through the WSL2
  %s bridge, and this batch can hold %s, which holds that device (%s)' "$(box_jobs_dxg_device)" "$2" "$3" ;;
    sdma) printf 'This box'"'"'s GPU has a small SDMA (copy-engine)
  queue pool, %s allocatable queues for the whole device per its KFD topology
  (%s), and this batch can hold %s (%s), whose every process that copies takes one' \
      "$(box_jobs_sdma_pool)" "$(box_jobs_kfd_topology)" "$2" "$3" ;;
    wide-sdma) printf 'This box'"'"'s AMD GPU reports %s allocatable SDMA
  queues per its KFD topology (%s), and this batch can hold %s (%s), which holds it' \
      "$(box_jobs_sdma_pool)" "$(box_jobs_kfd_topology)" "$2" "$3" ;;
    nvidia) printf 'This is a native NVIDIA boot
  (%s), and this batch can hold %s (%s), which holds its GPU' "$(box_jobs_nvidia_device)" "$2" "$3" ;;
    nvidia-cpu) printf 'This is the fleet'"'"'s rog-nv-linux, natively booted
  (%s), and this batch can hold %s (%s), which shares its correctness slots with the
  batches that hold the GPU' "$(box_jobs_nvidia_device)" "$2" "$3" ;;
  esac
}
hazard_why() { # <hazard>
  case $1 in
    dxg) printf 'At dune'"'"'s default
  width the bridge'"'"'s VM-bus ring overflows, and the suite comes back red in the
  same stanzas a real backend regression lands in (gh-ocannl-983). The cap lives
  in tools/box-jobs.sh, shared with tools/sweep.sh; the refusal signature and the
  recovery are the dxg bullet of
  docs/agent-notes/build-and-test.md#gpu-boxes-job-caps-and-runtime-refusals.' ;;
    sdma) printf 'At dune'"'"'s default width
  the pool can run out (kernel: `No more SDMA queue to allocate`), and a stanza
  aborts in ROCr like a backend regression (gh-ocannl-1029); -j %s keeps the
  fleet'"'"'s %s correctness slots on this box within the %s hip-width measured
  clean between them (lukstafi/ludics-lite#344). The cap lives in
  tools/box-jobs.sh; the signature is in the native-boot bullets of
  docs/agent-notes/build-and-test.md#gpu-boxes-job-caps-and-runtime-refusals.' \
      "$BOX_JOBS_SDMA_SLOT_CAP" "$BOX_JOBS_SDMA_SLOTS" "$BOX_JOBS_SDMA_BUDGET" ;;
    wide-sdma) printf 'The fleet'"'"'s %s correctness
  slots on this box were measured at -j %s each, %s hip-width between them;
  uncapped, a batch here runs at dune'"'"'s default width, and that many at once
  were never measured (lukstafi/ludics-lite#344). The cap lives in
  tools/box-jobs.sh; the evidence is in the native-boot bullets of
  docs/agent-notes/build-and-test.md#gpu-boxes-job-caps-and-runtime-refusals.' \
      "$BOX_JOBS_WIDE_SDMA_SLOTS" "$BOX_JOBS_WIDE_SDMA_SLOT_CAP" "$BOX_JOBS_WIDE_SDMA_BUDGET" ;;
    nvidia) printf 'The fleet runs %s GPU tokens of
  %s correctness slots on this box (lukstafi/ludics-lite#391): the %s cuda
  batches the tokens admit were measured at -j %s each, and three or four such
  batches at once hit a CUDA_ERROR_OUT_OF_MEMORY (lukstafi/ludics-lite#316 and ludics-lite#344); two
  batches at dune'"'"'s default width were never measured (gh-ocannl-1033). The
  cap lives in tools/box-jobs.sh; the evidence is in the native-boot bullets of
  docs/agent-notes/build-and-test.md#gpu-boxes-job-caps-and-runtime-refusals.' \
      "$BOX_JOBS_NATIVE_CUDA_TOKENS" \
      "$BOX_JOBS_NATIVE_NVIDIA_SLOTS" "$BOX_JOBS_NATIVE_CUDA_TOKENS" "$BOX_JOBS_NATIVE_CUDA_CAP" ;;
    nvidia-cpu) printf 'The fleet runs %s correctness slots on
  this box, %s of them GPU tokens (lukstafi/ludics-lite#391), and CPU batches
  there were only ever measured at -j %s; uncapped, %s of them would each run
  as many jobs as the box has cores (gh-ocannl-1065). This run takes its fleet
  slot itself, as `execution slot --cpu` when none of its backends holds a GPU
  (gh-ocannl-1004). The cap lives in tools/box-jobs.sh; the evidence
  is in the native-boot bullets of
  docs/agent-notes/build-and-test.md#gpu-boxes-job-caps-and-runtime-refusals.' \
      "$BOX_JOBS_NATIVE_NVIDIA_SLOTS" "$BOX_JOBS_NATIVE_CUDA_TOKENS" \
      "$BOX_JOBS_NATIVE_CPU_CAP" "$BOX_JOBS_NATIVE_NVIDIA_SLOTS" ;;
  esac
}

width_cap=        # the width to inject, empty for none
width_announce=   # what to say about it, on stderr and in the run's log
plan_width_cap() { # dune argv; after plan_batch
  local cap hazard backend
  width_cap= width_announce=
  [ -n "$batch_resolved" ] || return 0
  batch_width
  cap=$batch_width_cap hazard=$batch_width_hazard backend=$batch_width_backend
  [ -n "$cap" ] || return 0
  if explicit_jobs "$@"; then
    width_announce="$(hazard_name "$hazard"), and this batch can hold $backend:
  this command names its own dune width, so the -j $cap cap (tools/box-jobs.sh) was NOT injected."
    return 0
  fi
  width_cap=$cap
  width_announce="capping dune at -j $cap. $(hazard_found "$hazard" "$backend" "$(batch_why "$backend")"). $(hazard_why "$hazard") Pass an
  explicit -j to run at a width of your own."
}

# Resolves the batch's backends (tools/batch-backends.sh) where something reads
# them: a box where a backend meets a width cap, or a fleet slot, whose kind is
# the other reading. Elsewhere nothing is built and nothing is said. Called
# by `_resolve`, under the worktree lock and the run's cap, so dune's output
# from building the readers lands in the run's log.
plan_batch() { # <log> dune argv
  local log=$1
  shift
  batch_resolved=
  [ -n "$slot_fw" ] || batch_box_has_hazard || return 0
  batch_resolve "$DUNE" "$log" "$@"
}

# The fleet's run-time correctness slot (gh-ocannl-1004). On a fleet box --
# one whose deployed `fleet-worker.sh execution slot --probe` names it (the
# lukstafi/ludics-lite issue-wave skill) -- a `run`/`start` takes one of the
# box's correctness slots itself: the supervisor runs dune under
# `fleet-worker.sh execution slot --cpu|--gpu`, declaring `--cpu` only when
# none of the batch's resolved backends holds a GPU (plan_batch, the same
# resolution the width reads; tools/batch-backends.sh).
# No brief has to name the wrapper and no worker can forget it, or forget
# `--cpu` and hold a GPU token for a cc batch. A worker that still wraps the
# runner is harmless: the fleet's nested-slot rule runs this batch inside the
# wrapper's slot.
#
# The probe is also the capability check: a fleet-worker.sh from before the
# nested-slot rule has no `--probe` and is not used, so a worker's wrapper
# around this script can never cost two slots. Anything else the probe says --
# a machine outside the fleet, no skill deployed -- runs dune directly, as
# before, and silently. The candidates -- and OCANNL_TOOL_FLEET_WORKER, which
# names another fleet-worker.sh (the harness's fake) or turns the slot off with
# `none` -- are tools/fleet-worker-candidates.sh's, shared with tools/sweep.sh,
# which reads the fleet's registry through the same one. `repeat` never takes it:
# an isolation tool runs as given, like its width (wrap it yourself on a
# fleet box). The slot's wait comes out of the run's cap: it is the smaller of
# half what the resolution left of the cap and OCANNL_TOOL_SLOT_WAIT (600s),
# after which the slot refuses and the run reports SLOT REFUSED (exit 75),
# never a test verdict.
slot_fw=          # the fleet-worker.sh to take the slot through, empty for none
slot_wait=
slot_announce=
plan_slot() {
  local fw tag box= slots tokens
  slot_fw= slot_wait= slot_announce=
  # Each candidate in turn until one answers the probe: the two skill trees
  # are deployed independently, and one of them may predate the probe while
  # the other has it. The answer is one line, `EXECUTION SLOT PROBE <box>
  # <slots> <tokens>`: no lock, no registry read, so it costs nothing on a box
  # outside the fleet.
  while IFS= read -r fw; do
    [ -x "$fw" ] || continue
    # Bounded, so a wedged fleet-worker.sh costs this launch 30s and no slot
    # rather than its whole cap; answered through a file, so a descendant it
    # leaves behind cannot hold the read open (the run's process group reaps
    # that descendant with dune's). It is asked without the worktree lock.
    perl -e 'alarm shift; exec @ARGV or exit 127' 30 "$fw" execution slot --probe \
      </dev/null >"$run_dir/probe" 2>/dev/null 9>&- || continue
    read -r tag _ _ box slots tokens _ <"$run_dir/probe"
    [ "$tag" = EXECUTION ] && [ -n "$tokens" ] && { slot_fw=$fw; break; }
  done < <(fleet_worker_candidates)
  rm -f "$run_dir/probe"
  [ -n "$slot_fw" ] || return 0
  slot_wait=${OCANNL_TOOL_SLOT_WAIT:-600}
  case $slot_wait in '' | *[!0-9]*) slot_wait=600 ;; esac
  slot_box=$box slot_slots=$slots slot_tokens=$tokens
}
# At most half of what the resolution left of the cap: the cap's alarm has run
# since the supervisor started, and a busy slot must be able to refuse -- and
# be reported SLOT REFUSED -- before the alarm reports the run as a TIMEOUT
# instead (Codex review round 3 on PR #803).
clamp_slot_wait() { # after the resolution, in `_resolve`, whose SECONDS the cap's are
  local left
  [ -n "$slot_fw" ] && [ "$cap" -gt 0 ] || return 0
  left=$((cap - SECONDS))
  [ "$left" -ge 0 ] || left=0
  [ "$slot_wait" -le $((left / 2)) ] || slot_wait=$((left / 2))
}
slot_box= slot_slots= slot_tokens=
slot_kind=        # cpu or gpu, once plan_batch has resolved the batch
plan_slot_kind() { # after plan_batch
  local what
  [ -n "$slot_fw" ] || return 0
  slot_kind=$(batch_kind)
  if [ "$slot_kind" = cpu ]; then
    what="--cpu: none of its backends ($(batch_summary)) holds a GPU"
  elif [ "$slot_tokens" -lt "$slot_slots" ]; then
    what="--gpu, one of its $slot_tokens GPU tokens: it can hold $(batch_summary)"
  else
    what="--gpu (any slot may hold the GPU here): it can hold $(batch_summary)"
  fi
  slot_announce="requesting one of $slot_box's $slot_slots fleet correctness slots for this run (gh-ocannl-1004),
  as $what. Acquisition timeout: ${slot_wait}s; an enclosing slot may be reused immediately."
}

# Whether `run` resolves this batch's backends at all (plan_batch): only on a
# box where a backend meets a width cap, or under a fleet slot. `plan` resolves
# them on any box, so the two decisions below ask this rather than whether a
# resolution happened, or `plan` would report bounds `run` never applies.
run_reads_batch() { [ -n "$batch_resolved" ] && { [ -n "$slot_fw" ] || batch_box_has_hazard; }; }

# What the batch's backends say about the GPU it was meant to test
# (gh-ocannl-1433). A batch on a GPU box can resolve to the CPU backends alone
# -- OCANNL_BACKEND unset, the test configuration naming cc -- and then it is
# green and has tested nothing on the GPU, which is how two wave legs passed
# while proving nothing about the backend they were run for. So:
#   - `--expect-backend B` (repeatable; a backend name, an alias canonicalized)
#     is the caller's statement of what the batch must hold. A batch that does
#     not hold every B, or whose backends could not be read (an unread answer
#     is "any backend" for a width and a slot, but proves no backend held), is
#     refused before dune starts: BACKEND REFUSED, exit 2, and the digest
#     names what it held. The option makes `run` resolve the batch's backends
#     on any box, where otherwise only a width hazard or a fleet slot does.
#   - Without the option, on a GPU box -- one where cuda or hip meets a width
#     hazard (tools/box-jobs.sh's local view: the fleet's Linux GPU boxes, a
#     WSL guest's bridge), so never a Mac -- a batch whose resolved backends
#     are all CPU ones says so in a WARNING line, in the announcements, the
#     run's log and the digest. A batch holding no backend at all (the scans)
#     claims none and is not warned about; one whose `--expect-backend` holds
#     has said what it meant, cc included.
expect_backends=  # canonical, space-separated; from the run's `expect-backend`
backend_refusal=  # why the batch is refused, empty for none
backend_warning=  # the GPU box's warning, empty for none
box_gpu_backends() { # the GPU backends that meet a width hazard on this box
  local b out=
  for b in $BATCH_ALL_BACKENDS; do
    box_jobs_gpu_backend "$b" || continue
    [ -z "$(box_jobs_local_hazard "$b")" ] || out="$out${out:+ }$b"
  done
  printf '%s' "$out"
}
batch_holds_backend() { # <backend>; 0 iff the resolved batch holds it
  local b rest
  while IFS=' ' read -r b rest; do
    [ "$b" = "$1" ] && return 0
  done <<<"$batch_holds"
  return 1
}
plan_backend_check() { # after plan_batch, and batch_resolve when expect_backends is set
  local b missing= gpus select
  backend_refusal= backend_warning=
  if [ -n "$expect_backends" ]; then
    if [ -n "$batch_unknown" ]; then
      backend_refusal="--expect-backend $expect_backends: this batch's backends are unread ($batch_unknown), so nothing shows it holds $expect_backends"
      return 0
    fi
    for b in $expect_backends; do
      batch_holds_backend "$b" || missing="$missing${missing:+ }$b"
    done
    [ -z "$missing" ] ||
      backend_refusal="--expect-backend $expect_backends: this batch holds $(batch_summary), not $missing. Select it (OCANNL_BACKEND=$(set -- $missing; printf '%s' "$1"), or a test configuration naming it), or run the batch without the option"
    return 0
  fi
  [ -n "$batch_resolved" ] && [ -z "$batch_unknown" ] && [ -n "$batch_holds" ] || return 0
  [ "$(batch_kind)" = cpu ] || return 0
  gpus=$(box_gpu_backends)
  [ -n "$gpus" ] || return 0
  # A WSL guest's bridge is a hazard for cuda and hip alike, so there the box's
  # own GPU is not known here: name the one backend only when it is the only one.
  case $gpus in
    *' '*) gpus="${gpus// / or }" select="OCANNL_BACKEND=<this box's GPU backend>" ;;
    *) select="OCANNL_BACKEND=$gpus" ;;
  esac
  backend_warning="WARNING: this box has a GPU backend ($gpus), but this batch holds only $(batch_summary): it tests nothing on the GPU. Select it ($select) to test it here, or pass --expect-backend to refuse a batch that does not hold it (gh-ocannl-1433)."
}

# The device probe (gh-ocannl-1211; probe_perl says what it reads): every GPU
# backend of a batch `run` resolves, through `bin/device_props` -- the query
# that found tuf's wedged device -- or OCANNL_TOOL_DEVICE_PROBE, a stand-in
# taking the same `--ocannl_backend=<name>`, or `none` to probe nothing.
# Bounded by OCANNL_TOOL_DEVICE_PROBE_CAP (60s; a healthy query answers in
# well under a second). Everything happens INSIDE the fleet slot, as the first
# act of the command the slot runs (`_probe`): the build of `device_props`
# too, so a box under a measurement hold sees neither a compile nor device
# traffic from a batch it refuses, and the slot's wait is still the only thing
# between clamp_slot_wait and the slot. Not on macOS without a stand-in: a
# freshly linked `device_props` can sit in dlopen for minutes while XProtect
# scans it (runs launched over ssh or by launchd are not exempted), which
# would read as a hung device; metal batches keep the per-test cap.
probe_prog= probe_backends= probe_cap= probe_announce=
plan_device_probe() { # after plan_batch
  local b
  probe_prog= probe_backends= probe_announce=
  run_reads_batch && [ "${OCANNL_TOOL_DEVICE_PROBE:-}" != none ] || return 0
  if [ -n "${OCANNL_TOOL_DEVICE_PROBE:-}" ]; then
    probe_prog=$OCANNL_TOOL_DEVICE_PROBE
    case $probe_prog in /*) ;; *) probe_prog=$PWD/$probe_prog ;; esac
  else
    [ "$(uname -s)" != Darwin ] || return 0
    probe_prog=- # `_probe` builds bin/device_props
  fi
  while IFS= read -r b; do
    box_jobs_cpu_backend "$b" || probe_backends="$probe_backends $b"
  done < <(batch_backends)
  probe_backends=${probe_backends# }
  [ -n "$probe_backends" ] || return 0
  probe_cap=${OCANNL_TOOL_DEVICE_PROBE_CAP:-$DEVICE_PROBE_CAP_DEFAULT}
  case $probe_cap in '' | 0 | *[!0-9]*) probe_cap=$DEVICE_PROBE_CAP_DEFAULT ;; esac
  probe_announce="a fresh device query for $probe_backends before dune starts${slot_fw:+, inside the slot}; one
  that does not answer within ${probe_cap}s refuses the batch as DEVICE UNHEALTHY (gh-ocannl-1211)."
}

# The per-test cap's value (gh-ocannl-1211; test_cap_watch enforces it). An
# explicit one -- `--test-cap N` or OCANNL_TOOL_PER_TEST_CAP -- applies to any
# batch. The default, TEST_CAP_DEFAULT (tools/per-test-cap.sh, shared with the
# sweep's units, says why it is the number it is), applies only where the
# failure it exists for can happen and no legitimate test is known to come near
# it: a batch `run` resolves to hold a GPU backend, whose targets do not reach
# the `slow` aliases.
test_cap_why=
plan_test_cap() { # <requested, empty for the default> dune-argv
  local requested=$1 arg
  shift
  test_cap=0 test_cap_why=
  if [ -n "$requested" ]; then
    test_cap=$requested test_cap_why="asked for"
    [ "$test_cap" -gt 0 ] || test_cap_why="--test-cap 0"
    return 0
  fi
  if ! run_reads_batch; then
    test_cap_why="no default: run does not resolve this batch's backends here (no width hazard, no fleet slot)"
    return 0
  fi
  if [ "$(batch_kind)" != gpu ]; then
    test_cap_why="no default: a CPU batch"
    return 0
  fi
  if arg=$(test_cap_reaches_slow "$@"); then
    test_cap_why="no default: the targets reach the slow tests ($arg)"
    return 0
  fi
  test_cap=$TEST_CAP_DEFAULT test_cap_why="the default for a GPU batch"
}

# An alias dune does not know (gh-ocannl-1448). A misspelled `@DIR/runtest-NAME`
# used to be found by dune itself, after the launch had resolved the batch's
# backends (a build of the readers), taken the fleet's slot and probed the
# device: on a GPU box the slot's wait, at worst a full one, bought nothing but
# dune's "Alias ... is empty". So `_resolve` asks dune first, before any of
# those: `dune show aliases DIR...`, a listing that loads the directories' rules
# and builds nothing, for every `@` and `@@` operand of a `dune build` before
# dune's own `--`. An alias is known as dune resolves one on its command line:
# `@@DIR/NAME` where DIR's listing holds NAME, `@DIR/NAME` where DIR's or that of
# any directory below it does (the directories holding a dune file, as the
# tracked tree lists them, or as `find` does outside a checkout's top level); a
# DIR dune does not know ("Directory DIR does not exist") is no alias at all.
# Every unknown one refuses the batch: ALIAS REFUSED, exit 2, nothing ran.
#
# The check refuses only on dune's own, conclusive answer, and is otherwise
# silent and lets dune decide, as before: a listing that does not read as dune
# 3.20-3.24's (the listing on either stream, `DIR:` headers when it names more
# than one directory, one alias name per line, `Error: Directory DIR does not
# exist.`), a directory whose listing lacks dune's standard `default` alias, a
# failed or timed-out query, an argv carrying any dune option not known to
# leave the alias set alone (`--profile`, `--root`, `--workspace`, `-p`, ... --
# plan_aliases holds the allow-list), and an operand that is not a plain
# relative `DIR/NAME` (an absolute path, a `..` or an inner `.` segment, or a
# `_`-led directory such as a context's `_build/CONTEXT/`). The directories
# below an operand's are those holding a dune file and those its `(subdir ...)`
# stanzas give rules to (alias_dirs_perl); a dune file that does not read
# leaves the recursive misses to dune. The query's trace goes to a file of the
# run's, so the shared `_build/trace.csexp` is not rewritten.
alias_refusal=  # why the batch is refused, one unknown alias per line; empty for none
alias_listing_perl='
  # <raw listing> <dir>...: the listing of these directories, as "+ DIR" (dune
  # knows it), "- DIR" (it does not) and "= DIR NAME", tab-separated. Exit 3
  # where the listing does not read as dune; exit 4, with only the "- DIR"
  # lines, where it names directories dune does not know and lists nothing,
  # which is how dune answers a query naming one: ask again without them.
  my ($raw, @dirs) = @ARGV;
  my %dir = map { $_ => 1 } @dirs;
  open my $fh, "<", $raw or exit 3;
  my ($cur, %names, %absent, $listed);
  $cur = $dirs[0] if @dirs == 1;
  while (my $l = <$fh>) {
    $l =~ s/\r?\n\z//;
    next if $l eq "";
    if ($l =~ /^Error: Directory (.*) does not exist\.$/ && $dir{$1}) { $absent{$1} = 1; next }
    $listed = 1;
    if (@dirs > 1 && $l =~ /^(.*):$/ && $dir{$1}) { $cur = $1; $names{$cur} ||= []; next }
    exit 3 unless $l =~ /^[A-Za-z0-9_.+-]+$/ && defined $cur;
    push @{ $names{$cur} }, $l;
  }
  if (%absent && !$listed) {
    print "-\t$_\n" for grep { $absent{$_} } @dirs;
    exit 4;
  }
  for my $d (@dirs) {
    if ($absent{$d}) { print "-\t$d\n"; next }
    exit 3 unless grep { $_ eq "default" } @{ $names{$d} || [] };
    print "+\t$d\n";
    print "=\t$d\t$_\n" for @{ $names{$d} };
  }
'
alias_listing() { # <listing file> dir...; 0 with the parsed listing in it, 1 inconclusive
  local out=$1 rc d left
  shift
  : >"$out" || return 1
  while :; do
    rc=0
    promotion_bounded "$DUNE" show aliases --trace-file="$run_dir/aliases.trace" "$@" >"$out.raw" 2>&1 ||
      rc=1
    [ "$rc" != 0 ] || perl -e "$alias_listing_perl" "$out.raw" "$@" >>"$out" || rc=$?
    rm -f "$out.raw" "$run_dir/aliases.trace"
    [ "$rc" = 4 ] || break
    # Every directory the answer named is out; the others are asked again, and
    # each round asks for fewer.
    left=()
    for d do alias_listed - "$d" "" "$out" || left+=("$d"); done
    [ ${#left[@]} -gt 0 ] || { rc=0; break; }
    [ ${#left[@]} -lt $# ] || { rc=1; break; }
    set -- "${left[@]}"
  done
  [ "$rc" = 0 ]
}
# The directories a dune file's `(subdir NAME ...)` stanzas give rules to, which
# hold no dune file of their own (gh-ocannl-1448, Codex review round 2): a
# reader of dune's s-expressions -- `;` line comments, quoted and plain atoms,
# and nothing else -- that walks every stanza, nested `subdir`s included.
# Input: dune file paths (`dune`, or `dune-file` where a project accepts that
# name); output: their directories and the logical ones below them. It is
# fail-closed on what it does not model: exit 3 where a file does not read (an
# unbalanced form, a `#|` or `#;`, which dune versions read differently) or
# takes stanzas from another file (an `include` or `dynamic_include`, at any
# depth, Codex review round 3), and the caller then cannot list them all.
alias_dirs_perl='
  my $bad = 0;
  for my $f (@ARGV) {
    my $dir = $f =~ m{^dune(?:-file)?$} ? "." : ($f =~ m{^(.*)/dune(?:-file)?$} ? $1 : next);
    print "$dir\n";
    open my $fh, "<", $f or do { $bad = 1; next };
    local $/;
    my $t = <$fh>;
    close $fh;
    my (@tok, $ok);
    $ok = eval {
      pos($t) = 0;
      while (pos($t) < length $t) {
        if ($t =~ /\G\s+/gc || $t =~ /\G;[^\n]*/gc) { next }
        # `#|` and `#;` are where dune readings differ (comments in some
        # versions, atom text in others): a file holding one is not read.
        die if $t =~ /\G#[|;]/gc;
        if ($t =~ /\G([()])/gc) { push @tok, [$1]; next }
        if ($t =~ /\G"((?:[^"\\]|\\.)*)"/gcs) { push @tok, ["a", $1]; next }
        if ($t =~ /\G([^\s();"]+)/gc) { push @tok, ["a", $1]; next }
        die;
      }
      1;
    };
    unless ($ok) { $bad = 1; next }
    # Into nested lists.
    my @stack = ([]);
    for my $k (@tok) {
      if ($k->[0] eq "(") { push @stack, []; next }
      my $x;
      if ($k->[0] eq ")") { @stack > 1 or do { $bad = 1; last }; $x = pop @stack }
      else { $x = $k->[1] }
      push @{ $stack[-1] }, $x;
    }
    if (@stack != 1) { $bad = 1; next }
    my $walk;
    $walk = sub {
      my ($at, $forms) = @_;
      for my $s (@$forms) {
        # Stanzas read from another file -- `(include FILE)`, a generated
        # `(dynamic_include ...)` -- may hold subdirs this reader never sees.
        if (ref $s && @$s && !ref $s->[0] && $s->[0] =~ /^(?:include|dynamic_include)$/) { $bad = 1; next }
        next unless ref $s && @$s >= 2 && !ref $s->[0] && $s->[0] eq "subdir";
        my $name = $s->[1];
        if (ref $name || $name !~ m{^[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*$} || $name =~ m{(?:^|/)\.\.?(?:/|$)}) {
          $bad = 1;
          next;
        }
        my $sub = $at eq "." ? $name : "$at/$name";
        print "$sub\n";
        $walk->($sub, [ @$s[2 .. $#$s] ]);
      }
    };
    $walk->($dir, $stack[0]);
  }
  exit($bad ? 3 : 0);
'
alias_dune_files() { # the dune files of this tree, NUL-terminated
  (
    # This tree's own listing, whatever repository an inherited GIT_DIR names
    # (as record_checkout).
    # shellcheck disable=SC2046 # a list of variable names, split on purpose
    unset $(git rev-parse --local-env-vars 2>/dev/null)
    if [ "$(git rev-parse --is-inside-work-tree --show-prefix 2>/dev/null)" = true ]; then
      git ls-files -z --cached --others --exclude-standard -- dune '*/dune' dune-file '*/dune-file' 2>/dev/null
    else
      find . \( -name _build -o -name .git \) -prune -o \( -name dune -o -name dune-file \) -type f -print0 2>/dev/null
    fi
  )
}
alias_dune_dirs() { # the directories holding a dune file or a subdir's rules, `.` for the top; 1 incomplete
  local list=() f
  # NUL-delimited, so a path git would quote (a non-ASCII name) is read as
  # itself (Codex review round 5).
  while IFS= read -r -d '' f; do
    [ ! -f "$f" ] || list+=("$f")
  done < <(alias_dune_files)
  [ ${#list[@]} -gt 0 ] || return 1
  list=("${list[@]#./}")
  # Kept for alias_dynamic_include, whatever the reader says of them.
  printf '%s\0' "${list[@]}" >"$run_dir/aliases.files" || return 1
  perl -e "$alias_dirs_perl" "${list[@]}" >"$run_dir/aliases.dirs" || return 1
  sort -u "$run_dir/aliases.dirs"
  rm -f "$run_dir/aliases.dirs"
}
alias_imports() { # 0 iff a dune file alias_dune_dirs listed imports another file
  local f
  while IFS= read -r -d '' f; do
    perl -0777 -ne 'exit(/\(\s*(?:dynamic_)?include\s/ ? 0 : 1)' -- "$f" 2>/dev/null && return 0
  done <"$run_dir/aliases.files"
  return 1
}
plan_aliases() { # dune argv; in `_resolve`, before the slot: sets alias_refusal
  local arg spec rec d name i n all= unlisted=
  local specs=() recs=() spec_dirs=() names=() first=() rest=()
  alias_refusal=
  [ "${1:-}" = build ] || return 0
  shift
  # Every option before dune's `--` must be one known to leave the set of
  # aliases as `show aliases` sees it: a profile, a workspace, a root, a
  # package selection, a context -- anything else -- can change what dune
  # defines, so its presence leaves the whole batch to dune (an allow-list, so
  # an option dune adds later fails open). A separate value is skipped with it.
  local value=
  for arg do
    if [ -n "$value" ]; then value=; continue; fi
    case $arg in
      --) break ;;
      -j | --cache | --display | --diff-command | --trace-file | --error-reporting | --sandbox | --build-dir)
        value=1; continue ;;
      -j[0-9]* | --cache=* | --display=* | --diff-command=* | --trace-file=* | --error-reporting=* | \
        --sandbox=* | --build-dir=* | --force | -f | --auto-promote | --no-buffer | --stop-on-first-error | \
        --action-stdout-on-success=* | --action-stderr-on-success=*)
        continue ;;
      -*) return 0 ;;
      @*) ;;
      *) continue ;;
    esac
    spec=${arg#@} rec=1
    case $spec in @*) spec=${spec#@} rec=0 ;; esac
    case $spec in */*) d=${spec%/*} name=${spec##*/} ;; *) d=. name=$spec ;; esac
    while :; do case $d in ./*) d=${d#./} ;; */) d=${d%/} ;; *) break ;; esac; done
    [ -n "$d" ] || d=.
    case $name in '' | *[!A-Za-z0-9_.+-]*) continue ;; esac
    case /$d/ in //* | */../* | *//* | *$'\n'* | *$'\t'*) continue ;; esac
    # A `.` segment inside the path (`test/./NAME`) is dune's to resolve, and so
    # is a path through a directory dune keeps out of the source tree by
    # default, `_build/CONTEXT/...` (a context-qualified alias) among them.
    [ "$d" = . ] || case /$d/ in */./* | /_*) continue ;; esac
    specs+=("$arg") recs+=("$rec") spec_dirs+=("$d") names+=("$name")
  done
  n=${#specs[@]}
  [ "$n" -gt 0 ] || return 0
  # First each operand's own directory; then, only for a recursive one not
  # found there, every dune directory below it, in one more query.
  for d in "${spec_dirs[@]}"; do
    alias_member "$d" ${first[@]+"${first[@]}"} || first+=("$d")
  done
  # A generated `(dynamic_include ...)` is built by any dune query that loads
  # its directory's rules -- this one too, outside the slot (Codex review round
  # 4) -- and a static `(include FILE)` can import one, or a subdir, the census
  # does not read (round 5): a tree whose dune files import another file
  # anywhere leaves the batch to dune, before any query.
  rm -f "$run_dir/aliases.files"
  all=$(alias_dune_dirs) || { all=- unlisted=1; }
  if [ ! -s "$run_dir/aliases.files" ] || alias_imports; then
    rm -f "$run_dir/aliases.files"
    return 0
  fi
  rm -f "$run_dir/aliases.files"
  alias_listing "$run_dir/aliases.1" "${first[@]}" || return 0
  i=0
  while [ "$i" -lt "$n" ]; do
    d=${spec_dirs[$i]}
    if [ -z "$unlisted" ] && [ "${recs[$i]}" = 1 ] && alias_listed + "$d" "" "$run_dir/aliases.1" &&
       ! alias_listed = "$d" "${names[$i]}" "$run_dir/aliases.1"; then
      while IFS= read -r arg; do
        [ -n "$arg" ] && [ "$arg" != "$d" ] || continue
        [ "$d" = . ] || case $arg in "$d"/*) ;; *) continue ;; esac
        alias_member "$arg" ${first[@]+"${first[@]}"} ${rest[@]+"${rest[@]}"} || rest+=("$arg")
      done <<<"$all"
    fi
    i=$((i + 1))
  done
  : >"$run_dir/aliases.2"
  # An inconclusive second answer leaves the recursive operands it was for to
  # dune: only what the first answer decided refuses.
  [ ${#rest[@]} -eq 0 ] || alias_listing "$run_dir/aliases.2" "${rest[@]}" ||
    printf '?\n' >"$run_dir/aliases.2"
  [ -z "$unlisted" ] || printf '?\n' >"$run_dir/aliases.2"
  i=0
  while [ "$i" -lt "$n" ]; do
    d=${spec_dirs[$i]} name=${names[$i]}
    if alias_listed - "$d" "" "$run_dir/aliases.1"; then
      alias_refusal="$alias_refusal${alias_refusal:+$'\n'}${specs[$i]}: dune knows no directory $d"
    elif alias_listed = "$d" "$name" "$run_dir/aliases.1"; then
      :
    elif [ "${recs[$i]}" = 0 ]; then
      alias_refusal="$alias_refusal${alias_refusal:+$'\n'}${specs[$i]}: $d defines no alias $name"
    elif grep -qxF -e '?' "$run_dir/aliases.2"; then
      :
    elif ! awk -F '\t' -v d="$d" -v name="$name" \
           '$1 == "=" && $3 == name && (d == "." || index($2, d "/") == 1) { f = 1 } END { exit !f }' \
           "$run_dir/aliases.1" "$run_dir/aliases.2"; then
      alias_refusal="$alias_refusal${alias_refusal:+$'\n'}${specs[$i]}: neither $d nor any directory below it defines an alias $name"
    fi
    i=$((i + 1))
  done
  rm -f "$run_dir/aliases.1" "$run_dir/aliases.2"
}
alias_member() { # word word...; 0 iff the first is among the rest
  local w=$1 x
  shift
  for x do [ "$x" != "$w" ] || return 0; done
  return 1
}
alias_listed() { # kind dir name listing-file; 0 iff the parsed listing has that line
  local pat
  case $1 in
    =) pat=$(printf '=\t%s\t%s' "$2" "$3") ;;
    *) pat=$(printf '%s\t%s' "$1" "$2") ;;
  esac
  grep -qxF -e "$pat" "$4"
}

select_dune() {
  # On Windows the environment rewrite is required even if dune is already on
  # PATH: opam's native output otherwise leaves an MSYS shell half-configured.
  case ${OSTYPE:-} in
    msys* | cygwin*)
      . "$runner_root/tools/opam-env.sh" ||
        die "tools/opam-env.sh failed; refusing an unrewritten Windows toolchain"
      ;;
  esac
  command -v dune >/dev/null 2>&1 || . "$runner_root/tools/opam-env.sh"
  command -v dune >/dev/null 2>&1 || die "dune not found (opam environment not set up?)"
  # Preserve dune's status while filtering only the known Windows linker noise.
  case ${OSTYPE:-} in msys* | cygwin*) DUNE=$runner_root/tools/dune-quiet.sh ;; *) DUNE=dune ;; esac
  # Again, now that the switch's environment may have been sourced: its updates can set a
  # deny-listed variable the startup scrub never saw (gh-ocannl-1280).
  scrub_credentials
}

# Pin to the repo containing THIS script (promote.sh convention): dune then runs
# at this worktree's root no matter where the caller's cwd wandered, and the
# per-worktree lock below keys on the tree actually being tested.
# -P: the physical path, so the same worktree entered through a symlink and
# through its real path derive the same key, lock file, and recorded wt.
# Read-only queries interpret relative paths here, never through a caller's
# CDPATH (which can also make cd print an extra path on stdout).
case ${1:-} in paths | lock-status | affected) CDPATH= ;; esac
cd -P "$(dirname "$0")/.." || die "cannot cd to repo root"
# Where this script and everything it runs of its own live: the helpers below
# are sourced from here, and the run's own phases (`_resolve`, `_probe`) are
# this copy of the script. Normally also the tree under test; `bisect` points a
# run at a tree of its own instead (see TESTRUN_TREE below).
runner_root=$PWD

[ -r scripts/process-group.sh ] || die "cannot read scripts/process-group.sh"
# shellcheck source=../scripts/process-group.sh
. scripts/process-group.sh

# The per-box dune width cap, shared with tools/sweep.sh so the two cannot
# drift. See width_cap below for what this script does with it.
[ -r tools/box-jobs.sh ] || die "cannot read tools/box-jobs.sh"
# shellcheck source=box-jobs.sh
. tools/box-jobs.sh
# The per-test cap's default and its watcher, shared with tools/sweep.sh.
[ -r tools/per-test-cap.sh ] || die "cannot read tools/per-test-cap.sh"
# shellcheck source=per-test-cap.sh
. tools/per-test-cap.sh
# The device probe -- probe_perl, one bounded query, and device_probe_sh, the
# per-backend loop and its DEVICE UNHEALTHY refusal -- shared with tools/sweep.sh
# and tools/machine-verify.sh.
[ -r tools/device-probe.sh ] || die "cannot read tools/device-probe.sh"
# shellcheck source=device-probe.sh
. tools/device-probe.sh
# Which backends a batch holds -- the resolution the width and the fleet slot
# both read (plan_batch).
[ -r tools/batch-backends.sh ] || die "cannot read tools/batch-backends.sh"
# shellcheck source=batch-backends.sh
. tools/batch-backends.sh
# Where this host's fleet-worker.sh might be (plan_slot), shared with tools/sweep.sh.
[ -r tools/fleet-worker-candidates.sh ] || die "cannot read tools/fleet-worker-candidates.sh"
# shellcheck source=fleet-worker-candidates.sh
. tools/fleet-worker-candidates.sh
# The `bisect` verb and its steps (gh-ocannl-1225).
[ -r tools/bisect-steps.sh ] || die "cannot read tools/bisect-steps.sh"
# shellcheck source=bisect-steps.sh
. tools/bisect-steps.sh
# Credentials never reach dune, which records every spawned process's environment in the
# worktree's `_build/trace.csexp` (gh-ocannl-1280). Removed from this script's own environment,
# before any subcommand, so dune -- launched by the supervisor, by `_resolve`, `_probe`, `repeat`
# or a promotion query -- and everything it spawns inherit none; nothing this script runs reads
# one. select_dune scrubs again after sourcing the opam environment. The deny-list is shared with
# tools/sweep.sh and tools/machine-verify.sh.
[ -r tools/credential-env.sh ] || die "cannot read tools/credential-env.sh"
# shellcheck source=credential-env.sh
. tools/credential-env.sh
scrub_credentials() {
  eval "$(credential_env_scrub_text)" ||
    die "cannot remove credential variables from the environment:$credential_env_left"
}
scrub_credentials

# perl is load-bearing rather than a convenience: the per-worktree flock, the
# cap supervisor and the atomic rename behind the `last` pointer are all
# `perl -e` one-liners, and every subcommand reaches at least one of them.
#
# Checked ONCE, here, because the alternative is what this check was written
# from: with perl absent, `take_lock`'s perl exited 127, which is not one of the
# statuses that block assigns a meaning, so it fell through to the catch-all
# "another test-run is active in this worktree" -- a phantom lock holder,
# reported with recovery advice (`stop` it) that cannot work because there is
# nothing to stop. That is how the Windows CI smoke step refused with exit 2
# under the Cygwin bash `shell: bash` resolves to once setup-ocaml has prepended
# opam's cygwin to PATH (gh-ocannl-662).
#
# The module is loaded rather than the binary merely found: `use Fcntl ":flock"`
# is what the lock needs, and a perl that cannot provide it would fail in the
# same misread place.
perl -e 'use Fcntl ":flock"; exit 0' 2>/dev/null || die "perl with Fcntl not found; it carries the lock, the cap and the \`last\` pointer.
  On Windows run this from Git Bash, whose MSYS perl has both; a Cygwin bash
  ships neither unless its perl package is installed."

# The tree under test, when it is not this script's own (gh-ocannl-1225).
# `bisect` keeps a frozen copy of this script and its helpers beside the
# worktree it bisects, so every step is judged by one runner -- the one the
# bisect was started with, not the tested commit's, which may predate the slot,
# the caps or a fix to either -- and points each step's `run` at that worktree
# through OCANNL_TOOL_TESTRUN_TREE. Only `run` and `plan` take it; any other
# subcommand refuses it rather than answer about the wrong tree. It is consumed
# here: dune and the tests never see it, so a test that drives this script
# (tools/test-test-run.sh) is not redirected. A run's own phases work in the
# tree their run recorded (`wt`, written before the supervisor exists).
#
# The same goes for the variables a run's own phases are handed (gh-ocannl-1448):
# OCANNL_TOOL_TESTRUN_OWN, the run directory `_resolve` and `_probe` own, and
# OCANNL_TOOL_TESTRUN_RD, the step directory `_direct_plan` writes into, are
# read into shell variables here and unset, so dune and every test it starts
# see neither. The supervisor deletes them, with the rest of its own
# (OCANNL_TOOL_TESTRUN_BG, _RUNNER), before it forks anything (supervisor_perl),
# and a phase gets its variable back only on its own command line, as `env
# NAME=VALUE` ahead of the phase -- `_resolve` hands `_probe` its run that way.
testrun_own=${OCANNL_TOOL_TESTRUN_OWN:-} testrun_rd=${OCANNL_TOOL_TESTRUN_RD:-}
unset OCANNL_TOOL_TESTRUN_OWN OCANNL_TOOL_TESTRUN_RD
case ${1:-} in
  _resolve | _probe)
    # Without a run to own, the phase's own check below refuses it.
    if [ -n "$testrun_own" ]; then
      testrun_tree=$(cat "$testrun_own/wt" 2>/dev/null) && [ -n "$testrun_tree" ] ||
        die "cannot read the tree under test from $testrun_own/wt"
      cd -P "$testrun_tree" || die "cannot cd to the tree under test: $testrun_tree"
    fi ;;
  run | plan)
    if [ -n "${OCANNL_TOOL_TESTRUN_TREE:-}" ]; then
      cd -P "$OCANNL_TOOL_TESTRUN_TREE" || die "OCANNL_TOOL_TESTRUN_TREE: cannot cd to $OCANNL_TOOL_TESTRUN_TREE"
    fi ;;
  *)
    [ -z "${OCANNL_TOOL_TESTRUN_TREE:-}" ] ||
      die "OCANNL_TOOL_TESTRUN_TREE points run and plan at another tree; ${1:-this subcommand} does not take it (unset it)" ;;
esac
unset OCANNL_TOOL_TESTRUN_TREE

# OCANNL_TOOL_ is the namespace the library reserves for names that address OCANNL
# without being configuration: an OCANNL executable warns at startup about any other
# `OCANNL_...` variable it finds (gh-ocannl-629), and these are exported into the
# environment of every test this script runs.
RUNS=${OCANNL_TOOL_TEST_RUNS:-}
require_query_path() {
  case $1 in *$'\n'* | *$'\r'*) die "queries require single-line paths (no CR/LF)" ;; esac
}

query_physical_path() {
  # Resolve with launch-shell semantics, then inspect the physical PWD before
  # command substitution can strip a trailing newline from the path itself.
  (cd -- "$1" && cd -P . && require_query_path "$PWD" && pwd -P)
}

query_state_for() {
  # Absolute references carry their own recorded state. Do not inspect an
  # unrelated current root just to answer a question about that run.
  require_query_path "$1"
  recorded_absolute_path "$1" && return 0
  if [ -z "$RUNS" ]; then
    [ -n "${HOME:-}" ] || die "HOME is unavailable; set OCANNL_TOOL_TEST_RUNS for current-state queries"
    RUNS=$HOME/.ocannl-test-runs
  fi
  require_query_path "$RUNS"
  require_query_path "$PWD"
  # Let the same shell `cd -P` as launch resolve the existing prefix. In
  # particular, Perl File::Spec under MSYS does not treat C:/ as Bash does.
  # No mkdir: append only ordinary missing components. A missing prefix
  # followed by .. has no physical identity to query; do not simulate how
  # mkdir or platform-specific root traversal would interpret it.
  query_prefix=$RUNS query_suffix=
  while [ ! -d "$query_prefix" ]; do
    [ ! -e "$query_prefix" ] && [ ! -L "$query_prefix" ] ||
      die "not a directory: $query_prefix"
    query_parent=$(dirname -- "$query_prefix")
    # dirname C:/missing returns C:, which MSYS directory predicates do
    # not recognize as the drive root. Preserve the explicit root slash.
    case $query_prefix:$query_parent in
      [A-Za-z]:/*:[A-Za-z]:) query_parent=$query_parent/ ;;
    esac
    # An unavailable UNC host/share must not fall back to a local / path.
    case ${OSTYPE:-} in
      msys* | cygwin*)
        case $query_prefix in
          //[!/]*) case $query_parent in / | //) die "cannot resolve UNC root: $RUNS" ;; esac ;;
        esac ;;
    esac
    [ "$query_parent" != "$query_prefix" ] || die "cannot resolve $RUNS"
    query_part=$(basename -- "$query_prefix")
    case $query_part in
      ..) die "cannot resolve missing state path containing ..: $RUNS" ;;
      .) ;;
      *) query_suffix=$query_part${query_suffix:+/$query_suffix} ;;
    esac
    query_prefix=$query_parent
  done
  RUNS=$(query_physical_path "$query_prefix") || die "cannot resolve $RUNS"
  if [ -n "$query_suffix" ]; then
    # mkdir traverses symlink/.. physically. A logical cd of the existing
    # prefix may name another directory; do not predict its missing child.
    query_physical_prefix=$(cd -P -- "$query_prefix" && require_query_path "$PWD" && pwd -P) || die "cannot resolve missing state prefix"
    [ "$query_physical_prefix" = "$RUNS" ] || die "cannot resolve missing state path with ambiguous symlink parent traversal"
    RUNS=${RUNS%/}/$query_suffix
  fi
  LOCK=$RUNS/lock-$wt_key
  OWNER=$RUNS/owner-$wt_key
  LAST=$RUNS/last-$wt_key
}
case ${1:-} in
  # Initialize lazily, only if the query needs this root; `_resolve` and
  # `_probe` work in the run directory their supervisor names.
  paths | lock-status | _resolve | _probe | affected) ;;
  *) RUNS=${RUNS:-$HOME/.ocannl-test-runs}
     mkdir -p "$RUNS" || die "cannot create $RUNS"
     RUNS=$(cd "$RUNS" && pwd -P) || die "cannot resolve $RUNS" ;;
esac
# Canonicalized: run identities (owner pointer, `last`, wt cross-references)
# are compared as strings, so a relative override must not record a
# different spelling than a later absolute reference resolves to.
# The worktree key makes every per-worktree fact under $RUNS -- the lock, its
# owner pointer and the `last` pointer -- per-checkout, so concurrent sessions
# in different worktrees never read each other's verdicts or contend on each
# other's lock. The readable basename is for humans listing $RUNS; the crc of
# the full path is what keeps two paths differing only in punctuation from
# sharing a key. One definition, applied by the launcher to its own root and
# by `stop` and retention to the root a run RECORDED, so a sibling worktree
# derives the same key from the same path.
wt_key_of() { # <worktree root, physical path> -> its key
  printf '%s' "$(basename "$1" | tr -c 'A-Za-z0-9' '_')$(printf %s "$1" | cksum | awk '{print $1}')"
}
wt_key=$(wt_key_of "$PWD")
# Every file this script keeps per worktree lives under $RUNS, keyed as above,
# and NOT in the worktree: a run must leave nothing behind in the tree it
# tested. A gitignored dotfile that survives the run reads as "ignored local
# data" to anything that judges a worktree finished by its cleanliness, and
# the teardown after every worktree-based session refused over three of them
# (gh-ocannl-606). The cost, stated: OCANNL_TOOL_TEST_RUNS relocates the lock
# together with the diagnostics, so two sessions testing the SAME worktree
# see each other's lock only under the same override. A split there is caught
# one layer down, by dune's own `_build/.lock` -- two dune instances cannot
# both build one tree -- but without this script's pointer at the run to stop.
LOCK=$RUNS/lock-$wt_key   # the flock target: fd 9 of every process of a run
OWNER=$RUNS/owner-$wt_key # the run directory holding the lock (see take_lock)
LAST=$RUNS/last-$wt_key   # the run this worktree published most recently

# The supervisor: ONE perl process that owns a run from the instant it
# inherits lock fd 9 until the verdict is on disk -- the second of a run's two
# parties, the launching shell being the first. It caps the run, relays
# INT/TERM to dune's process group and reaps it, records its own identity (pid
# and start token) for status/stop, and on every exit path writes the `exit:`
# sentinel and the verdict file, then exits -- which is the lock's release
# (see take_lock). The sibling of sweep.sh's supervisor (see the rationale
# there). `perl -e 'alarm N; exec ...'` alone is not enough: alarm survives
# exec, so SIGALRM would reach only dune while every compiler it spawned kept
# running and holding _build locks. Exits 142 on expiry.
#
# This used to be three parties. A wrapper subshell between launcher and
# supervisor published the verdict and held the lock through a publication
# handshake with the launcher -- an arbiter file, an acknowledgement, an
# abandonment marker and a metadata lock, each closing the interleaving the
# previous one had opened (gh-ocannl-606). It existed because bash 3.2 has no
# BASHPID for a subshell to record itself by. Perl knows its pid, so the
# wrapper's duties moved here, and the handshake went with its reason: the
# launcher publishes the run's pointers BEFORE this process exists, holding
# the lock itself, so no fast run can finish and release under a publication
# still in flight. What the collapse gives up: a SIGKILL aimed at this process
# alone leaves no party to record a verdict for it -- `status` then reports
# the run as died, and `stop` reaps the dune group it recorded, exactly as
# after a group-wide SIGKILL before.
#
# Signals: a foreground run gets HUP relayed as TERM by its launcher, and a
# detached run must survive its session -- so HUP is ignored here whenever
# OCANNL_TOOL_TESTRUN_BG is set, and defaulted back in dune's child. Once a
# signal or dune's own exit has started the finish, every later signal is
# ignored: the verdict write is the one critical section, and KILL remains
# available. setpgrp is eval-guarded and the group kill falls back to a plain
# kill, for MSYS perl where process groups are shaky. The child deliberately
# KEEPS lock fd 9: whatever dune leaves behind -- in its own group or setsid'd
# out of it -- goes on holding the worktree lock while it can still mutate
# _build, and the lock clears exactly when the last such holder exits. The
# child records its pgid (only once confirmed to LEAD its own group, so a
# group-kill can never hit the caller where setpgrp failed) and its start
# token, so `stop` can reap a group that outlived this process.
#
# For `run`/`start`/`plan` the child is this script's `_resolve` first: the
# batch's resolution runs as the supervisor's first phase, under this cap,
# group and signal handling -- it builds the readers, a dune run in the
# worktree -- and then runs dune as its own child, in the group it leads, so
# the pid and group recorded here are the child's throughout (gh-ocannl-1106);
# once dune exits it records dune's promotion list and exits with dune's
# status (gh-ocannl-1087). It writes `resolved` when the first phase ends; a
# run cut short before that says in its log that dune was not started.
#
# OCANNL_TOOL_TESTRUN_RD: where pgid/gtoken go (the run directory, or a repeat
# iteration's). OCANNL_TOOL_TESTRUN_OWN: the run directory whose identity and
# verdict this process owns -- unset for a repeat iteration, whose coordinator
# owns both and reads this process's exit status instead.
supervisor_perl='
  use POSIX ();
  my $cap = shift;
  # The root of the runner, where the slowest-actions reader lives -- not
  # necessarily the tree under test (a bisect step tests another) -- taken out
  # of the environment before anything is forked: nothing the run starts sees it.
  # Nor any other variable addressed to this process (gh-ocannl-1448): a phase
  # of the run that needs its directory is handed it on its own command line
  # (`env OCANNL_TOOL_TESTRUN_OWN=...`), and consumes it there.
  my $runner = delete $ENV{OCANNL_TOOL_TESTRUN_RUNNER};
  my $rd = delete $ENV{OCANNL_TOOL_TESTRUN_RD};
  my $own = delete $ENV{OCANNL_TOOL_TESTRUN_OWN};
  my $bg = delete $ENV{OCANNL_TOOL_TESTRUN_BG};
  my ($pid, $finishing);
  my $code_of = sub { my $st = shift; ($st & 127) ? 128 + ($st & 127) : $st >> 8 };
  # The start token of THIS process, in the rendering ps_token recomputes for
  # any pid: Linux reads /proc/self/stat (clock ticks); elsewhere ps renders
  # lstart under the same pinned locale/TZ and squeeze. Sampled by the process
  # itself, never by a parent that could catch a recycled pid after a fast
  # exit.
  my $self_token = sub {
    my $tok = "";
    if (open my $sf, "<", "/proc/self/stat") {
      my $s = <$sf>;
      if ($s =~ /\)\s+(.*)$/) {
        my @f = split /\s+/, $1;
        $tok = defined $f[19] ? $f[19] : "";
      }
    } else {
      local $ENV{LC_ALL} = "C";
      local $ENV{TZ} = "UTC";
      $tok = qx{ps -o lstart= -p $$ 2>/dev/null};
      $tok =~ s/\s+$//;
      $tok =~ s/ +/ /g;
    }
    $tok;
  };
  my $write = sub { # path, content -> 1, or 0 with $! set
    open(my $fh, ">", $_[0]) or return 0;
    print $fh $_[1] or return 0;
    close $fh or return 0;
    1;
  };
  # The slowest actions of the run (see explicit_trace_file): the five longest
  # processes of the trace the batch wrote into the run directory, read once
  # the child has exited (dune with it), and published as `slowest` only when
  # the reader succeeds. The reader is a child of its own, in a group of its
  # own so its python goes with it, holding no descriptor past stdio (the
  # worktree lock is fd 9) and with every check of the trace made there, so
  # nothing here can block on the file system: it is polled for its bound (30
  # seconds), then KILLed and polled again, five seconds, and abandoned if
  # even that does not reap it -- the verdict publishes and the lock clears
  # regardless. A reader that fails or times out leaves `slowest-skipped`,
  # its last diagnostic line (the reader quotes offsets, never trace bytes) or
  # the bound it missed, which the digest prints in place of the table -- a
  # dune older than 3.24 writes a trace format it refuses. The trace is
  # deleted on every path, a failed fork included. OCANNL_TOOL_SLOWEST_CAP and OCANNL_TOOL_SLOWEST_READER
  # are seams for tools/test-test-run.sh; an ambient value can only shorten
  # the bound (anything but 1-30 reads as 30), and the reader it names is
  # bounded and KILLed like the real one.
  my $slowest = sub {
    my $trace = "$own/trace.csexp";
    my $bound = $ENV{OCANNL_TOOL_SLOWEST_CAP};
    $bound = 30 unless defined $bound && $bound =~ /^[1-9][0-9]?$/ && $bound <= 30;
    my $reader = $ENV{OCANNL_TOOL_SLOWEST_READER};
    $reader = (defined $runner && length $runner ? "$runner/" : "") . "tools/action-durations.sh"
      unless defined $reader && length $reader;
    my $c = fork();
    unless (defined $c) { unlink $trace; return }
    if (!$c) {
      $SIG{$_} = "DEFAULT" for qw(ALRM INT TERM HUP);
      POSIX::close($_) for 3 .. 255;
      eval { setpgrp(0, 0) };
      POSIX::_exit(1) unless -f $trace;
      open(STDIN, "<", "/dev/null");
      open(STDERR, ">", "$own/slowest.err");
      open(STDOUT, ">", "$own/slowest.tmp") or POSIX::_exit(126);
      exec("bash", $reader, "-n", "5", $trace);
      POSIX::_exit(127);
    }
    my $st;
    for (1 .. 10 * $bound) {
      if (waitpid($c, POSIX::WNOHANG()) == $c) { $st = $?; last }
      select undef, undef, undef, 0.1;
    }
    unless (defined $st) {
      kill("KILL", -$c) or kill("KILL", $c);
      for (1 .. 50) {
        last if waitpid($c, POSIX::WNOHANG()) != 0;
        select undef, undef, undef, 0.1;
      }
    }
    my $why;
    if (defined $st && $st == 0) {
      rename("$own/slowest.tmp", "$own/slowest") or unlink "$own/slowest.tmp";
    } else {
      unlink "$own/slowest.tmp";
      if (!defined $st) {
        $why = "the reader did not finish within ${bound}s";
      } elsif (open(my $ef, "<", "$own/slowest.err")) {
        my @lines = grep { /\S/ } <$ef>;
        close $ef;
        $why = $lines[-1] if @lines;
      }
    }
    unlink "$own/slowest.err";
    if (defined $why) {
      chomp $why;
      $write->("$own/slowest-skipped", substr($why, 0, 300) . "\n");
    }
    unlink $trace;
  };
  # Every exit path ends here. The verdict file is written aside and renamed:
  # its EXISTENCE is the completion signal status/wait key on, so it must
  # never be observable empty. Retried with backoff -- a filesystem that
  # filled up during the run may clear, and giving up silently would
  # downgrade a real verdict into a generic "died without a verdict".
  my $finish = sub {
    my $code = shift;
    $finishing = 1;
    $SIG{$_} = "IGNORE" for qw(ALRM INT TERM HUP);
    alarm 0;
    if ($own) {
      $slowest->();
      # A `fresh` run builds in a directory of its own, which goes with the
      # run on every path -- the caps and cancellations included -- once dune
      # is gone; the digest reports one that survived.
      if (open(my $mf, "<", "$own/mode")) {
        my $m = <$mf>;
        close $mf;
        if (defined $m && $m eq "fresh\n" && -d "$own/build" && !-l "$own/build") {
          require File::Path;
          File::Path::remove_tree("$own/build");
        }
      }
      print STDOUT "exit: $code\n";
      my $ok = 0;
      for my $try (1 .. 3) {
        if ($write->("$own/exit.tmp", "$code\n") && rename("$own/exit.tmp", "$own/exit")) {
          $ok = 1;
          last;
        }
        sleep 2 if $try < 3;
      }
      print STDOUT "test-run: FAILED to record verdict $code (filesystem?)\n" unless $ok;
    }
    exit $code;
  };
  my $blast = sub { my $sig = shift; kill($sig, -$pid) or kill($sig, $pid) };
  my $reap = sub {
    my $code = shift;
    # A signal landing once the finish has begun -- after dune was reaped, or
    # during the verdict write -- must neither blast the stale pid/group (a
    # recycled pid could make that an innocent process) nor replace the real
    # verdict with the signal code.
    return if $finishing;
    $finishing = 1;
    if ($pid) {
      # The WNOHANG probe covers the statements between the reaping waitpid
      # and the bookkeeping -- there, $? still holds the status the main flow
      # has not yet read (captured before our own waitpid resets it).
      my $saved = $?;
      my $r = waitpid($pid, POSIX::WNOHANG());
      $finish->($code_of->($saved)) if $r == -1;
      $finish->($code_of->($?)) if $r == $pid;
      $blast->("TERM");
      # Grace for the WHOLE group, not just the leader: a leader that exits
      # fast must not collapse the grace of its descendants to one polling
      # interval. (Where the child has no group of its own, the group probe
      # fails and this degrades to the plain leader wait.)
      my $gone = 0;
      for (1 .. 50) {
        $gone = 1 if !$gone && waitpid($pid, POSIX::WNOHANG()) != 0;
        last if $gone && !kill(0, -$pid);
        select undef, undef, undef, 0.1;
      }
      if (!$gone || kill(0, -$pid)) {
        $blast->("KILL");
        # Bounded reap: a child stuck in uninterruptible kernel I/O keeps
        # even SIGKILL pending, and a blocking waitpid here would hang the
        # supervisor -- the cap must report its 142/143 regardless; the
        # stuck process then keeps the worktree lock through its fd 9.
        unless ($gone) {
          for (1 .. 50) {
            last if waitpid($pid, POSIX::WNOHANG()) != 0;
            select undef, undef, undef, 0.1;
          }
        }
        # Then let the group VANISH before finishing, bounded: the KILLed
        # members are orphans now, and until init reaps them a Linux group
        # probe still counts their corpses -- a repeat coordinator asking
        # right after this exit would find a reachable group with no leader
        # to verify and refuse to reuse its build tree. Under an init that
        # never reaps, the corpses are permanent and this wait ends on its
        # bound; that refusal is then the fail-closed answer it was built
        # to give.
        for (1 .. 10) {
          last unless kill(0, -$pid);
          select undef, undef, undef, 0.1;
        }
      }
    }
    # Cut short in the first phase: the child was still resolving the
    # batch (it marks the end of that phase with `resolved`), so no dune ran.
    if ($own && !-e "$own/resolved") {
      print STDOUT "test-run: " . ($code == 142 ? "the cap expired" : "cancelled")
        . " while the launch resolved the batch\x27s backends; dune was not started\n";
    }
    $finish->($code);
  };
  # Armed BEFORE the identity record and the fork: a stop landing in the
  # launch window is honoured -- a pre-fork signal finishes with its code
  # before dune ever starts, and the verdict says CANCELLED.
  $SIG{ALRM} = sub { $reap->(142) };
  $SIG{INT} = sub { $reap->(130) };
  $SIG{TERM} = sub { $reap->(143) };
  $SIG{HUP} = $bg ? "IGNORE" : sub { $reap->(129) };
  if ($own) {
    # STDOUT is the run log, shared with dune: unbuffered, so the sentinel
    # lands before the verdict file appears.
    $| = 1;
    # Identity first, token before pid: a reader that finds the pid finds the
    # token too. A run whose supervisor cannot record itself would be
    # invisible to status and uncancellable by stop while dune ran on, so
    # dune is not started: nothing ran, and the verdict (126) says so.
    unless ($write->("$own/ptoken", $self_token->() . "\n") && $write->("$own/pid", "$$\n")) {
      print STDOUT "test-run: cannot record the supervisor identity in $own: $!\n";
      $finish->(126);
    }
  }
  # Perl defers handlers to safe points, including the return from fork
  # before its assignment. Keep the signals pending until each process has
  # established its ownership: the parent knows the child pid; the child has
  # reset the inherited handlers. Restore the original mask, not an empty one.
  my $fork_signals = POSIX::SigSet->new(POSIX::SIGINT(), POSIX::SIGTERM(),
    POSIX::SIGHUP(), POSIX::SIGALRM());
  my $old_mask = POSIX::SigSet->new();
  POSIX::sigprocmask(POSIX::SIG_BLOCK(), $fork_signals, $old_mask)
    or die "test-run: block fork signals: $!\n";
  $pid = fork();
  if (!defined $pid || $pid) {
    POSIX::sigprocmask(POSIX::SIG_SETMASK(), $old_mask)
      or die "test-run: restore parent signal mask: $!\n";
  }
  unless (defined $pid) {
    print STDOUT "test-run: fork: $!\n" if $own;
    $finish->(126);
  }
  if (!$pid) {
    # SIG_IGN survives fork AND exec (HUP is ignored above in detached mode,
    # and the launching shell may have inherited an ignored INT): without this
    # reset dune would start deaf to the very signals the cap and `stop` rely
    # on, degrading every cancellation to the KILL escalation.
    $SIG{$_} = "DEFAULT" for qw(ALRM INT TERM HUP);
    POSIX::sigprocmask(POSIX::SIG_SETMASK(), $old_mask)
      or die "test-run: restore child signal mask: $!\n";
    # Perl otherwise reserves the right to close inherited descriptors above
    # $^F at exec. fd 9 is the worktree lock and repeat additionally supplies
    # fd 6/8 as its descendant witness; keep that containment state attached
    # to Dune and to anything Dune launches.
    $^F = 9;
    eval { setpgrp(0, 0) };
    eval {
      if ($rd && getpgrp(0) == $$) {
        open my $fh, ">", "$rd/pgid" or die;
        print $fh $$;
        close $fh;
        # The leader publishes its OWN start token before exec: a token
        # sampled later by a parent could capture a recycled pid if the
        # leader exited first.
        my $tok = $self_token->();
        if ($tok ne "") {
          open my $gf, ">", "$rd/gtoken" or die;
          print $gf "$tok\n";
          close $gf;
        }
      }
    };
    exec @ARGV;
    exit 127;
  }
  alarm $cap if $cap > 0;
  waitpid($pid, 0);
  my $st = $?;
  # The finish flagged BEFORE anything else -- a handler firing in between
  # reads the status through its WNOHANG probe and finishes with the real
  # code -- and the group reaped BEFORE the verdict: a run is not finished
  # while a descendant dune left in its group still runs, holding lock fd 9
  # (every later launch refused) or, having closed it, mutating _build
  # under no lock at all. Only a group the child confirmed LEADING (its
  # pgid record) is reaped: where setpgrp failed there is no group of ours
  # to signal, only whatever process the number lands on next. The group
  # id is the pid of the reaped leader, which POSIX forbids reusing while
  # the group exists, so a reachable group is what dune left behind -- the
  # residual, stated: a group that emptied and whose id a fresh leader took
  # within this two-second grace. (An escapee that called setsid is beyond
  # this and every census; the lock it inherited is what stops the next
  # launch, and `stop` reaps it.)
  $finishing = 1;
  my $led = 0;
  if ($rd && open(my $pf, "<", "$rd/pgid")) {
    my $recorded = <$pf>;
    $led = 1 if defined $recorded && $recorded =~ /^\s*(\d+)\s*$/ && $1 == $pid;
  }
  if ($led && kill(0, -$pid)) {
    kill("TERM", -$pid);
    for (1 .. 20) {
      last unless kill(0, -$pid);
      select undef, undef, undef, 0.1;
    }
    kill("KILL", -$pid) if kill(0, -$pid);
  }
  $pid = 0;
  $finish->($code_of->($st));
'

# Take the per-worktree lock on fd 9, non-blocking. perl takes it and exits;
# the lock lives on the open file DESCRIPTION, which every process of the
# run inherits through fd 9 -- released by the kernel when the last holder
# closes: the supervisor's exit in the common case, otherwise the exit of the
# last descendant dune left behind, with nothing to reclaim after a crash
# (see sweep.sh). The launcher closes its own copy as soon as the supervisor
# has inherited it.
#
# Acquisition and the owner pointer's rewrite happen inside ONE process, so
# the pointer names THIS run from the instant the lock is held: there is no
# moment at which the lock is held without an owner. A stop of the previous
# run cannot blame its leftovers for our presence on the lock file and TERM
# this very launcher, and a launcher stalled anywhere after acquisition is
# reachable through `stop <run>` on the owner it published. That is why
# new_run creates the run directory BEFORE the lock is taken: the directory
# is the pointer's target. Exit codes: 1 lock busy, 2 pointer unwritable.
take_lock() {
  # Transitional (delete, with the `.gitignore` rule for these files, once
  # no run launched before gh-ocannl-606 can be in flight): a `start`/
  # `repeat` of the previous version holds the lock that version kept
  # beside the worktree, which this acquisition would not see -- and two
  # managed runs in one worktree is what the lock exists to refuse. So
  # where that file exists it is TAKEN, on fd 5, not merely probed: held
  # through this acquisition and inherited by the run like fd 9, it keeps
  # a launcher of the previous version refused for as long as this run
  # holds the new lock. Its owner pointer sits beside it, so a refusal can
  # still name the run. Once held, the residue that version left -- lock,
  # owner and launcher record -- is removed: unlocked, it belongs to no run,
  # and it is what made every worktree that ran a suite ignored-dirty.
  if [ -e "$PWD/.test-run.lock" ]; then
    exec 5>>"$PWD/.test-run.lock" || die "cannot open the previous version's lock file"
    if ! perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 1)' <&5; then
      rm -rf "$run_dir"
      owner=$(cat "$PWD/.test-run.lock.owner" 2>/dev/null)
      owner=$(printf %q "${owner:-last}")
      echo "test-run: a test-run of the previous version is still active in this worktree" >&2
      echo "  (it holds $PWD/.test-run.lock); check it with: tools/test-run.sh status $owner" >&2
      echo "(a stale one can be stopped with: tools/test-run.sh stop $owner)" >&2
      exit 2
    fi
    rm -f "$PWD/.test-run.lock" "$PWD/.test-run.lock.owner" "$PWD/.test-run.lock.launcher"
  fi
  exec 9>>"$LOCK" || die "cannot open lock file $LOCK"
  # The pointer is written aside and renamed into place: a reader refused by
  # the lock in the same instant sees the previous owner or this run, never
  # a truncated or half-written path that would send it to `last` instead
  # of the actual holder.
  perl -e '
    use Fcntl ":flock";
    exit 1 unless flock(STDIN, LOCK_EX | LOCK_NB);
    my $tmp = "$ARGV[0].tmp.$$";
    open(my $fh, ">", $tmp) or exit 2;
    print $fh "$ARGV[1]\n" or exit 2;
    close $fh or exit 2;
    rename($tmp, $ARGV[0]) or do { unlink $tmp; exit 2 };
    exit 0;
  ' "$OWNER" "$run_dir" <&9
  case $? in
    0) ;;
    2)
      rm -rf "$run_dir"
      die "cannot record the lock owner pointer at $OWNER (state directory not writable?)"
      ;;
    *)
      # A run directory made for a launch that is refused was never
      # published; it goes, or `list` would show a dead run that never was.
      rm -rf "$run_dir"
      # The owner pointer, not `last`: `last` is the run published most
      # recently, and what the reader needs to inspect or stop is the lock's
      # HOLDER. %q, so a runs directory containing spaces or metacharacters
      # survives copy-pasting the recovery commands.
      owner=$(cat "$OWNER" 2>/dev/null)
      owner=$(printf %q "${owner:-last}")
      echo "test-run: another test-run is active in this worktree; check it with:" >&2
      echo "  tools/test-run.sh status $owner" >&2
      echo "(a stale one can be stopped with: tools/test-run.sh stop $owner)" >&2
      exit 2
      ;;
  esac
}

# The revision a run tests, as a launch-time fact (gh-ocannl-992): `head` holds
# the checkout's HEAD commit and `dirty` its `git status --porcelain` -- empty
# for a clean tree, otherwise one line per uncommitted path, untracked ones
# included. Without them a run directory said what ran and how it ended but not
# on which source, and a reader had to infer it: from the worktree's HEAD now
# (which a reset or a later commit moves, and a commit-time guard cannot tell
# apart), or from a worker's transcript. Written in that order, so `head`'s
# presence means both are on record; they are optional for every reader, since
# a run outside a Git checkout, or one git cannot describe, records neither and
# launches as before. What they cannot say: an edit made after the launch --
# during a fleet slot's wait, or while dune runs -- is not in them; nor is
# anything git does not track or would not: an ignored file (a root
# `ocannl_config`, whose settings a test can pick up through the ancestor
# search) and the environment (`OCANNL_*`) are configuration, not source, and
# a report that depends on them states them itself.
#
# Recorded only where this script's root IS the checkout's top level: a copy
# sitting inside some other repository (a harness fixture under a checkout)
# would otherwise record that repository's HEAD for a tree it does not
# describe. The repository-selecting environment (GIT_DIR, GIT_WORK_TREE,
# GIT_INDEX_FILE, ...: git's own --local-env-vars list) is dropped for the
# same reason -- a launch from inside a git hook inherits them. Untracked files
# are listed whatever `status.showUntrackedFiles` says: a new test source is
# exactly the edit a dirty record exists to show. And the status takes no
# optional lock: git would otherwise rewrite the index to refresh its stat
# cache, which is a write into the checkout (and a transient index.lock that
# can fail the caller's own concurrent `git commit`).
#
# HEAD is read on both sides of the status, and the pair retaken while it
# moved: a commit or reset landing in between would otherwise pair the OLD
# commit with the NEW tree's clean status -- a record certifying a revision
# that did not run. Three attempts, then neither file (a checkout committing
# that fast is not one a launch-time fact can describe).
record_checkout() {
  (
    command -v git >/dev/null 2>&1 || exit 0
    # shellcheck disable=SC2046 # a list of variable names, split on purpose
    unset $(git rev-parse --local-env-vars 2>/dev/null)
    [ "$(git rev-parse --is-inside-work-tree --show-prefix 2>/dev/null)" = true ] || exit 0
    attempt=1
    while :; do
      head=$(git rev-parse --verify --quiet HEAD 2>/dev/null) || exit 0
      case $head in '' | *[!0-9a-f]*) exit 0 ;; esac
      if ! GIT_OPTIONAL_LOCKS=0 git status --porcelain=v1 --untracked-files=normal \
           >"$run_dir/dirty" 2>/dev/null; then
        rm -f "$run_dir/dirty"
        exit 0
      fi
      [ "$(git rev-parse --verify --quiet HEAD 2>/dev/null)" = "$head" ] && break
      [ "$attempt" -lt 3 ] || { rm -f "$run_dir/dirty"; exit 0; }
      attempt=$((attempt + 1))
    done
    printf '%s\n' "$head" >"$run_dir/head"
  )
}

# Dune's own promotion list, recorded at the run's end (gh-ocannl-1087): the
# `promotions` file holds one source path per line, the files `dune promote`
# would update right then, and is empty when dune has nothing to promote. The
# digest used to infer this from the log, and four review rounds of
# staging#827 each found a log shape the inference misread (colored and
# patdiff headers, a chosen --diff-command, a quoted `.corrected` stanza, a
# hunk above the scanned tail): dune's output is not a contract, and `dune
# promotion list` (since 3.14, below the 3.20 floor) is. It printed on stderr
# before 3.22 and on stdout since, so `dune --version` picks the stream. The
# run's last phase writes it (see `_resolve`), after dune exits and before
# the supervisor publishes the verdict, so every reader of a verdict finds
# it. It is kept in the run directory because dune's list is not: the next
# build in the worktree, an unrelated one included, replaces it, after which
# `dune promote` says "Nothing to promote" for a run whose digest offered
# promotion (staging#840).
#
# Optional, like `head`: without it the digest falls back to reading the log.
# It is not written where the list cannot be taken faithfully: the argv names
# another build directory or root (`--build-dir`, `--root`, or any
# abbreviation cmdliner accepts; erring towards "names one" costs only the
# record), dune's version or list cannot be read, the cap is nearly spent
# (recording must never turn a finished run into a TIMEOUT), or a line is not
# a path in this worktree (a warning on the pre-3.22 stream). A missing build
# directory is an empty list without asking dune: its list lives there, and
# asking would recreate the directory a `dune clean` just removed. The list's
# own trace goes to a file of the run's, so the build's `_build/trace.csexp`
# survives it (dune before 3.22 still rewrites `_build/log`, as any later
# dune command in the worktree would). What the record is NOT: this run's
# diffs. It is dune's list when the run ended, which for a run that ran
# nothing (a refused slot or invocation) is an older build's, so the digest
# reads it only under a pass or FAIL verdict.
explicit_build_root() { # dune argv; 0 iff it names a build dir or root before dune's `--`
  for arg do
    case $arg in
      --) return 1 ;;
      --bu* | --ro*) return 0 ;;
    esac
  done
  return 1
}
record_promotions() { # in `_resolve`, after dune exited, whose SECONDS the cap's are
  local build_dir=${DUNE_BUILD_DIR:-_build} version major minor stream
  if [ ! -d "$build_dir" ]; then
    : >"$run_dir/promotions" 2>/dev/null || rm -f "$run_dir/promotions"
    return 0
  fi
  version=$(promotion_bounded "$DUNE" --version 2>/dev/null) || return 0
  case $version in [0-9]*.[0-9]*) ;; *) return 0 ;; esac
  major=${version%%.*} minor=${version#*.}
  minor=${minor%%[!0-9]*}
  case $major in *[!0-9]*) return 0 ;; esac
  if [ "$major" -gt 3 ] || { [ "$major" = 3 ] && [ "$minor" -ge 22 ]; }; then
    stream=stdout
  else
    stream=stderr
  fi
  # Before 3.22, list recomputes diffs; a presentation command such as `-`
  # would hide registered corrections. Discovery uses an ordinary diff.
  set -- "$DUNE" promotion list --diff-command=diff --trace-file="$run_dir/promotions.trace"
  if [ "$stream" = stdout ]; then
    promotion_bounded "$@" >"$run_dir/promotions.tmp" 2>/dev/null
  else
    promotion_bounded "$@" 2>"$run_dir/promotions.tmp" >/dev/null
  fi && promotion_lines <"$run_dir/promotions.tmp" >"$run_dir/promotions.list" &&
    mv -f "$run_dir/promotions.list" "$run_dir/promotions"
  rm -f "$run_dir/promotions.tmp" "$run_dir/promotions.list" "$run_dir/promotions.trace"
  # Corrected bytes belong to this record too, before the lock is released.
  # All-or-nothing publication: a query/copy failure keeps the useful list,
  # but must never advertise an incomplete set as recoverable.
  [ -s "$run_dir/promotions" ] || return 0
  case $rc in 0 | 1) ;; *) return 0 ;; esac
  ! dune_refusal "$run_dir/log" "$(cat "$run_dir/prelude" 2>/dev/null)" >/dev/null || return 0
  local saved="$run_dir/promotion-files.tmp" line n=0 ok=1
  mkdir "$saved" || return 0
  cp "$run_dir/promotions" "$saved/paths" || ok=0
  while IFS= read -r line; do
    [ "$ok" = 1 ] || break
    n=$((n + 1))
    # `show` adds a framing newline (and is absent on 3.20). The public
    # diff command receives the actual original/correction paths instead.
    # Copy bytes via its second argument; exit 1 means "diff found" to Dune.
    # The destination travels through the environment, never shell source.
    OCANNL_TOOL_PROMOTION_CAPTURE="$saved/$n.corrected" promotion_bounded "$DUNE" promotion diff \
      --trace-file="$run_dir/promotions.trace" \
      --diff-command="perl -MFile::Copy -e 'copy(\$ARGV[1], \$ENV{OCANNL_TOOL_PROMOTION_CAPTURE}) or exit 2; exit 1'" \
      -- "$line" >/dev/null 2>&1 || ok=0
    [ -f "$saved/$n.corrected" ] || ok=0
    if [ -f "$line" ]; then
      promotion_bounded cp -- "$line" "$saved/$n.original" || ok=0
    elif [ -e "$line" ] || [ -L "$line" ]; then
      ok=0
    else
      : >"$saved/$n.absent" || ok=0
    fi
  done <"$run_dir/promotions"
  rm -f "$run_dir/promotions.trace"
  if [ "$ok" = 1 ]; then
    mv "$saved" "$run_dir/promotion-files" || rm -rf "$saved"
  else
    rm -rf "$saved"
  fi
}
# Copies dune's list, failing on the first line that is not a relative path
# whose directory exists here (every promotion targets a file beside its dune
# stanza) -- the one guard against reading a warning as a path.
promotion_lines() {
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}
    case $line in '' | /* | \\* | [A-Za-z]:* | . | .. | ./* | ../* | */./* | */../* | */. | */..) return 1 ;; esac
    case $line in */*) [ -d "${line%/*}" ] || return 1 ;; esac
    printf '%s\n' "$line" || return 1
  done
}
# One dune query, bounded by what the cap leaves (five seconds kept back for
# the supervisor) and by 30 seconds; refused outright when nothing is left.
promotion_bounded() { # command...
  local bound=30 left
  if [ "$cap" -gt 0 ]; then
    left=$((cap - SECONDS - 5))
    [ "$left" -ge 1 ] || return 1
    [ "$left" -ge "$bound" ] || bound=$left
  fi
  perl -e 'alarm shift; exec @ARGV or exit 127' "$bound" "$@" </dev/null
}

# The run's slowest actions, recorded once it ends: `slowest` holds what
# tools/action-durations.sh prints for the five longest processes in the
# batch's own dune trace, and the digest shows it, so per-action duration data
# (the evidence behind TEST_CAP_DEFAULT) accumulates in every run directory
# with no manual step, and a TIMEOUT digest shows how close the other tests
# came. The reader decodes only timing fields, never the environment a trace
# also records (gh-ocannl-1280). The trace is the run's own: `_resolve` hands
# the batch's dune `--trace-file=<run dir>/trace.csexp`, right after its
# subcommand, so the worktree's shared `_build/trace.csexp` -- which a
# concurrent manual dune may be writing, under a build lock this script does
# not hold -- is neither read nor touched by the batch, and a run whose dune
# never started (a refused or cancelled slot wait) has no trace to misread.
# (The launch's other dune invocations -- the readers' build in
# batch_resolve, a GPU batch's device-probe build -- write the shared trace
# as any dune command does, after taking dune's build lock; what must not
# happen is a write or unlink outside that lock.) The supervisor
# reads it as the verdict's last step, once the child has exited, on every
# exit path -- the caps' included, where `_resolve` itself is killed -- with
# the caps disarmed, so recording never turns a finished run into a TIMEOUT;
# the reader is bounded (30 seconds, then KILL and a bounded reap) and never
# holds the worktree lock, so a reader that cannot even be reaped still lets
# the verdict publish and the lock clear (see supervisor_perl). The trace is
# then deleted: the record is the five rows, and an environment dump per run
# would pile up under the runs directory. Nothing is recorded where the argv
# names a trace file of its own (`--trace-file`, or any abbreviation of it;
# erring towards "names one" costs only the record), or where the first word
# is not one of the subcommands known to take the option.
explicit_trace_file() { # dune argv; 0 iff it names a trace file before dune's `--`
  for arg do
    case $arg in
      --) return 1 ;;
      --tr*) return 0 ;;
    esac
  done
  return 1
}

# `fresh`'s operands (see the header): per-test aliases, each once, and `-j N`,
# into dune's argv `build --force --cache=disabled ALIAS... [-j N]` (fresh_args)
# and the alias list (fresh_aliases). The run's build directory goes in at
# `_resolve`, once the run directory exists. DIR must be a plain relative
# path, each segment starting with a letter, digit or `_` -- so no `.` or `..`
# segment and nothing absolute -- since it names where the test's action runs.
fresh_argv() { # operands
  local arg jobs= seen verb=fresh form='fresh [--cap N] [--test-cap N] @DIR/KIND-NAME... [-j N]' other='use `run`'
  if [ "$sub" = repeat ]; then
    verb="repeat --direct" form='repeat [--cap N] [--test-cap N] [--alone] --direct N @DIR/KIND-NAME... [-j N]'
    other='repeat it without --direct'
  fi
  fresh_aliases=() fresh_args=() fresh_jobs=
  while [ $# -gt 0 ]; do
    arg=$1
    shift
    case $arg in
      -j)
        [ $# -gt 0 ] || die "$verb: -j needs a width"
        jobs=$1
        shift ;;
      -j*) jobs=${arg#-j} ;;
      @*)
        if ! [[ $arg =~ ^@@?([A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*)/(runtest|slow|train)-[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]]; then
          die "$verb: not a per-test alias: $arg
  name each test as @DIR/runtest-NAME, @DIR/slow-NAME or @DIR/train-NAME (@@ for DIR alone);
  an aggregate (@runtest, @DIR/slow, @DIR/scans) or a directory-less alias is not one -- $other"
        fi
        for seen in ${fresh_aliases[@]+"${fresh_aliases[@]}"}; do
          [ "$seen" != "$arg" ] || die "$verb: $arg is named twice"
        done
        fresh_aliases+=("$arg") ;;
      *)
        die "$verb takes per-test aliases and -j N only, not: $arg
  usage: tools/test-run.sh $form
  (it passes --force, --cache=disabled and a build directory of its own; see the header)" ;;
    esac
  done
  [ ${#fresh_aliases[@]} -gt 0 ] ||
    die "$verb: name at least one per-test alias (@DIR/runtest-NAME, @DIR/slow-NAME, @DIR/train-NAME)"
  if [ -n "$jobs" ]; then
    case $jobs in '' | *[!0-9]* | 0*) die "$verb: -j wants a positive width, got: $jobs" ;; esac
  fi
  fresh_jobs=$jobs
  fresh_args=(build --force --cache=disabled "${fresh_aliases[@]}")
  [ -z "$jobs" ] || fresh_args+=(-j "$jobs")
}

# Which processes of a trace are a named test's action: the rule `fresh` times
# a test by (see the header) and `repeat --direct` replays a test by, one copy
# for both. Input: the alias list, then tools/action-durations.sh's `--tsv`
# rows. `mode=timings` prints `fresh`'s per-alias line (fresh_timings);
# `mode=rows` prints `ALIAS<TAB>ROW` for each row (1-based) that is the
# alias's action (_direct_plan).
attribution_awk='
    function ends(s, t) { return length(s) >= length(t) && substr(s, length(s) - length(t) + 1) == t }
    function named(i) { # the alias i: a, rec (a recursive @), p, d (its DIR), base, name
      a = alias[i]; rec = substr(a, 1, 2) != "@@"; p = substr(a, rec ? 2 : 3)
      d = p; sub(/\/[^\/]*$/, "", d); base = substr(p, length(d) + 2)
      name = base; sub(/^[^-]*-/, "", name)
    }
    function in_dir(x) { return x == d || (rec && index(x, d "/") == 1) }
    function dir_of(s, x) { # the directory a row names: @D/ALIAS, D/FILE or (in D)
      if (s ~ /^\(in .*\)$/) return substr(s, 5, length(s) - 5)
      x = s; sub(/^@/, "", x); sub(/\/[^\/]*$/, "", x)
      return x
    }
    # A target is an output named after the test when its file name is
    # PRE.<anything> and it is not one of the compile artifacts the tests
    # executable leaves under the same name (NAME.exe, NAME.pp.ml, ...).
    function produces(s, pre, b) {
      b = s; sub(/^.*\//, "", b)
      if (substr(s, 1, 1) == "@" || substr(s, 1, 1) == "(" || !in_dir(dir_of(s))) return 0
      if (b ~ /\.(exe|bc|bc-for-jsoo|js|cm[a-z]*|o|a|so|dll|d|pp\.ml|pp\.mli)$/) return 0
      return index(b, pre ".") == 1
    }
    function action_of(j, k, s) { # a target of row j is the alias action, or the test output
      for (k = 1; k <= ntg[j]; k++) {
        s = tg[j, k]
        if (s == "@" p || (rec && index(s, "@" d "/") == 1 && ends(s, "/" base))) return 1
        if (produces(s, name)) return 1
        # BASE-VARIANT names BASE.VARIANT.* (runtest-top_down_prec-extension)
        if (match(name, /-[^-]*$/) && produces(s, substr(name, 1, RSTART - 1) "." substr(name, RSTART + 1))) return 1
      }
      return 0
    }
    NR == FNR { alias[++n] = $0; next }
    {
      secs[++m] = $1; ex[m] = $2; prog[m] = $3; act[m] = $4; sub(/ \+[0-9]+$/, "", act[m])
      if (NF < 5) { ntg[m] = 1; tg[m, 1] = act[m] }
      else { ntg[m] = NF - 4; for (k = 5; k <= NF; k++) tg[m, k - 4] = $k }
    }
    END {
      for (i = 1; i <= n; i++) {
        named(i)
        for (j = 1; j <= m; j++) if (action_of(j)) { hits[i] = hits[i] " " j; claimed[j] = 1 }
        how[i] = "action"
      }
      # Only for a test none of whose processes was found so: its program
      # NAME.exe, in its directory, in a row no other named test claimed.
      for (i = 1; i <= n; i++) {
        if (hits[i] != "") continue
        named(i)
        how[i] = "program"
        for (j = 1; j <= m; j++)
          if (!claimed[j] && prog[j] == name ".exe" && in_dir(dir_of(act[j]))) {
            hits[i] = hits[i] " " j; claimed[j] = 1
          }
      }
      # `rows` mode, for the plan of repeat --direct: the rows that are each named action.
      if (mode == "rows") {
        for (i = 1; i <= n; i++) {
          nh = split(hits[i], hl, " ")
          for (q = 1; q <= nh; q++) printf "%s\t%d\n", alias[i], hl[q]
        }
        exit 0
      }
      for (i = 1; i <= n; i++) {
        nh = split(hits[i], hl, " "); total = 0; exits = ""
        for (q = 1; q <= nh; q++) {
          j = hl[q]; total += secs[j]
          if (index("," exits ",", "," ex[j] ",") == 0) exits = exits (exits == "" ? "" : ",") ex[j]
        }
        state = nh ? "ran" : "not-run"
        if (exits == "") exits = "-"
        if (!nh) how[i] = "-"
        printf "%s\t%s\t%.2f\t%d\t%s\t%s\n", alias[i], state, total, nh, exits, how[i]
      }
    }'

# A `fresh` run's per-test times (see the header), in `_resolve` once dune has
# exited, from the run's own trace: the shipping reader's `--tsv` rows -- the
# same whitelisted fields its table prints (gh-ocannl-1280), every target of a
# process included -- bounded like the promotion list, within what is left of
# the run's cap (a test that ends near the cap is UNTIMED, never a TIMEOUT
# after dune passed), then attributed to the named aliases. `timings`
# holds one line per alias, `ALIAS<TAB>ran|not-run<TAB>SECONDS<TAB>PROCS<TAB>EXITS<TAB>HOW`,
# HOW naming the rule that found its processes (`action`, `program`, `-`);
# a reader that fails leaves `timings-skipped` with its last diagnostic
# instead. A trace holding no process at all is the reader's refusal, and
# means every named test is NOT RUN, not that the times are unreadable.
fresh_timings() { # <trace file>
  local tsv=$run_dir/fresh-processes.tsv err=$run_dir/fresh-processes.err why
  rm -f "$run_dir/timings" "$run_dir/timings.tmp" "$run_dir/timings-skipped"
  if [ "$cap" -gt 0 ] && [ $((cap - SECONDS - 5)) -lt 1 ]; then
    echo "the run's cap was nearly spent when dune exited" >"$run_dir/timings-skipped"
    return 0
  fi
  if ! promotion_bounded bash "$runner_root/tools/action-durations.sh" --tsv -n 0 "$1" >"$tsv" 2>"$err" 9>&-; then
    if grep -q '^action-durations.sh: no process in ' "$err" 2>/dev/null; then
      : >"$tsv"
    else
      why=$(grep . "$err" 2>/dev/null | tail -n 1 | cut -c1-300)
      [ -n "$why" ] || why="the trace reader failed, or did not finish within its bound"
      printf '%s\n' "$why" >"$run_dir/timings-skipped"
      rm -f "$tsv" "$err"
      return 0
    fi
  fi
  awk -F '\t' -v mode=timings "$attribution_awk" "$run_dir/fresh-aliases" "$tsv" >"$run_dir/timings.tmp" 2>"$err" &&
    mv -f "$run_dir/timings.tmp" "$run_dir/timings" || {
      rm -f "$run_dir/timings.tmp"
      printf 'the per-test times could not be recorded: %s\n' \
        "$(grep . "$err" 2>/dev/null | head -n 1 | cut -c1-200)" >"$run_dir/timings-skipped"
    }
  rm -f "$tsv" "$err"
}

new_run() {
  run_dir=$RUNS/$(date -u +%Y%m%dT%H%M%SZ)-$$
  mkdir "$run_dir" || die "cannot create $run_dir"
  # Every pre-launch metadata write is checked: a quota hit or squatter that
  # slipped through here would let the launch report success while status,
  # wait and the supervisor all operate on a run that cannot be tracked.
  { { printf '%q ' "$@"; echo; } >"$run_dir/cmd" &&
    printf '%s\n' "$cap" >"$run_dir/cap" &&
    printf '%s\n' "$PWD" >"$run_dir/wt" &&
    printf '%s\n' "$RUNS" >"$run_dir/runs" &&
    record_checkout &&
    if explicit_diff_command "$@"; then
      echo 'on the command line' >"$run_dir/diff-command"
    elif [ -n "${DUNE_DIFF_COMMAND:-}" ]; then
      printf 'DUNE_DIFF_COMMAND=%s\n' "$DUNE_DIFF_COMMAND" >"$run_dir/diff-command"
    fi &&
    : >"$run_dir/log"; } || die "cannot write run metadata in $run_dir"
  # `diff-command` records that the run chose its own diff presentation, on
  # dune's command line or through its environment -- a launch-time fact the
  # digest's reading of the log depends on (see `digest`).
  # `runs` is the state root this run's lock and pointers live under: `stop`
  # and retention read them from there, whichever OCANNL_TOOL_TEST_RUNS the
  # caller has -- and its absence marks a run of the version that kept them
  # beside the worktree (see lock_paths_of).
  # Runs are throwaway diagnostics; reap old ones so the directory cannot grow
  # without bound. Deletion demands the full run schema -- the timestamped
  # name AND this script's metadata files -- never mere position under $RUNS,
  # because the override can point $RUNS at a directory with other tenants
  # and "has a file named exit" must not mark those for deletion. Age is the
  # VERDICT file's, since a verdict-less directory may be a live run
  # (`--cap 0` is supported and unbounded); those are reaped only on a much
  # longer leash, as crash leftovers.
  # -mindepth 1 plus the explicit guard: find emits the starting directory
  # itself at depth 0, and an override pointing $RUNS at a directory that
  # happens to match the schema must not get the whole state root deleted.
  find "$RUNS" -mindepth 1 -maxdepth 1 -type d -name '2*Z-*' 2>/dev/null |
    while IFS= read -r d; do
      [ "$d" = "$RUNS" ] && continue
      # The find glob is only a pre-filter; deletion requires the EXACT
      # generated shape -- YYYYMMDDTHHMMSSZ-<pid> -- so a stray directory
      # like `2-oldZ-backup` cannot qualify even if it contains files with
      # the metadata names.
      b=$(basename "$d")
      case $b in
        [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z-*) ;;
        *) continue ;;
      esac
      # NON-greedy strip: the prefix pattern pins the first `Z-` to position
      # 15, so this is the entire remainder of the basename -- a name like
      # `...Z-archiveZ-456` leaves `archiveZ-456` here and is rejected,
      # where a greedy strip would leave `456` and pass it.
      case ${b#*Z-} in '' | *[!0-9]*) continue ;; esac
      [ -f "$d/cmd" ] && [ -f "$d/cap" ] || continue
      if [ -f "$d/exit" ]; then
        if [ -n "$(find "$d/exit" -mtime +7 2>/dev/null)" ]; then
          # A completed run whose leftovers still hold ITS worktree's lock
          # keeps its metadata -- deleting it would strand that worktree
          # with the advertised status/stop recovery pointing at nothing.
          lock_still_owned "$d" && continue
          rm -rf "$d"
        fi
      else
        # A verdict-less directory may be a LIVE run (--cap 0 is supported and
        # unbounded, and $RUNS is shared across worktrees): never reap while
        # its owner still runs, and age it by the newest file INSIDE --
        # appending to `log` does not touch the directory's mtime.
        sup_alive "$d" && continue
        # Same held-lock guard as the completed branch: a crashed run's
        # descendant can hold its worktree lock without a verdict on record.
        lock_still_owned "$d" && continue
        [ -z "$(find "$d" -mtime -30 2>/dev/null | head -1)" ] && rm -rf "$d"
      fi
    done
}

# "Is the recorded supervisor still running?" -- answered with the pid AND a
# start-time token, because a bare `kill -0` latches onto whatever process
# recycled the pid after a reboot or a supervisor crash: `status`/`list` would
# report a stale run as active forever and `stop` would TERM an innocent
# process. An empty token (MSYS ps without lstart) degrades to the plain
# pid check rather than failing.
lock_held() { # <lock-file>; exits 0 iff some process holds its flock
  perl -e 'use Fcntl ":flock";
           open(my $fh, ">>", $ARGV[0]) or exit 1;
           exit(flock($fh, LOCK_EX | LOCK_NB) ? 1 : 0)' "$1" 2>/dev/null
}

# Non-writing lock snapshot shared by the public queries and idle. Missing
# locks are idle; inspection errors must never be mistaken for absence.
probe_locks() {
  perl -e '
      use Fcntl ":flock";
      use Errno qw(EWOULDBLOCK EAGAIN ENOENT);
      my @handles;
      for my $path (@ARGV) {
        unless (lstat $path) {
          next if $! == Errno::ENOENT;
          exit 2;
        }
        -f $path or exit 2;
        # Read/write access matches the existing lock on MSYS too, but this
        # probe never writes, creates, truncates, or unlinks a lock file.
        open my $fh, "+<", $path or exit 2;
        flock($fh, LOCK_EX | LOCK_NB)
          or exit(($! == EWOULDBLOCK || $! == EAGAIN) ? 3 : 2);
        push @handles, $fh;
      }
      exit 0;
  ' "$@"
}

# Is the RECORDED worktree of run-dir $1 still locked with $1 as the named
# owner? Then $1's leftovers are what is holding it -- grounds both for
# reaping them (stop) and for keeping $1's metadata alive (retention).
lock_still_owned() {
  lock_paths_of "$1" || return 1
  [ "$(cat "$run_owner" 2>/dev/null)" = "$1" ] && lock_held "$run_lock"
}
recorded_absolute_path() {
  case $1 in
    /*) ;;
    [A-Za-z]:/*) case ${OSTYPE:-} in msys* | cygwin*) ;; *) return 1 ;; esac ;;
    *) return 1 ;;
  esac
  case $1 in *$'\n'* | *$'\r'*) return 1 ;; esac
}

read_query_record() {
  local record
  [ -f "$1" ] && [ -r "$1" ] || return 1
  # Validate the bytes before shell capture strips trailing LF. A record is
  # exactly one nonempty, LF-terminated path; no CR, NUL, or second record.
  record=$(perl -e '
    open my $fh, "<:raw", $ARGV[0] or exit 1;
    my $line = <$fh>;
    defined($line) && $line =~ /\A[^\x00\r\n]+\n\z/ && eof($fh) or exit 1;
    binmode STDOUT;
    print $line;
  ' -- "$1") || return 1
  recorded_absolute_path "$record" || return 1
  printf '%s\n' "$record"
}

# Where run-dir $1's lock and owner pointer live: under the state root it
# recorded in `runs`, keyed by its worktree. A run with no `runs` on record
# was launched by the version that kept both BESIDE the worktree
# (`.test-run.lock`, `.test-run.lock.owner`); its leftovers, if any, hold
# THAT lock, and only those paths can attribute and reap them. Sets
# run_wt, run_runs, run_lock, run_owner; fails when the run recorded no worktree.
# Queries additionally reject present but invalid metadata rather than treating
# it as an old run. Other callers retain their historical best-effort behavior.
lock_paths_of() {
  local r k
  if [ "${2:-}" = query ]; then
    run_wt=$(read_query_record "$1/wt") || return 1
    r=
    if [ -e "$1/runs" ] || [ -L "$1/runs" ]; then
      r=$(read_query_record "$1/runs") || return 1
    fi
  else
    run_wt=$(cat "$1/wt" 2>/dev/null) || return 1
    [ -n "$run_wt" ] || return 1
    r=$(cat "$1/runs" 2>/dev/null)
  fi
  run_runs=$r
  if [ -n "$r" ]; then
    k=$(wt_key_of "$run_wt")
    run_lock=$r/lock-$k
    run_owner=$r/owner-$k
  else
    run_lock=$run_wt/.test-run.lock
    run_owner=$run_wt/.test-run.lock.owner
  fi
}

# One fixed rendering for start-time tokens: lstart is locale- AND
# timezone-formatted, so an unpinned rendering makes the same process print
# differently from a shell in another TZ -- classifying a live run as dead
# and letting stop refuse to reach it.
ps_token() {
  # Linux: starttime in clock ticks from /proc/<pid>/stat -- lstart's
  # one-second resolution cannot tell a pid recycled within the same second
  # from the original. Elsewhere: lstart under a pinned locale/TZ.
  if [ -r "/proc/$1/stat" ]; then
    perl -e '
      open my $fh, "<", "/proc/$ARGV[0]/stat" or exit 0;
      my $s = <$fh>;
      $s =~ /\)\s+(.*)$/ or exit 0;   # comm may contain spaces/parens
      my @f = split /\s+/, $1;        # $f[0] is field 3 (state)
      print defined $f[19] ? $f[19] : "";
    ' "$1" 2>/dev/null
  else
    # Squeezed AND trimmed: ps pads lstart to a fixed width, and tokens are
    # compared as strings against records written by other renderers (the
    # group leader trims its own) -- one canonical form for all writers.
    LC_ALL=C TZ=UTC ps -o lstart= -p "$1" 2>/dev/null |
      tr -s ' ' | sed 's/^ *//; s/ *$//'
  fi
}

proc_identity_matches() { # pid-file token-file -- zombies retain identity
  local pid tok now
  pid=$(cat "$1" 2>/dev/null) || return 1
  # A corrupted or forged pid file must never reach kill: 0 and negative
  # values are POSIX kill specials (caller's group / broadcast), so only a
  # positive decimal integer is a pid at all.
  case $pid in '' | *[!0-9]* | 0) return 1 ;; esac
  kill -0 "$pid" 2>/dev/null || return 1
  tok=$(tr -s ' ' <"$2" 2>/dev/null | sed 's/^ *//; s/ *$//') || tok=
  # No RECORDED identity (MSYS ps without lstart) degrades to the plain pid
  # check -- but where one was recorded, a pid that can no longer produce a
  # token (exited between kill -0 and ps_token, possibly recycled) must
  # fail, not pass.
  [ -z "$tok" ] && return 0
  now=$(ps_token "$pid")
  [ -n "$now" ] || return 1
  [ "$tok" = "$now" ]
}
proc_alive() { # pid-file token-file
  local pid
  proc_identity_matches "$1" "$2" || return 1
  pid=$(cat "$1" 2>/dev/null) || return 1
  # A zombie retains identity but will never publish anything; status/wait
  # must not treat it as live. Empty state (ps without the column) falls
  # through as the portable over-reporting fallback.
  case $(ps -o state= -p "$pid" 2>/dev/null | tr -d ' ') in Z*) return 1 ;; esac
  return 0
}
# The run's OWNER: the supervisor of a `run`/`start`, the coordinator of a
# `repeat` -- the process that publishes the verdict and that `stop` TERMs.
# Transitional: a run recorded by the previous version (no `runs`, see
# lock_paths_of) kept its owner -- the wrapper subshell, or the repeat
# coordinator, in `wpid`/`wtoken` -- apart from the supervisor in `pid`,
# and that owner outlives the supervisor while it publishes, or between
# repeat iterations; delete with the legacy paths.
sup_alive() { proc_alive "$1/pid" "$1/ptoken" || legacy_owner_alive "$1"; }
legacy_owner_alive() { [ ! -f "$1/runs" ] && proc_alive "$1/wpid" "$1/wtoken"; }
# Group signaling demands a RECORDED leader token: proc_alive's empty-token
# fallback exists for supervisor pids on platforms without lstart,
# and is too weak to aim a signal at a whole, possibly recycled, process
# group.
group_verified() { [ -s "$1/gtoken" ] && proc_alive "$1/pgid" "$1/gtoken"; }
group_identity_matches() {
  [ -s "$1/gtoken" ] && proc_identity_matches "$1/pgid" "$1/gtoken"
}

# "Is anything in this process group still RUNNING?" -- the group-scale twin of
# the zombie filter proc_alive applies per pid, and for the same reason: a
# zombie is a process-table entry, so `kill -0 -- -PGID` succeeds on a group
# whose every member has already exited, and group_verified does not rescue the
# check either -- a zombie leader still prints its recorded lstart. Reading such
# a group as alive makes `stop` announce "orphaned process group N ignored TERM"
# for a group holding nothing but corpses, which is exactly the report someone
# consults when working out why a worktree lock will not clear. Under an init
# that reaps, the corpse is transient and the bare probe merely lost a race with
# it; under one that does not -- the ordinary container case -- it is PERMANENT,
# so waiting it out was never the fix (gh-ocannl-742; the same misreading cost
# scripts/setup-ocaml-env.sh 30s on every failed ssh fetch; both callers now
# source the same `scripts/process-group.sh` ladder).
#
# States are not signal-visible, so they are read where the system publishes
# them: /proc on Linux (with the shell's own `read`, no fork per process), `ps`
# on the BSDs and macOS. Where neither answers -- a Cygwin `ps` takes no `-o`,
# and that shell is supported here -- it degrades to the signal alone, which
# over-reports; Cygwin reaps its own children, so the motivating case does not
# arise on that path.
#
# The signal probe stays as a NECESSARY condition rather than only a fallback:
# every caller follows this with a kill to the same group, and keeping it first
# makes this predicate a strict narrowing of the bare `kill -0` it replaces --
# it can turn a phantom alive into dead and never the reverse, whatever the
# state reader sees (a group of another uid, say, which `ps -A` lists and no
# kill of ours could reach).
#
# It remains a CENSUS, and a census is a SNAPSHOT: the glob (or the `ps`) is
# taken at one instant, so a child forked while it is being read is not in it,
# while a leader that exited into a zombie during that instant is. Both callers
# are therefore written so that this answer can shorten a reap or reword a
# report but never SKIP one -- the signals go out on reachability alone. A wrong
# answer then costs a less graceful shutdown, never a survivor left mutating
# _build behind a released worktree lock (Codex review round 1, P1).
# Make the launched run discoverable as `last`. The launcher calls this while
# it holds the lock and BEFORE the supervisor exists: publication and the
# run's completion are then ordered by construction -- no run can finish and
# release the lock under a pointer write still in flight, which is what the
# former three-party publication handshake existed to prevent (gh-ocannl-606).
# The price is a window of milliseconds in which `last` names a run whose
# supervisor has not yet recorded itself: `status` reports that state by
# name, and the launcher reports the launch only once the record exists.
publish_run() { # 0 published; 1 error
  # A PLAIN FILE holding the run directory's path, not a symlink: under MSYS
  # (Git Bash) with the default `winsymlinks` mode, `ln -s` does not create a
  # link at all -- it silently COPIES, so a directory target left a full copy
  # of the run directory at `last-<key>` and every `last` lookup afterwards
  # resolved to nothing. Native symlinks there need a privilege the shell may
  # not hold, so the portable pointer is the file. Rename-based, hence still
  # atomic: a reader sees the old path or the new one, never a partial write.
  #
  # Anything at the pointer path that is neither a regular file nor a
  # symlink left by an earlier version is a squatter -- notably the
  # DIRECTORY the copying `ln -s` used to leave behind. (`-f` follows, so a
  # symlink to a run directory needs the explicit `-L` arm.)
  { [ ! -e "$LAST" ] || [ -f "$LAST" ] || [ -L "$LAST" ]; } || {
    echo "test-run: $LAST exists and is not a regular file; remove it" >&2
    return 1
  }
  # Written through a temporary and put in place with rename(2), which
  # REPLACES whatever the path names -- a previous pointer, or a symlink one
  # written before this script stopped using symlinks -- in a single atomic
  # step, then re-read to prove the pointer names this run before the launch
  # reports success. Neither of the obvious spellings works here: `mv` stats
  # the destination THROUGH the link, so a symlink to a run directory makes
  # it move the pointer INSIDE that directory; and unlinking first would
  # leave `last` absent for an interval, which is the very gap the
  # old-or-new guarantee exists to rule out (a launcher killed inside it
  # would strand a recorded, possibly live, run with no `last` at all).
  # rename(2) also refuses a directory destination outright, so the squatter
  # above cannot be absorbed even if the guard is somehow raced.
  { printf '%s\n' "$run_dir" >"$LAST.tmp.$$" &&
    perl -e 'rename($ARGV[0], $ARGV[1]) or exit 1' "$LAST.tmp.$$" "$LAST" &&
    [ "$(cat "$LAST" 2>/dev/null)" = "$run_dir" ]; } || {
    rm -f "$LAST.tmp.$$"
    echo "test-run: cannot update $LAST" >&2
    return 1
  }
}

resolve_run() {
  local ref=${1:-last}
  case $sub in
    paths | lock-status)
      # An absolute reference never falls back to an unrelated state entry.
      if recorded_absolute_path "$ref" && [ ! -d "$ref" ]; then
        die "no such run: $ref"
      fi ;;
  esac
  if [ "$ref" = last ]; then
    # The pointer is a plain file (see publish_run). A symlink there was
    # written by a version predating that change and may still name a run
    # that is LIVE -- `stop last` has to be able to reach one, and only a
    # later publication replaces the link -- so both spellings are read.
    #
    # Deliberately NOT a `-L` test selecting the matching read: publish_run's
    # rename can replace the symlink with the pointer file between the test
    # and the read, and a reader must not fail on a path that named a valid
    # run throughout. Trying the reads themselves has no such gap, in THIS
    # order: a symlink can only ever become a file (publish_run writes
    # nothing else), so a failed `readlink` means `cat` now applies. The
    # reverse order would still race -- `cat` fails on a symlink to a
    # directory, and the rename could land before the `readlink` retry.
    case $sub in
      # Resolve the symlink itself before capture: readlink's output can lose
      # trailing LF and accidentally select another existing run directory.
      paths | lock-status)
        run_dir=
        if [ -L "$LAST" ]; then
          run_dir=$(query_physical_path "$LAST" 2>/dev/null) || run_dir=
        fi ;;
      *) run_dir=$(readlink "$LAST" 2>/dev/null) || run_dir= ;;
    esac
    if [ -z "$run_dir" ]; then
      case $sub in
        paths | lock-status) run_dir=$(read_query_record "$LAST" 2>/dev/null) || run_dir= ;;
        *) run_dir=$(cat "$LAST" 2>/dev/null) || run_dir= ;;
      esac
    fi
    [ -n "$run_dir" ] || die "no runs recorded for this worktree"
    [ -d "$run_dir" ] || die "no such run: $run_dir"
  elif [ -d "$ref" ]; then
    # Canonicalized (physically -- symlink spellings differ per referrer)
    # for the same reason as $RUNS: identity is compared as a string
    # against recorded pointers.
    case $sub in
      paths | lock-status) run_dir=$(query_physical_path "$ref") || die "cannot resolve $ref" ;;
      *) run_dir=$(cd "$ref" && pwd -P) || die "cannot resolve $ref" ;;
    esac
  elif [ -d "$RUNS/$ref" ]; then
    # The bare identifiers `list` prints resolve here.
    run_dir=$RUNS/$ref
  else
    die "no such run: $ref (neither a directory nor an entry under $RUNS)"
  fi
  # Only directories this runner created may be trusted: without this, an
  # arbitrary directory's stray `pid` file (with the missing token read as
  # the documented fallback) could get an unrelated process TERMed by stop.
  [ -f "$run_dir/cmd" ] && [ -f "$run_dir/cap" ] ||
    die "not a test-run directory (no cmd/cap metadata): $run_dir"
}

# A dune invocation dune's own command-line parser refused -- an unknown option
# or subcommand, a missing or malformed operand -- is a stable two-part shape
# on stderr: `dune: <complaint>` (continued onto indented lines when the parser
# wraps it) followed by `Usage: dune ...`, then exit 1 with nothing built or
# run. Neither an `Error:` nor a `File "..."` line ever accompanies it, so the
# fingerprint has nothing to quote and the digest used to fall through to
# `FAIL (exit 1)` plus a raw log tail -- the verdict of a red suite, read at
# the moment the reader decides between "read the failures" and "fix the
# command line" (gh-ocannl-944). The shape is required WHOLE, FIRST and ALONE:
# a `dune:` line alone could be a test's own output; dune's usage errors are
# emitted before anything else is; and a refusal that ran nothing leaves
# nothing after itself -- past `Usage:` only dune's own `Try '... --help'`
# line, blank lines and this script's `exit: N` sentinel may follow. Anything
# else after it is evidence that something DID run (a `dune exec` program
# printing a nested dune usage error and then failing, say), and the log is
# then an ordinary red run whose fingerprint the digest must show. Prints the
# complaint (the lines before `Usage:`), 0 iff FILE is such a refusal.
#
# "First" means first after the launch's own prelude, which the log carries
# before dune starts: what the launch's first phase wrote there (the readers'
# build, the batch's backends, the width, the slot) ends at the byte offset
# `_resolve` records in `prelude` just before it runs the command, and is
# skipped by position, never by pattern -- a program's own `test-run: ` line
# is output like any other. The fleet slot's `EXECUTION SLOT ` admission
# lines, written by fleet-worker.sh between that exec and dune's, are the one
# prelude read by pattern (Codex review rounds 1-2 on PR #832).
dune_refusal() { # FILE [PRELUDE BYTES]
  tail -c +$(( ${2:-0} + 1 )) "$1" 2>/dev/null | head -c 20000 | awk '
    !started && /^EXECUTION SLOT / { next }
    !started { started = 1; if ($0 !~ /^dune: /) exit 1 }
    { n++ }
    !found && /^Usage: dune/ { found = 1; next }
    !found && n > 20 { exit 1 }
    !found { print; next }
    /^Try \047.*--help/ || /^exit: [0-9]+$/ || /^$/ { next }
    { found = 0; exit 1 }
    END { exit found ? 0 : 1 }'
}

# The compact report `run`, `wait` and `status` all end with. Fingerprint in
# the sweep.sh sense: the `File "..."` and `Error ...` lines, deduplicated, so
# a new failure is distinguishable from a standing one without opening the log.
# Both of dune's location spellings, since a stanza-level diagnostic -- what a
# failing explicit-rule test produces -- says `lines N-M` and would otherwise
# fall through to the raw log tail. The line numbers are kept as printed: this
# report is read against the tree that produced it, unlike sweep.sh's, which is
# diffed across commits and so normalizes a dune location to its stanza.
# Sets `digest_rc`, the status `run` and `wait` exit with: the recorded status,
# except 2 for a refused invocation (see the header's exit-code contract).
# A run that took the fleet's slot (it has a `slot` record) and never got
# one: fleet-worker.sh refused (exit 1: no slot or GPU token before the
# deadline, a measurement holding the box, a spec it could not read) or could
# not read the anchor's registry (exit 4), and says so in a line of its own --
# while a run it admitted logs the admission before dune starts: the slot it
# holds, the enclosing slot it nests in, or the live measurement hold it runs
# under (`execution hold --request <id> -- ...` exports FLEET_MEASUREMENT_HELD,
# and the slot then admits the batch inside it; lukstafi/ludics-lite#480). A
# refusal line with no admission is the slot's verdict, not the suite's.
slot_refusal() { # <run dir>; 0 iff the slot was refused and dune never ran
  [ -f "$1/slot" ] || return 1
  grep -Eq '^EXECUTION SLOT (REFUSED|UNREACHABLE) ' "$1/log" 2>/dev/null || return 1
  ! grep -Eq '^EXECUTION SLOT [^ ]+: (slot [0-9]+ of [0-9]+.* held for|inside slot [0-9]+ of [0-9]+|inside measurement [^ ]+)' "$1/log"
}

# The fixed start of the line a GPU backend's device-init deadline writes
# before it kills its process: Utils.device_init_deadline_tag, which
# test/operations/device_init_deadline checks this spelling against.
device_init_deadline_tag='OCANNL device_init_deadline:'

digest_rc=
digest() {
  local dir=$1 rc verdict fp complaint= refusal_src prelude_bytes
  rc=$(cat "$dir/exit" 2>/dev/null) || die "no verdict recorded in $dir"
  digest_rc=$rc
  # Where dune's stderr opens: the run log for `run`/`start`; for a repeat the
  # log opens with the iteration banner, and the refusal (if any) is the first
  # iteration's own stderr, which is also the only iteration a refusal leaves.
  refusal_src=$dir/log prelude_bytes=$(cat "$dir/prelude" 2>/dev/null)
  case $prelude_bytes in '' | *[!0-9]*) prelude_bytes=0 ;; esac
  [ "$(cat "$dir/mode" 2>/dev/null)" = repeat ] && refusal_src=$dir/iteration-1/stderr prelude_bytes=0
  # A direct set's first dune is its build phase.
  [ "$(cat "$dir/direct" 2>/dev/null)" = 1 ] && refusal_src=$dir/build-phase/stderr
  # 142 is the ONLY code the cap produces (the supervisor's SIGALRM exit), so
  # only it may say "timeout" -- the run's cap, or the per-test cap, whose
  # watcher sends the same SIGALRM and leaves its `test-cap` record behind
  # (test_cap_watch). 69 is only the device probe's (`_probe`). 137 is a SIGKILL -- an OOM kill or a forced
  # external kill -- and labeling it a timeout would send triage hunting a
  # hang that never happened. And 1 is the only status dune's parser exits
  # with, so only it is examined for a refusal: a refused-looking log under
  # any other code still reports that code's verdict.
  case $rc in
    0) verdict=pass ;;
    2)
      if [ -s "$dir/alias-refused" ]; then
        verdict="ALIAS REFUSED (dune knows no such alias; nothing ran)"
      elif [ -s "$dir/backend-refused" ]; then
        verdict="BACKEND REFUSED (the batch does not hold the backend --expect-backend names; nothing ran)"
      else
        verdict=FAIL
      fi ;;
    1 | 4)
      if slot_refusal "$dir"; then
        verdict="SLOT REFUSED (the fleet's run-time slot was not taken; nothing ran)"
        digest_rc=75
      elif [ "$rc" = 1 ] && complaint=$(dune_refusal "$refusal_src" "$prelude_bytes"); then
        verdict="INVOCATION REFUSED (dune rejected the arguments; nothing ran)"
        digest_rc=2
      else
        verdict=FAIL
      fi
      ;;
    142)
      if [ -s "$dir/test-cap" ]; then
        verdict="TIMEOUT (a test outlived the per-test cap; run was killed, not judged)"
      else
        verdict="TIMEOUT (cap expired; run was killed, not judged)"
      fi ;;
    69) verdict="DEVICE UNHEALTHY (a fresh device query did not answer; nothing ran)" ;;
    137) verdict="KILLED (SIGKILL: OOM or forced kill; not judged)" ;;
    129 | 130 | 143) verdict="CANCELLED (run was killed, not judged)" ;;
    126 | 127) verdict="ERROR (toolchain/setup: nothing ran)" ;;
    *) verdict=FAIL ;;
  esac
  # A GPU backend's in-process deadline on device initialization
  # (gh-ocannl-1281, Utils.with_device_init_deadline): past it the process
  # writes one line opening with the tag below and dies by SIGALRM (142 in a
  # shell). Under dune that death is a red action, exit 1, and dune's default
  # display does not name the signal -- so the line is the evidence, read only
  # at a line's start (a test quoting a child's stderr indents it). A red run
  # carrying it is a wedged device, not a test result; any other code keeps its
  # own verdict and gets the same lines. The WHOLE log is searched, unlike the
  # 10MB tail the rest of the digest reads: the line is the only evidence, and
  # parallel actions can write far more than that after it. A fixed-string
  # grep stays cheap at any size; the line-start test is awk's, over its hits.
  local wedged=
  if [ "$rc" != 0 ]; then
    wedged=$(LC_ALL=C grep -a -F -- "$device_init_deadline_tag" "$dir/log" 2>/dev/null |
             awk -v tag="$device_init_deadline_tag" 'index($0, tag) == 1' | sort -u | head -n 3)
    [ -z "$wedged" ] || [ "$verdict" != FAIL ] ||
      verdict="WEDGED DEVICE (a GPU device initialization outlived device_init_deadline and died by SIGALRM; not judged)"
  fi
  # A `fresh` run is green only where every named test has a fresh time (see
  # fresh_timings): one with no process of its own is NOT RUN, the refusal's
  # code, and times the trace could not give are UNTIMED.
  local fresh= fresh_missing=0
  [ "$(cat "$dir/mode" 2>/dev/null)" != fresh ] || fresh=1
  if [ -n "$fresh" ] && [ "$verdict" = pass ]; then
    if [ -s "$dir/timings" ]; then
      fresh_missing=$(awk -F '\t' '$2 != "ran"' "$dir/timings" | grep -c '')
      if [ "$fresh_missing" -gt 0 ]; then
        verdict="NOT RUN (no process of this run is the action of $fresh_missing named test(s): no fresh time for them)"
        digest_rc=2
      fi
    else
      verdict="UNTIMED (dune passed, but the per-test times could not be read from its trace)"
      digest_rc=1
    fi
  fi
  echo "command: dune $(cat "$dir/cmd")"
  echo "verdict: $verdict (exit $digest_rc)"
  echo "log:     $dir/log"
  # The source the run tested, from the launch-time record (record_checkout),
  # so a digest quoted as evidence carries its revision. Nothing is printed
  # for a run that recorded none (outside Git, or launched before the record).
  if [ -s "$dir/head" ]; then
    local src_head src_n
    src_head=$(sed -n 1p "$dir/head")
    if [ ! -f "$dir/dirty" ]; then
      echo "source:  $src_head (uncommitted state not recorded)"
    elif [ ! -s "$dir/dirty" ]; then
      echo "source:  $src_head (clean)"
    else
      src_n=$(grep -c '' "$dir/dirty")
      if [ "$src_n" = 1 ]; then
        echo "source:  $src_head + 1 uncommitted path"
      else
        echo "source:  $src_head + $src_n uncommitted paths"
      fi
    fi
  fi
  [ ! -s "$dir/alias-refused" ] || sed 's/^/alias refused: /' "$dir/alias-refused"
  if [ -s "$dir/backend-refused" ]; then
    echo "backend refused: $(cat "$dir/backend-refused")"
  elif [ -s "$dir/backend-warning" ]; then
    cat "$dir/backend-warning"
  fi
  if [ -s "$dir/device-probe" ]; then
    echo "device probe:"
    sed 's/^/  /' "$dir/device-probe"
  fi
  if [ "$rc" = 142 ] && [ -s "$dir/test-cap" ]; then
    echo "hung test: $(sed -n 's/^command //p' "$dir/test-cap")"
    echo "  ran $(sed -n 's/^elapsed //p' "$dir/test-cap")s, past the $(sed -n 's/^cap //p' "$dir/test-cap")s per-test cap$(sed -n 's/^dir / (in /p' "$dir/test-cap" | sed 's/$/)/')"
    echo "  legitimately long? rerun with --test-cap 0 (no per-test cap) or a larger --test-cap"
    ! grep -qx 'kind gpu' "$dir/test-cap" ||
      echo "  on this GPU batch, read the kernel journal before rerunning (journalctl -k: gfxhub/CPC faults, ring timeouts, NVRM Xid)"
  fi
  if [ -n "$wedged" ]; then
    echo "wedged device (the device_init_deadline's own line):"
    printf '%s\n' "$wedged" | sed 's/^/  /'
    echo "  stop GPU work on this box and read the kernel journal before rerunning (journalctl -k:" \
         "gfxhub/CPC faults, ring timeouts, NVRM Xid); a legitimately slow cold start? raise" \
         "device_init_deadline (0 disables)"
  fi
  [ ! -s "$dir/test-cap-off" ] ||
    echo "per-test cap: off for this run ($(cat "$dir/test-cap-off")); only the run's cap bounded it"
  # The batch's slowest processes (see explicit_trace_file), under a verdict
  # dune's own run produced: a refused slot, invocation or device probe, or a
  # setup error, ran no batch, so a record beside one is not this verdict's.
  # `open` marks a process the run's end cut short, its seconds a lower bound.
  case $verdict in
    INVOCATION* | SLOT* | DEVICE* | ERROR* | BACKEND* | ALIAS*) ;;
    *)
      if [ -s "$dir/slowest" ]; then
        echo "slowest actions (this run's dune trace, by tools/action-durations.sh):"
        sed -n '1,7p' "$dir/slowest" | sed 's/^/  /'
      elif [ -s "$dir/slowest-skipped" ]; then
        echo "no slowest-actions record: $(head -c 300 "$dir/slowest-skipped" | head -n 1)"
      fi
      if [ -n "$fresh" ]; then
        if [ -s "$dir/timings" ]; then
          echo "per-test wall time (fresh: own build directory, --force, --cache=disabled; this run's dune trace):"
          printf '  %9s %6s %5s  %-7s  %s\n' seconds procs exit matched test
          awk -F '\t' '{ if ($2 == "ran") printf "  %9s %6s %5s  %-7s  %s\n", $3, $4, $5, $6, $1
                          else printf "  %9s %6s %5s  %-7s  %s\n", "NOT RUN", 0, "-", "-", $1 }' "$dir/timings"
        elif [ -s "$dir/timings-skipped" ]; then
          echo "no per-test times: $(head -c 300 "$dir/timings-skipped" | head -n 1)"
        fi
        if [ "$fresh_missing" -gt 0 ]; then
          echo "a test NOT RUN had no process here that is its action: not the alias's own action, not" \
               "the producer of an output in DIR named after it (NAME.<ext>, not a compile artifact)," \
               "and no unclaimed NAME.exe in DIR. Name the per-test alias" \
               "whose action runs it; this tool cannot time a test it cannot see run."
        fi
        [ ! -e "$dir/build" ] ||
          echo "the fresh build directory survived its run (remove it): $dir/build"
      fi ;;
  esac
  case $verdict in
    ALIAS*)
      # The refusal lines above say what dune does not know; no rule ran.
      echo "fix the alias and run again -- \`dune show aliases DIR\` lists a directory's;" \
           "nothing was built or judged"
      return 0 ;;
  esac
  local refused_invocation=
  case $verdict in INVOCATION*) refused_invocation=1 ;; esac
  if [ -n "$refused_invocation" ]; then
    # dune's own words name the fix; nothing below (promotion diffs, the
    # fingerprint, a log tail) could apply to a run in which no rule ran.
    echo "dune said:"
    printf '%s\n' "$complaint" | sed 's/^/  /'
    echo "fix the command line and run again -- no test was built or judged" \
         "(dune's own status was $rc; this script exits 2, its usage code)"
    return 0
  fi
  # Digest sits on `wait`'s deadline path, so it examines at most the last
  # 10MB of the log rather than scaling with an arbitrarily noisy run.
  scan_log() { tail -c 10000000 "$dir/log" 2>/dev/null; }
  # Promotion is dune's own answer where the run recorded it (`promotions`,
  # see record_promotions), read only under a verdict dune gave: a refused
  # slot ran nothing, and the list it found is an older build's. The log is
  # read only for runs that recorded none.
  local promo_n=
  case $verdict in
    pass | FAIL) [ ! -f "$dir/promotions" ] || promo_n=$(grep -c '' "$dir/promotions") ;;
  esac
  # Without it, promotion is offered only on a diff dune actually printed: a
  # `--- ` header line directly followed by a `+++ ` one (git diff and diff
  # -u; patdiff's `------ `/`++++++ ` too; color escapes stripped first). Merely NAMING a
  # `.expected` or `.corrected` file is not one -- dune quotes the failing
  # stanza, `(diff? x.ml x.ml.corrected)` included, for a rule whose action
  # failed before any diff ran, and sending that reader to `dune promote`
  # hides the real failure (gh-ocannl-1055). Two runs cannot be told "no diff"
  # from their log: one that chose its own diff presentation (`--diff-command`,
  # or DUNE_DIFF_COMMAND at launch; `-` prints nothing, and an inline-expect
  # rule's remaining location names only its `.ml`), whose failure may be a
  # pending promotion whatever the log says; and one whose log outgrew the
  # scanned tail, which may hold a hunk above it.
  local log_bytes
  log_bytes=$(wc -c <"$dir/log" 2>/dev/null | tr -d ' ')
  if [ -n "$fresh" ]; then
    # A fresh run built in a directory that is gone: nothing of it is
    # promotable from the worktree, whatever its log shows.
    [ "$verdict" != FAIL ] ||
      echo "a fresh run's corrections are not promotable (its build directory is removed);" \
           "rerun the failing alias with \`tools/test-run.sh run\` to promote it"
  elif [ -n "$promo_n" ] && [ "$promo_n" -gt 0 ]; then
    echo "promotion diffs present -- dune's promotion list at the run's end names $promo_n" \
         "file(s); inspect the log, accept with \`dune promote\` (tools/promote.sh on Windows):"
    sed -n '1,20p' "$dir/promotions" | sed 's/^/  /'
    [ "$promo_n" -le 20 ] || echo "  ... and $((promo_n - 20)) more, in $dir/promotions"
    if [ -d "$dir/promotion-files" ]; then
      printf '  saved corrected outputs: tools/promote.sh --from-run %q\n' "$dir"
    else
      echo "  (corrected outputs were not saved; promote before another build replaces dune's list)"
    fi
  elif [ -n "$promo_n" ]; then
    [ "$verdict" != FAIL ] ||
      echo "action failed -- dune's promotion list at the run's end is empty, nothing to" \
           "promote; read the failure below"
  elif scan_log | awk 'BEGIN { esc = sprintf("%c", 27) }
                     { gsub(esc "\\[[0-9;]*m", "") }
                     prev ~ /^---+ / && /^\+\+\++ / { found = 1; exit }
                     { prev = $0 }
                     END { exit found ? 0 : 1 }'; then
    echo "promotion diffs present -- inspect the log, accept with \`dune promote\`" \
         "(tools/promote.sh on Windows)"
  elif [ "$verdict" = FAIL ] && [ -s "$dir/diff-command" ]; then
    echo "promotion diffs possible -- the run chose its own diff command, whose" \
         "output this digest cannot read; inspect the log before \`dune promote\`"
  elif [ "$verdict" = FAIL ] && [ "${log_bytes:-0}" -gt 10000000 ]; then
    echo "action failed; no diff in the log's last 10MB, but the log is longer --" \
         "search it whole before concluding there is nothing to promote"
  elif [ "$verdict" = FAIL ] && [ "$(cat "$dir/mode" 2>/dev/null)" != repeat ]; then
    # Not for a repeat set: its red can be drift between iterations that each
    # passed, which no failed action explains.
    echo "action failed (no diff) -- nothing to promote; read the failure below"
  fi
  if [ "$rc" != 0 ]; then
    fp=$({ scan_log | grep -oE '^File "[^"]+", lines? [0-9]+(-[0-9]+)?'
           scan_log | grep -oE '^(Error|Fatal error|Exception)[^,]*'
         } 2>/dev/null | sort -u | head -40)
    if [ -n "$fp" ]; then
      echo "fingerprint:"
      printf '%s\n' "$fp" | sed 's/^/  /'
    else
      echo "no Error/File lines matched; log tail:"
      # Byte-capped BEFORE the line cap: a giant newline-free record would
      # otherwise ride through `tail -n` whole.
      tail -c 4000 "$dir/log" | tail -n 25 | sed 's/^/  /'
    fi
  fi
}

finish_run() { # rc -> append sentinel, record verdict
  printf 'exit: %s\n' "$1" >>"$run_dir/log"
  # Written aside and renamed: the verdict file's EXISTENCE is the completion
  # signal `status`/`wait` key on, so it must never be observable empty. The
  # write is retried with backoff -- a filesystem that filled up during the
  # run may clear, and giving up silently would downgrade a real verdict
  # into a generic "died without recording a verdict".
  local n=0
  until printf '%s\n' "$1" >"$run_dir/exit.tmp" &&
        mv -f "$run_dir/exit.tmp" "$run_dir/exit"; do
    n=$(( n + 1 ))
    if [ "$n" -ge 3 ]; then
      printf 'test-run: FAILED to record verdict %s (filesystem?)\n' "$1" \
        >>"$run_dir/log" 2>/dev/null
      return 1
    fi
    sleep 2
  done
}

sub=${1:-}
[ -n "$sub" ] || die "usage: tools/test-run.sh run|start|plan|repeat|fresh|bisect|status|wait|stop|list|idle|paths|lock-status|affected ... (see header)"
shift

case $sub in
  affected)
    # Read-only, but it asks dune about this worktree, and a dune started
    # beside a run's dune contends for `_build`. So it takes the worktree's run
    # lock, as a run does, and holds it on fd 9 through the helper and every
    # dune it starts: a run refuses meanwhile, and this refuses during a run.
    # (No owner pointer is published: this is not a run to point at.) Where the
    # previous version's in-tree lock exists it is held too, on fd 5, as
    # take_lock holds it, so a launcher of that version is refused meanwhile;
    # it is never created, nor removed, here.
    query_state_for ""
    if [ -e "$PWD/.test-run.lock" ]; then
      exec 5>>"$PWD/.test-run.lock" || die "cannot open the previous version's lock file"
      perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 3)' <&5 ||
        die "a run of the previous version is active in this worktree; affected asks dune too, so wait for it"
    fi
    mkdir -p "$RUNS" || die "cannot create $RUNS"
    exec 9>>"$LOCK" || die "cannot open lock file $LOCK"
    perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 3)' <&9 ||
      die "a run is active in this worktree; affected asks dune too, so wait for it (status last)"
    select_dune
    exec bash tools/affected-tests.sh "$@"
    ;;
  paths)
    [ $# -ge 1 ] && [ $# -le 2 ] || die "usage: paths FIELD [RUN|last]"
    field=$1
    case $field in run | worktree | runs | lock | owner | last) ;;
      *) die "unknown path field: $field (run, worktree, runs, lock, owner, last)" ;;
    esac
    query_state_for "${2:-}"
    if [ "$field" = run ]; then
      resolve_run "${2:-last}"
      # Legacy last symlinks may spell a run through a symlinked directory.
      query_physical_path "$run_dir" || die "cannot resolve $run_dir"
      exit 0
    fi
    query_wt=$PWD query_runs=$RUNS query_lock=$LOCK query_owner=$OWNER query_last=$LAST
    if [ $# -eq 2 ]; then
      resolve_run "$2"
      lock_paths_of "$run_dir" query || die "run has unavailable or invalid path metadata: $run_dir"
      query_wt=$run_wt query_lock=$run_lock query_owner=$run_owner
      query_runs=$run_runs
      # Legacy runs have no stored state root; the run directory identifies it.
      [ -n "$query_runs" ] || query_runs=$(query_physical_path "$run_dir/..") || die "cannot resolve legacy state root"
      query_last=$query_runs/last-$(wt_key_of "$query_wt")
    fi
    case $field in
      worktree) query_out=$query_wt ;;
      runs) query_out=$query_runs ;;
      lock) query_out=$query_lock ;;
      owner) query_out=$query_owner ;;
      last) query_out=$query_last ;;
    esac
    require_query_path "$query_out"
    printf '%s\n' "$query_out"
    ;;
  lock-status)
    [ $# -le 1 ] || die "usage: lock-status [RUN|last]"
    query_state_for "${1:-}"
    if [ $# -eq 1 ]; then
      resolve_run "$1"
      lock_paths_of "$run_dir" query || die "run has unavailable or invalid path metadata: $run_dir"
      probe_locks "$run_lock"
    else
      probe_locks "$LOCK" "$PWD/.test-run.lock"
    fi
    query_rc=$?
    case $query_rc in
      0) echo idle ;;
      3) echo held ;;
      *) die "cannot inspect worktree lock" ;;
    esac
    exit "$query_rc"
    ;;
  repeat)
    cap=${OCANNL_TOOL_TEST_CAP:-3600}
    # As for `run`: empty is the default, which the resolution phase decides.
    test_cap=${OCANNL_TOOL_PER_TEST_CAP:-}
    alone=0
    direct=0
    while [ $# -gt 0 ]; do
      case $1 in
        --cap) [ $# -ge 2 ] || die "--cap requires a value"; cap=$2; shift 2 ;;
        --test-cap) [ $# -ge 2 ] || die "--test-cap requires a value"; test_cap=$2; shift 2 ;;
        --alone) alone=1; shift ;;
        --direct) direct=1; shift ;;
        --) shift; break ;;
        *) break ;;
      esac
    done
    normalize_cap
    [ -z "$test_cap" ] || normalize_cap test_cap --test-cap
    [ $# -gt 0 ] || die "repeat requires a count of at least 2"
    repeats=$1
    shift
    case $repeats in '' | *[!0-9]*) die "repeat count must be an integer of at least 2" ;; esac
    [ ${#repeats} -le 6 ] || die "repeat count too large (max 6 digits)"
    repeats=$(( 10#$repeats ))
    [ "$repeats" -ge 2 ] || die "repeat count must be at least 2"
    reject_misplaced_options "$@"
    if [ "$direct" = 1 ]; then
      # The operands `fresh` takes (fresh_argv): the replay plans per-test aliases only.
      fresh_argv "$@"
      set -- build "${fresh_aliases[@]}"
      [ -z "$fresh_jobs" ] || set -- "$@" -j "$fresh_jobs"
      command -v python3 >/dev/null 2>&1 ||
        die "repeat --direct needs python3 (the trace reader and tools/replay-rules.sh)"
    fi
    [ $# -gt 0 ] || set -- runtest
    select_dune
    new_run "$@"
    take_lock
    # The coordinator is the run's OWNER: its identity is what status reads
    # as running and what stop TERMs -- the set-wide cancellation bit lives
    # here, not in any one iteration's supervisor, which the coordinator
    # records under the iteration instead.
    { printf '%s\n' repeat >"$run_dir/mode" &&
      printf '%s\n' "$repeats" >"$run_dir/repeats" &&
      printf '%s\n' "$alone" >"$run_dir/alone" &&
      printf '%s\n' "$direct" >"$run_dir/direct" &&
      { [ "$direct" = 0 ] || printf '%s\n' "${fresh_aliases[@]}" >"$run_dir/direct-aliases"; } &&
      ps_token "$$" >"$run_dir/ptoken" &&
      printf '%s\n' "$$" >"$run_dir/pid"; } ||
      die "cannot record repeat metadata in $run_dir"
    repeat_sup=
    repeat_cancelled=
    completed=0
    first_nonzero=0
    repeat_refused=
    repeat_signal() {
      repeat_cancelled=$1
      [ -n "$repeat_sup" ] && kill "-$1" "$repeat_sup" 2>/dev/null
    }
    trap 'repeat_signal INT' INT
    trap 'repeat_signal TERM' TERM
    trap 'repeat_signal TERM' HUP
    # Once published, every coordinator exit must publish a verdict too. This
    # is also the group-cancellation backstop: a foreground finalizer child
    # (diff/cmp/remove_tree) shares the terminal's process group and can die
    # from the same signal even though the coordinator traps it. If that makes
    # a checked finalizer command call die, EXIT still records cancellation.
    repeat_exit() {
      local rc=$?
      trap - EXIT
      # Verdict publication itself is the final, tiny critical section. Ignore
      # another terminal/group signal here so finish_run's children inherit
      # that disposition and the atomic exit file cannot be interrupted.
      trap '' INT TERM HUP
      if [ -n "$repeat_cancelled" ]; then
        if [ "$first_nonzero" != 0 ]; then
          rc=$first_nonzero
        else
          case $repeat_cancelled in INT) rc=130 ;; *) rc=143 ;; esac
        fi
        if [ -z "${repeat_reported_cancelled:-}" ]; then
          printf 'repeat result: CANCELLED -- completed %s of %s iterations\n' \
            "$completed" "$repeats"
          printf 'repeat result: CANCELLED -- completed %s of %s iterations\n' \
            "$completed" "$repeats" >>"$run_dir/log"
        fi
        printf 'repeat cancellation observed: %s\n' "$repeat_cancelled" >>"$run_dir/log"
      fi
      if [ ! -f "$run_dir/exit" ]; then
        finish_run "$rc" ||
          printf 'test-run: repeat exited %s but its verdict could not be recorded\n' "$rc" >&2
      fi
      exec 9>&-
      # Same contract as `run`: the RECORDED status is dune's own, the process
      # status of a refused invocation is the usage code (header).
      [ -n "$repeat_cancelled" ] || [ -z "$repeat_refused" ] || rc=2
      exit "$rc"
    }
    trap repeat_exit EXIT

    reap_repeat_group() { # iteration dir -- no verified survivor may share repeat_build
      local iter_dir=$1 pg n
      pg=$(cat "$iter_dir/pgid" 2>/dev/null) || return 0
      # Reachability gates cleanup. group_alive is a fallible census and may
      # miss a member that forks/exits while /proc is enumerated; it must never
      # turn a surviving group into permission to reuse repeat_build.
      kill -0 -- "-$pg" 2>/dev/null || return 0
      group_identity_matches "$iter_dir" ||
        die "iteration group $pg survived without a verifiable identity; refusing to reuse $repeat_build"
      printf 'repeat: iteration group %s survived its supervisor; reaping before reuse\n' "$pg" \
        | tee -a "$run_dir/log"
      kill -TERM -- "-$pg" 2>/dev/null
      for n in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        kill -0 -- "-$pg" 2>/dev/null || return 0
        sleep 0.1
      done
      # Revalidate immediately before the destructive escalation. A zombie
      # leader still has the original token and safely identifies its group;
      # a missing leader while the group remains reachable can also mean a
      # live descendant survived TERM, so it is not permission to reuse.
      if ! group_identity_matches "$iter_dir"; then
        # Do not ask the fallible process census to authorize reuse here. A
        # descendant can fork and exit while /proc is being enumerated, making
        # a still-live chain momentarily look zombie-only. Without the recorded
        # leader we can neither safely aim KILL nor prove the build tree inert.
        die "leaderless iteration group $pg survived TERM; refusing to reuse $repeat_build"
      fi
      kill -KILL -- "-$pg" 2>/dev/null
      for n in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 -- "-$pg" 2>/dev/null || return 0
        if ! group_identity_matches "$iter_dir"; then
          # KILL already reached the verified original group, so a live
          # leaderless member is now an unkillable survivor; only a zombie-only
          # residue is inert enough to release the shared build tree.
          group_alive "$pg" &&
            die "leaderless iteration group $pg survived KILL; refusing to reuse $repeat_build"
          return 0
        fi
        sleep 0.1
      done
      # KILL has already reached the identity-verified original group, so no
      # live member can subsequently fork. A remaining verified group whose
      # census is now zombie-only is inert residue under a non-reaping init,
      # not a reason to discard the iteration's real verdict.
      group_alive "$pg" || return 0
      die "iteration group $pg survived TERM/KILL; refusing to reuse $repeat_build"
    }

    # A repeat can be long enough to manage from another shell. Its complete
    # cancellation state and exit finalizer are armed BEFORE it becomes `last`:
    # every externally discoverable coordinator can therefore publish a
    # verdict even if stop or a group signal lands in the publication gap.
    publish_run || die "cannot publish repeat run $run_dir"

    # The two bounds `run` gives a GPU batch (gh-ocannl-1252), decided once, as `run` decides them
    # (`_resolve`'s repeat mode: the same resolution of the batch's backends, under the same cap,
    # in a phase of its own with the coordinator's cancellation), and applied to every iteration:
    # the device probe before it starts (`_probe`, recording into the iteration's directory), the
    # per-test cap's watcher beside its dune. A wedged device hangs every test after it, so a probe
    # that finds one, or a test the cap ends, ends the set (repeat_bounded below).
    resolve_dir=$run_dir/resolution
    mkdir "$resolve_dir" || die "cannot create $resolve_dir"
    # Named before the phase: reap_repeat_group's refusals name it (Codex review round 2 on PR #1172).
    repeat_build=$run_dir/build
    OCANNL_TOOL_TESTRUN_BG=0 OCANNL_TOOL_TESTRUN_RD=$resolve_dir \
      perl -e "$supervisor_perl" -- "$cap" env "OCANNL_TOOL_TESTRUN_OWN=$run_dir" \
      "${BASH:-bash}" "$runner_root/tools/test-run.sh" _resolve repeat "$cap" "$test_cap" "$DUNE" "$@" \
      </dev/null >>"$run_dir/log" 2>&1 &
    repeat_sup=$!
    [ -z "$repeat_cancelled" ] || kill "-$repeat_cancelled" "$repeat_sup" 2>/dev/null
    { printf '%s\n' "$repeat_sup" >"$resolve_dir/pid" &&
      ps_token "$repeat_sup" >"$resolve_dir/ptoken"; } ||
      kill -TERM "$repeat_sup" 2>/dev/null
    while :; do
      wait "$repeat_sup"
      resolve_rc=$?
      proc_alive "$resolve_dir/pid" "$resolve_dir/ptoken" || break
    done
    repeat_sup=
    # The resolution touches nothing repeat_build holds, so a group of only zombies it leaves behind
    # is inert: a process substitution's child its shell never reaped, handed at its exit to a PID
    # 1 that reaps no orphans -- a leaderless group the iterations' reap must refuse (Codex review
    # round 3 on PR #1172). Anything live, or a group still led by its own leader, goes through
    # that fail-closed reap as before.
    resolve_pg=$(cat "$resolve_dir/pgid" 2>/dev/null) || resolve_pg=
    if [ -n "$resolve_pg" ] && kill -0 -- "-$resolve_pg" 2>/dev/null &&
       ! group_identity_matches "$resolve_dir" && ! group_alive "$resolve_pg"; then
      printf 'repeat: the resolution left only zombies in group %s: inert, not reaped\n' "$resolve_pg" \
        >>"$run_dir/log"
    else
      reap_repeat_group "$resolve_dir"
    fi
    repeat_test_cap=0 repeat_watch_kind=cpu repeat_probe_cap= repeat_probe_prog= repeat_probe_backends=
    if [ -z "$repeat_cancelled" ]; then
      # A resolution its supervisor ended -- its cap (142), a kill -- ran no iteration, and the set
      # exits with that status, as a run whose resolution was cut short does (repeat_exit records
      # it); a resolution that finished without its bounds is this script's own failure.
      if [ "$resolve_rc" != 0 ] && [ -s "$run_dir/alias-refused" ]; then
        sed 's/^/test-run: repeat: ALIAS REFUSED: /; $s/$/; no iteration ran/' "$run_dir/alias-refused"
        exit "$resolve_rc"
      elif [ "$resolve_rc" != 0 ]; then
        printf 'test-run: repeat: the bounds were not resolved (exit %s; see %s/log), so no iteration ran\n' \
          "$resolve_rc" "$run_dir" | tee -a "$run_dir/log"
        exit "$resolve_rc"
      fi
      [ -s "$run_dir/bounds" ] || die "the repeat's resolution recorded no bounds (see $run_dir/log)"
      { read -r repeat_test_cap && read -r repeat_watch_kind && read -r repeat_probe_cap &&
          read -r repeat_probe_prog && read -r repeat_probe_backends; } <"$run_dir/bounds" ||
        die "cannot read the repeat's bounds from $run_dir/bounds"
    fi
    repeat_bounded=
    repeat_bounded_where=
    repeat_direct_refused=
    repeat_direct_rc=
    # Each step of the set -- an iteration, and for --direct the build phase and the plan --
    # is one supervised launch of step_cmd into its own directory: its stdout, stderr and exit,
    # under the per-iteration cap and the coordinator's cancellation, its group reaped before
    # the shared build tree is reused (repeat_launch). step_rc is its status.
    repeat_argv=("$@")
    repeat_isolation=(--force --cache=disabled --build-dir="$repeat_build")
    [ "$alone" = 0 ] || repeat_isolation+=(-j 1)
    repeat_dune_cmd() { # <step dir>: dune's argv for a step, into repeat_cmd
      local iso=("${repeat_isolation[@]}") repeat_arg repeat_separator=0
      # A direct set reads the build phase's processes from a trace of its own (_direct_plan);
      # each iteration's names what its dune ran (`dune-processes`).
      [ "$direct" = 0 ] || iso+=(--trace-file="$1/trace.csexp")
      # Dune's first `--` ends Dune option parsing (`dune exec PROG -- ARGS`).
      # Isolation options belong immediately before it; appending them would
      # silently hand them to PROG and run Dune in its ambient build context.
      repeat_cmd=("$DUNE")
      for repeat_arg in ${repeat_argv[@]+"${repeat_argv[@]}"}; do
        if [ "$repeat_separator" = 0 ] && [ "$repeat_arg" = -- ]; then
          repeat_cmd+=("${iso[@]}")
          repeat_separator=1
        fi
        repeat_cmd+=("$repeat_arg")
      done
      [ "$repeat_separator" = 1 ] || repeat_cmd+=("${iso[@]}")
    }
    # The step that runs dune: `clean` (a clean build context, then dune -- every iteration
    # without --direct), `build` (the same for the build phase of a direct set, whose dune is
    # never exec'd: bash's exec drops the SHLVL and `_` it would otherwise hand dune, and so
    # every action, which the replay's actions get as a child of this shell would), or `replay`
    # (the plan's rules replayed into the built tree, then dune's --force re-running the alias
    # actions) -- beside
    # the per-test cap's watcher, whose resolver is this `bash -c` and whose supervisor ($PPID)
    # the step's, and, on a probed batch, after `_probe`, which execs it in its own place once
    # the device answered. The watcher times what dune starts and, through `test-parents`,
    # what the replay starts. The shell outlives its children, a signal to the group included:
    # the supervisor's TERM ends dune and the watcher too, and the trap (run once the
    # foreground child has exited) reaps the watcher before the shell exits, so no member of
    # the step's group is left to a parent that may never reap it -- a zombie-only group
    # without its leader, which reap_repeat_group must refuse, under a PID 1 that does not
    # reap orphans (Codex review round 1 on PR #1172). A replay that fails a rule (1) still
    # lets dune diff what it wrote; any other replay failure ends the step with its status.
    repeat_iter_cmd() { # <step dir> <clean|build|replay>: into step_cmd
      step_cmd=(/bin/bash -c '
        dune=$1 build=$2 tcap=$3 watch=$4 kind=$5 rd=$6 pre=$7 replay=$8 plan=$9
        shift 9
        if [ "$pre" != replay ]; then "$dune" clean --build-dir="$build" || exit 126; fi
        [ "$tcap" -gt 0 ] || [ "$pre" != clean ] || exec "$dune" "$@"
        watcher=
        trap '\''[ -z "$watcher" ] || { kill "$watcher"; wait "$watcher"; } 2>/dev/null; exit 143'\'' TERM INT HUP
        if [ "$tcap" -gt 0 ]; then
          perl -e "$watch" "$tcap" "$$" "$PPID" "$rd" "$kind" </dev/null 9>&- &
          watcher=$!
        fi
        rc=0
        if [ "$pre" = replay ]; then
          bash "$replay" run --plan "$plan" --out "$rd" --parents "$rd/test-parents" </dev/null
          rc=$?
        fi
        case $rc in
          0 | 1)
            "$dune" "$@"
            drc=$?
            [ "$drc" = 0 ] || rc=$drc ;;
        esac
        if [ -n "$watcher" ]; then
          kill "$watcher" 2>/dev/null
          wait "$watcher" 2>/dev/null
        fi
        exit "$rc"' \
        -- "$DUNE" "$repeat_build" "$repeat_test_cap" "$test_cap_watch" "$repeat_watch_kind" "$1" "$2" \
        "$runner_root/tools/replay-rules.sh" "$run_dir/plan.json" "${repeat_cmd[@]:1}")
      if [ -n "$repeat_probe_backends" ]; then
        # shellcheck disable=SC2086  # the backend list is word-split on purpose
        step_cmd=(env "OCANNL_TOOL_TESTRUN_OWN=$1" "${BASH:-bash}" "$runner_root/tools/test-run.sh" _probe
          "$DUNE" "$repeat_probe_prog" "$repeat_probe_cap" $repeat_probe_backends -- "${step_cmd[@]}")
      fi
    }
    repeat_launch() { # <step dir> <label>: one supervised step; sets step_rc
      local sdir=$1 label=$2 descendant_fifo descendant_rc
      # The step directory is what a probed step's `_probe` owns
      # (OCANNL_TOOL_TESTRUN_OWN), and a run's own phase reads the tree under
      # test from its owner's `wt` (gh-ocannl-1225): the run's, recorded again.
      cp "$run_dir/wt" "$sdir/wt" || die "cannot record the tree under test in $sdir"
      # A process-group reap cannot see a Dune action that calls setsid. Give
      # every descendant an inherited FIFO writer as a second containment
      # witness: after the supervisor exits, EOF proves no session-escaped
      # descendant remains able to mutate this step's shared build tree.
      # The read/write anchor makes the two one-way opens nonblocking; it is
      # closed before launch and never reaches the child.
      descendant_fifo=$sdir/descendants.fifo
      mkfifo "$descendant_fifo" || die "cannot create descendant witness for $label"
      exec 7<>"$descendant_fifo" || die "cannot anchor descendant witness for $label"
      exec 8<"$descendant_fifo" || die "cannot read descendant witness for $label"
      exec 6>"$descendant_fifo" || die "cannot write descendant witness for $label"
      exec 7>&-
      OCANNL_TOOL_TESTRUN_BG=0 OCANNL_TOOL_TESTRUN_RD=$sdir \
        perl -e "$supervisor_perl" -- "$cap" "${step_cmd[@]}" \
        >"$sdir/stdout" 2>"$sdir/stderr" &
      repeat_sup=$!
      exec 6>&-
      [ -z "$repeat_cancelled" ] || kill "-$repeat_cancelled" "$repeat_sup" 2>/dev/null
      { printf '%s\n' "$repeat_sup" >"$sdir/pid" &&
        ps_token "$repeat_sup" >"$sdir/ptoken"; } ||
        kill -TERM "$repeat_sup" 2>/dev/null
      while :; do
        wait "$repeat_sup"
        step_rc=$?
        proc_alive "$sdir/pid" "$sdir/ptoken" || break
      done
      # The numeric pid is no longer ours once wait/proc_alive prove the
      # supervisor gone. Clear it before the slower orphan-group reap so a
      # cancellation cannot signal a recycled, unrelated process.
      repeat_sup=
      # SIGKILL can remove the supervisor without reaching the Dune process
      # group it owned. Reap that identity-verified group before recording the
      # step, starting another one, or deleting their shared build tree.
      reap_repeat_group "$sdir"
      # A zero-byte read is EOF. Bash 3.2 returns status 1 for BOTH EOF and a
      # `read -t` timeout, so use the Perl already required by this harness to
      # distinguish them: 0 is EOF, 2 is timeout, 1 is data/error. Anything
      # but EOF retains the build context and refuses concurrent reuse.
      perl -e '
        $SIG{ALRM} = sub { exit 2 };
        alarm 2;
        my $n = sysread(STDIN, my $byte, 1);
        exit(defined($n) && $n == 0 ? 0 : 1);
      ' <&8
      descendant_rc=$?
      exec 8>&-
      [ "$descendant_rc" = 0 ] ||
        die "$label left a session-escaped descendant; refusing to reuse $repeat_build"
      rm -f "$descendant_fifo"
      printf '%s\n' "$step_rc" >"$sdir/exit" || die "cannot record $label verdict"
    }
    # A step a bound ended (gh-ocannl-1252): the device probe found the device wedged (69,
    # nothing ran), or a test outlived the per-test cap (142 with its `test-cap` record), and on
    # a wedged device every later step would hang the same way. Its records are copied up to the
    # run, where the digest reads them, as a run's own would be.
    repeat_bound_of() { # <step dir> <status> <where>: 0 (setting repeat_bounded) iff a bound ended it
      if { [ "$2" = 69 ] && grep -q ': NO ANSWER within ' "$1/device-probe" 2>/dev/null; } ||
         { [ "$2" = 142 ] && [ -s "$1/test-cap" ]; }; then
        repeat_bounded=$2 repeat_bounded_where=$3
        for record in device-probe test-cap; do
          [ ! -s "$1/$record" ] || cp "$1/$record" "$run_dir/$record" ||
            die "cannot copy $3's $record record"
        done
        return 0
      fi
      return 1
    }

    # --direct (gh-ocannl-1377): one build, then replays. The build phase is a clean iteration
    # whose dune writes a trace; the plan reads from it which rules are each named test's own
    # action (the rule `fresh` times a test by) and asks dune for them (tools/replay-rules.sh,
    # which refuses what it cannot reproduce: a sandboxed rule, an action form outside its
    # subset, a test the build phase never ran), then snapshots their targets: dune's own run
    # of the same rules, the reference the first replay is compared with.
    if [ "$direct" = 1 ] && [ -z "$repeat_cancelled" ]; then
      bphase=$run_dir/build-phase
      mkdir "$bphase" || die "cannot create $bphase"
      repeat_dune_cmd "$bphase"
      {
        printf '%q clean --build-dir=%q && ' "$DUNE" "$repeat_build"
        printf '%q ' "${repeat_cmd[@]}"
        echo
      } >"$bphase/cmd" || die "cannot record the build phase's command"
      printf 'repeat: build phase -- dune, once, into %s%s\n' "$repeat_build" \
        "$([ "$alone" = 1 ] && printf ' (alone, -j 1)' || :)"
      repeat_iter_cmd "$bphase" build
      repeat_launch "$bphase" "the build phase"
      build_rc=$step_rc
      {
        printf '=== repeat build phase stdout ===\n'
        cat "$bphase/stdout"
        printf '=== repeat build phase stderr ===\n'
        cat "$bphase/stderr"
        printf '=== repeat build phase exit %s ===\n' "$build_rc"
      } >>"$run_dir/log" || die "cannot append the build phase to $run_dir/log"
      printf 'repeat: build phase exit %s; stdout=%s stderr=%s\n' \
        "$build_rc" "$bphase/stdout" "$bphase/stderr"
      # An invocation dune's parser refused ran nothing: there is nothing to replay.
      if [ "$build_rc" = 1 ] && dune_refusal "$bphase/stderr" >/dev/null; then
        repeat_refused=1 first_nonzero=1
      elif repeat_bound_of "$bphase" "$build_rc" "the build phase"; then
        :
      elif [ -z "$repeat_cancelled" ]; then
        pdir=$run_dir/plan
        mkdir "$pdir" || die "cannot create $pdir"
        step_cmd=(env "OCANNL_TOOL_TESTRUN_RD=$pdir" "${BASH:-bash}" "$runner_root/tools/test-run.sh" _direct_plan
          "$run_dir" "$DUNE")
        { printf '%q ' "${step_cmd[@]}"; echo; } >"$pdir/cmd" || die "cannot record the plan's command"
        repeat_launch "$pdir" "the replay plan"
        plan_rc=$step_rc
        # The trace holds the environment dune ran with; only the plan reads it.
        rm -f "$bphase/trace.csexp"
        { cat "$pdir/stdout"; cat "$pdir/stderr"; } | tee -a "$run_dir/log"
        why=$(sed -n 's/^replay-rules: //p' "$pdir/stderr" | tail -n 1)
        case $plan_rc in
          0) ;;
          3 | 5)
            repeat_direct_refused="DIRECT REFUSED -- $why; repeat it without --direct"
            repeat_direct_rc=2 ;;
          4)
            if [ "$build_rc" != 0 ]; then
              repeat_direct_refused="BUILD FAILED -- the build phase exited $build_rc, and ${why#NOT RUN -- } (see $bphase/stderr)"
              repeat_direct_rc=$build_rc
            else
              repeat_direct_refused="$why"
              repeat_direct_rc=2
            fi ;;
          *)
            [ -n "$repeat_cancelled" ] ||
              repeat_direct_refused="ERROR -- the replay could not be planned (exit $plan_rc; see $pdir/stderr)"
            repeat_direct_rc=$plan_rc ;;
        esac
      fi
    fi
    i=1
    while [ "$i" -le "$repeats" ] && [ -z "$repeat_cancelled" ] && [ -z "$repeat_refused" ] &&
          [ -z "$repeat_bounded" ] && [ -z "$repeat_direct_refused" ]; do
      iter=$run_dir/iteration-$i
      mkdir "$iter" || die "cannot create $iter"
      repeat_dune_cmd "$iter"
      {
        if [ "$direct" = 1 ]; then
          printf 'bash %q run --plan %q --out %q; ' "$runner_root/tools/replay-rules.sh" "$run_dir/plan.json" "$iter"
        else
          printf '%q clean --build-dir=%q && ' "$DUNE" "$repeat_build"
        fi
        printf '%q ' "${repeat_cmd[@]}"
        echo
      } >"$iter/cmd" ||
        die "cannot record iteration $i command"
      printf 'repeat: iteration %s/%s -- %s%s\n' "$i" "$repeats" \
        "$([ "$direct" = 1 ] && printf 'replay, then dune --force' || printf dune)" \
        "$([ "$alone" = 1 ] && printf ' (alone, -j 1)' || :)"
      # A stop can land after the loop condition. Refuse a launch already
      # known to be cancelled, then recheck immediately after starting the
      # supervisor to close the unavoidable signal-between-commands window.
      [ -z "$repeat_cancelled" ] || break
      if [ "$direct" = 1 ]; then repeat_iter_cmd "$iter" replay; else repeat_iter_cmd "$iter" clean; fi
      repeat_launch "$iter" "iteration $i"
      iter_rc=$step_rc
      if [ "$direct" = 1 ]; then
        # What the iteration's dune ran itself, its program and action per process (the
        # whitelisted fields; the trace itself, which holds the environment, is not kept):
        # the alias actions it re-ran, and no compile -- it rebuilt nothing.
        if [ -f "$iter/trace.csexp" ]; then
          bash "$runner_root/tools/action-durations.sh" --tsv -n 0 "$iter/trace.csexp" 2>/dev/null |
            cut -f3,4 >"$iter/dune-processes"
          rm -f "$iter/trace.csexp"
        fi
      fi
      [ "$first_nonzero" != 0 ] || [ "$iter_rc" = 0 ] || first_nonzero=$iter_rc
      {
        printf '=== repeat iteration %s/%s stdout ===\n' "$i" "$repeats"
        cat "$iter/stdout"
        printf '=== repeat iteration %s/%s stderr ===\n' "$i" "$repeats"
        cat "$iter/stderr"
        if [ "$direct" = 1 ]; then
          printf '=== repeat iteration %s/%s replayed rules ===\n' "$i" "$repeats"
          cat "$iter/statuses" 2>/dev/null
          cat "$iter/replay-stderr" 2>/dev/null
        fi
        printf '=== repeat iteration %s/%s exit %s ===\n' "$i" "$repeats" "$iter_rc"
      } >>"$run_dir/log" || die "cannot append iteration $i to $run_dir/log"
      printf 'repeat: iteration %s/%s exit %s; %sstdout=%s stderr=%s\n' \
        "$i" "$repeats" "$iter_rc" "$([ "$direct" = 1 ] && printf 'outputs=%s ' "$iter/outputs" || :)" \
        "$iter/stdout" "$iter/stderr"
      completed=$i
      # An invocation dune's parser refused ran nothing, so there is nothing
      # whose stability the remaining iterations could measure: stop after the
      # first, and let the verdict below say so rather than reporting N
      # identical refusals as IDENTICAL. Only the first iteration can decide
      # this -- every iteration runs the same argv (a direct set's build phase
      # decided it already).
      if [ "$direct" = 0 ] && [ "$i" = 1 ] && [ "$iter_rc" = 1 ] && dune_refusal "$iter/stderr" >/dev/null; then
        repeat_refused=1
        break
      fi
      # Nor after a bound ended the iteration (repeat_bound_of).
      repeat_bound_of "$iter" "$iter_rc" "iteration $i" && break
      [ -z "$repeat_cancelled" ] || break
      i=$(( i + 1 ))
    done
    # The isolated build tree can be very large (a training target pulls most
    # of the library graph). Only the diagnostic streams and pairwise diffs are
    # promised artifacts, so remove this exact generated child before keeping
    # the run directory for seven days.
    [ "$repeat_build" = "$run_dir/build" ] || die "refusing an unexpected repeat build path"
    perl -MFile::Path=remove_tree -e 'remove_tree($ARGV[0])' "$repeat_build" ||
      die "cannot remove repeat build context $repeat_build"
    [ ! -e "$repeat_build" ] || die "repeat build context survived cleanup: $repeat_build"

    mkdir "$run_dir/diffs" || die "cannot create $run_dir/diffs"
    differing=0
    stderr_only=0
    identical=0
    # What a pair compares: a difference in a red record makes it `differing`, in a diagnostic
    # one alone `stderr-only`. A direct iteration's records add the replayed rules' outputs,
    # their statuses and what they printed (tools/replay-rules.sh) to dune's own streams.
    red_records=(stdout exit)
    diag_records=(stderr)
    if [ "$direct" = 1 ]; then
      red_records+=(outputs statuses replay-stdout)
      diag_records+=(replay-stderr)
    fi
    same_record() { # <left> <right>: 0 iff equal (two absent records are equal)
      [ -e "$1" ] || [ -e "$2" ] || return 0
      cmp -s "$1" "$2"
    }
    diff_record() { # <left> <right> <out> <what>
      diff -u -N "$1" "$2" >"$3" || [ $? -eq 1 ] || die "cannot diff $4"
    }
    i=1
    while [ "$i" -lt "$completed" ]; do
      j=$(( i + 1 ))
      while [ "$j" -le "$completed" ]; do
        left=$run_dir/iteration-$i
        right=$run_dir/iteration-$j
        pair=$i-$j
        red_moved=() diag_moved=()
        for record in "${red_records[@]}"; do
          same_record "$left/$record" "$right/$record" || red_moved+=("$record")
        done
        for record in "${diag_records[@]}"; do
          same_record "$left/$record" "$right/$record" || diag_moved+=("$record")
        done
        if [ ${#red_moved[@]} = 0 ] && [ ${#diag_moved[@]} = 0 ]; then
          pair_result=identical
          identical=$(( identical + 1 ))
        elif [ ${#red_moved[@]} = 0 ]; then
          pair_result=stderr-only
          stderr_only=$(( stderr_only + 1 ))
        else
          pair_result=differing
          differing=$(( differing + 1 ))
        fi
        for record in ${red_moved[@]+"${red_moved[@]}"} ${diag_moved[@]+"${diag_moved[@]}"}; do
          diff_record "$left/$record" "$right/$record" "$run_dir/diffs/$pair.$record" \
            "$record for iterations $i and $j"
        done
        printf 'repeat: pair %s/%s: %s\n' "$i" "$j" "$pair_result" |
          tee -a "$run_dir/log"
        j=$(( j + 1 ))
      done
      i=$(( i + 1 ))
    done
    # The replay's fidelity witness: dune's own run of the replayed rules (the build phase's
    # snapshot) against the first replay. Replays that agree with each other but not with
    # dune are a replay that does not reproduce the rule's conditions, never a stable test.
    replay_witness=
    if [ "$direct" = 1 ] && [ "$completed" -ge 1 ] && [ -f "$run_dir/build-phase/outputs" ]; then
      if same_record "$run_dir/build-phase/outputs" "$run_dir/iteration-1/outputs"; then
        replay_witness=identical
      else
        replay_witness=differing
        diff_record "$run_dir/build-phase/outputs" "$run_dir/iteration-1/outputs" \
          "$run_dir/diffs/build-1.outputs" "the build phase's outputs and iteration 1's"
      fi
      printf "repeat: dune's own run vs replay 1: %s\n" "$replay_witness" | tee -a "$run_dir/log"
    fi

    if [ -n "$repeat_cancelled" ]; then
      repeat_result="CANCELLED -- completed $completed of $repeats iterations"
      repeat_reported_cancelled=1
      final_rc=$first_nonzero
      [ "$final_rc" != 0 ] || final_rc=143
    elif [ -n "$repeat_refused" ]; then
      repeat_result="INVOCATION REFUSED -- dune rejected the arguments; nothing ran, so nothing was repeated"
      final_rc=$first_nonzero
    elif [ -n "$repeat_direct_refused" ]; then
      repeat_result=$repeat_direct_refused
      final_rc=$repeat_direct_rc
    # A bound's verdict outranks every comparison, and its code the first red: the set did not
    # finish, and the box needs attention before it is run again.
    elif [ "$repeat_bounded" = 69 ]; then
      repeat_result="DEVICE UNHEALTHY -- $repeat_bounded_where's device probe got no answer, and the set stopped there ($completed of $repeats iterations started)"
      final_rc=69
    elif [ "$repeat_bounded" = 142 ]; then
      repeat_result="TIMEOUT -- in $repeat_bounded_where a test outlived the per-test cap ($(sed -n 's/^command //p' "$run_dir/test-cap")), and the set stopped there ($completed of $repeats iterations started)"
      final_rc=142
    elif [ "$differing" -gt 0 ]; then
      if [ "$direct" = 1 ]; then
        repeat_result="DIFFERING -- stdout, exit status or a replayed rule's output moved across $differing pair(s)"
      else
        repeat_result="DIFFERING -- stdout or exit status moved across $differing pair(s)"
      fi
      final_rc=$first_nonzero
      [ "$final_rc" != 0 ] || final_rc=1
    elif [ "$replay_witness" = differing ]; then
      repeat_result="REPLAY DIVERGED -- every replay agreed, but not with dune's own run of the same rules ($run_dir/diffs/build-1.outputs): the replay did not reproduce the rules' conditions; repeat it without --direct"
      final_rc=$first_nonzero
      [ "$final_rc" != 0 ] || final_rc=1
    elif [ "$stderr_only" -gt 0 ]; then
      repeat_result="STDERR-ONLY -- stdout and exit status were stable; stderr moved across $stderr_only pair(s)"
      final_rc=$first_nonzero
    else
      repeat_result="IDENTICAL -- stdout, stderr and exit status matched across all $identical pair(s)"
      [ "$direct" = 0 ] || repeat_result="IDENTICAL -- the replayed outputs, stdout, stderr and exit status matched across all $identical pair(s), and dune's own run"
      final_rc=$first_nonzero
    fi
    printf 'repeat result: %s\n' "$repeat_result" | tee -a "$run_dir/log"
    if [ -n "$repeat_refused" ]; then
      refused_stderr=$run_dir/iteration-1/stderr
      [ "$direct" = 0 ] || refused_stderr=$run_dir/build-phase/stderr
      { echo "dune said:"
        dune_refusal "$refused_stderr" | sed 's/^/  /'
        echo "fix the command line and run again -- no test was built or judged" \
             "(dune's own status was $final_rc; this script exits 2, its usage code)"
      } | tee -a "$run_dir/log"
    fi
    printf 'repeat artifacts: %s (pairwise diffs under %s/diffs)\n' "$run_dir" "$run_dir" |
      tee -a "$run_dir/log"
    # repeat_exit keeps cancellation deferred through cleanup/comparison and
    # publishes the atomic verdict while signals are ignored.
    exit "$final_rc"
    ;;
  _resolve)
    # The run's first phase, and not a command for callers (gh-ocannl-1106):
    # the supervisor's child, in the run's own process group, under its lock,
    # its cap and its signal handling -- the ones dune then runs under, so the
    # resolution needs no process machinery of its own. It resolves the
    # batch's backends where something reads them (plan_batch), decides the
    # width and the fleet slot, and records them in the run: the capped
    # command rewritten into `cmd`, so the width is part of the RECORDED
    # command and of every later digest (gh-ocannl-1066), the announcements
    # into the log and into `resolved`, which marks the end of the phase for
    # the launcher and the supervisor. Then it runs dune (through the slot's
    # fleet-worker.sh) as its child, in the group it leads, so the pid and
    # group the supervisor caps, signals and reaps stay the same, and records
    # dune's promotion list once dune exits. For `plan` it resolves whatever
    # this box is, writes the report to `plan` and exits. A GPU batch's device
    # probe runs inside the slot, ahead of dune (plan_device_probe, `_probe`),
    # and the per-test cap's watcher beside dune (test_cap_watch).
    [ $# -ge 4 ] && [ -n "$testrun_own" ] ||
      die "_resolve is the supervisor's first phase, not a command"
    mode=$1 cap=$2 test_cap=$3 DUNE=$4
    shift 4
    run_dir=$testrun_own
    # An alias dune does not know is refused first (plan_aliases): before the
    # backends' resolution builds anything, and before the slot and the probe.
    plan_aliases "$@"
    if [ -n "$alias_refusal" ] && [ "$mode" != plan ]; then
      { printf '%s\n' "$alias_refusal" >"$run_dir/alias-refused.tmp" &&
          mv -f "$run_dir/alias-refused.tmp" "$run_dir/alias-refused"; } ||
        { echo "test-run: cannot record the alias refusal in $run_dir"; exit 126; }
      printf '%s\n' "$alias_refusal" | sed 's/^/test-run: ALIAS REFUSED: /'
      exit 2
    fi
    plan_slot
    expect_backends=$(cat "$run_dir/expect-backend" 2>/dev/null) || expect_backends=
    if [ "$mode" = plan ]; then
      batch_resolve "$DUNE" "$run_dir/log" "$@"
    else
      plan_batch "$run_dir/log" "$@"
      # An expectation is judged on a resolution wherever the run is.
      [ -z "$expect_backends" ] || [ -n "$batch_resolved" ] || batch_resolve "$DUNE" "$run_dir/log" "$@"
    fi
    plan_backend_check
    if [ -n "$backend_refusal" ] && [ "$mode" != plan ]; then
      # Before the slot, the probe and dune: nothing ran, and the digest says why.
      { printf '%s\n' "$backend_refusal" >"$run_dir/backend-refused.tmp" &&
          mv -f "$run_dir/backend-refused.tmp" "$run_dir/backend-refused"; } ||
        { echo "test-run: cannot record the backend refusal in $run_dir"; exit 126; }
      printf 'test-run: BACKEND REFUSED: %s\n' "$backend_refusal"
      exit 2
    fi
    if [ -n "$backend_warning" ] && [ "$mode" != plan ] && [ "$mode" != repeat ]; then
      printf '%s\n' "$backend_warning" >"$run_dir/backend-warning" ||
        { echo "test-run: cannot record the backend warning in $run_dir"; exit 126; }
    fi
    clamp_slot_wait
    plan_width_cap "$@"
    plan_slot_kind
    plan_device_probe
    plan_test_cap "$test_cap" "$@"
    # A repeat's resolution phase (gh-ocannl-1252) ends here: the two bounds `run` would give this
    # batch, for the coordinator to apply to every iteration (`repeat`), and said in the log. No
    # width and no slot: a repeat runs as given.
    if [ "$mode" = repeat ]; then
      watch_kind=cpu
      ! run_reads_batch || [ "$(batch_kind)" != gpu ] || watch_kind=gpu
      { printf '%s\n' "$test_cap" "$watch_kind" "${probe_cap:-0}" "${probe_prog:--}" "$probe_backends" \
          >"$run_dir/bounds.tmp" && mv -f "$run_dir/bounds.tmp" "$run_dir/bounds"; } ||
        { echo "test-run: cannot record the repeat's bounds in $run_dir"; exit 126; }
      if [ "$test_cap" -gt 0 ]; then
        echo "test-run: repeat: per-test cap: ${test_cap}s for each process dune starts ($test_cap_why); one past it ends the iteration and the set"
      else
        echo "test-run: repeat: per-test cap: none ($test_cap_why)"
      fi
      if [ -n "$probe_backends" ]; then
        echo "test-run: repeat: device probe: $probe_backends before each iteration, bounded at ${probe_cap}s; one that does not answer ends the set as DEVICE UNHEALTHY"
      fi
      exit 0
    fi
    # A fresh run's own build directory (see the header), recorded in its
    # command: the digest's `command:` line names where the run built.
    if [ "$mode" = fresh ]; then
      fresh_sub=$1
      shift
      set -- "$fresh_sub" "--build-dir=$run_dir/build" "$@"
    fi
    # The width goes immediately after dune's subcommand, where dune accepts
    # it whatever the target is, and always before dune's own `--`.
    if [ -n "$width_cap" ]; then
      width_sub=$1
      shift
      set -- "$width_sub" -j "$width_cap" "$@"
    fi
    if [ "$mode" = plan ]; then
      {
        echo "command: dune $*"
        echo "backends: $(batch_summary)"
        sed -n 's/^test-run: batch: /  /p' "$run_dir/log"
        if [ -n "$alias_refusal" ]; then
          echo "aliases: run would refuse this batch (ALIAS REFUSED, exit 2)"
          printf '%s\n' "$alias_refusal" | sed 's/^/  /'
        fi
        if [ -n "$backend_refusal" ]; then
          echo "expect-backend: run would refuse this batch (BACKEND REFUSED, exit 2)"
          printf '  %s\n' "$backend_refusal"
        elif [ -n "$expect_backends" ]; then
          echo "expect-backend: $expect_backends, held"
        fi
        [ -z "$backend_warning" ] || printf '%s\n' "$backend_warning"
        if [ -n "$width_cap" ]; then
          echo "width: -j $width_cap, injected ($batch_width_hazard hazard, for $batch_width_backend)"
        elif explicit_jobs "$@"; then
          if [ -n "$batch_width_cap" ]; then
            echo "width: the caller's (a -j $batch_width_cap cap applies here, for $batch_width_backend)"
          else
            echo "width: the caller's (no backend of the batch meets a cap on this box)"
          fi
        else
          echo "width: dune's default (no backend of the batch meets a cap on this box)"
        fi
        [ -z "$width_announce" ] || printf '  %s\n' "$width_announce"
        if [ -n "$slot_fw" ]; then
          echo "slot: --$slot_kind"
          printf '  %s\n' "$slot_announce"
        else
          echo "slot: none (not a fleet box, or the slot is turned off)"
        fi
        if [ -n "$probe_backends" ]; then
          echo "device probe: $probe_backends, bounded at ${probe_cap}s"
        else
          echo "device probe: none (no GPU backend resolved, or the probe is turned off)"
        fi
        if [ "$test_cap" -gt 0 ]; then
          echo "test cap: ${test_cap}s for each process dune starts, $test_cap_why; one past it ends the run"
        else
          echo "test cap: none ($test_cap_why)"
        fi
      } >"$run_dir/plan" || { echo "test-run: cannot write the plan in $run_dir"; exit 126; }
      exit 0
    fi
    if [ -n "$width_cap" ] || [ "$mode" = fresh ]; then
      { { printf '%q ' "$@"; echo; } >"$run_dir/cmd.tmp" &&
        mv -f "$run_dir/cmd.tmp" "$run_dir/cmd"; } ||
        { echo "test-run: cannot record the capped command in $run_dir"; exit 126; }
    fi
    # Into the run's log (this process's stdout) and, through `resolved`, onto
    # the launcher's stderr: the cap is in the artifact triage reads rather
    # than only in the launching terminal's scrollback.
    : >"$run_dir/resolved.tmp" || { echo "test-run: cannot write in $run_dir"; exit 126; }
    for line in "$backend_warning" "$width_announce" "$slot_announce" "$probe_announce"; do
      [ -z "$line" ] || printf 'test-run: %s\n' "$line" | tee -a "$run_dir/resolved.tmp"
    done
    # Decided on the caller's argv, before the slot's words are put in front.
    promotions_recordable=1
    ! explicit_build_root "$@" || promotions_recordable=
    # The batch's dune writes its trace into the run directory, for the run's
    # slowest actions (see explicit_trace_file) -- after `cmd` was recorded,
    # so the command the digest prints is still the caller's. Not where the
    # argv already sends the trace elsewhere, nor for a subcommand not known
    # to take the option, where dune would refuse it. Absolute, whatever
    # `--root` the argv names.
    if ! explicit_trace_file "$@"; then
      case $1 in
        build | runtest | test | exec)
          trace_file=$run_dir/trace.csexp
          case $trace_file in /*) ;; *) trace_file=$PWD/$trace_file ;; esac
          trace_sub=$1
          shift
          set -- "$trace_sub" "--trace-file=$trace_file" "$@" ;;
      esac
    fi
    # A probed batch's command is `_probe`, which queries the devices and
    # execs dune in its place; under a fleet slot it is fleet-worker.sh, which
    # takes the slot and execs that command in its place. So dune's status is
    # the slot's, and the probe's own refusal (69) is the only other.
    set -- "$DUNE" "$@"
    if [ -n "$probe_backends" ]; then
      # shellcheck disable=SC2086  # the backend list is word-split on purpose
      set -- env "OCANNL_TOOL_TESTRUN_OWN=$run_dir" "${BASH:-bash}" "$runner_root/tools/test-run.sh" _probe \
        "$DUNE" "$probe_prog" "$probe_cap" \
        $probe_backends -- "$@"
    fi
    if [ -n "$slot_fw" ]; then
      printf '%s\n' "$slot_fw" >"$run_dir/slot" 2>/dev/null || :
      set -- "$slot_fw" execution slot --wait "$slot_wait" "--$slot_kind" -- "$@"
    fi
    # Where the launch's own prelude in the log ends (dune_refusal).
    wc -c <"$run_dir/log" | tr -d ' ' >"$run_dir/prelude" 2>/dev/null || :
    mv -f "$run_dir/resolved.tmp" "$run_dir/resolved" ||
      { echo "test-run: cannot record the end of the resolution in $run_dir"; exit 126; }
    # Dune runs as this phase's child, not in its place, for the run's last
    # phase: recording dune's promotion list once it exits (record_promotions,
    # gh-ocannl-1087), before the supervisor publishes the verdict. The group
    # the supervisor caps, signals and reaps is still this process's, which
    # leads it; a signal that kills dune ends this shell with the same status
    # an exec'd dune would have had, and dune's own status is the one it
    # exits with. The per-test cap's watcher runs beside it, without the
    # worktree lock, and is gone before the promotion list is asked for.
    watch=
    if [ "$test_cap" -gt 0 ]; then
      watch_kind=cpu
      ! run_reads_batch || [ "$(batch_kind)" != gpu ] || watch_kind=gpu
      perl -e "$test_cap_watch" "$test_cap" "$$" "$PPID" "$run_dir" "$watch_kind" </dev/null 9>&- &
      watch=$!
    fi
    "$@"
    rc=$?
    if [ -n "$watch" ]; then
      kill "$watch" 2>/dev/null
      wait "$watch" 2>/dev/null
    fi
    [ -z "$promotions_recordable" ] || record_promotions
    [ "$mode" != fresh ] || [ -z "${trace_file:-}" ] || fresh_timings "$trace_file"
    exit "$rc"
    ;;
  _direct_plan)
    # A direct repeat's plan (gh-ocannl-1377), and not a command for callers: a step the
    # coordinator supervises like an iteration, after the build phase. It reads the build
    # phase's processes from its trace (tools/action-durations.sh --tsv --dirs: the whitelisted
    # fields, plus each process's directory), keeps those that are a named test's action by
    # the rule `fresh` times a test by (attribution_awk, rows mode), and has
    # tools/replay-rules.sh ask dune for their rules (`plan.json`; its exit 3/4/5 refusals pass
    # through) and snapshot their targets as dune's own run left them (`build-phase/outputs`).
    [ $# -eq 2 ] && [ -n "$testrun_rd" ] ||
      die "_direct_plan is a step of repeat --direct, not a command"
    run_dir=$1 DUNE=$2
    pdir=$testrun_rd
    if ! bash "$runner_root/tools/action-durations.sh" --tsv --dirs -n 0 "$run_dir/build-phase/trace.csexp" \
         >"$pdir/rows-dirs.tsv" 2>"$pdir/rows.err"; then
      grep -q '^action-durations.sh: no process in ' "$pdir/rows.err" ||
        { cat "$pdir/rows.err" >&2; echo "test-run: the build phase's trace could not be read" >&2; exit 126; }
      : >"$pdir/rows-dirs.tsv"
    fi
    cut -f2- "$pdir/rows-dirs.tsv" >"$pdir/rows.tsv" &&
      awk -F '\t' -v mode=rows "$attribution_awk" "$run_dir/direct-aliases" "$pdir/rows.tsv" >"$pdir/hits" &&
      awk -F '\t' 'NR == FNR { row[FNR] = $0; next } { print $1 "\t" row[$2] }' \
        "$pdir/rows-dirs.tsv" "$pdir/hits" >"$pdir/rows" ||
      { echo "test-run: the build phase's processes could not be attributed" >&2; exit 126; }
    bash "$runner_root/tools/replay-rules.sh" plan --dune "$DUNE" --build-dir "$run_dir/build" \
      --rows "$pdir/rows" --aliases "$run_dir/direct-aliases" --out "$run_dir/plan.json" </dev/null ||
      exit $?
    exec bash "$runner_root/tools/replay-rules.sh" run --plan "$run_dir/plan.json" \
      --out "$run_dir/build-phase" --snapshot </dev/null
    ;;
  _probe)
    # A probed GPU batch's first act inside its slot, and not a command for
    # callers (gh-ocannl-1211; plan_device_probe decides, tools/device-probe.sh asks):
    # one bounded device query per GPU backend, then dune in this process's
    # place. A program of `-` is bin/device_props, built here, under the slot,
    # with its build output in `device-probe-build.log`. Each answer goes to
    # `device-probe` in the run directory, for the digest, and to the log only
    # when it refuses -- the log's first line after the prelude must stay
    # dune's (dune_refusal). Asked from test/config, so the query reads the
    # configuration the tests read, with the backend pinned on its command line.
    [ $# -ge 6 ] && [ -n "$testrun_own" ] ||
      die "_probe is a run's own phase, not a command"
    run_dir=$testrun_own
    DUNE=$1 probe_prog=$2 probe_cap=$3
    shift 3
    probe_backends=
    while [ $# -gt 0 ] && [ "$1" != -- ]; do probe_backends="$probe_backends $1"; shift; done
    [ $# -ge 2 ] || die "_probe: no command after the backends"
    shift
    if [ "$probe_prog" = - ]; then
      # Where dune puts it (see batch_resolve on DUNE_BUILD_DIR).
      probe_prog=${DUNE_BUILD_DIR:-_build}
      case $probe_prog in /*) ;; *) probe_prog=$PWD/$probe_prog ;; esac
      probe_prog=$probe_prog/default/bin/device_props.exe
      if ! "$DUNE" build ./bin/device_props.exe </dev/null >"$run_dir/device-probe-build.log" 2>&1 ||
         [ ! -x "$probe_prog" ]; then
        printf 'not probed: %s did not build (%s)\n' "$probe_prog" "$run_dir/device-probe-build.log" \
          >>"$run_dir/device-probe"
        exec "$@"
      fi
    fi
    # shellcheck disable=SC2086  # the backend list is word-split on purpose
    sh -c "$device_probe_sh" device-probe test-run "$probe_perl" "$probe_cap" "$run_dir" \
      "$probe_prog" $probe_backends
    [ $? -ne 69 ] || exit 69
    exec "$@"
    ;;
  run | start | plan | fresh)
    # `plan` is what `run` would do with this argv, without running it: the
    # batch's resolved backends and why, the width it would inject, and the
    # fleet slot it would take. It is a launch whose first phase is its last
    # (see `_resolve`): the readers' build is a dune run in this worktree, so
    # it holds the worktree lock throughout like a launch does (a run started
    # meanwhile is refused, not raced; Codex review round 1 on PR #832), under
    # a run directory that is never published and is removed at the end, and
    # its --cap bounds the build as a run's would.
    cap=${OCANNL_TOOL_TEST_CAP:-3600}
    # Empty is the default, which `_resolve` decides once it knows the batch
    # (plan_test_cap); a value here is explicit and applies to any batch.
    test_cap=${OCANNL_TOOL_PER_TEST_CAP:-}
    expect=
    while [ $# -gt 0 ]; do
      case $1 in
        --cap) [ $# -ge 2 ] || die "--cap requires a value"; cap=$2; shift 2 ;;
        --test-cap) [ $# -ge 2 ] || die "--test-cap requires a value"; test_cap=$2; shift 2 ;;
        --expect-backend)
          [ $# -ge 2 ] || die "--expect-backend requires a backend"
          batch_known_backend "$2" ||
            die "--expect-backend $2: not a backend (cc, multidev_cc, cuda, hip, metal)"
          case " $expect " in
            *" $(batch_canonical "$2") "*) ;;
            *) expect="$expect${expect:+ }$(batch_canonical "$2")" ;;
          esac
          shift 2 ;;
        --) shift; break ;;
        *) break ;;
      esac
    done
    normalize_cap
    [ -z "$test_cap" ] || normalize_cap test_cap --test-cap
    reject_misplaced_options "$@"
    # `fresh` is `run` over an argv of its own making (see the header and
    # fresh_argv); from here on it differs only where `_resolve` and `digest`
    # read the run's `mode`.
    if [ "$sub" = fresh ]; then
      fresh_argv "$@"
      set -- "${fresh_args[@]}"
    fi
    [ $# -gt 0 ] || set -- runtest
    # Toolchain checks gate only launches: status/wait/stop/list remain usable
    # from a shell whose opam environment is no longer active.
    select_dune
    # Cancellation is armed BEFORE the lock is taken, for every mode: from
    # here on the launcher holds state a signal must not abandon halfway (the
    # lock, then a published run). For `run` and `plan` the signal is
    # forwarded to the supervisor -- at once when there is one, and right
    # after the launch for one that arrived before; for `start` it is merely
    # deferred past the launch: the launcher is about to exit anyway, and the
    # run is MEANT to survive it.
    cancelled= sup=
    forward_cancel() {
      [ "$sub" != start ] && [ -n "$sup" ] || return 0
      # The supervisor is signalled only while it can be identified: as this
      # shell's own unreaped child before it has recorded its identity, and
      # under its recorded start token afterwards -- never by a bare pid that
      # a process may have recycled after the reaping wait.
      { [ ! -f "$run_dir/pid" ] || sup_alive "$run_dir"; } || return 0
      # Per-signal, so an interrupt records 130 and not a generic 143: INT is
      # relayed as INT (the supervisor maps it to 130); HUP relays as TERM
      # because the detached-mode supervisor deliberately ignores HUP.
      case $cancelled in
        INT) kill -INT "$sup" 2>/dev/null ;;
        *) kill -TERM "$sup" 2>/dev/null ;;
      esac
    }
    fwd_sig() { cancelled=$1; sigs_seen=$(( sigs_seen + 1 )); forward_cancel; }
    # A signal sent to the launcher's process group -- Ctrl-C or a closed
    # terminal, `kill -TERM -<pgid>` -- reaches this shell's own probe
    # children as well as this shell and the supervisor. A cat or ps it kills
    # makes sup_alive answer "gone" for a supervisor that is alive, reaping
    # its group on that same signal and about to publish the verdict; the
    # attached wait then reported "died without recording a verdict" with
    # the lock still held (gh-ocannl-1362). So a "gone" counts only from a
    # probe no trapped signal landed in: bash runs the trap before the
    # interrupted probe returns, so the count moving means ask again.
    sigs_seen=0
    sup_alive_settled() {
      local seen
      while :; do
        seen=$sigs_seen
        sup_alive "$1" && return 0
        [ "$seen" != "$sigs_seen" ] || return 1
      done
    }
    trap 'fwd_sig INT' INT
    trap 'fwd_sig TERM' TERM
    trap 'fwd_sig HUP' HUP
    new_run "$@"
    if [ -n "$expect" ]; then
      printf '%s\n' "$expect" >"$run_dir/expect-backend" ||
        { rm -rf "$run_dir"; die "cannot record the expected backends in $run_dir"; }
    fi
    if [ "$sub" = fresh ]; then
      { printf '%s\n' fresh >"$run_dir/mode" &&
        printf '%s\n' "${fresh_aliases[@]}" >"$run_dir/fresh-aliases"; } ||
        { rm -rf "$run_dir"; die "cannot record the fresh run's metadata in $run_dir"; }
    fi
    take_lock
    # After the supervisor exits: the report, and the run directory removed --
    # the lock went with the supervisor, so the owner pointer never names a
    # deleted directory while the lock is held (Codex review round 8).
    plan_finish() { # dune argv
      local rc
      rc=$(cat "$run_dir/exit" 2>/dev/null) || rc=
      case $rc in
        0) cat "$run_dir/plan" ;;
        142)
          echo "command: dune $*"
          echo "backends: unresolved (the cap expired while they resolved)"
          echo "slot: none -- resolving the backends used the whole cap, so run would record the cap's verdict (142) without starting dune"
          ;;
        129 | 130 | 143) ;;
        *) { echo "test-run: plan failed (exit ${rc:-unrecorded}); its log:"
             sed 's/^/  /' "$run_dir/log"; } >&2 ;;
      esac
      rm -rf "$run_dir"
      case $rc in 0 | 142) exit 0 ;; 129 | 130 | 143) exit "$rc" ;; *) exit 2 ;; esac
    }
    if [ "$sub" != plan ]; then
      publish_run || { rm -rf "$run_dir"; die "cannot publish $run_dir"; }
    fi
    # The supervisor inherits lock fd 9 and owns the run from here: it records
    # its identity, runs its child under the cap -- first the batch's
    # resolution (`_resolve`), which then runs dune -- and publishes the
    # verdict (see supervisor_perl). Nothing about the fate of THIS shell --
    # HUP from a closed terminal, harness cancellation, a plain kill -- can
    # lose the verdict; `run` differs from `start` only in staying attached to
    # wait and digest.
    OCANNL_TOOL_TESTRUN_BG=1 OCANNL_TOOL_TESTRUN_RD=$run_dir OCANNL_TOOL_TESTRUN_OWN=$run_dir \
      OCANNL_TOOL_TESTRUN_RUNNER=$runner_root perl -e "$supervisor_perl" -- "$cap" \
        env "OCANNL_TOOL_TESTRUN_OWN=$run_dir" "${BASH:-bash}" "$runner_root/tools/test-run.sh" _resolve "$sub" "$cap" "$test_cap" "$DUNE" "$@" \
        </dev/null >>"$run_dir/log" 2>&1 &
    sup=$!
    # The launcher's own fd 9 copy served its purpose the moment the
    # supervisor inherited the lock's description: close it, so an attached
    # launcher never appears in a leftover census -- a concurrent stop must
    # not reap the process that is about to deliver the digest.
    exec 9>&-
    # A signal flagged before the supervisor existed is converted now.
    [ -z "$cancelled" ] || forward_cancel
    # The launch is reported only once the supervisor has recorded itself:
    # `wait last` from the very next command must find a live owner, not a
    # published run with no supervisor. Bounded. A supervisor that died first,
    # or gave up (it records a verdict for that, and dune never ran), ends
    # the wait at once; one wedged before its first write is reported after
    # a minute, with the run left to it.
    i=0
    while [ ! -s "$run_dir/pid" ] && [ ! -f "$run_dir/exit" ]; do
      kill -0 "$sup" 2>/dev/null || break
      case $(ps -o state= -p "$sup" 2>/dev/null | tr -d ' ') in Z*) break ;; esac
      i=$(( i + 1 ))
      if [ "$i" -gt 600 ]; then
        echo "test-run: the supervisor (pid $sup) has not recorded itself after 60s;" >&2
        echo "  the run is left to it: $run_dir (log: $run_dir/log)" >&2
        exit 2
      fi
      sleep 0.1
    done
    if [ ! -s "$run_dir/pid" ]; then
      # Gone, or given up, before recording itself: its verdict, if it left
      # one, says why (nothing ran); otherwise only its log can.
      wait "$sup" 2>/dev/null
      sup=
      trap - INT TERM HUP
      [ "$sub" != plan ] || plan_finish "$@"
      if [ -f "$run_dir/exit" ]; then
        digest "$run_dir"
        exit "$digest_rc"
      fi
      echo "run died before its supervisor recorded itself: $run_dir (log: $run_dir/log)"
      exit 1
    fi
    if [ "$sub" != plan ]; then
      # The launch is reported once its first phase has resolved the batch
      # (`resolved`): its announcements then reach this terminal before dune
      # starts, and `start` prints the command dune is given. A run that ended
      # in that phase -- cancelled, or its cap spent -- has its verdict instead.
      while [ ! -f "$run_dir/resolved" ] && [ ! -f "$run_dir/exit" ] && sup_alive_settled "$run_dir"; do
        sleep 0.1
      done
      [ ! -f "$run_dir/resolved" ] || cat "$run_dir/resolved" >&2
    fi
    if [ "$sub" != start ]; then
      # Attached: wait for the supervisor -- its exit means the verdict file
      # is on disk. A trapped signal returns from `wait` early, hence the
      # retry loop; sup_alive rather than a bare kill -0, since after the
      # supervisor is reaped its pid can be recycled and probing the number
      # alone would spin this loop on an unrelated process; settled, since a group
      # signal landing mid-probe kills the probe too (sup_alive_settled).
      while sup_alive_settled "$run_dir"; do wait "$sup" 2>/dev/null; done
      sup=
      trap - INT TERM HUP
      [ "$sub" != plan ] || plan_finish "$@"
      if [ -f "$run_dir/exit" ]; then
        digest "$run_dir"
        exit "$digest_rc"
      fi
      echo "run died without recording a verdict (supervisor killed?): $run_dir"
      exit 1
    fi
    trap - INT TERM HUP
    if [ ! -f "$run_dir/resolved" ]; then
      # Over before dune started.
      if [ -f "$run_dir/exit" ]; then
        digest "$run_dir"
        exit "$digest_rc"
      fi
      echo "run died before dune started: $run_dir (log: $run_dir/log)"
      exit 1
    fi
    disown
    echo "started: $run_dir"
    echo "  command: dune $(sed 's/ *$//' "$run_dir/cmd")"
    echo "  log:     $run_dir/log"
    echo "  check:   tools/test-run.sh status last    # from this worktree; never blocks"
    echo "  gate:    tools/test-run.sh wait last      # bounded; exits with dune's status"
    ;;
  bisect) bisect_main "$@" ;;
  _bisect_step) bisect_step "$@" ;;
  idle)
    [ $# -eq 0 ] || die "idle takes no arguments"
    # Use the same flock as run/start/repeat, including inherited holders after
    # a launcher or supervisor dies. This snapshot does not reserve the tree.
    probe_locks "$LOCK" "$PWD/.test-run.lock"
    idle_rc=$?
    case $idle_rc in
      0) ;;
      3) echo "test-run: worktree lock is held: $LOCK" >&2 ;;
      *) echo "test-run: cannot inspect worktree lock: $LOCK" >&2; idle_rc=2 ;;
    esac
    exit "$idle_rc"
    ;;
  status)
    resolve_run "${1:-last}"
    # One-shot and honest -- no sleeping in `status`. Ordered by LIVENESS,
    # never by pid-file presence: a supervisor killed after recording itself
    # must read as dead, not as running forever. The explicit `exit 0` on the
    # finished paths is the header contract: `status` reports PUBLICATION,
    # not the run's verdict, so it must not inherit whatever `digest`'s last
    # command happened to return.
    if [ -f "$run_dir/exit" ]; then
      digest "$run_dir"
      exit 0
    elif proc_alive "$run_dir/pid" "$run_dir/ptoken"; then
      echo "running: dune $(cat "$run_dir/cmd")  (log: $run_dir/log)"
      exit 3
    elif legacy_owner_alive "$run_dir"; then
      # A run of the previous version whose wrapper outlives its supervisor
      # to publish the verdict (or, for a repeat, its coordinator between
      # iterations): in flight, not dead.
      echo "managed, verdict pending (the previous version's wrapper is publishing): $run_dir"
      exit 3
    elif [ -f "$run_dir/exit" ]; then
      digest "$run_dir" # published between the checks above
      exit 0
    elif [ ! -f "$run_dir/pid" ] && lock_still_owned "$run_dir"; then
      # Published, but no supervisor on record, and the run holds its
      # worktree lock: the milliseconds between a launcher's publication and
      # its supervisor's first write. In flight -- the launcher is alive and
      # holding the lock -- so the running code, not the dead one.
      echo "launch in progress (supervisor not yet recorded): $run_dir"
      exit 3
    elif [ ! -f "$run_dir/pid" ]; then
      echo "launch interrupted before its supervisor started: $run_dir"
      exit 1
    else
      echo "run died without recording a verdict (killed externally?): $run_dir"
      exit 1
    fi
    ;;
  wait)
    ref=last budget=
    while [ $# -gt 0 ]; do
      case $1 in
        --timeout)
          [ $# -ge 2 ] || die "--timeout requires a value"
          budget=$2 budget_given=1
          shift 2
          ;;
        *) ref=$1; shift ;;
      esac
    done
    resolve_run "$ref"
    if [ -n "${budget_given:-}" ]; then
      # Same traps --cap had: leading zeros reach bash arithmetic as octal,
      # and oversized values wrap. Gated on the option being GIVEN, so an
      # explicitly empty value ('--timeout ""' from an unset harness
      # variable) is rejected rather than silently using the default budget.
      case $budget in '' | *[!0-9]*) die "--timeout must be a nonnegative integer of seconds" ;; esac
      [ ${#budget} -le 9 ] || die "--timeout too large (max 9 digits)"
      budget=$(( 10#$budget ))
    fi
    # Bounded by construction: default budget is every capped iteration plus
    # one cleanup allowance, so a managed repeat does not time out merely
    # because `cap` is per iteration. Ordinary runs have an implicit count 1.
    cap=$(cat "$run_dir/cap" 2>/dev/null) || cap=3600
    [ -n "$cap" ] || cap=3600
    cap=$(( 10#$cap )) # decimal, whatever an older run recorded
    wait_repeats=1
    if [ "$(cat "$run_dir/mode" 2>/dev/null)" = repeat ]; then
      wait_repeats=$(cat "$run_dir/repeats" 2>/dev/null) || wait_repeats=1
      case $wait_repeats in '' | *[!0-9]*) wait_repeats=1 ;; esac
      wait_repeats=$(( 10#$wait_repeats ))
      [ "$wait_repeats" -gt 0 ] || wait_repeats=1
      # A direct set's build phase and plan are capped steps of their own.
      [ "$(cat "$run_dir/direct" 2>/dev/null)" != 1 ] || wait_repeats=$(( wait_repeats + 2 ))
    fi
    [ -n "$budget" ] || budget=$(( cap > 0 ? cap * wait_repeats + 120 : 7200 ))
    waited=0
    while [ ! -f "$run_dir/exit" ]; do
      # Every sleep here -- the poll AND the dead-supervisor grace -- is
      # bounded by the remaining budget, so a short --timeout is honored to
      # the second rather than rounded up to an interval.
      remaining=$(( budget - waited ))
      if ! sup_alive "$run_dir"; then
        # The owner is gone: no process is left to publish. One short settle,
        # bounded like every sleep here, covers a verdict renamed into place
        # between the checks above (the supervisor writes it last, and only
        # then exits); an already-expired budget reports the documented
        # timeout.
        step=$(( remaining < 1 ? remaining : 1 ))
        [ "$step" -gt 0 ] && sleep "$step"
        waited=$(( waited + step ))
        [ -f "$run_dir/exit" ] && break
        # Re-asked after the settle: `wait last` can arrive in the
        # milliseconds between a launcher's publication and its
        # supervisor's first write, and an owner that has appeared since is
        # a running run, not a dead one.
        sup_alive "$run_dir" && continue
        # Recomputed AFTER the settle: consuming the final second must report
        # the documented timeout, not slip through on the pre-sleep value.
        [ $(( budget - waited )) -le 0 ] && { echo "wait timed out after ${budget}s: $run_dir"; exit 124; }
        echo "run died without recording a verdict (killed externally?): $run_dir"
        exit 1
      fi
      [ "$remaining" -le 0 ] && { echo "wait timed out after ${budget}s: $run_dir"; exit 124; }
      step=$(( remaining < 5 ? remaining : 5 ))
      sleep "$step"
      waited=$(( waited + step ))
    done
    digest "$run_dir"
    exit "$digest_rc"
    ;;
  stop)
    resolve_run "${1:-last}"
    # Leftover recovery, shared by the finished and dead-without-verdict
    # branches. Descendants of dune may hold the run's worktree lock through
    # inherited fd 9, possibly having setsid'd out of the recorded group.
    # Everything uses the run's RECORDED worktree and state root, not the
    # caller's -- an explicit run directory may belong to another checkout,
    # or to another OCANNL_TOOL_TEST_RUNS. The gate:
    # the lock is HELD with the owner pointer naming THIS run, so whatever
    # holds it is this run's leftovers. The recorded group is signaled only
    # under a matching leader token (a recycled numeric pgid is never
    # trusted); the lsof census on the lock file covers detached holders
    # the group cannot see. reap_cycle reports honestly when the lock is
    # STILL held afterwards (no lsof on this system, or unkillable holders).
    lock_paths_of "$run_dir" || {
      # No worktree on record: nothing can attribute leftovers to this run,
      # so the leftover branches below stay closed and only the recorded
      # owner and group can be signalled.
      run_wt=$PWD run_lock= run_owner=
    }
    # The census prefers /proc/locks (the flock matched by device AND inode
    # -- inode numbers repeat across filesystems); lsof is the fallback and
    # lists any process with the file open. But /proc/locks names the pid
    # that TOOK the flock, not the ones holding it now: the lock lives on the
    # shared description, and our take_lock perl, its acquirer, exits at
    # once, so for a run's lock the named pid is dead -- or recycled by an
    # unrelated process -- while the supervisor, dune and whatever dune left
    # behind hold it through inherited fd 9. So a named pid counts only once
    # fd_holds_lock confirms it holds the lock now, and when none is left
    # the census sweeps every process's fdinfo rather than concluding
    # "nobody" (gh-ocannl-1107: trusting any named pid returned the dead
    # perl, and `stop` reaped none of the live holders). A confirmed named
    # pid is returned alone, missing whatever inherited from it: only an
    # acquirer still running can be one, and test-run's never is.
    lock_holder_pids() {
      local out p live=
      out=$(perl -e '
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
      ' "$run_lock" 2>/dev/null)
      for p in $out; do
        fd_holds_lock "$p" && live="$live$p
"
      done
      if [ -n "$live" ]; then
        printf '%s' "$live"
        return 0
      fi
      if [ -d /proc ]; then
        # /proc/locks named no live holder, yet the lock may be held -- sweep
        # every process's fdinfo instead of depending on lsof, which need
        # not be installed.
        for pd in /proc/[0-9]*; do
          p=${pd#/proc/}
          fd_holds_lock "$p" && printf '%s\n' "$p"
        done
        return 0
      fi
      # macOS: lsof lists OPENERS -- the accepted approximation, there being
      # no portable flock-holder API.
      lsof -t -- "$run_lock" 2>/dev/null
    }
    fd_holds_lock() { # Linux: does <pid> hold a FLOCK on the lock file NOW?
      perl -e '
        my ($pid, $target) = @ARGV;
        my @st = stat($target) or exit 1;
        opendir(my $dh, "/proc/$pid/fd") or exit 1;
        for my $fd (readdir $dh) {
          next if $fd =~ /^\./;
          my @fs = stat("/proc/$pid/fd/$fd") or next;
          next unless $fs[0] == $st[0] && $fs[1] == $st[1];
          open(my $fi, "<", "/proc/$pid/fdinfo/$fd") or next;
          while (<$fi>) { exit 0 if /^lock:.*FLOCK/ }
        }
        exit 1
      ' "$1" "$run_lock" 2>/dev/null
    }
    holds_lock_now() { # revalidated at SIGNAL time, not census time
      if [ -d /proc ]; then
        fd_holds_lock "$1"
      else
        lsof -t -- "$run_lock" 2>/dev/null | grep -qx "$1"
      fi
    }
    reap_leftovers() { # <signal>
      # Ownership rechecked at entry (and per census kill below): if the
      # lock changed hands since the caller's gate, nothing here may fire.
      lock_still_owned "$run_dir" || return 0
      if group_verified "$run_dir"; then
        kill "-$1" -- "-$(cat "$run_dir/pgid")" 2>/dev/null
      fi
      for p in $(lock_holder_pids); do
        case $p in '' | *[!0-9]* | 0) continue ;; esac
        # Re-verified per pid immediately before the signal: the census ran
        # as one batch, and a holder that exited meanwhile could have had
        # its pid recycled by an unrelated process. The ownership recheck
        # closes the handoff race too -- if a NEW launch acquired the lock
        # mid-reap, take_lock pointed the owner at the new run atomically
        # with acquisition, so the pointer can no longer name this run and
        # the new run's holders are never signaled.
        holds_lock_now "$p" || continue
        lock_still_owned "$run_dir" || return 0
        kill "-$1" "$p" 2>/dev/null
      done
    }
    reap_cycle() { # exits 0 iff this run's hold on the lock is gone
      lock_still_owned "$run_dir" || return 0
      reap_leftovers TERM
      sleep 2
      if lock_still_owned "$run_dir"; then
        reap_leftovers KILL
        sleep 1
      fi
      ! lock_still_owned "$run_dir"
    }
    report_reap() {
      if reap_cycle; then
        echo "$1, but its leftover processes held the worktree lock; reaped them"
      else
        echo "$1, but leftover processes STILL hold $run_wt's lock" \
             "(no lsof on this system, or unkillable holders); inspect with:" \
             "lsof $(printf %q "$run_lock")"
      fi
    }
    if [ -f "$run_dir/exit" ]; then
      lock_still_owned "$run_dir" && report_reap "finished"
      echo "already finished:"
      digest "$run_dir"
      exit 0
    fi
    # A stop landing in the launch window -- the run published, its
    # supervisor not yet on record -- must not read the live launcher as
    # leftovers and reap it: give the record the moment it needs, and let
    # the owner branch take the TERM. (A bounded wait, so a launch that
    # really died there still reaches the recovery below.)
    if [ ! -f "$run_dir/pid" ] && [ ! -f "$run_dir/exit" ] && lock_still_owned "$run_dir"; then
      for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
        [ -f "$run_dir/pid" ] && break
        sleep 0.1
      done
    fi
    if sup_alive "$run_dir"; then
      # The run's owner takes the TERM: the supervisor reaps dune and records
      # the cancellation; a repeat's coordinator owns the set-wide
      # cancellation bit -- killing only its current iteration's supervisor
      # would produce exit 143 and then let the outer loop launch every
      # remaining iteration while stop claimed success. For a run of the
      # previous version that coordinator is the recorded wrapper (see
      # sup_alive), and its `pid` is only the current iteration.
      # Name the run explicitly (%q-quoted): `last` may resolve to a
      # DIFFERENT run when this stop targeted an identifier from another
      # worktree's history.
      if [ "$(cat "$run_dir/mode" 2>/dev/null)" = repeat ] && legacy_owner_alive "$run_dir"; then
        kill -TERM "$(cat "$run_dir/wpid")" 2>/dev/null
        echo "sent TERM to the repeat coordinator; confirm with: tools/test-run.sh wait $(printf %q "$run_dir")"
      elif proc_alive "$run_dir/pid" "$run_dir/ptoken"; then
        kill -TERM "$(cat "$run_dir/pid")" 2>/dev/null
        if [ "$(cat "$run_dir/mode" 2>/dev/null)" = repeat ]; then
          echo "sent TERM to the repeat coordinator; confirm with: tools/test-run.sh wait $(printf %q "$run_dir")"
        else
          echo "sent TERM; confirm with: tools/test-run.sh wait $(printf %q "$run_dir")"
        fi
      else
        # The previous version's wrapper, publishing its verdict after the
        # supervisor exited: it ignores TERM by design, and there is nothing
        # left to cancel.
        echo "run is finishing (verdict publication in flight); confirm with:" \
             "tools/test-run.sh wait $(printf %q "$run_dir")"
      fi
    elif group_verified "$run_dir" &&
         pg=$(cat "$run_dir/pgid") && kill -0 -- "-$pg" 2>/dev/null; then
      # SIGKILL can remove the supervisor around a dune that survives in its
      # own recorded group -- still holding the worktree lock, beyond
      # its cap. Identity is the group LEADER's recorded start token (the
      # same mechanism as every other liveness check here); a recycled pgid
      # -- even one leading a process named dune -- fails the token and is
      # never signaled. A leaderless surviving group is refused too, the
      # conservative side: clean that up by hand. TERM gets a bounded grace,
      # then a revalidated KILL -- an orphan that ignores TERM has lost its
      # cap and would otherwise hold the worktree lock indefinitely.
      #
      # Reachability, not liveness, opens this branch, and every signal below
      # is sent on it alone: a census is a snapshot, so a child forked while it
      # was taken is missing from it, and letting that veto -- or downgrade --
      # the reap would leave the child holding the lock, or take away the TERM
      # it needed to shut down cleanly. group_alive decides only the GRACE and
      # the WORDING, which is where the phantom lived (gh-ocannl-742): a group
      # of unreaped corpses announced as a runaway dune ignoring TERM.
      kill -TERM -- "-$pg" 2>/dev/null
      # Asked AFTER the TERM, so a member the earlier census could have missed
      # is included: a grace has a point only where something can still act on
      # the signal, and corpses under a non-reaping init would otherwise cost
      # every stop two seconds.
      group_alive "$pg" && sleep 2
      if group_verified "$run_dir" && kill -0 -- "-$pg" 2>/dev/null; then
        # Something is still there. It may be running work or only unreaped
        # corpses; the KILL goes out either way, and one sentence covers both
        # because a second census can distinguish them only through a race.
        kill -KILL -- "-$pg" 2>/dev/null
        echo "orphaned process group $pg survived TERM (possibly only as" \
             "unreaped exited processes); escalated to KILL"
      else
        echo "sent TERM to the orphaned process group $pg; re-run stop to confirm"
      fi
    elif lock_still_owned "$run_dir"; then
      # Dead without a verdict, yet its leftovers still hold the worktree
      # lock (a setsid escapee outliving a killed supervisor) --
      # the same recovery as the finished branch, or later runs stay
      # refused forever.
      report_reap "run is dead without a verdict"
    else
      echo "nothing left to signal (supervisor gone; no identity-verified surviving group)"
    fi
    ;;
  list)
    found=0
    for d in "$RUNS"/2*/; do
      [ -d "$d" ] || continue
      found=1
      d=${d%/}
      if [ -f "$d/exit" ]; then state="exit $(cat "$d/exit")"
      elif sup_alive "$d"; then state=running
      else state=dead
      fi
      printf '%s  %-8s  dune %s\n' "$(basename "$d")" "$state" "$(cat "$d/cmd" 2>/dev/null)"
    done
    [ "$found" = 1 ] || echo "no recorded runs in $RUNS"
    ;;
  *) die "unknown subcommand: $sub (run|start|plan|repeat|fresh|bisect|status|wait|stop|list|idle|paths|lock-status)" ;;
esac
