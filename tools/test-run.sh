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
#     (gh-ocannl-1211; see test_cap_watch and probe_perl).
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
#   tools/test-run.sh run   [--cap N] [--test-cap N] [DUNE ARGS...]   # foreground; digest; dune's status (2: refused)
#   tools/test-run.sh start [--cap N] [--test-cap N] [DUNE ARGS...]   # detached; survives the session
#   tools/test-run.sh plan  [--cap N] [--test-cap N] [DUNE ARGS...]   # what `run` would inject and take; runs nothing
#   tools/test-run.sh repeat [--cap N] [--alone] N [DUNE ARGS...]
#                                                      # compare N isolated runs
#   tools/test-run.sh status [RUN|last]                # one-shot, never blocks
#   tools/test-run.sh wait   [RUN|last] [--timeout N]  # bounded; exits with the run's status
#   tools/test-run.sh stop   [RUN|last]                # TERM the run's process group
#   tools/test-run.sh list                             # recent runs and their states
#   tools/test-run.sh paths FIELD [RUN|last]          # one absolute path (see below)
#   tools/test-run.sh lock-status [RUN|last]          # idle/held; exits 0/3 (2 error)
#   tools/test-run.sh idle                             # 0 idle, 3 locked, 2 unreadable
#                                                     # snapshot, not a reservation
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
# is not red. Any red dune iteration keeps a nonzero dune status.
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
# the run's or a test's, 69 = a `run`/`start` GPU batch's device probe found
# the device wedged and dune never started, 143/130 = cancelled, 137 = SIGKILLed, 124 = `wait` itself
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
# `--cap 0` disables it (then supply your own bound). `--test-cap N` bounds each
# process dune starts (each test, in a test batch) and ENDS the run when one
# outlives it, as the run's own cap would; N (or $OCANNL_TOOL_PER_TEST_CAP)
# applies to any batch, 0 lifts it, and by default only a GPU batch not
# reaching `@slow` gets 1500s (see plan_test_cap, test_cap_watch). A GPU batch
# also first asks each of its GPU backends for a fresh device query, and
# refuses to start dune when one does not answer (DEVICE UNHEALTHY, exit 69;
# see probe_perl). Neither bound applies to `repeat`, which runs as given.
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
  # dune has no `--cap`, `--test-cap` or `--alone`, so any of them among the arguments
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
      --cap | --cap=* | --alone | --alone=* | --test-cap | --test-cap=*)
        case $sub in
          repeat) form='repeat [--cap N] [--alone] N [DUNE ARGS...]' ;;
          *) form="$sub [--cap N] [--test-cap N] [DUNE ARGS...]" ;;
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
  probe_cap=${OCANNL_TOOL_DEVICE_PROBE_CAP:-60}
  case $probe_cap in '' | 0 | *[!0-9]*) probe_cap=60 ;; esac
  probe_announce="a fresh device query for $probe_backends before dune starts${slot_fw:+, inside the slot}; one
  that does not answer within ${probe_cap}s refuses the batch as DEVICE UNHEALTHY (gh-ocannl-1211)."
}

# The per-test cap's value (gh-ocannl-1211; test_cap_watch enforces it). An
# explicit one -- `--test-cap N` or OCANNL_TOOL_PER_TEST_CAP -- applies to any
# batch. The default applies only where the failure it exists for can happen
# and no legitimate test is known to come near it: a batch `run` resolves to
# hold a GPU backend, whose targets do not reach the `slow` aliases. 1500s
# clears every runtest/train action in the fleet's dune traces (the longest,
# 1021s, under correctness-slot load), but `slow-cifar_conv` runs ~300s solo
# and loads of 5-15x were measured, plus its dataset download -- so `@slow`
# and CPU batches keep only the run's cap unless asked. Re-derive it from a
# box's traces with tools/action-durations.sh, or from the `slowest` record
# every run directory keeps (see explicit_trace_file).
TEST_CAP_DEFAULT=1500
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
  for arg; do
    case $arg in
      --) break ;;
      @slow | @@slow | @*/slow | @@*/slow | @slow-* | @@slow-* | @*/slow-* | @@*/slow-*)
        test_cap_why="no default: the targets reach the slow tests ($arg)"
        return 0 ;;
    esac
  done
  test_cap=$TEST_CAP_DEFAULT test_cap_why="the default for a GPU batch"
}

select_dune() {
  # On Windows the environment rewrite is required even if dune is already on
  # PATH: opam's native output otherwise leaves an MSYS shell half-configured.
  case ${OSTYPE:-} in
    msys* | cygwin*)
      . tools/opam-env.sh ||
        die "tools/opam-env.sh failed; refusing an unrewritten Windows toolchain"
      ;;
  esac
  command -v dune >/dev/null 2>&1 || . tools/opam-env.sh
  command -v dune >/dev/null 2>&1 || die "dune not found (opam environment not set up?)"
  # Preserve dune's status while filtering only the known Windows linker noise.
  case ${OSTYPE:-} in msys* | cygwin*) DUNE=tools/dune-quiet.sh ;; *) DUNE=dune ;; esac
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
case ${1:-} in paths | lock-status) CDPATH= ;; esac
cd -P "$(dirname "$0")/.." || die "cannot cd to repo root"

[ -r scripts/process-group.sh ] || die "cannot read scripts/process-group.sh"
# shellcheck source=../scripts/process-group.sh
. scripts/process-group.sh

# The per-box dune width cap, shared with tools/sweep.sh so the two cannot
# drift. See width_cap below for what this script does with it.
[ -r tools/box-jobs.sh ] || die "cannot read tools/box-jobs.sh"
# shellcheck source=box-jobs.sh
. tools/box-jobs.sh
# Which backends a batch holds -- the resolution the width and the fleet slot
# both read (plan_batch).
[ -r tools/batch-backends.sh ] || die "cannot read tools/batch-backends.sh"
# shellcheck source=batch-backends.sh
. tools/batch-backends.sh
# Where this host's fleet-worker.sh might be (plan_slot), shared with tools/sweep.sh.
[ -r tools/fleet-worker-candidates.sh ] || die "cannot read tools/fleet-worker-candidates.sh"
# shellcheck source=fleet-worker-candidates.sh
. tools/fleet-worker-candidates.sh
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
  paths | lock-status | _resolve | _probe) ;;
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
  my $rd = $ENV{OCANNL_TOOL_TESTRUN_RD};
  my $own = $ENV{OCANNL_TOOL_TESTRUN_OWN};
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
  # worktree lock is fd 9) and with every check of the trace made there, so nothing here can block on
  # the file system: it is polled for its bound (OCANNL_TOOL_SLOWEST_CAP,
  # 30 seconds), then KILLed and polled again, five seconds, and abandoned if
  # even that does not reap it -- the verdict publishes and the lock clears
  # regardless. OCANNL_TOOL_SLOWEST_READER and OCANNL_TOOL_SLOWEST_KILL (the
  # escalation signal; 0 sends none) are seams for tools/test-test-run.sh,
  # which stands a reader that KILL cannot reap in with them. The trace is
  # deleted either way.
  my $slowest = sub {
    my $trace = "$own/trace.csexp";
    my $bound = $ENV{OCANNL_TOOL_SLOWEST_CAP};
    $bound = 30 unless defined $bound && $bound =~ /^[1-9][0-9]{0,4}$/;
    my $esc = $ENV{OCANNL_TOOL_SLOWEST_KILL};
    $esc = "KILL" unless defined $esc && $esc =~ /^(KILL|0)$/;
    my $reader = $ENV{OCANNL_TOOL_SLOWEST_READER};
    $reader = "tools/action-durations.sh" unless defined $reader && length $reader;
    my $c = fork();
    return unless defined $c;
    if (!$c) {
      $SIG{$_} = "DEFAULT" for qw(ALRM INT TERM HUP);
      POSIX::close($_) for 3 .. 255;
      eval { setpgrp(0, 0) };
      POSIX::_exit(1) unless -f $trace;
      open(STDIN, "<", "/dev/null");
      open(STDERR, ">", "/dev/null");
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
      kill($esc, -$c) or kill($esc, $c);
      for (1 .. 50) {
        last if waitpid($c, POSIX::WNOHANG()) != 0;
        select undef, undef, undef, 0.1;
      }
    }
    if (defined $st && $st == 0) {
      rename("$own/slowest.tmp", "$own/slowest") or unlink "$own/slowest.tmp";
    } else {
      unlink "$own/slowest.tmp";
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
  $SIG{HUP} = $ENV{OCANNL_TOOL_TESTRUN_BG} ? "IGNORE" : sub { $reap->(129) };
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

# The per-test cap (gh-ocannl-1211): the run's cap, applied to each process
# dune starts. A GPU fault on tuf (gfxhub/CPC page fault, then the test's
# HIP_ERROR_ILLEGAL_ADDRESS) left the device wedged, and every test dune
# started after it spun a core in the runtime's device init with no output --
# eight at once, for 32 minutes, until a person cancelled the run; the run's
# own cap had another half hour to go. A device that wedges mid-batch hangs
# every LATER test the same way, so a test that outlives this cap ends the
# whole run rather than just itself: killing one hung test would only let
# dune start the next.
#
# Run by `_resolve` beside dune, in the run's process group with the worktree
# lock closed, it lists the processes every few seconds (`ps`, the portable
# reader: /proc on Linux, the BSD ps on macOS) and takes as a test every
# child of a `dune` descending from `_resolve` -- dune starts each action as
# its own process-group leader, so the run's group kill reaches dune but not
# them, while dune's own TERM handling kills every action it has running
# (measured on dune 3.24, a TERM-ignoring action included). Children of dune
# rather than "every group leader in the run", because the fleet slot holds
# its sleep inhibitor in a group of its own for the whole run. Past the cap it
# records the test in `test-cap`, says so in the log, and sends the
# supervisor the SIGALRM the run's own cap sends: the same reap, the same 142,
# with the digest naming the test from the record. It never signals once its
# parent is no longer `_resolve` (dune exited and `_resolve` moved on, or the
# run is being reaped), so it cannot outlive the supervisor and signal a
# recycled pid. Where `ps` cannot list processes (Git Bash) it records that in
# `test-cap-off`, for the digest, and exits, leaving the run's cap as the only
# bound -- never in the log, whose first line after the prelude must stay
# dune's own (dune_refusal).
test_cap_watch='
  my ($cap, $resolver, $sup, $rd, $kind) = @ARGV;
  $| = 1;
  $ENV{LC_ALL} = "C";
  my $poll = $cap < 60 ? 1 : 5;
  # ps elapsed time: [[dd-]hh:]mm:ss
  my $seconds = sub {
    my $t = shift;
    my $d = ($t =~ s/^(\d+)-//) ? $1 : 0;
    my $s = 0;
    $s = $s * 60 + $_ for split /:/, $t;
    $d * 86400 + $s;
  };
  while (getppid() == $resolver) {
    my @ps = qx{ps -A -o pid= -o ppid= -o etime= -o comm= 2>/dev/null};
    my (%ppid, %etime, %comm, %kids);
    for (@ps) {
      next unless /^\s*(\d+)\s+(\d+)\s+([\d:-]+)\s+(.*?)\s*$/;
      ($ppid{$1}, $etime{$1}, $comm{$1}) = ($2, $3, $4);
      push @{ $kids{$2} }, $1;
    }
    unless (exists $ppid{$$}) {
      if (open my $fh, ">", "$rd/test-cap-off") { print $fh "ps cannot list this host\x27s processes\n"; close $fh }
      exit 0;
    }
    my (%dune, @queue);
    @queue = ($resolver);
    while (@queue) {
      for my $k (@{ $kids{ shift @queue } || [] }) {
        push @queue, $k;
        # By name: a `dune` on PATH that is a shebang WRAPPER would be named
        # dune too (Linux names a script by its basename), and the real dune
        # below it would then be timed as a test.
        $dune{$k} = 1 if $comm{$k} =~ m{(?:^|/)dune(?:\.exe)?$};
      }
    }
    for my $p (sort { $a <=> $b } keys %ppid) {
      next unless $dune{ $ppid{$p} };
      my $e = $seconds->($etime{$p});
      next if $e < $cap;
      my $args = qx{ps -o args= -p $p 2>/dev/null};
      chomp $args;
      $args = $comm{$p} if $args eq "";
      my $dir = readlink("/proc/$p/cwd");
      $dir = "" unless defined $dir;
      if (open my $fh, ">", "$rd/test-cap.tmp") {
        print $fh "cap $cap\nelapsed $e\npid $p\nkind $kind\ncommand $args\n", ($dir ne "" ? "dir $dir\n" : "");
        close $fh;
        rename "$rd/test-cap.tmp", "$rd/test-cap";
      }
      print "test-run: per-test cap: pid $p has run ${e}s, past the ${cap}s cap: $args",
        ($dir ne "" ? " (in $dir)" : ""), "\n",
        "test-run: ending the run, not judged: a test that outlives the per-test cap is taken as hung",
        ($kind eq "gpu" ? ", and on a GPU batch a wedged device hangs every test after it" : ""),
        " (gh-ocannl-1211). Legitimately long? Rerun with --test-cap 0, or a larger cap.\n";
      kill "ALRM", $sup if getppid() == $resolver;
      exit 0;
    }
    sleep $poll;
  }
'

# A GPU batch's device probe (gh-ocannl-1211): one fresh device query, bounded.
# A wedged device does not fail a query, it never answers one -- the recovery
# trip after the tuf fault found `device_props` spinning at 100% CPU before
# printing anything -- so the answer read here is whether the query FINISHED
# within the bound, and a query that fails fast (no such device on this box)
# is recorded and passed over: its tests fail fast too, rather than spin.
# Prints `exit <code> <seconds>`, or `hung <seconds>` for a query KILLed at
# the bound; the query's own output goes to the file given. Polled rather
# than waited for, so a query stuck where even KILL cannot reach it (in the
# driver, uninterruptibly) still gets its verdict on time; it stays in the
# run's process group, which the supervisor reaps.
probe_perl='
  use POSIX ();
  use Time::HiRes ();
  my ($bound, $out, @cmd) = @ARGV;
  my $t0 = Time::HiRes::time();
  my $pid = fork();
  defined $pid or do { print "error fork: $!\n"; exit 0 };
  if (!$pid) {
    open STDIN, "<", "/dev/null";
    open STDOUT, ">", $out or exit 126;
    open STDERR, ">&", \*STDOUT;
    exec @cmd or exit 127;
  }
  my $st;
  while (1) {
    if (waitpid($pid, POSIX::WNOHANG()) == $pid) { $st = $?; last }
    last if Time::HiRes::time() - $t0 >= $bound;
    Time::HiRes::sleep(0.05);
  }
  my $took = sprintf "%.1f", Time::HiRes::time() - $t0;
  if (defined $st) {
    print "exit ", (($st & 127) ? 128 + ($st & 127) : $st >> 8), " $took\n";
  } else {
    kill "KILL", $pid;
    for (1 .. 50) { last if waitpid($pid, POSIX::WNOHANG()) != 0; Time::HiRes::sleep(0.1) }
    print "hung $took\n";
  }
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
# not hold -- is neither read nor touched, and a run whose dune never started
# (a refused or cancelled slot wait) has no trace to misread. The supervisor
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
  [ ! -s "$dir/test-cap-off" ] ||
    echo "per-test cap: off for this run ($(cat "$dir/test-cap-off")); only the run's cap bounded it"
  # The batch's slowest processes (see explicit_trace_file), under a verdict
  # dune's own run produced: a refused slot, invocation or device probe, or a
  # setup error, ran no batch, so a record beside one is not this verdict's.
  # `open` marks a process the run's end cut short, its seconds a lower bound.
  case $digest_rc:$verdict in
    2:* | *:SLOT* | *:DEVICE* | *:ERROR*) ;;
    *)
      if [ -s "$dir/slowest" ]; then
        echo "slowest actions (this run's dune trace, by tools/action-durations.sh):"
        sed -n '1,7p' "$dir/slowest" | sed 's/^/  /'
      fi ;;
  esac
  if [ "$digest_rc" = 2 ]; then
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
  if [ -n "$promo_n" ] && [ "$promo_n" -gt 0 ]; then
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
[ -n "$sub" ] || die "usage: tools/test-run.sh run|start|plan|repeat|status|wait|stop|list|idle|paths|lock-status ... (see header)"
shift

case $sub in
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
    alone=0
    while [ $# -gt 0 ]; do
      case $1 in
        --cap) [ $# -ge 2 ] || die "--cap requires a value"; cap=$2; shift 2 ;;
        --alone) alone=1; shift ;;
        --) shift; break ;;
        *) break ;;
      esac
    done
    normalize_cap
    [ $# -gt 0 ] || die "repeat requires a count of at least 2"
    repeats=$1
    shift
    case $repeats in '' | *[!0-9]*) die "repeat count must be an integer of at least 2" ;; esac
    [ ${#repeats} -le 6 ] || die "repeat count too large (max 6 digits)"
    repeats=$(( 10#$repeats ))
    [ "$repeats" -ge 2 ] || die "repeat count must be at least 2"
    reject_misplaced_options "$@"
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

    repeat_build=$run_dir/build
    i=1
    while [ "$i" -le "$repeats" ] && [ -z "$repeat_cancelled" ]; do
      iter=$run_dir/iteration-$i
      mkdir "$iter" || die "cannot create $iter"
      # Dune's first `--` ends Dune option parsing (`dune exec PROG -- ARGS`).
      # Isolation options belong immediately before it; appending them would
      # silently hand them to PROG and run Dune in its ambient build context.
      repeat_cmd=("$DUNE")
      repeat_separator=0
      for repeat_arg in "$@"; do
        if [ "$repeat_separator" = 0 ] && [ "$repeat_arg" = -- ]; then
          repeat_cmd+=(--force --cache=disabled --build-dir="$repeat_build")
          [ "$alone" = 0 ] || repeat_cmd+=(-j 1)
          repeat_separator=1
        fi
        repeat_cmd+=("$repeat_arg")
      done
      if [ "$repeat_separator" = 0 ]; then
        repeat_cmd+=(--force --cache=disabled --build-dir="$repeat_build")
        [ "$alone" = 0 ] || repeat_cmd+=(-j 1)
      fi
      {
        printf '%q clean --build-dir=%q && ' "$DUNE" "$repeat_build"
        printf '%q ' "${repeat_cmd[@]}"
        echo
      } >"$iter/cmd" ||
        die "cannot record iteration $i command"
      printf 'repeat: iteration %s/%s -- dune%s\n' "$i" "$repeats" \
        "$([ "$alone" = 1 ] && printf ' (alone, -j 1)' || :)"
      # A stop can land after the loop condition. Refuse a launch already
      # known to be cancelled, then recheck immediately after starting the
      # supervisor to close the unavoidable signal-between-commands window.
      [ -z "$repeat_cancelled" ] || break
      # A process-group reap cannot see a Dune action that calls setsid. Give
      # every descendant an inherited FIFO writer as a second containment
      # witness: after the supervisor exits, EOF proves no session-escaped
      # descendant remains able to mutate this iteration's shared build tree.
      # The read/write anchor makes the two one-way opens nonblocking; it is
      # closed before launch and never reaches the child.
      descendant_fifo=$iter/descendants.fifo
      mkfifo "$descendant_fifo" || die "cannot create descendant witness for iteration $i"
      exec 7<>"$descendant_fifo" || die "cannot anchor descendant witness for iteration $i"
      exec 8<"$descendant_fifo" || die "cannot read descendant witness for iteration $i"
      exec 6>"$descendant_fifo" || die "cannot write descendant witness for iteration $i"
      exec 7>&-
      OCANNL_TOOL_TESTRUN_BG=0 OCANNL_TOOL_TESTRUN_RD=$iter \
        perl -e "$supervisor_perl" -- "$cap" /bin/bash -c \
        'dune=$1; build=$2; shift 2; "$dune" clean --build-dir="$build" || exit 126; exec "$dune" "$@"' \
        -- "$DUNE" "$repeat_build" "${repeat_cmd[@]:1}" \
        >"$iter/stdout" 2>"$iter/stderr" &
      repeat_sup=$!
      exec 6>&-
      [ -z "$repeat_cancelled" ] || kill "-$repeat_cancelled" "$repeat_sup" 2>/dev/null
      { printf '%s\n' "$repeat_sup" >"$iter/pid" &&
        ps_token "$repeat_sup" >"$iter/ptoken"; } ||
        kill -TERM "$repeat_sup" 2>/dev/null
      while :; do
        wait "$repeat_sup"
        iter_rc=$?
        proc_alive "$iter/pid" "$iter/ptoken" || break
      done
      # The numeric pid is no longer ours once wait/proc_alive prove the
      # supervisor gone. Clear it before the slower orphan-group reap so a
      # cancellation cannot signal a recycled, unrelated process.
      repeat_sup=
      # SIGKILL can remove the supervisor without reaching the Dune process
      # group it owned. Reap that identity-verified group before recording the
      # iteration, starting another one, or deleting their shared build tree.
      reap_repeat_group "$iter"
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
        die "iteration $i left a session-escaped descendant; refusing to reuse $repeat_build"
      rm -f "$descendant_fifo"
      printf '%s\n' "$iter_rc" >"$iter/exit" || die "cannot record iteration $i verdict"
      [ "$first_nonzero" != 0 ] || [ "$iter_rc" = 0 ] || first_nonzero=$iter_rc
      {
        printf '=== repeat iteration %s/%s stdout ===\n' "$i" "$repeats"
        cat "$iter/stdout"
        printf '=== repeat iteration %s/%s stderr ===\n' "$i" "$repeats"
        cat "$iter/stderr"
        printf '=== repeat iteration %s/%s exit %s ===\n' "$i" "$repeats" "$iter_rc"
      } >>"$run_dir/log" || die "cannot append iteration $i to $run_dir/log"
      printf 'repeat: iteration %s/%s exit %s; stdout=%s stderr=%s\n' \
        "$i" "$repeats" "$iter_rc" "$iter/stdout" "$iter/stderr"
      completed=$i
      # An invocation dune's parser refused ran nothing, so there is nothing
      # whose stability the remaining iterations could measure: stop after the
      # first, and let the verdict below say so rather than reporting N
      # identical refusals as IDENTICAL. Only the first iteration can decide
      # this -- every iteration runs the same argv.
      if [ "$i" = 1 ] && [ "$iter_rc" = 1 ] && dune_refusal "$iter/stderr" >/dev/null; then
        repeat_refused=1
        break
      fi
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
    i=1
    while [ "$i" -lt "$completed" ]; do
      j=$(( i + 1 ))
      while [ "$j" -le "$completed" ]; do
        left=$run_dir/iteration-$i
        right=$run_dir/iteration-$j
        pair=$i-$j
        stdout_same=0; stderr_same=0; exit_same=0
        cmp -s "$left/stdout" "$right/stdout" && stdout_same=1
        cmp -s "$left/stderr" "$right/stderr" && stderr_same=1
        cmp -s "$left/exit" "$right/exit" && exit_same=1
        if [ "$stdout_same" = 1 ] && [ "$stderr_same" = 1 ] && [ "$exit_same" = 1 ]; then
          pair_result=identical
          identical=$(( identical + 1 ))
        elif [ "$stdout_same" = 1 ] && [ "$exit_same" = 1 ]; then
          pair_result=stderr-only
          stderr_only=$(( stderr_only + 1 ))
          diff -u "$left/stderr" "$right/stderr" >"$run_dir/diffs/$pair.stderr" ||
            [ $? -eq 1 ] || die "cannot diff stderr for iterations $i and $j"
        else
          pair_result=differing
          differing=$(( differing + 1 ))
          if [ "$stdout_same" = 0 ]; then
            diff -u "$left/stdout" "$right/stdout" >"$run_dir/diffs/$pair.stdout" ||
              [ $? -eq 1 ] || die "cannot diff stdout for iterations $i and $j"
          fi
          if [ "$stderr_same" = 0 ]; then
            diff -u "$left/stderr" "$right/stderr" >"$run_dir/diffs/$pair.stderr" ||
              [ $? -eq 1 ] || die "cannot diff stderr for iterations $i and $j"
          fi
          if [ "$exit_same" = 0 ]; then
            diff -u "$left/exit" "$right/exit" >"$run_dir/diffs/$pair.exit" ||
              [ $? -eq 1 ] || die "cannot diff exits for iterations $i and $j"
          fi
        fi
        printf 'repeat: pair %s/%s: %s\n' "$i" "$j" "$pair_result" |
          tee -a "$run_dir/log"
        j=$(( j + 1 ))
      done
      i=$(( i + 1 ))
    done

    if [ -n "$repeat_cancelled" ]; then
      repeat_result="CANCELLED -- completed $completed of $repeats iterations"
      repeat_reported_cancelled=1
      final_rc=$first_nonzero
      [ "$final_rc" != 0 ] || final_rc=143
    elif [ -n "$repeat_refused" ]; then
      repeat_result="INVOCATION REFUSED -- dune rejected the arguments; nothing ran, so nothing was repeated"
      final_rc=$first_nonzero
    elif [ "$differing" -gt 0 ]; then
      repeat_result="DIFFERING -- stdout or exit status moved across $differing pair(s)"
      final_rc=$first_nonzero
      [ "$final_rc" != 0 ] || final_rc=1
    elif [ "$stderr_only" -gt 0 ]; then
      repeat_result="STDERR-ONLY -- stdout and exit status were stable; stderr moved across $stderr_only pair(s)"
      final_rc=$first_nonzero
    else
      repeat_result="IDENTICAL -- stdout, stderr and exit status matched across all $identical pair(s)"
      final_rc=$first_nonzero
    fi
    printf 'repeat result: %s\n' "$repeat_result" | tee -a "$run_dir/log"
    if [ -n "$repeat_refused" ]; then
      { echo "dune said:"
        dune_refusal "$run_dir/iteration-1/stderr" | sed 's/^/  /'
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
    [ $# -ge 4 ] && [ -n "${OCANNL_TOOL_TESTRUN_OWN:-}" ] ||
      die "_resolve is the supervisor's first phase, not a command"
    mode=$1 cap=$2 test_cap=$3 DUNE=$4
    shift 4
    run_dir=$OCANNL_TOOL_TESTRUN_OWN
    plan_slot
    if [ "$mode" = plan ]; then
      batch_resolve "$DUNE" "$run_dir/log" "$@"
    else
      plan_batch "$run_dir/log" "$@"
    fi
    clamp_slot_wait
    plan_width_cap "$@"
    plan_slot_kind
    plan_device_probe
    plan_test_cap "$test_cap" "$@"
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
    if [ -n "$width_cap" ]; then
      { { printf '%q ' "$@"; echo; } >"$run_dir/cmd.tmp" &&
        mv -f "$run_dir/cmd.tmp" "$run_dir/cmd"; } ||
        { echo "test-run: cannot record the capped command in $run_dir"; exit 126; }
    fi
    # Into the run's log (this process's stdout) and, through `resolved`, onto
    # the launcher's stderr: the cap is in the artifact triage reads rather
    # than only in the launching terminal's scrollback.
    : >"$run_dir/resolved.tmp" || { echo "test-run: cannot write in $run_dir"; exit 126; }
    for line in "$width_announce" "$slot_announce" "$probe_announce"; do
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
      set -- "${BASH:-bash}" "$PWD/tools/test-run.sh" _probe "$DUNE" "$probe_prog" "$probe_cap" \
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
    exit "$rc"
    ;;
  _probe)
    # A probed GPU batch's first act inside its slot, and not a command for
    # callers (gh-ocannl-1211; plan_device_probe decides, probe_perl reads):
    # one bounded device query per GPU backend, then dune in this process's
    # place. A program of `-` is bin/device_props, built here, under the slot,
    # with its build output in `device-probe-build.log`. Each answer goes to
    # `device-probe` in the run directory, for the digest, and to the log only
    # when it refuses -- the log's first line after the prelude must stay
    # dune's (dune_refusal). Asked from test/config, so the query reads the
    # configuration the tests read, with the backend pinned on its command line.
    [ $# -ge 6 ] && [ -n "${OCANNL_TOOL_TESTRUN_OWN:-}" ] ||
      die "_probe is a run's own phase, not a command"
    run_dir=$OCANNL_TOOL_TESTRUN_OWN
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
    wedged=
    for b in $probe_backends; do
      answer=$(cd test/config &&
        perl -e "$probe_perl" "$probe_cap" "$run_dir/device-probe-$b.out" \
          "$probe_prog" "--ocannl_backend=$b" 9>&-)
      read -r how code took <<<"$answer"
      case $how in
        exit)
          if [ "$code" = 0 ]; then line="$b: answered in ${took}s"
          else line="$b: failed fast (exit $code in ${took}s; no such device here?), not counted against the device"
          fi ;;
        hung)
          line="$b: NO ANSWER within ${probe_cap}s; the query was killed"
          wedged="$wedged $b" ;;
        *) line="$b: the query could not be run ($answer)" ;;
      esac
      printf '%s\n' "$line" >>"$run_dir/device-probe"
    done
    if [ -n "$wedged" ]; then
      printf 'test-run: device probe: a fresh device query for%s did not answer within %ss: the\n' \
        "$wedged" "$probe_cap"
      echo "  device is taken as wedged, and dune was not started -- a faulted GPU makes every test spin"
      echo "  rather than fail (gh-ocannl-1211). Stop GPU work on this box and read its kernel journal"
      echo "  (journalctl -k: gfxhub/CPC page faults, ring timeouts, NVRM Xid); recovery may need a reboot."
      sed 's/^/  /' "$run_dir/device-probe"
      exit 69
    fi
    exec "$@"
    ;;
  run | start | plan)
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
    while [ $# -gt 0 ]; do
      case $1 in
        --cap) [ $# -ge 2 ] || die "--cap requires a value"; cap=$2; shift 2 ;;
        --test-cap) [ $# -ge 2 ] || die "--test-cap requires a value"; test_cap=$2; shift 2 ;;
        --) shift; break ;;
        *) break ;;
      esac
    done
    normalize_cap
    [ -z "$test_cap" ] || normalize_cap test_cap --test-cap
    reject_misplaced_options "$@"
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
    fwd_sig() { cancelled=$1; forward_cancel; }
    trap 'fwd_sig INT' INT
    trap 'fwd_sig TERM' TERM
    trap 'fwd_sig HUP' HUP
    new_run "$@"
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
      perl -e "$supervisor_perl" -- "$cap" \
        "${BASH:-bash}" "$PWD/tools/test-run.sh" _resolve "$sub" "$cap" "$test_cap" "$DUNE" "$@" \
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
      while [ ! -f "$run_dir/resolved" ] && [ ! -f "$run_dir/exit" ] && sup_alive "$run_dir"; do
        sleep 0.1
      done
      [ ! -f "$run_dir/resolved" ] || cat "$run_dir/resolved" >&2
    fi
    if [ "$sub" != start ]; then
      # Attached: wait for the supervisor -- its exit means the verdict file
      # is on disk. A trapped signal returns from `wait` early, hence the
      # retry loop; sup_alive rather than a bare kill -0, since after the
      # supervisor is reaped its pid can be recycled and probing the number
      # alone would spin this loop on an unrelated process.
      while sup_alive "$run_dir"; do wait "$sup" 2>/dev/null; done
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
  *) die "unknown subcommand: $sub (run|start|plan|repeat|status|wait|stop|list|idle)" ;;
esac
