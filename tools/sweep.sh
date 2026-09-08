#!/usr/bin/env bash
# Cross-machine test sweep: runs the suite once per (machine, backend) pair on
# whichever machines are reachable, and records a compact result row plus a
# failure fingerprint for each.
#
# This exists because GitHub CI covers exactly one backend: test/config's
# ocannl_config pins `backend=cc`, and the runners have no GPU. Metal, CUDA, HIP
# and multidev_cc have no automated coverage at all without this -- the last of
# those despite needing no hardware, since only OCANNL_BACKEND selects it.
#
# The GPU boxes are often asleep or powered off, and an unreachable machine is
# recorded as `skip` rather than an error. They are Wake-on-LAN armed, though,
# so a skip means nobody woke them -- not that the coverage was unavailable.
# Wake them before a run that is meant to cover cuda or hip, and kick WSL after
# waking (it starts on demand or at login, never at boot, so the `-wsl` hosts
# this sweep addresses lag the box being up). The caller is expected to notice
# when a backend has been skipped for too long.
#
# Deliberately does NOT exit non-zero on test failures: the point is to record
# every unit's outcome, including the ones after a failing one. Only a usable
# harness failure (no local repo, etc.) aborts.
#
# Usage:
#   tools/sweep.sh                     # cc + multidev_cc + metal locally, cuda/hip if up
#   tools/sweep.sh --slow              # also `dune build @slow`
#   tools/sweep.sh --force             # cold rebuild and re-execute every test alias
#   tools/sweep.sh --only metal        # one backend (repeatable)
#   tools/sweep.sh --target test/einsum  # narrower dune target, for smoke-testing
#   tools/sweep.sh --ref origin/master   # what to test (default: origin/master)
#   OCANNL_TOOL_SWEEP_LOCAL_BOX=m4-max tools/sweep.sh  # required stable local box ID

set -uo pipefail

# The knobs below live in the OCANNL_TOOL_ namespace, which the library reserves for
# names that address OCANNL without being configuration: every OCANNL executable this
# script launches walks the environment at startup and warns about an `OCANNL_...` name
# that is not a config key (gh-ocannl-629), and a warning nobody can act on is how a
# useful one gets ignored.
STATE=${OCANNL_TOOL_SWEEP_STATE:-$HOME/.ocannl-sweep}
HISTORY=$STATE/history.tsv
LOGS=$STATE/logs
UNIT_STATES=$STATE/unit-state
MAIN=${OCANNL_TOOL_SWEEP_REPO:-$HOME/ocannl-staging}
SWEEP_TOOLS=$(cd "$(dirname "$0")" && pwd)
AGGREGATE_SKIPS=$SWEEP_TOOLS/aggregate-skips.sh
REF=origin/master
TARGET=
SLOW=0
FORCE=0
ONLY=()
# Per-unit wall-clock cap, enforced by the perl supervisor below on both the
# local and the remote side: macOS has no timeout(1) at all, and where timeout(1)
# does exist it is not necessarily one whose -k reaches the process group.
CAP=${OCANNL_TOOL_SWEEP_CAP:-5400}
# The budget for the post-unit RTC context collection, deliberately separate from
# CAP: see collect_rtc_context for why sharing the unit's deadline would let a
# diagnostic overwrite the verdict it is explaining.
CONTEXT_CAP=${OCANNL_TOOL_SWEEP_CONTEXT_CAP:-300}
LOCAL_BOX=${OCANNL_TOOL_SWEEP_LOCAL_BOX:-}

while [ $# -gt 0 ]; do
  case $1 in
    --slow) SLOW=1 ;;
    --force) FORCE=1 ;;
    --only) ONLY+=("$2"); shift ;;
    --target) TARGET=$2; shift ;;
    --ref) REF=$2; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# Errexit is off so that a FAILING TEST does not abort the remaining units --
# that is the whole point. It must not extend to the harness: anything that
# would make an outcome unrecordable, or make a recorded outcome describe a tree
# that was not the one under test, has to be loud. A sweep that silently reports
# coverage it did not perform is worse than one that does not run.
die() { echo "sweep: $*" >&2; exit 2; }

# The local path cannot infer a stable fleet identity from a hostname or CPU
# model: either can name a different machine the same way, and the DIGESTS names
# are operator-owned aliases. Require the launcher to bind this physical host to
# its declared ID before any log or history row can be attributed to it.
case $LOCAL_BOX in
  "" | [!A-Za-z0-9]* | *[!A-Za-z0-9._-]*)
    die "set OCANNL_TOOL_SWEEP_LOCAL_BOX to this host's portable measurement-box ID"
    ;;
esac

# (measurement-box, backend, ssh-host) -- ssh-host empty means run locally. The
# box identifiers are the stable names declared by benchmarks/fixtures/DIGESTS.txt;
# that declaration is validated against this execution map below rather than
# restated as the aggregation matrix.
# The WSL sides of the GPU boxes, not the native-Windows ones: plain Linux
# toolchain, and Windows portability is covered by the scheduled CI job.
#
# multidev_cc is local and needs no GPU, but it is here for the same reason the
# GPU boxes are: nothing else runs it. It keeps its OWN debug-log golden --
# `test/operations/micrograd_demo_logging-multidev_cc-0-0.log.expected`, whose
# statement order the scheduler is free to differ on -- and `dune runtest`
# exercises that golden only when OCANNL_BACKEND says so, which the pinned
# `backend=cc` in test/config's ocannl_config and CI never do. gh-ocannl-700 is
# what that costs: an ordering change landed, the cc golden was re-promoted with
# it, and the multidev leg stayed red on master for six weeks with nothing to
# notice. A backend with its own goldens and no leg here is a silent regression
# channel whether or not it needs hardware.
UNITS=(
  "$LOCAL_BOX:cc:"
  "$LOCAL_BOX:multidev_cc:"
  "$LOCAL_BOX:metal:"
  "rog-nv:cuda:rog-nv-wsl"
  "minix:hip:minix-amd-wsl"
)

# Dune's job count for the TEST phase of a unit, empty for dune's default (one
# per core). The hip box's GPU is an iGPU reached through WSL2's dxg bridge --
# every allocation and every module load is a synchronous message over a Hyper-V
# VM bus ring -- and that ring overflows when the suite's test executables hold
# the device at once (dune's default on that box is 32 jobs, 16 of them GPU
# processes at the moment it was measured). The kernel logs
# `dxgvmb_send_sync_msg: vmbus_sendpacket failed: fffffff5` (-EAGAIN) and the
# HIP runtime surfaces the lost messages as HIP_ERROR_INVALID_DEVICE at
# hip_init, HIP_ERROR_NO_BINARY_FOR_GPU at module load, and failed stream
# creation: 60+ red tests in a suite whose logic never ran, with a device that
# passes every standalone probe (2026-09-05, minix after two host resumes kept
# the VM alive with a degraded bridge; a fresh VM tolerated the full width for
# eleven daily sweeps before that). Measured on that degraded bridge the same
# day: full width 67 red, `-j 4` still 27 red with 120 kernel-side refusals,
# `-j 2` zero of either over a forced full unit in 18.5 minutes, `-j 1` clean
# on every rerun. A single GPU serialises the kernels anyway, so the cap costs
# the test phase little; the compile phase stays uncapped (`test_cmd` runs
# `@check` first). Override for one run with OCANNL_TOOL_SWEEP_JOBS=<n>, which
# then applies to every unit.
unit_jobs() {
  if [ -n "${OCANNL_TOOL_SWEEP_JOBS:-}" ]; then
    printf '%s' "$OCANNL_TOOL_SWEEP_JOBS"
    return
  fi
  case "$1:$2" in
    minix:hip) printf 2 ;;
    *) ;;
  esac
}

# Successful forced full-suite units are the only logs from which absence of a
# skip announcement means execution. Incremental Dune runs may serve a cached
# test without replaying its stderr, and a red or interrupted unit may not have
# reached every test. Keep the qualifying evidence from THIS invocation rather
# than recovering it by timestamp from history (two invocations can begin in
# the same second in the integration harness).
SKIP_RUN_BACKENDS=()
SKIP_RUN_BOXES=()
SKIP_RUN_LOGS=()

contains() {
  local wanted=$1 item
  shift
  for item in "$@"; do [ "$item" = "$wanted" ] && return 0; done
  return 1
}

# The scope columns are load-bearing rather than bookkeeping. The consumer ages
# the most recent `pass` per backend, so without them a narrow smoke run --
# `--target test/einsum` while debugging this script, say -- refreshes that age
# exactly as a full suite would, certifying coverage that never ran. `slow` is
# separate for the same reason: a weekday sweep must not make Sunday's slow
# coverage look current.
header_line() {
  printf 'when\tmachine\tbackend\tref\toutcome\tseconds\ttarget\tslow\tlog\texecution\n'
}
old_header_line() { printf 'when\tmachine\tbackend\tref\toutcome\tseconds\ttarget\tslow\tlog\n'; }

mkdir -p "$LOGS" "$UNIT_STATES" || die "cannot create state directories under $STATE"

# Ask git rather than inspecting `.git`'s file type: in a linked worktree -- a
# layout this project uses constantly -- `.git` is a regular file, and a -d test
# rejects a repository every later `git -C` call would have handled fine.
git -C "$MAIN" rev-parse --git-dir >/dev/null 2>&1 ||
  die "no git repository at $MAIN (set OCANNL_TOOL_SWEEP_REPO)"

# An --only typo must not look like a clean sweep: without this, `--only cudaa`
# selects nothing, records nothing, and exits 0 having tested nothing.
#
# A pure-shell match over the unit table, deliberately not a `printf | grep -qx`
# pipeline: that spelling once refused `cc` -- the FIRST backend listed -- on
# the ubuntu CI leg while printing a list that visibly contained it
# (gh-ocannl-949). Under `pipefail` a pipeline's verdict belongs to every
# element, so the check answered for the writer's fate (a builtin in a pipeline
# runs in a forked subshell) as well as the reader's match; a match that forks
# nothing has no second party to answer for. The forked spelling was never
# reproduced -- tens of thousands of iterations on bash 3.2, and a writer this
# short cannot outlive the reader's single read (see the PR) -- so this is the
# removal of a question, not the fix of a diagnosed bug.
known_backend() {
  local unit
  for unit in "${UNITS[@]}"; do
    case $unit in *:"$1":*) return 0 ;; esac
  done
  return 1
}
known_backends=$(for u in "${UNITS[@]}"; do b=${u#*:}; printf '%s\n' "${b%%:*}"; done)
if [ ${#ONLY[@]} -gt 0 ]; then
  for b in "${ONLY[@]}"; do
    known_backend "$b" ||
      die "unknown backend '$b'; known: $(printf '%s' "$known_backends" | tr '\n' ' ')"
  done
fi

# One sweep at a time. Every local unit reuses a single fixed worktree, so an
# overlapping invocation -- a manual run started while the scheduled one is
# going, which is exactly how these collide -- would reset and clean that tree
# under a running dune. The earlier run's row would then describe a mixture of
# revisions: the precise failure this script exists to make impossible. A
# colliding sweep therefore refuses to start rather than queueing, so the miss is
# loud and the routine reports it, instead of two runs quietly corrupting each
# other. Per-invocation worktrees would also fix it, at the cost of the _build
# reuse that makes a daily cadence affordable.
#
# The lock is an flock on an inherited descriptor rather than a directory plus a
# pid file, so the KERNEL owns its lifetime. That removes the whole class of
# problems a hand-rolled lock has: nothing to reclaim after a crash, no window
# between creating the lock and publishing ownership, and no dependence on
# `kill -0`, which answers only "does SOME process hold this pid" -- a recycled
# pid belonging to an unrelated long-lived process would otherwise refuse every
# sweep for as long as that process lived.
#
# perl takes the lock and exits, but a lock belongs to the open file DESCRIPTION,
# which this shell still holds through fd 9; it is released when the last holder
# closes it. Children inherit fd 9 deliberately: if the sweep is killed outright
# mid-build, the orphaned dune keeps the lock, which is right -- the worktree
# really is still in use -- and the lock clears by itself when that orphan exits.
#
# It lives beside the WORKTREE, not under $STATE. OCANNL_TOOL_SWEEP_STATE is a
# supported override -- it is how this script gets tested against a throwaway
# history -- and keying the lock there would split the lock namespace while
# leaving the worktree shared, so a run with its own state directory would walk
# straight past the lock and reset the tree under the scheduled sweep. A lock
# belongs in the same namespace as the resource it protects.
LOCAL_WT=$HOME/ocannl-staging-worktrees/sweep
LOCK=$LOCAL_WT.lock
# The parent will not exist on a machine that has never run a sweep, and
# bootstrapping is the one path a developer machine never exercises: without this
# the open below fails, and the failure would surface as a confusing complaint
# about another sweep rather than as a missing directory.
mkdir -p "$(dirname "$LOCK")" || die "cannot create $(dirname "$LOCK")"
exec 9>"$LOCK" || die "cannot open $LOCK"
perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 1)' <&9 ||
  die "another sweep is running; refusing to share the worktree"

# History validation and migration belong under the same run-wide lock as the
# rows themselves. Otherwise two launches can both observe the old header: one
# replaces it while the other is reading, and the loser can publish an
# eleven-column body under the ten-column header before it is refused below.
if [ -f "$HISTORY" ]; then
  # The one supported migration is exact and append-only: preserving the first
  # nine columns keeps positional consumers working, while `unknown` refuses to
  # pretend that a historical pass proves execution we did not measure. Write a
  # sibling and rename it, so an interrupted migration leaves the old file whole.
  if [ "$(head -1 "$HISTORY")" = "$(old_header_line)" ]; then
    migrated=$HISTORY.migrate.$$
    {
      header_line
      tail -n +2 "$HISTORY" |
        awk -F '\t' 'BEGIN { OFS="\t" } { if ($5 == "pass") $5="legacy-pass"; print $0, "unknown" }'
    } >"$migrated" && mv "$migrated" "$HISTORY" ||
      die "cannot migrate $HISTORY to the execution-aware schema"
  fi
  # A file written by an older schema would be silently mis-columned by the
  # consumer, which is worse than refusing to append to it.
  [ "$(head -1 "$HISTORY")" = "$(header_line)" ] ||
    die "$HISTORY has a different schema; archive it and let this run start a new one"
else
  header_line >"$HISTORY" || die "cannot write $HISTORY"
fi
# Probe once up front, so a read-only or full state filesystem is reported here
# with a clear message rather than as a run whose rows silently went nowhere.
printf '' >>"$HISTORY" || die "cannot append to $HISTORY"

# Signals aimed at THIS pid rather than at the process group -- `kill $pid` from a
# supervisor, or the scheduler cancelling the task -- never reach the capped
# supervisor on their own: bash defers a trap while it waits on a foreground
# command, so the tests would run to completion (or to the 90-minute cap) holding
# the lock, and every sweep queued behind them would be refused. Group signals
# reach the supervisor directly and it forwards them itself; this is the other
# half. Each unit therefore runs asynchronously and the trap relays to it.
#
# TERM rather than the signal received: bash sets SIGINT to ignored for
# asynchronous children, so relaying INT could be a no-op, while the supervisor
# installs its own TERM handler unconditionally.
UNIT_PID=
relay() {
  if [ -n "$UNIT_PID" ]; then
    kill -TERM "$UNIT_PID" 2>/dev/null
    wait "$UNIT_PID" 2>/dev/null
  fi
  exit "$1"
}
trap 'relay 130' INT
trap 'relay 143' TERM

# Every long-running step goes through this, so that publishing UNIT_PID is not
# re-derived per call site: a site that forgets it silently loses cancellation
# and keeps holding the lock, which is how the preparation leg came to differ
# from the test legs.
run_capped() {
  local budget=$1 rc
  shift
  capped_bg "$budget" "$@" &
  UNIT_PID=$!
  wait "$UNIT_PID"
  rc=$?
  UNIT_PID=
  return "$rc"
}

stamp=$(date -u +%Y%m%dT%H%M%SZ)
# Resolve the ref to a commit ONCE, here, and pin every machine to that commit.
# Letting each box resolve `origin/master` itself would have them testing
# different commits whenever a merge lands mid-sweep, which is exactly the
# ambiguity a sweep exists to remove. It does mean --ref must name something
# reachable from origin/master, since that is all the remotes fetch.
#
# The fetch is checked: an unchecked one that fails transiently still leaves
# `origin/master` resolvable at whatever it pointed to last time, so the sweep
# would pin every machine to a stale commit and record green coverage for a tree
# nobody asked about -- the exact silent non-coverage this script exists to make
# impossible.
git -C "$MAIN" fetch -q origin master || die "cannot fetch origin master in $MAIN"
full_sha=$(git -C "$MAIN" rev-parse "$REF" 2>/dev/null) ||
  die "cannot resolve $REF in $MAIN"
run_sha=$(git -C "$MAIN" rev-parse --short "$full_sha")

# Read the environment matrix from the exact commit every unit is about to run.
# The Python CLI is the checked-in DIGESTS parser, so the sweep does not grow a
# second implementation of the header grammar. A pre-gh-ocannl-850 ref has no
# declaration and deliberately leaves environment coverage unaggregated while
# preserving backend aggregation for that historical run.
known_boxes=()
matrix_document=$(mktemp "${TMPDIR:-/tmp}/ocannl-measurement-boxes.XXXXXX") ||
  die "cannot create temporary measurement-box document"
if git -C "$MAIN" show "$full_sha:benchmarks/fixtures/DIGESTS.txt" >"$matrix_document" 2>/dev/null; then
  matrix_output=$(python3 "$SWEEP_TOOLS/../benchmarks/fixture_digest.py" \
    --list-declared-measurement-boxes --digests "$matrix_document")
  matrix_rc=$?
  rm -f "$matrix_document"
  [ "$matrix_rc" -eq 0 ] || die "cannot parse measurement boxes at $run_sha"
  while IFS= read -r box; do
    [ -n "$box" ] && known_boxes+=("$box")
  done <<<"$matrix_output"
else
  rm -f "$matrix_document"
fi

# The declaration owns which boxes constitute completeness; UNITS owns how to
# reach them. Relate the two so adding or renaming a declared box cannot leave a
# matrix member that no sweep unit can ever satisfy. Units absent from a historical
# target's declaration still contribute backend evidence but not environment evidence.
if [ ${#known_boxes[@]} -gt 0 ]; then
  scheduled_boxes=()
  for unit in "${UNITS[@]}"; do
    IFS=: read -r box _ <<<"$unit"
    contains "$box" "${scheduled_boxes[@]:-}" || scheduled_boxes+=("$box")
  done
  for box in "${known_boxes[@]}"; do
    contains "$box" "${scheduled_boxes[@]}" ||
      die "declared measurement box '$box' has no sweep unit"
  done
fi

wanted() {
  [ ${#ONLY[@]} -eq 0 ] && return 0
  local b
  for b in "${ONLY[@]}"; do [ "$b" = "$1" ] && return 0; done
  return 1
}

# The dune invocation, shared by the local and remote paths so the two cannot
# drift. Unpiped inside the shell that runs it: piping dune to anything reports
# the pipe's status, not dune's, and a promotion diff then reads as green.
#
# runtest and @slow are run UNCONDITIONALLY, their statuses combined, rather than
# chained with &&. Metal's regular operations suite is known-red, so chaining
# would mean the Sunday slow sweep never runs a single slow test on the one
# backend whose slow tests are least covered elsewhere -- while its history row
# reported only the already-known regular failure.
#
# The cap wraps the WHOLE unit -- capped() locally, one `timeout` remotely -- not
# each dune call. Per-call caps would let a --slow unit run for twice the budget
# the script advertises.
#
# A timeout in either suite still outranks a failure in the other when the two
# statuses combine. Otherwise, on a backend whose regular suite is known-red, a
# slow suite that hung would be filed forever as the already-known regular
# failure, and the distinct `timeout` verdict -- the one that says coverage was
# lost rather than merely red -- would never appear.
#
# Everything below is written with printf and single quotes so that `$?` and the
# arithmetic survive into the shell that finally runs them. The result is spliced
# into the remote string via command substitution, which bash does not rescan.
test_cmd() {
  local backend=$1 wt=$2 jobs=${3:-} force_arg= jobs_arg=
  [ "$FORCE" = 1 ] && force_arg=--force
  [ -n "$jobs" ] && jobs_arg="-j $jobs"
  # 127, not a generic failure: a worktree that is not there means nothing ran,
  # which the outcome mapping treats as non-coverage rather than a red suite.
  printf 'cd "%s" || exit 127; ' "$wt"
  # Dune's alias --force does not reliably invalidate ppx_expect inline tests.
  # A forced pass therefore starts from an empty build tree, under the worktree
  # lock already held on both local and remote paths. Failure to establish that
  # precondition is harness non-coverage, not a red suite.
  if [ "$FORCE" = 1 ]; then
    printf 'opam exec -- dune clean; clean_rc=$?; [ $clean_rc -eq 0 ] || exit 126; '
  fi
  # A full-suite unit also builds @train, the training-integration tier that
  # lives off the runtest path (test/training/dune says why); one dune call, so
  # the two suites share a build graph and rc1 stays one verdict. A narrow
  # --target run keeps its narrow meaning -- the `target` column already marks
  # it as refreshing no coverage.
  # A capped full-suite unit compiles at full width first: `@check` runs no
  # test action (it is the compile-only alias), so the cap -- which exists to
  # bound how many test executables hold the GPU at once -- does not also
  # serialise the build. Its status is deliberately dropped: a compile failure
  # reaches the verdict through the capped call, which rebuilds the same cone and
  # fails the same way. A --target run gets no prebuild: a workspace-wide
  # `@check` ahead of a narrow target would spend the unit's deadline compiling
  # code the target never reaches, and its cone is small enough to compile under
  # the cap (Codex P2 on PR #658).
  if [ -n "$jobs" ] && [ -z "$TARGET" ]; then
    printf 'OCANNL_BACKEND=%s opam exec -- dune build @check; ' "$backend"
  fi
  if [ -z "$TARGET" ]; then
    printf 'OCANNL_BACKEND=%s opam exec -- dune build %s%s @runtest @train; rc1=$?; ' \
      "$backend" "$jobs_arg${jobs_arg:+ }" "$force_arg"
  else
    printf 'OCANNL_BACKEND=%s opam exec -- dune runtest %s%s %s; rc1=$?; ' \
      "$backend" "$jobs_arg${jobs_arg:+ }" "$force_arg" "$TARGET"
  fi
  if [ "$SLOW" = 1 ]; then
    printf 'OCANNL_BACKEND=%s opam exec -- dune build %s%s @slow; rc2=$?; ' \
      "$backend" "$jobs_arg${jobs_arg:+ }" "$force_arg"
  else
    printf 'rc2=0; '
  fi
  printf 'for r in $rc1 $rc2; do case $r in 124|137|142) exit $r ;; esac; done; '
  printf 'exit $(( rc1 != 0 ? rc1 : rc2 ))'
}

# What a failing GPU unit is missing when someone reads its fingerprint the next
# morning: the flags the kernels were compiled under, and which toolkit did it.
# gh-ocannl-735 was found as a schedule-dependent numeric mismatch and took a long
# hunt to reach "the optimizer reassociated the recurrence"; the option vector was
# assembled inline in the backend, visible nowhere, and the ROCm version was
# whatever the box happened to have. Both belong beside the failure.
#
# Emitted as shell text, and run on the machine that OWNS the worktree -- the
# versions are the ones that just compiled the kernels, not the sweep host's --
# under the same lock and PATH, appending to `$log`, which `fingerprint` then
# carries into the digest.
#
# It runs as its OWN phase after the unit's row has been recorded, never inside
# `test_cmd`, and that separation is load-bearing rather than tidiness. Folded
# into the unit it would share the unit's `CAP`: a suite that fails a minute
# before the deadline would have this forced Dune build cross it, and the
# supervisor's 142 would then REPLACE the already-decided test status -- filing a
# red suite as `timeout`, the verdict that says coverage was lost, and losing the
# context it was collecting on the way out (Codex P2 on PR #510). Best-effort
# diagnosis must not be able to change the verdict it exists to explain, so it
# gets its own budget (OCANNL_TOOL_SWEEP_CONTEXT_CAP) outside the unit's, and its
# status is discarded.
#
# The option vector is not restated here. It is produced by the repository's own
# GPU-free option tests, which call the production builders in `Compiler_options`
# and print got/want vectors on stderr; a copy in shell would be a second source
# of truth that no test compares against the first. `--force` because the alias is
# certainly cached by the run that just failed.
#
# What those tests print is the option POLICY, and the block says so rather than
# letting a reader take it for the failing compile's own command line (Codex P2 on
# PR #510). The builders take two slots the tests fill with sentinels -- the
# discovered CUDA/ROCm include directory, and the source-dependent architecture
# target -- so `-I/cuda/include` and `--gpu-architecture=compute_80` in the output
# below are fixture values, and a fingerprint that presented them as this box's
# would misattribute. The three things worth having beside a red unit survive that
# honestly: which flags the builder ALWAYS emits, which it NEVER emits (the
# reassociation opt-in, the membership claim gh-ocannl-784 rests on), and whether
# the debug variant was in play. The per-slot inputs are printed as what they are,
# environment readings from the owning box.
#
# Where the effective vector genuinely exists the block points at it:
# `cuda_to_ptx` and `hip_to_code` re-raise their runtime compiler's exception
# with the vector appended to its message. The hardware-backed compile-failure
# probes pin that each backend writes the line this block promises (gh-ocannl-849).

# Which runtime compiler the BACKEND loads, not which one happens to be first on
# PATH (Codex P2 on PR #510). cudajit and hipjit reach nvrtc/hiprtc through a
# ctypes stub library that carries the soname as a NEEDED entry and no RPATH, so
# the file is chosen by the dynamic loader -- LD_LIBRARY_PATH, then the ldconfig
# cache -- and `nvcc --version` reports an unrelated toolkit that need not even
# be installed: the rog-nv box compiles CUDA kernels with no nvcc on PATH at all,
# which is exactly the misattribution this replaces. `ldd` on the stub answers
# with the file the backend will load, and its realpath carries the version in
# its name (libnvrtc.so.13.3.33 -- strictly more than nvrtcVersion's 13.3).
loaded_rtc_cmd() {
  local rtc=$1 tmpl
  # Written once with an `RTC` placeholder: the two arms differ only in the name,
  # and the CUDA one is the arm executed on the box this was verified on.
  tmpl='so=$(ldd "$(opam var lib 2>/dev/null)"/stublibs/dll*RTC*stubs.so 2>/dev/null'
  tmpl=$tmpl' | sed -n "s/.*=> *\([^ ]*libRTC[^ ]*\).*/\1/p" | head -1); '
  tmpl=$tmpl'if [ -n "$so" ]; then echo "loaded RTC: $(readlink -f "$so")"; '
  tmpl=$tmpl'else echo "loaded RTC: unresolved -- no RTC stub in this opam switch, or no ldd"; fi; '
  printf '%s' "${tmpl//RTC/$rtc}"
}

rtc_context_cmd() {
  local backend=$1 alias_name=
  case $backend in
    cuda) alias_name=@arrayjit/test/runtest-test_cuda_compile_options ;;
    hip) alias_name=@arrayjit/test/runtest-test_hip_compile_options ;;
    metal) alias_name=@arrayjit/test/runtest-test_metal_compile_options ;;
  esac
  printf 'echo "=== rtc-context (%s) ==="; ' "$backend"
  case $backend in
    cuda)
      loaded_rtc_cmd nvrtc
      printf 'command -v nvidia-smi >/dev/null 2>&1 && '
      printf 'nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv 2>&1; '
      # The include slot's input, read from this box rather than inferred: the
      # builder's own fallback is documented in `cuda_include_options`, and
      # re-deriving it in shell is exactly the second source of truth this block
      # avoids elsewhere.
      printf 'echo "discovery input: CUDA_PATH=${CUDA_PATH:-(unset)}"; '
      ;;
    hip)
      loaded_rtc_cmd hiprtc
      printf 'command -v rocminfo >/dev/null 2>&1 && rocminfo 2>&1 | grep -m2 -E "gfx|Runtime Version"; '
      printf 'echo "discovery input: ROCM_PATH=${ROCM_PATH:-(unset)} HIP_PATH=${HIP_PATH:-(unset)}"; '
      ;;
    metal)
      printf 'command -v sw_vers >/dev/null 2>&1 && sw_vers 2>&1; '
      printf 'command -v xcrun >/dev/null 2>&1 && xcrun -sdk macosx metal --version 2>&1 | head -3; '
      ;;
  esac
  if [ -n "$alias_name" ]; then
    # Labelled, because the got/want vectors below are the option POLICY that the
    # GPU-free builder test prints under sentinel inputs -- not the command line of
    # the compile that just failed. Four lines, not a paragraph: `fingerprint`
    # carries this block under a line bound, and prose that crowded the vectors out
    # of it would cost more than it explains.
    # CUDA/HIP builders contain discovered include/architecture slots filled with sentinels by
    # their tests; Metal's property sequence has no discovered slot and is the effective policy.
    case $backend in
      cuda)
        printf 'echo "rtc option policy from %s; the include dir and"; ' "${alias_name#@}"
        printf 'echo "any arch target below are TEST SENTINELS, not this box\x27s: those come from the"; '
        printf 'echo "discovery input above and the failing kernel arch markers."; '
        printf 'echo "A failed nvrtc compile also logged its OWN vector, on an \x27nvrtc options:\x27 line."; '
        ;;
      hip)
        printf 'echo "rtc option policy from %s; the include dir and"; ' "${alias_name#@}"
        printf 'echo "any arch target below are TEST SENTINELS, not this box\x27s: those come from the"; '
        printf 'echo "discovery input above and the failing kernel arch markers."; '
        printf 'echo "A failed hiprtc compile also logged its OWN vector, on a \x27hiprtc options:\x27 line."; '
        ;;
      metal)
        printf 'echo "rtc option policy from %s; exact MTLCompileOptions property sequence."; ' "${alias_name#@}"
        printf 'echo "A failed Metal compile also logged its OWN state, on a \x27metal options:\x27 line."; '
        ;;
    esac
    printf 'opam exec -- dune build %s --force 2>&1 | sed "s/^/rtc /"; ' "$alias_name"
  fi
  printf 'echo "=== end rtc-context ==="; true'
}

# Run that block for one finished unit, appending to its log. Called only after
# `record` has written the row, so nothing here can reach the outcome; the status
# is discarded for the same reason, and the whole phase is bounded by its own
# CONTEXT_CAP rather than by what is left of the unit's.
#
# Only for a `fail`. A `timeout` had its process group destroyed and may still
# have the box -- and the far-side worktree lock -- busy, and an `error` never
# reached dune at all: neither has kernels whose flags would explain anything.
collect_rtc_context() {
  local backend=$1 host=$2 wt=$3 log=$4 path_prefix=${5:-} cmd
  # `exit 0`, not a failure: the worktree is gone or unreadable, which the row
  # already records; this phase has nothing to add and nothing to complain about.
  cmd="cd \"$wt\" || exit 0; $(rtc_context_cmd "$backend")"
  if [ -n "$host" ]; then
    # Bounded on both sides and re-taking the far-side worktree lock, for the
    # reasons the unit's own remote call documents: the unit's ssh has exited, so
    # its lock is released, and a dune running there unlocked is exactly what the
    # next sweep's preparation would reset the tree underneath.
    run_capped "$(( CONTEXT_CAP + 120 ))" ssh -o BatchMode=yes -o ConnectTimeout=8 \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      "$host" "$(remote_capped "$CONTEXT_CAP" "$path_prefix $(remote_lock_cmd "$wt") $cmd")" \
      >>"$log" 2>&1
  else
    run_capped "$CONTEXT_CAP" /bin/sh -c "$cmd" >>"$log" 2>&1
  fi
  return 0
}

# POSIX single-quoting, so a generated command can be handed to `sh -c` on the
# far side without the remote shell re-splitting or re-expanding any of it.
sq() { printf "'%s'" "$(printf %s "$1" | sed "s/'/'\\\\''/g")"; }

# The far-side counterpart of fd 9. A local lock cannot protect a worktree on
# another machine, and the gap is reachable: when keepalives detect a blackholed
# connection, ssh returns 255 EARLY -- well inside the outer budget -- so the
# sweep records `error` and exits while the remote dune keeps running under its
# own timeout for the rest of CAP. The next sweep's preparation would then reset
# and clean that remote worktree underneath a live build. With this it is refused
# instead, and recorded as non-coverage.
#
# 126 so the outcome mapping files a busy remote as `error`: nothing was judged.
#
# flock(1) rather than the perl the local side needs -- the remote boxes are WSL
# Linux, where util-linux provides it. Orphaned remote processes inherit the
# descriptor, so the lock outlives the ssh session that took it and clears when
# the last of them exits: the same property the local lock relies on.
remote_lock_cmd() {
  printf 'mkdir -p "$(dirname "%s")" && exec 9>"%s.lock" && flock -n 9 || exit 126; ' "$1" "$1"
}

# Cap a local command, killing its whole process group when the cap expires.
# `perl -e 'alarm N; exec ...'` is not enough on its own: alarm survives exec, so
# SIGALRM reaches only the immediate child while dune and every compiler it
# spawned keep running -- holding _build locks that the NEXT unit on this machine
# (the local backends share one worktree) would then contend with, turning one timeout
# into a cascade. Exits 142 on expiry, matching the outcome mapping below.
#
# INT, TERM and HUP are forwarded to that group and REAPED before this exits.
# Putting the child in its own group is what makes forwarding necessary: a signal
# sent to the sweep's group (a terminal Ctrl-C, the scheduler cancelling the
# task) reaches bash and this supervisor but not the child, so without this dune
# would survive while the caller's trap released the lock -- letting the next
# sweep reset the worktree under a still-running build, which is the corruption
# the lock exists to prevent.
#
# The TERM->KILL grace is measured on the GROUP, not on the leader. The leader is
# a `sh -c` that dies on the first TERM, so waiting for IT to exit would end the
# grace in ~0.1s -- and the descendants are the entire reason to escalate: a dune
# worker or a compiler with a TERM handler wants a moment to unlink its
# temporaries and drop its _build lock cleanly. `kill 0, -$pid` counts what is
# still in the group, and the leader is reaped on each pass so its zombie does
# not read as a survivor and hold the grace open for the full interval.
capped_perl='
  use POSIX ();
  my $cap = shift;
  my $pid = fork();
  die "fork: $!" unless defined $pid;
  if (!$pid) { setpgrp(0, 0); exec @ARGV; exit 127 }
  my $reap = sub {
    my $code = shift;
    kill "TERM", -$pid;
    for (1 .. 100) {
      waitpid($pid, POSIX::WNOHANG());
      last unless kill 0, -$pid;
      select undef, undef, undef, 0.1;
    }
    kill "KILL", -$pid;
    waitpid($pid, 0);
    exit $code;
  };
  $SIG{ALRM} = sub { $reap->(142) };
  $SIG{INT} = sub { $reap->(130) };
  $SIG{TERM} = sub { $reap->(143) };
  $SIG{HUP} = sub { $reap->(129) };
  alarm $cap;
  waitpid($pid, 0);
  my $st = $?;
  alarm 0;
  exit($st & 127 ? 128 + ($st & 127) : $st >> 8);
'
# First argument is the budget in seconds, so the remote path can allow for
# far-side cleanup and ssh teardown on top of its own cap.
capped() { perl -e "$capped_perl" -- "$@"; }

# The same supervisor, for the backgrounded unit calls. `exec` is the point:
# backgrounding a shell FUNCTION runs it in a subshell, so `capped ... &` would
# put the SUBSHELL's pid in $! -- and a relayed signal would kill that wrapper
# while the supervisor, and dune under it, carried on. exec replaces the subshell
# so $! names the supervisor itself. Valid only backgrounded: called
# synchronously it would replace this script.
capped_bg() { exec perl -e "$capped_perl" -- "$@"; }

# The FAR-SIDE cap, emitted as shell text for the remote shell to run. It is the
# same perl supervisor, and for the same reason: `timeout -k` is not a
# group-killing bound everywhere it exists. uutils coreutils -- Ubuntu's default
# since 25.10, and what rog-nv's WSL side runs -- delivers the TERM phase to the
# group but escalates the -k KILL to the DIRECT CHILD only, so a descendant that
# ignores or outlives TERM (a wedged pool worker, a CUDA call that never returns)
# is reparented and keeps running while `timeout` reports 137 (gh-ocannl-727).
# The unit would then be filed as `timeout` -- coverage lost -- while still
# holding the GPU and the remote worktree lock that the NEXT sweep's preparation
# must take. perl(1) is a firmer assumption than GNU-vs-uutils semantics: the
# sweep hosts are WSL Linux, where it is part of the base system, and
# tools/test-run.sh already requires it there.
#
# The supervisor exits 142 on expiry -- capped()'s code, which the outcome
# mapping below already reads as `timeout` -- so both sides now report a hang
# identically. It also reaps on HUP, which is what the remote end gets when the
# local ssh is killed by the outer run_capped, so a lost connection tears the
# remote unit down instead of orphaning it.
remote_capped() {
  printf 'perl -e %s -- %s sh -c %s' "$(sq "$capped_perl")" "$1" "$(sq "$2")"
}

# Put a reused worktree exactly on $full_sha, and PROVE it rather than assume it.
# `checkout --detach` is not sufficient on its own: a tracked edit that does not
# conflict with the target survives the checkout, which still exits 0 -- so the
# suite would run against a tree that is not the commit the history row names.
# `reset --hard` drops such edits and `clean -fd` drops untracked strays, while
# leaving IGNORED files alone: `_build` is ignored, and reusing it is what makes
# a daily cadence affordable. The porcelain check at the end is the proof.
#
# Emitted as shell text, and used by BOTH paths, so local and remote preparation
# cannot drift apart.
prep_cmd() {
  local repo=$1 wt=$2
  printf 'git -C "%s" worktree prune && ' "$repo"
  printf '{ git -C "%s" rev-parse --git-dir >/dev/null 2>&1 || ' "$wt"
  printf 'git -C "%s" worktree add -q --detach "%s" %s; } && ' "$repo" "$wt" "$full_sha"
  printf 'git -C "%s" checkout -q --detach %s && ' "$wt" "$full_sha"
  printf 'git -C "%s" reset -q --hard %s && ' "$wt" "$full_sha"
  printf 'git -C "%s" clean -qfd && ' "$wt"
  printf '[ -z "$(git -C "%s" status --porcelain)" ]' "$wt"
}

# Fatal on write failure: the history file is deliberately the only verdict, so
# a row that did not land is indistinguishable downstream from a unit that never
# ran. Better to abort mid-sweep, loudly, than to hand the consumer a partial
# history it will read as coverage.
record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$stamp" "$1" "$2" "$run_sha" "$3" "$4" "${TARGET:-<all>}" "$SLOW" \
    "${5:--}" "${6:-none}" >>"$HISTORY" ||
    die "cannot record $1/$2 outcome in $HISTORY"
}

# A compact, diffable summary of what went wrong, so a caller can tell a NEW
# failure from a standing one. Metal's operations suite carries known-red tests,
# and a sweep that shouts on every red is a sweep nobody reads.
fingerprint() {
  {
    # Error SITES, in BOTH of dune's spellings: a diagnostic anchored to one
    # line says `line N`, one anchored to a span -- notably a whole stanza whose
    # action exited non-zero, which is how every explicit-rule test here fails --
    # says `lines N-M`. Matching only the singular left a unit whose ONLY failure
    # had that shape with an EMPTY fingerprint, and empty compares equal to
    # empty, so the consumer that diffs against the previous non-pass run read a
    # red suite as "unchanged since the last sweep" and said nothing.
    #
    # A location in a dune FILE is additionally reduced to the stanza it names.
    # Line numbers there shift under any edit to that file, so a fingerprint
    # keyed on them reports wholesale change whenever an unrelated stanza is
    # inserted above -- overstating exactly the thing the diff is asked to
    # measure. The stanza's own alias/name survives such edits, and is what a
    # reader needs anyway. A stanza is named by whichever of alias/name/target
    # it declares first -- a bare `(rule (target x.actual) ...)` has no alias to
    # give. Dune elides the middle of a long excerpt, so nothing identifying is
    # always quoted; the location stands in when none was.
    awk '
      function flush() {
        if (loc == "") return
        if (name != "") print prefix ", " name; else print loc
        loc = ""; name = ""; want = ""; opened = 0
      }
      /^File "[^"]+", lines? [0-9]+/ {
        flush()
        match($0, /^File "[^"]+", lines? [0-9]+(-[0-9]+)?/)
        here = substr($0, 1, RLENGTH)
        match($0, /^File "[^"]+"/)
        head = substr($0, 1, RLENGTH)
        if (head ~ /\/dune"$/ || head == "File \"dune\"") {
          loc = here; prefix = head; next
        }
        print here
        next
      }
      loc != "" {
        # The quoted excerpt: numbered source lines, plus the elision marker
        # dune prints for a long one. Anything else ends the excerpt, which
        # then never named its stanza.
        if ($0 ~ /^\.\.\.+$/) { want = ""; opened = 0; next }
        if ($0 !~ /^[0-9 ]*[0-9] \|/) { flush(); next }
        if (name != "") next
        text = $0
        sub(/^[0-9 ]*[0-9] \| ?/, "", text)
        # Tokenized rather than matched as one regex, because the identifier is
        # not reliably a bare word sitting on its keywords line: it can be
        # quoted, and dune wraps a long field so that `(targets` ends one line
        # and its first target begins the next. A same-line regex reads both as
        # unnamed and falls back to the shifting span -- which is the failure
        # this normalization exists to avoid.
        gsub(/\(/, " ( ", text)
        gsub(/\)/, " ) ", text)
        n = split(text, tok, /[ \t]+/)
        for (i = 1; i <= n; i++) {
          if (tok[i] == "") continue
          # A dune comment runs to end of line: never the stanzas identifier.
          if (tok[i] ~ /^;/) break
          # An opening paren abandons a pending keyword: the field held a
          # nested form, as `(alias (name slow))` does, and the name is inside.
          if (tok[i] == "(") { opened = 1; want = ""; continue }
          if (tok[i] == ")") { opened = 0; want = ""; continue }
          if (want != "") { name = want " " tok[i]; break }
          if (opened && tok[i] ~ /^(alias|name|names|target|targets)$/) want = tok[i]
          opened = 0
        }
        next
      }
      END { flush() }
    ' "$1"
    grep -hoE '^(Error|Fatal error|Exception)[^,]*' "$1"
    # A production compiler option vector appended to the exception message by
    # `cuda_to_ptx`, `hip_to_code`, or `compile_metal_source`. The selectors above
    # cannot reach it (it starts neither at an error site nor at
    # `Error`/`Fatal error`/`Exception`), so match the prefix each backend writes.
    # A changed option set then appears as a fingerprint diff rather than as a
    # missing line (gh-ocannl-849; Codex P2 on PR #510).
    grep -hoE '^(nvrtc|hiprtc|metal) options: .*' "$1"
  } 2>/dev/null | sort -u | head -60
  # The rtc-context block a failing GPU unit appended (see rtc_context_cmd),
  # verbatim and unsorted: it is a small fixed-size report whose ORDER is what
  # makes it readable, not a set of error sites to deduplicate. Carried into the
  # fingerprint rather than left in the log because the fingerprint is what a
  # caller diffs against yesterday's -- a toolkit upgrade or a changed option
  # vector then shows up as a diff beside the failure it explains, which is the
  # whole point (gh-ocannl-784).
  sed -n '/^=== rtc-context /,/^=== end rtc-context ===$/p' "$1" 2>/dev/null | head -40
}

# An outcome that is not a pass, with nothing extractable from its log, is its
# own condition -- not a fingerprint of zero failures. The consumer diffs this
# file against the previous non-pass run's, and an empty file compares equal to
# an empty file, so such a unit was filed as "unchanged since the last sweep"
# and reported to nobody; that is how the missing `lines N-M` spelling above
# survived two sweeps. The sentinel makes the file differ from a real
# fingerprint in either direction, and the summary line is what a human
# actually sees: the scheduled routine quotes sweep output, so a finding that
# lives only in a written file is one nobody reads (gh-ocannl-792).
EMPTY_FINGERPRINT='(no fingerprintable diagnostics -- read the log)'

write_fingerprint() {
  local log=$1 label=$2 fp=${1%.log}.fingerprint
  fingerprint "$log" >"$fp"
  if [ ! -s "$fp" ]; then
    printf '%s\n' "$EMPTY_FINGERPRINT" >"$fp"
    echo "  $label: $EMPTY_FINGERPRINT -- $log"
  fi
  WRITTEN_FINGERPRINT=$fp
}

# The history remains the append-only coverage record. This smaller state is
# the comparison cursor for one exact unit scope: its immediately previous
# verdict, and the previous failing fingerprint plus the commit that last
# touched each failing golden. A target smoke must not become the predecessor
# of a full sweep (nor a weekday run the predecessor of a slow one), hence all
# scope columns participate in the key.
unit_state_path() { # machine backend -> path
  local raw readable crc
  # The requested logical ref is part of the experiment scope. A one-off old
  # or feature ref must not become origin/master's green/red predecessor even
  # when both happen to resolve to related commits.
  raw=$(printf '%s\t%s\t%s\t%s\t%s' "$1" "$2" "${TARGET:-<all>}" "$SLOW" "$REF")
  readable=$(printf '%s' "$1-$2-${TARGET:-all}-$SLOW-$REF" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-96)
  crc=$(printf '%s' "$raw" | cksum | awk '{print $1}')
  printf '%s/%s-%s.state' "$UNIT_STATES" "$readable" "$crc"
}

state_field() { # state key -> first value
  awk -F '\t' -v key="$2" '$1 == key { print $2; exit }' "$1"
}

goldens_from_log() { # log destination -- source-tree paths proved to have failed a diff
  local log=$1 destination=$2 candidates token
  candidates=$destination.candidates.$$
  : >"$destination" || die "cannot stage failing golden paths"

  # An ordinary `(test)` failure names its expected file directly. Explicit
  # rules name only their dune stanza, but a diff that ACTUALLY RAN and found a
  # mismatch emits a resolved `diff --git` header. Reading that header—rather
  # than guessing from the stanza—distinguishes a failed diff from an earlier
  # command in a run-then-diff `progn`, and naturally carries `%{read:...}`
  # expansions plus PPX's `*_expected.ml` naming.
  {
    sed -n 's/^File "\([^"]*\.expected\)".*/\1/p' "$log"
    sed -n 's|^diff --git a/_build/default/\([^ ]*\) b/_build/default/.*$|\1|p' "$log"
    # Inline ppx_expect compares the source baseline directly with a generated
    # .corrected file, so its first operand has no _build/default prefix.
    sed -n 's|^diff --git a/\([^ ]*\) b/_build/default/[^ ]*\.corrected$|\1|p' "$log"
  } | sort -u >"$candidates" || die "cannot extract proven failing goldens"
  while IFS= read -r token; do
    [ -n "$token" ] || continue
    token=${token#./}
    git -C "$MAIN" cat-file -e "$full_sha:$token" 2>/dev/null || continue
    printf '%s\n' "$token" >>"$destination" ||
      die "cannot stage failing golden path $token"
  done <"$candidates"
  sort -u "$destination" -o "$destination" || die "cannot normalize failing golden paths"
  rm -f "$candidates"
}

update_unit_state() { # machine backend outcome [fingerprint] [log]
  local machine=$1 backend=$2 outcome=$3 fp=${4:-} log=${5:-}
  local label state stage previous_verdict previous_failure_ref
  local previous_fp current_goldens golden_paths path commit old_commit short old_short
  label=$machine/$backend
  # skip/error/timeout are recorded outcomes but not verdicts: they judged no
  # test result. Letting one replace a prior green would hide the next red's
  # regression transition merely because a machine slept or a run timed out.
  case $outcome in skip | error | timeout) return 0 ;; esac
  state=$(unit_state_path "$machine" "$backend")
  stage=$state.stage.$$
  previous_fp=$state.previous-fingerprint.$$
  current_goldens=$state.current-goldens.$$
  golden_paths=$state.golden-paths.$$
  : >"$current_goldens" || die "cannot stage unit state for $label"

  if [ -f "$state" ]; then
    [ "$(head -1 "$state")" = "$(printf 'schema\t1')" ] ||
      die "$state has an unknown unit-state schema"
    previous_verdict=$(state_field "$state" last_verdict)
    [ -n "$previous_verdict" ] || die "$state has no last verdict"
    previous_failure_ref=$(state_field "$state" last_failure_ref)
    if [ -n "$previous_failure_ref" ] &&
       ! awk -F '\t' '$1 == "fingerprint" { found=1 } END { exit !found }' "$state"; then
      die "$state has a previous failure but no fingerprint"
    fi
  else
    previous_verdict=
    previous_failure_ref=
  fi

  if [ "$outcome" = fail ]; then
    [ -n "$fp" ] && [ -s "$fp" ] || die "no current failure fingerprint for $label"
    [ -n "$log" ] && [ -f "$log" ] || die "no current failure log for $label"
    case $previous_verdict in
      pass | incremental-pass | legacy-pass)
        echo "  $label: REGRESSION OR FIX DID NOT TAKE -- previous verdict was $previous_verdict"
        ;;
    esac

    # Compare with the previous FAILURE, even if green or unavailable runs sat
    # between it and this one. That is the experiment whose identity matters:
    # same failure vs a moving one, not merely same as yesterday's outcome.
    if [ -n "$previous_failure_ref" ]; then
      awk -F '\t' '$1 == "fingerprint" { sub(/^[^\t]*\t/, ""); print }' \
        "$state" >"$previous_fp" || die "cannot read the previous fingerprint for $label"
      if [ -s "$previous_fp" ] && ! cmp -s "$previous_fp" "$fp"; then
        short=$(printf '%s' "$previous_failure_ref" | cut -c1-8)
        echo "  $label: fingerprint moved since the previous failure at $short"
      fi
    fi

    goldens_from_log "$log" "$golden_paths"
    while IFS= read -r path; do
      [ -n "$path" ] || continue
      commit=$(git -C "$MAIN" log -1 --format=%H "$full_sha" -- "$path" 2>/dev/null) ||
        die "cannot read golden history for $path"
      [ -n "$commit" ] || continue
      printf 'golden\t%s\t%s\n' "$commit" "$path" >>"$current_goldens" ||
        die "cannot stage golden state for $label"
      if [ -f "$state" ]; then
        old_commit=$(awk -F '\t' -v path="$path" \
          '$1 == "golden" && $3 == path { print $2; exit }' "$state")
        if [ -n "$old_commit" ] && [ "$old_commit" != "$commit" ]; then
          short=$(printf '%s' "$commit" | cut -c1-8)
          old_short=$(printf '%s' "$old_commit" | cut -c1-8)
          echo "  $label: REGRESSION OR FIX DID NOT TAKE -- $path last changed at $short (previous failing copy: $old_short)"
        fi
      fi
    done <"$golden_paths"
  fi

  {
    printf 'schema\t1\n'
    printf 'last_verdict\t%s\n' "$outcome"
    printf 'last_ref\t%s\n' "$full_sha"
    if [ "$outcome" = fail ]; then
      printf 'last_failure_ref\t%s\n' "$full_sha"
      while IFS= read -r line; do printf 'fingerprint\t%s\n' "$line"; done <"$fp"
      cat "$current_goldens"
    elif [ -f "$state" ] && [ -n "$previous_failure_ref" ]; then
      awk -F '\t' '$1 == "last_failure_ref" || $1 == "fingerprint" || $1 == "golden"' "$state"
    fi
  } >"$stage" && mv "$stage" "$state" ||
    die "cannot publish unit state for $label"
  rm -f "$previous_fp" "$current_goldens" "$golden_paths"
}

if [ "$FORCE" = 1 ]; then
  execution=forced
else
  execution=incremental
fi

echo "sweep $stamp  ref=$REF ($run_sha)  slow=$SLOW  target=${TARGET:-<all>}  execution=$execution"
echo

for unit in "${UNITS[@]}"; do
  IFS=: read -r machine backend host <<<"$unit"
  wanted "$backend" || continue
  WRITTEN_FINGERPRINT=

  log=$LOGS/$stamp-$machine-$backend.log
  started=$(date +%s)

  if [ -n "$host" ]; then
    # The reachability probe doubles as the way the remote home is resolved, so
    # every generated path is a literal and the whole test command can be
    # single-quoted for `sh -c`. Leaving `$HOME` in it would force double quotes
    # on the far side and leave the command open to re-expansion.
    # ConnectTimeout alone does not bound this: ssh_config(5) scopes it to
    # establishing the connection, the handshake and key exchange -- not to
    # running the remote command. A box that accepts the connection and then
    # wedges its shell would hang the whole sweep here, before any unit records
    # anything, so every ssh in this loop gets an outer bound as well.
    #
    # The one step NOT routed through run_capped: its output is captured, and
    # command substitution runs in a subshell, so a UNIT_PID published there
    # would be invisible to the trap in this shell. Its 60s budget bounds how
    # long a cancellation can be delayed here, which is the reason that is
    # tolerable where a 900s preparation leg was not.
    if ! remote_home=$(capped 60 ssh -o BatchMode=yes -o ConnectTimeout=8 \
         -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
         "$host" 'printf %s "$HOME"' 2>/dev/null) ||
       [ -z "$remote_home" ]; then
      echo "  $machine/$backend: skip (unreachable)"
      record "$machine" "$backend" skip 0
      update_unit_state "$machine" "$backend" skip
      continue
    fi
    wt="$remote_home/ocannl-staging-worktrees/sweep"
    # rog needs the CUDA and WSL lib dirs on PATH; harmless elsewhere.
    path_prefix="export PATH=/usr/local/cuda/bin:/usr/lib/wsl/lib:\$PATH;"
    # Preparation is its own ssh round trip so that its failure -- a connection
    # dropped after the probe, a full disk, a wedged worktree -- is recorded as
    # `error`, matching the local path. Folded into the test command it would
    # have surfaced as a non-zero status in the generic branch below and been
    # written down as a FAILING SUITE, which is the opposite of the truth: a
    # remote that never got as far as dune tested nothing at all.
    remote_repo="$remote_home/ocannl-staging"
    remote_prep="git -C \"$remote_repo\" fetch -q origin master && $(prep_cmd "$remote_repo" "$wt")"
    # Bounded on BOTH sides, for the reason the test leg documents below: an
    # outer bound only kills the local ssh, and a wedged `git fetch` left running
    # on the far side can finish later and reset the shared remote worktree --
    # possibly while a subsequent sweep is building in it. The far-side cap is
    # what actually stops that; the outer budget is the backstop for a connection
    # that dies without the remote noticing, and is larger so it cannot pre-empt
    # the inner one. Generous overall, since a cold fetch on a slow link is
    # legitimate work.
    if ! run_capped 900 ssh -o BatchMode=yes \
         -o ServerAliveInterval=30 -o ServerAliveCountMax=10 \
         "$host" "$(remote_capped 600 "$path_prefix $(remote_lock_cmd "$wt") $remote_prep")" \
         >"$log" 2>&1; then
      echo "  $machine/$backend: error (cannot pin $host to $run_sha)"
      record "$machine" "$backend" error "$(( $(date +%s) - started ))" "$log"
      write_fingerprint "$log" "$machine/$backend"
      update_unit_state "$machine" "$backend" error "$WRITTEN_FINGERPRINT"
      continue
    fi
    # The cap is applied on the FAR side: killing the local ssh would leave the
    # remote dune running. ONE cap around the whole unit -- the same perl
    # supervisor capped() uses locally, see remote_capped -- because a
    # per-dune-call cap would let a --slow unit run for twice the budget the
    # script advertises.
    remote="$(remote_capped "$CAP" "$path_prefix $(remote_lock_cmd "$wt") $(test_cmd "$backend" "$wt" "$(unit_jobs "$machine" "$backend")")")"
    # The far-side cap does not bound the LOCAL ssh: if the connection blackholes
    # after the command starts -- the box suspends, the WiFi drops -- the remote
    # cap may kill dune while this ssh sits waiting for a status that will
    # never arrive. OpenSSH's defaults do not rescue it (`ssh -G` reports
    # serveraliveinterval 0 and connecttimeout none), and because the unit is in
    # the foreground, the whole sweep stalls behind it: no later units, no rows.
    #
    # Keepalives detect a dead peer in ~5min, and capped() is the backstop for
    # the case where the connection is alive but the far side never returns. Its
    # budget deliberately exceeds $CAP, so it can only fire after the remote cap
    # has had its chance plus room for cleanup and teardown; otherwise it
    # would cut legitimate long runs short and call them timeouts.
    run_capped "$(( CAP + 300 ))" \
      ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 \
      "$host" "$remote" >"$log" 2>&1
    rc=$?
  else
    wt=$LOCAL_WT
    # Checked, and fatal for this UNIT rather than the run. The worktree is
    # reused, so a checkout that fails -- a conflicting edit, a half-removed
    # worktree -- leaves the previous revision's tree on disk; running the suite
    # against it would record a pass under $run_sha for a commit that was never
    # tested.
    #
    # One error here is sticky and worth recognising: if the directory survives
    # while its administrative entry does not (`git worktree prune` reaps the
    # entry whenever the path is temporarily absent or replaced), every
    # subsequent run reports `fatal: '<path>' already exists`. Recover by hand --
    # move `_build` aside, remove the directory, `git worktree add --detach` it
    # again, move `_build` back. Deliberately not automated: deleting a
    # multi-gigabyte build tree unattended is worse than a loud repeated error.
    if ! /bin/sh -c "$(prep_cmd "$MAIN" "$wt")" >"$log" 2>&1; then
      echo "  $machine/$backend: error (cannot pin $wt to $run_sha)"
      record "$machine" "$backend" error "$(( $(date +%s) - started ))" "$log"
      write_fingerprint "$log" "$machine/$backend"
      update_unit_state "$machine" "$backend" error "$WRITTEN_FINGERPRINT"
      continue
    fi
    run_capped "$CAP" /bin/sh -c "$(test_cmd "$backend" "$wt" "$(unit_jobs "$machine" "$backend")")" >"$log" 2>&1
    rc=$?
  fi

  elapsed=$(( $(date +%s) - started ))
  # A hang, a lost connection and a red test call for different responses, so
  # keep them apart. 142 is the supervisor's expiry (128+SIGALRM), on either
  # side of the ssh now that the remote unit runs under it too. 124 and 137 are
  # kept as timeouts anyway: they are what a timeout(1) still on the far side of
  # an older worktree would report, and 137 is also what an OOM kill looks like
  # -- all three mean the run was destroyed rather than judged, which is the
  # distinction the outcome is carrying.
  #
  # ssh reserves 255 for its own transport errors, so on the remote path that is
  # a connection lost mid-run: nothing was judged there either, which is `error`
  # (non-coverage) rather than a failing suite. Locally 255 is just an exit code.
  #
  # 126 and 127 mean the shell could not run what it was asked to: no opam, no
  # dune in the selected switch, no worktree to cd into. Nothing was judged, so
  # that is non-coverage, not a red suite -- a distinction that matters most on a
  # GPU box used rarely enough for its switch to rot unnoticed.
  case $rc in
    # `pass` is reserved for the mode that makes Dune re-execute every action
    # attached to the selected aliases. An incremental success is useful, but
    # is an unknown mixture of execution and cache hits, so it must not refresh
    # a consumer that ages actual backend coverage by the latest `pass` row.
    0) [ "$FORCE" = 1 ] && outcome=pass || outcome=incremental-pass ;;
    124 | 137 | 142) outcome=timeout ;;
    126 | 127) outcome=error ;;
    255) [ -n "$host" ] && outcome=error || outcome=fail ;;
    *) outcome=fail ;;
  esac
  echo "  $machine/$backend: $outcome (${elapsed}s; execution=$execution)"
  record "$machine" "$backend" "$outcome" "$elapsed" "$log" "$execution"
  if [ "$outcome" = pass ] && [ -z "$TARGET" ]; then
    SKIP_RUN_BACKENDS+=("$backend")
    SKIP_RUN_BOXES+=("$machine")
    SKIP_RUN_LOGS+=("$log")
  fi
  # Diagnosis, strictly after the row and the elapsed time it reports: this phase
  # has its own budget, and nothing it does can reach $outcome or $elapsed. It
  # runs before the fingerprint so that what it appends to the log is carried in.
  case $outcome:$backend in
    fail:cuda | fail:hip | fail:metal)
      collect_rtc_context "$backend" "$host" "$wt" "$log" "${path_prefix:-}"
      ;;
  esac
  case $outcome in
    fail | timeout | error) write_fingerprint "$log" "$machine/$backend" ;;
  esac
  update_unit_state "$machine" "$backend" "$outcome" "${WRITTEN_FINGERPRINT:-}" "$log"
  WRITTEN_FINGERPRINT=
done

echo
if [ "$FORCE" = 1 ] && [ -z "$TARGET" ]; then
  [ -x "$AGGREGATE_SKIPS" ] || die "skip aggregator is not executable: $AGGREGATE_SKIPS"
  aggregate_args=()
  while IFS= read -r backend; do
    [ -n "$backend" ] && aggregate_args+=(--known "$backend")
  done <<<"$known_backends"
  for box in "${known_boxes[@]:-}"; do
    [ -n "$box" ] && aggregate_args+=(--known-box "$box")
  done
  for ((i = 0; i < ${#SKIP_RUN_BACKENDS[@]}; i++)); do
    aggregate_args+=(--run "${SKIP_RUN_BACKENDS[$i]}" "${SKIP_RUN_BOXES[$i]}" \
      "${SKIP_RUN_LOGS[$i]}")
  done

  report=$LOGS/$stamp-skip-coverage.txt
  report_stage=$report.stage.$$
  scope='@runtest + @train'
  [ "$SLOW" = 1 ] && scope="$scope + @slow"
  {
    echo "skip coverage for $run_sha ($scope; forced execution)"
    "$AGGREGATE_SKIPS" "${aggregate_args[@]}"
  } >"$report_stage"
  aggregate_rc=$?
  case $aggregate_rc in
    0 | 1) mv "$report_stage" "$report" || die "cannot publish $report" ;;
    *) rm -f "$report_stage"; die "skip aggregation failed (exit $aggregate_rc)" ;;
  esac
  if [ "$aggregate_rc" -eq 1 ]; then
    echo "skip coverage: FAIL -- $report"
    echo "sweep: skip coverage FAIL -- $report" >&2
  else
    aggregate_status=$(grep '^status:' "$report" | head -1)
    echo "skip coverage: ${aggregate_status:-report written} -- $report"
  fi
  # The verdict and the findings themselves, not only the report path: the
  # scheduled routine's report and notification quote sweep output, and a
  # zero-coverage claim that lives only behind a file path is one no human
  # reads (gh-ocannl-792). Indented, so the `skip coverage:` line above stays
  # the one line consumers extract the path from. Unbounded on purpose: the
  # findings are the intersection across backends, already small by
  # construction, and a cap here would silently hide the very claims this
  # exists to surface.
  grep -E '^(result|environment result|FAIL|POTENTIAL): ' "$report" | sed 's/^/  /'
else
  echo "skip coverage: not aggregated (requires --force with no --target)"
fi
echo "history: $HISTORY"
echo "logs:    $LOGS/$stamp-*"
echo "state:   $UNIT_STATES"
