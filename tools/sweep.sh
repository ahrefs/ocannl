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
# Each machine's units run as one LANE, and the lanes run concurrently (see
# run_lane): units that share a box run one after another, units on different
# boxes do not wait for each other.
#
# Usage:
#   tools/sweep.sh                     # cc + metal locally; cuda on rog-nv, hip + multidev_cc on minix, if up
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
# The per-box width cap, shared with tools/test-run.sh so the two cannot drift.
[ -r "$SWEEP_TOOLS/box-jobs.sh" ] || {
  echo "sweep: cannot read $SWEEP_TOOLS/box-jobs.sh" >&2
  exit 2
}
# shellcheck source=box-jobs.sh
. "$SWEEP_TOOLS/box-jobs.sh"
# The dxg window filter and its burst count, shared with the harness that pins it.
[ -r "$SWEEP_TOOLS/dxg-window.sh" ] || {
  echo "sweep: cannot read $SWEEP_TOOLS/dxg-window.sh" >&2
  exit 2
}
# shellcheck source=dxg-window.sh
. "$SWEEP_TOOLS/dxg-window.sh"

# ---------------------------------------------------------------- the lab lock
# The WSL boxes are shared, and `wsl.exe --shutdown` on one of them is HOST-GLOBAL: it destroys the
# whole VM, so every session on that box dies with it. On 2026-09-16 that cost this sweep both GPU
# units. Run 20260916T074913Z was 13 minutes into rog-nv/cuda and minix/hip when a second session,
# verifying an unrelated fix to wake-lab.sh against the real lab, ran `wake-lab.sh --wait
# --restart-wsl rog minix` at 08:02:30Z; both guests rebooted seconds later and both units died on
# a reset connection. They were recorded as `error`, which reads as a box fault, and the failure
# was attributed to the GPU autotune tests for two days -- they are the longest-running tests in
# the suite, so a randomly-timed external kill lands in them far more often than anywhere else.
#
# So a remote lane RESERVES its box for as long as it is using it, and wake-lab.sh refuses to
# destroy a VM whose box is reserved. The contract between the two tools is deliberately just a
# directory, a filename and a one-line description -- this sweep takes its own flock and never
# calls wake-lab.sh, so a checkout on a box that has no ~/bin/wake-lab.sh still reserves correctly.
# `wake-lab.sh lock-path <box>` answers the same path for anyone who would rather ask than derive.
LAB_LOCK_DIR=${WAKE_LAB_LOCK_DIR:-$HOME/.local/state/wake-lab}
# How long a lane waits for a box someone else is using. Sized against what the holder is most
# likely doing: a `--restart-wsl` is a shutdown plus a cold VM start plus the tailscaled wait
# behind it, which is minutes rather than seconds. A lane that waits longer than this skips its
# units rather than running them on a box that is being torn down underneath it.
LAB_LOCK_WAIT=${OCANNL_TOOL_SWEEP_LAB_LOCK_WAIT:-300}
# The identity probe: a single `cat` of the guest's boot id, asked when nothing else could say
# which guest is there. Three budgets, and each answers a different way the far side misbehaves.
# Plain constants rather than knobs; nothing outside this file has a reason to retune them.
#   CAP     one attempt, so a far side that connects and then wedges cannot hold the lane. Small
#           on purpose, and deliberately NOT the diagnostic cap the window query carries.
#   WINDOW  the whole probe, so a guest that never comes back is bounded too.
#   PAUSE   between attempts. A guest that is still booting REFUSES connections rather than
#           dropping them, and a refusal returns at once -- so without this the attempts all land
#           inside the same millisecond and there is no retry window at all.
DXG_IDENTITY_CAP=30
DXG_IDENTITY_WINDOW=90
DXG_IDENTITY_PAUSE=10

# The wake-lab box whose lock covers an ssh alias. The two real ones are named rather than derived,
# so a renamed alias fails loudly here instead of silently reserving a box nobody checks; the
# fallback is the alias's first component, which is the convention the aliases already follow.
lab_box_of() { # ssh-alias
  case $1 in
    rog-nv-wsl) printf 'rog' ;;
    minix-amd-wsl) printf 'minix' ;;
    *) printf '%s' "${1%%-*}" ;;
  esac
}

# Reserve a box on fd 8, for as long as this shell lives. Called only in a LANE subshell, so the
# reservation is released when the lane ends however it ends -- there is nothing to reclaim after a
# crash, and no state that can outlive the process that made it. The flock idiom is the run lock's
# above: perl takes it and exits, and the lock survives because it belongs to the open file
# DESCRIPTION behind fd 8, which this shell keeps open.
# Three-valued, because two of the outcomes mean opposite things to the operator: 0 the box is
# reserved, 1 another holder still has it after the wait (contention -- a fact about the lab), 2
# this harness could not make a lock at all (a read-only or full state directory -- a fact about
# THIS machine). Collapsing 2 into 1 would publish `skip (box ... reserved by ...)` over a local
# failure, which reads as another run legitimately holding the box and hides the broken harness
# behind coverage the operator would not think to question.
take_lab_lock() { # box -- 0 reserved, 1 held by someone else, 2 this harness cannot lock at all
  local box=$1 path deadline=$((SECONDS + LAB_LOCK_WAIT)) waited=0
  path=$LAB_LOCK_DIR/$box.lock
  mkdir -p "$LAB_LOCK_DIR" 2>/dev/null || return 2
  exec 8>>"$path" || return 2
  while :; do
    if perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 1)' <&8; then
      # The holder line is advisory -- it names who to go and look at, and nothing reads it to make
      # a decision. Written after the lock is held, so two racing lanes cannot interleave into it.
      printf 'ocannl sweep %s (pid %s, since %s)\n' "$stamp" "$$" \
        "$(date -u +%Y%m%dT%H%M%SZ)" >"$path" 2>/dev/null
      [ "$waited" -gt 0 ] && say "  $box: reserved after waiting ${waited}s for the previous holder"
      return 0
    fi
    [ "$SECONDS" -ge "$deadline" ] && { exec 8>&-; return 1; }
    sleep 5
    waited=$((waited + 5))
  done
}

# Who holds a box, for the skip message. Never trusted for the decision.
lab_lock_holder() { # box
  local line
  line=$(head -1 "$LAB_LOCK_DIR/$1.lock" 2>/dev/null | tr -d '\000-\037')
  printf '%s' "${line:-an unnamed holder}"
}

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
# Table order is execution order within a box. Boxes run concurrently (see
# run_lane), so the order ACROSS boxes decides nothing about timing.
#
# multidev_cc needs no GPU, but it is here for the same reason the GPU boxes
# are: nothing else runs it. It keeps its OWN debug-log golden --
# `test/operations/micrograd_demo_logging-multidev_cc-0-0.log.expected`, whose
# statement order the scheduler is free to differ on -- and `dune runtest`
# exercises that golden only when OCANNL_BACKEND says so, which the pinned
# `backend=cc` in test/config's ocannl_config and CI never do. gh-ocannl-700 is
# what that costs: an ordering change landed, the cc golden was re-promoted with
# it, and the multidev leg stayed red on master for six weeks with nothing to
# notice. A backend with its own goldens and no leg here is a silent regression
# channel whether or not it needs hardware.
#
# The two CPU backends deliberately run on DIFFERENT boxes: cc on the local
# macOS host, multidev_cc on minix's Linux side. With lanes, the longest lane
# sets the sweep's wall-clock, and the local one -- carrying metal's long
# suite -- is that lane, so moving a CPU unit off it is the load balance that
# shortens the run; minix's lane stays shorter even with both of its units. It
# also exercises the CPU code generators and their goldens under both operating
# systems every day. hip goes first on its box: it is the unit that needs the
# freshly restarted WSL VM the scheduled routine hands over (`wake-lab.sh
# --restart-wsl`), and the one whose hardware nothing else covers.
UNITS=(
  "$LOCAL_BOX:cc:"
  "$LOCAL_BOX:metal:"
  "rog-nv:cuda:rog-nv-wsl"
  "minix:hip:minix-amd-wsl"
  "minix:multidev_cc:minix-amd-wsl"
)

# Dune's job count for the TEST phase of a unit, empty for dune's default (one
# per core). The cap itself, which boxes it covers and why it is the number it
# is now live in tools/box-jobs.sh, the single source this and tools/test-run.sh
# both read: the same bridge overflows a MANUAL GPU suite on such a box, and a
# cap only the sweep knew about was rediscovered the hard way (gh-ocannl-983).
# Applied to the test phase only -- the compile phase stays uncapped, since
# `test_cmd` runs `@check` first and the cap bounds GPU-holding processes, not
# the build. Override for one run with OCANNL_TOOL_SWEEP_JOBS=<n>, which then
# applies to every unit.
unit_jobs() {
  if [ -n "${OCANNL_TOOL_SWEEP_JOBS:-}" ]; then
    printf '%s' "$OCANNL_TOOL_SWEEP_JOBS"
    return
  fi
  box_jobs_sweep_cap "$1" "$2"
}

# The failure names that mean the ENVIRONMENT refused the run rather than a
# test judging it. On minix the WSL2 dxg bridge (tools/box-jobs.sh, and the dxg
# bullet in docs/agent-notes/build-and-test.md) surfaces its lost
# messages as three HIP exceptions; rog-nv's CUDA reaches its GPU through the
# same WSL2 bridge, so the cudajit checks at the same three call sites are
# listed by analogy, plus the primary-context retain every CUDA process makes
# first -- where rog-nv's one lost bridge message was reported, as
# CUDA_ERROR_OUT_OF_MEMORY on a device with VRAM to spare. A unit whose log
# carries any of them is environment-red: its stanzas were refused a device,
# not judged, and whatever test-logic failures it also holds are hidden under
# that noise until the environment is repaired -- days, for a WSL box. The 2026-09-05 wide run
# hid a genuine hip-only regression that way (gh-ocannl-943; the executable
# segfaulted after its hip_init was refused, so no fingerprint could have shown
# it), and only rerunning the failing stanzas at `-j 1` on the box told the two
# apart: 4 of 27 stayed red on their own. So such a unit gets exactly that rerun
# (serial_rerun below, gh-ocannl-945): every failing stanza again under `-j 1`,
# and `serial rerun:` lines in its log and fingerprint saying which stayed red.
# Keyed as dune prints an uncaught binding error, `Fatal error: exception
# <name>:` with the status on the next line. The statuses each name has been
# seen with, and how to read a rerun's verdict, are the signature table in
# docs/agent-notes/build-and-test.md (the record half of gh-ocannl-927).
ENVIRONMENT_REFUSALS='hip_init
hip_module_load_data_ex
hip_stream_create_with_priority
cu_init
cu_device_primary_ctx_retain
cu_module_load_data_ex
cu_stream_create_with_priority'

# Either the name list or the kernel's own evidence (dxg_window_red, which reads
# the unit's collected-evidence sidecar and answers only to a positive count). The second arm is what lets a
# call site nobody has seen yet -- or a failure with no exception name at all,
# such as the SEGV the 2026-09-15 minix runs produced -- get its serial rerun on
# the FIRST miss instead of after one (gh-ocannl-979). The rerun's `still red` /
# `all clean` verdict remains the judge either way: this decides that the unit is
# rerun, never that its failures were the environment's.
environment_red() { # log
  local name
  while IFS= read -r name; do
    [ -n "$name" ] && grep -q "^Fatal error: exception $name:" "$1" && return 0
  done <<<"$ENVIRONMENT_REFUSALS"
  dxg_window_red "$1" && return 0
  return 1
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
#
# The same relay serves both levels of the run. In the top-level shell the
# children are the lanes (LANE_PIDS), each of which installs this trap again for
# its own in-flight supervisor (UNIT_PID) -- a subshell does not inherit caught
# traps. A lane is signalled and then WAITED for, so the top level exits only
# after every lane's supervisor has reaped its process group: the run lock on fd
# 9 is held by every lane too, and a sweep that returned while a lane was still
# tearing down would let the next one take a worktree that is still in use.
# Space-separated strings rather than arrays: bash 3.2 (macOS's /bin/bash, the
# scheduled host's) refuses an empty array expansion under `set -u`.
#
# A registered pid is signalled only while this shell's own job table still
# lists it as running (`jobs -rp`, which also answers inside the command
# substitution). bash reaps an exited child in the background, so a finished
# lane's pid is free for the system to reuse while the top level is still waiting
# on a slower lane -- 90 minutes, on a hung unit -- and a relay that trusted the
# bare pid could TERM an unrelated process. The job table is the parent's own
# record of child completion, so it holds however the child ended, SIGKILL
# included, where a marker the child removes on its way out would not.
UNIT_PID=
LANE_PIDS=
# Which level of the run this shell is: the top level (0) or a lane subshell
# (1). Only the top level owns run-wide artifacts -- the run record, in
# particular -- and a lane installs the same traps, so the two cannot be told
# apart by the trap alone.
IN_LANE=0
# The record's locator, printed at most once however the run ends: a cancellation
# arriving after the completing path has already announced it would otherwise
# print a second line, and a caller extracting "the" locator would read two paths
# where the contract promises one. A cancelled run is exactly the one whose
# record an operator cannot otherwise find, since it ends before the summary
# block, so the announcement belongs on both paths -- once.
RUN_RECORD_ANNOUNCED=0
announce_run_record() {
  [ "$RUN_RECORD_ANNOUNCED" = 0 ] || return 0
  RUN_RECORD_ANNOUNCED=1
  echo "run:     $RUN_RECORD"
}
# Declared before the traps below are installed, not where it is later given its
# path: `relay` reads it, and under `set -u` a signal arriving in between would
# abort the trap with an unbound variable -- turning a cancellation into a
# harness error. Empty means no record is owed yet (see write_run_record).
RUN_RECORD=
relay() {
  local pid running live=
  running=" $(jobs -rp | tr '\n' ' ') "
  for pid in $UNIT_PID $LANE_PIDS; do
    case $running in *" $pid "*) live="$live $pid" ;; esac
  done
  for pid in $live; do kill -TERM "$pid" 2>/dev/null; done
  for pid in $live; do wait "$pid" 2>/dev/null; done
  # A cancelled run still ended, and the rows its lanes managed to write before
  # the signal are real; the record says which those were. Only the top level
  # writes it -- a lane runs this same relay for its own supervisor -- and only
  # once there is a coordination directory to read the lanes' evidence from. A
  # failure to write it must not replace the cancellation's exit status, so it
  # is reported rather than fatal.
  if [ -n "$RUN_RECORD" ] && [ "$IN_LANE" = 0 ]; then
    if write_run_record cancelled; then
      announce_run_record
    else
      echo "sweep: cannot write $RUN_RECORD" >&2
    fi
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

# The stamp names every per-run artifact -- each unit's log, its fingerprint, the
# skip-coverage report and the run record -- and at one-second resolution two
# invocations can choose the same one. They cannot RUN concurrently (the
# per-worktree lock sees to that), but a run that ends inside a second releases
# the lock inside it too, so a retry -- after a cancellation, notably -- can
# start in the same second and overwrite the artifacts of the run it is
# retrying, leaving two invocations' history rows behind one invocation's
# evidence. So the stamp is advanced until it names nothing that exists yet.
# Advanced rather than made unique with a pid or subsecond suffix: the stamp is
# the run's identity in the history's `when` column as well, where consumers
# read it as a UTC timestamp, so it has to stay exactly this format.
# A previous invocation claims a stamp by either of the two things it leaves: an
# artifact named after it, or a history row carrying it. The rows matter as much
# as the files now that the record derives its units from them -- an invocation
# killed after recording a remote unit's `skip` leaves a row and no file at all,
# and a retry reusing that stamp would put the dead run's units in its record.
stamp_taken() { # stamp -- does anything of a previous run already use it?
  local candidate
  for candidate in "$LOGS/$1-"*; do
    [ -e "$candidate" ] && return 0
  done
  awk -F '\t' -v s="$1" '$1 == s { found = 1; exit } END { exit !found }' "$HISTORY"
}
stamp=$(date -u +%Y%m%dT%H%M%SZ) || die "cannot read the clock"
stamp_offset=0
while stamp_taken "$stamp"; do
  stamp_offset=$((stamp_offset + 1))
  [ "$stamp_offset" -le 600 ] || die "no free run stamp near $stamp in $LOGS"
  stamp=$(utc_of "$(( $(date +%s) + stamp_offset ))") || die "cannot advance the run stamp"
done
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

# Appended to the unit's log, and carried into its fingerprint, like the
# rtc-context block. Its own budget, for the reason collect_rtc_context documents:
# a diagnostic must not be able to overwrite the verdict it explains.
# A sidecar appears complete or not at all. A cancellation reaching the lane
# mid-write would otherwise leave the top level to build the run record from a
# half-written file -- known bounds with no count line, which the record would
# then spell `-`, the value that means no window was ever collected. Staged and
# renamed, like the run record and the unit-state files.
# Its failures are silent on stderr, as is the fallback's: the lane buffers its
# summary lines but not its stderr, so the shell's own complaint about a failed
# redirection would reach the terminal at once, unlabelled and ahead of the unit's
# block, while the WARNING that names the unit waits for that block's flush. The
# `2>/dev/null` goes BEFORE each redirection it covers, which is processed first.
publish_dxg_sidecar() { # sidecar log writer args... -- stdin is the writer's
  local sidecar=$1 log=$2
  shift 2
  "$@" 2>/dev/null >"$sidecar.stage.$$" && mv "$sidecar.stage.$$" "$sidecar" 2>/dev/null || {
    rm -f "$sidecar.stage.$$"
    return 1
  }
  cat "$sidecar" 2>/dev/null >>"$log"
}

# A publication that failed must not leave the unit looking like one where no
# collection was attempted. The sidecar is the only provenance channel, so an
# absent one reads as `-` in the record -- "no window" -- which is the reading
# that loses the rerun for a bridge failure nobody listed. One direct retry with
# the unavailable marker, since the staged write is what failed; if even that
# cannot be written, the disk is gone and the only honest thing left is to say so
# where a human reads the run.
publish_dxg_unavailable_fallback() { # sidecar log reason label [boot-verdict]
  dxg_window_unavailable - - "$3" "${5:-}" 2>/dev/null >"$2.dxg-fallback.$$" &&
    mv "$2.dxg-fallback.$$" "$1" 2>/dev/null && {
      cat "$1" 2>/dev/null >>"$2"
      return 0
    }
  rm -f "$2.dxg-fallback.$$"
  say "  $4: WARNING -- could not record the dxg window ($3); its record fields say no window"
  return 1
}

# Which guest is on the box now, or nothing if it could not be asked inside the window.
#
# Written to a file rather than captured in a command substitution, for the reason collect_dxg_window
# gives about its own query: a substitution runs in a SUBSHELL, so the UNIT_PID that `run_capped`
# publishes there is invisible to the lane -- a cancellation could then neither relay TERM to the
# supervisor nor reap it, and it would hold the inherited locks until its own cap expired. The
# pause between attempts goes through run_capped for exactly the same reason: a bare `sleep` is
# invisible to the lane's trap, and an orphaned one keeps the lab reservation and the worktree lock
# alive after the lane has gone.
# The answer is LEFT IN THE FILE rather than printed, and the callers read it from there: a
# function that printed it would have to be called in a command substitution, which is the very
# subshell this is avoiding. GUEST_ID carries it for a caller that wants one line instead.
remote_guest_id() { # host scratch-path -- leaves the boot id in the file, and in $GUEST_ID
  local host=$1 probe=$2 deadline=$(( SECONDS + DXG_IDENTITY_WINDOW ))
  GUEST_ID=
  while :; do
    run_capped "$DXG_IDENTITY_CAP" ssh -o BatchMode=yes -o ConnectTimeout=8 \
      "$host" 'cat /proc/sys/kernel/random/boot_id 2>/dev/null' >"$probe" 2>/dev/null
    read -r GUEST_ID < "$probe" 2>/dev/null || GUEST_ID=
    [ -n "$GUEST_ID" ] && return 0
    [ "$SECONDS" -ge "$deadline" ] && return 0
    run_capped "$(( DXG_IDENTITY_PAUSE + 5 ))" sleep "$DXG_IDENTITY_PAUSE"
  done
}

collect_dxg_window() { # host log remote-start-epoch label start-boot-id
  local host=$1 log=$2 remote_start=$3 label=$4 start_boot=${5:-}
  local kernel rc bounds start_utc end_utc sidecar end_boot boot=unknown
  sidecar=$(dxg_sidecar "$log")
  # No start instant from the box means no window to bound. Reported as a failed
  # collection, which is what it is, rather than guessed.
  if [ -z "$remote_start" ]; then
    publish_dxg_sidecar "$sidecar" "$log" dxg_window_unavailable - - \
      "no clock reading from $host" ||
      publish_dxg_unavailable_fallback "$sidecar" "$log" "no clock reading from $host" "$label"
    return 0
  fi
  # Written to a file rather than captured in a command substitution, which runs
  # in a SUBSHELL: the UNIT_PID `run_capped` publishes would be invisible to the
  # lane, so a cancellation could neither relay TERM to this supervisor nor reap
  # it, and it would hold the inherited lock until its own cap expired. The
  # unit's own ssh calls document the same trap.
  kernel=$log.dxg.$$
  run_capped "$(( CONTEXT_CAP + 60 ))" ssh -o BatchMode=yes -o ConnectTimeout=8 \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
    "$host" "$(remote_capped "$CONTEXT_CAP" "$(dxg_window_cmd "$remote_start")")" \
    >"$kernel" 2>/dev/null
  rc=$?
  # The bounds the REMOTE used, in its own clock domain -- the only one the log's
  # timestamps are in. Reported rather than recomputed here, so the block and the
  # record row name the window that was actually queried.
  bounds=$(sed -n 's/^dxg-window-bounds \([0-9][0-9]*\) \([0-9][0-9]*\)$/\1 \2/p' \
    "$kernel" 2>/dev/null | head -1)
  # Which guest answered THIS collection, against the one the unit started on. The
  # comparison is three-valued on purpose: `replaced` is a finding, `same` is a
  # finding, and a boot id missing from either end is neither -- a box that cannot
  # report one is exactly as trustworthy as it was before this check existed, and
  # turning that into an alarm would retire a working window on every host whose
  # kernel does not publish the file.
  end_boot=$(sed -n 's/^dxg-window-boot \(.*\)$/\1/p' "$kernel" 2>/dev/null | head -1)
  # A collection that failed leaves no boot id, and the guest most likely to refuse it is exactly
  # the one this exists to catch: a replacement drops the unit's connection the moment the old VM
  # dies, and the new one is often still starting when the collector arrives. So ask the cheap
  # question on its own -- WHICH guest is there now -- rather than retrying the whole collection.
  #
  # Retrying the collection is what the previous round did, and it multiplied the wrong budget: the
  # window query carries the five-minute diagnostic cap, so three attempts at it could hold the lane
  # and its lab reservation for the better part of twenty minutes whenever the remote WEDGED rather
  # than failing fast. This probe reads one small file, so its own short cap bounds each attempt no
  # matter how the far side misbehaves, and three of them cannot add up to more than a minute and a
  # half. There is no sleep between attempts either: a bare `sleep` here is invisible to the
  # cancellation supervisor, so a TERM would leave it holding the inherited lab and worktree locks
  # after the lane had gone -- and the connect timeout of a box that is still down already spaces
  # the attempts by about as much as a sleep would.
  if [ -z "$end_boot" ] && [ -n "$start_boot" ]; then
    remote_guest_id "$host" "$log.boot.$$"
    end_boot=$GUEST_ID
    rm -f "$log.boot.$$"
  fi
  if [ -n "$start_boot" ] && [ -n "$end_boot" ]; then
    if [ "$start_boot" = "$end_boot" ]; then boot=same; else boot=replaced; fi
  fi
  if [ -n "$bounds" ]; then
    start_utc=$(utc_of "${bounds%% *}")
    end_utc=$(utc_of "${bounds##* }")
  else
    # No bounds line means the far side never got as far as printing one, so there
    # is no window to report and nothing trustworthy to filter.
    start_utc=-
    end_utc=-
    [ "$rc" -eq 0 ] && rc=1
  fi
  # A collection that did not happen is NOT a clean window. The ssh can time out
  # or lose the connection after a unit that ran for an hour, and an empty answer
  # filtered as kernel lines would report zero bursts -- "the bridge was fine" --
  # over a box nobody read, losing the rerun for exactly the unlisted failure this
  # trigger exists to catch.
  # The block is written to the SIDECAR, which is the collector's own channel and
  # the only one the trigger, the fingerprint and the record read, and copied into
  # the log for whoever reads that. See dxg_sidecar for why provenance cannot come
  # from the log itself.
  if [ "$rc" -ne 0 ]; then
    rm -f "$kernel"
    publish_dxg_sidecar "$sidecar" "$log" dxg_window_unavailable "$start_utc" "$end_utc" \
      "kernel log unreadable on $host (exit $rc)" "$boot" ||
      publish_dxg_unavailable_fallback "$sidecar" "$log" \
        "kernel log unreadable on $host (exit $rc)" "$label" "$boot"
  else
    # Filtered HERE rather than on the far side: the filter is the part with a
    # judgement in it, so it belongs where a fixture can feed it lines directly
    # instead of behind an ssh no test can reach. The publication's status is read
    # BEFORE the cleanup below, which would otherwise replace it with its own.
    publish_dxg_sidecar "$sidecar" "$log" dxg_window_summary "$start_utc" "$end_utc" "$boot" <"$kernel"
    rc=$?
    rm -f "$kernel"
    [ "$rc" -eq 0 ] ||
      publish_dxg_unavailable_fallback "$sidecar" "$log" \
        "the collected window could not be published" "$label" "$boot"
  fi
}

# Collect a remote GPU unit's dxg window and say what it shows about the guest.
#
# Factored out because it has to happen on EVERY path that ends a remote unit, not only the one
# that ran dune. A guest replaced during the up-to-600s remote PREPARATION ends the unit through
# the preparation's own `error` return, which collected nothing at all -- so an uncoordinated
# reboot in that window produced neither `vm-replaced` evidence nor the warning, on the path least
# likely to be looked at afterwards. Any future early return from a remote unit belongs here too.
#
# Called BEFORE write_fingerprint on each path, so the fingerprint carries the window: a replaced
# guest is part of what distinguishes this failure from the same failure on a healthy box.
finish_remote_window() { # machine backend host log outcome remote-start-epoch start-boot-id
  local machine=$1 backend=$2 host=$3 log=$4 outcome=$5 remote_start=$6 start_boot=$7
  REMOTE_GUEST_REPLACED=0
  # A local unit has no remote guest, and a unit that never ran (`skip`) has nothing to account for.
  [ -n "$host" ] || return 0
  case $outcome in skip) return 0 ;; esac
  # The dxg WINDOW is GPU-only -- the bridge is what /dev/dxg is, so minix's multidev_cc (CPU, on a
  # WSL box) would only ever collect another unit's noise -- but the GUEST is not. multidev_cc runs
  # in the same replaceable VM as hip, so a replacement takes it down just the same, and before
  # this it was recorded as a bare `error` with nothing saying the machine had gone. The two
  # questions are separate and only one of them is about the bridge, so only one of them is gated
  # on the backend.
  case $backend in
    cuda | hip)
      collect_dxg_window "$host" "$log" "$remote_start" "$machine/$backend" "$start_boot"
      ;;
    *)
      # No sidecar for a non-GPU unit: the dxg channel is the bridge's, and writing one here would
      # make a CPU unit environment-red through dxg_window_red and put a bridge verdict in its
      # record row. The identity answer is reported below instead, which is the part an operator
      # acts on.
      if [ -n "$start_boot" ]; then
        remote_guest_id "$host" "$log.boot.$$"
        [ -n "$GUEST_ID" ] && [ "$GUEST_ID" != "$start_boot" ] && REMOTE_GUEST_REPLACED=1
        rm -f "$log.boot.$$"
      fi
      ;;
  esac
  # A replaced guest on an outcome that cannot be rerun. `vm-replaced` makes a unit
  # environment-red, and on a `fail` that is the whole story: serial_rerun reads the window and the
  # unit gets its second run. On an `error` -- which is what a mid-unit replacement actually
  # produces, the ssh dying with the VM and run_capped returning 255 -- there is nothing for
  # serial_rerun to do: it reruns the FAILING STANZAS under -j 1, and an error never reached dune,
  # so no stanza was ever recorded. Rerunning the whole unit is a different mechanism, and not one
  # to add here: the row and the elapsed time are already published, so a second attempt would need
  # a second history row for one unit in one run, which is exactly the kind of second channel the
  # record's own design refuses (see write_run_record).
  #
  # What was missing is that nobody was TOLD. The sidecar knew, and on 2026-09-16 the operator did
  # not: two units read `error (818s)` and the day went to the GPU. So say it where the scheduled
  # routine quotes the sweep's output, for the reason the empty fingerprint is said there too
  # (gh-ocannl-792) -- a finding that lives only in a written file is one nobody reads.
  case $outcome in
    error | timeout)
      { [ "$REMOTE_GUEST_REPLACED" = 1 ] || dxg_guest_replaced "$log"; } &&
        say "  $machine/$backend: the guest was REPLACED mid-unit -- this unit tested nothing, and its result is about the box, not the code; rerun it (reruns are incremental)"
      ;;
  esac
  return 0
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
#
# Lanes record concurrently, so the append takes an exclusive flock on the
# history file for the one write of the whole row: a row is never interleaved
# with another lane's. Rows land in COMPLETION order, not table order; every
# consumer keys on the machine/backend columns (the routine's diff and staleness
# steps take the most recent row per backend), none on position within a run.
#
# This append is also what the run record reads a unit's outcome from: the record
# derives its unit rows from the rows of THIS stamp rather than from a second
# per-unit channel written beside them (see write_run_record).
record() {
  local row
  row=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
    "$stamp" "$1" "$2" "$run_sha" "$3" "$4" "${TARGET:-<all>}" "$SLOW" \
    "${5:--}" "${6:-none}")
  perl -e 'use Fcntl ":flock";
    open(my $h, ">>", $ARGV[0]) or exit 1;
    flock($h, LOCK_EX) or exit 1;
    print $h "$ARGV[1]\n" or exit 1;
    close($h) or exit 1;' "$HISTORY" "$row" ||
    die "cannot record $1/$2 outcome in $HISTORY"
}

# The per-run record (gh-ocannl-977): one machine-readable file beside the logs,
# holding the three facts a consumer otherwise re-derives from this script's
# stdout prose -- which kind of exit happened, which rows a stopped lane still
# wrote, and which box runs a backend TODAY. The scheduled routine that gates
# five backends reconstructed all three by parsing summary lines against
# history.tsv, and every sweep feature (lanes, a moved backend) reopened the
# derivation; five review rounds went into keeping that prose true.
#
# A SEPARATE file rather than columns on history.tsv: history rows are per unit
# and append-only, so the run-level exit kind and today's backend->box map have
# no unit row to live on -- and an ABSENT record is exactly the startup-refusal
# signal. A run that died before any lane started swept nothing and says so by
# leaving no file; a cancellation that early is indistinguishable from it and
# means the same thing.
#
# Kind-tagged rows, as the unit-state files are: one `run` row; one `unit` row
# per SELECTED unit, so `no-row` names a unit that should have run and did not
# rather than one `--only` excluded; and one `backend` row per unit of the
# table, selected or not, because staleness must be aged by rows from the box
# that runs that backend today whether or not this run touched it. Columns are
# documented in docs/agent-notes/build-and-test.md.
#
# A unit's outcome comes from the HISTORY ROWS OF THIS RUN, read back under the
# same lock that writes them -- not from a per-unit file staged beside them. That
# is what makes `no-row` mean exactly "no history row": with a second channel the
# two can disagree in both directions, and every way of ordering the two writes
# leaves a window where a signal lands between them (a lane's deferred TERM trap,
# a group TERM reaching the writer itself) and the record then either hides a
# real outcome or asserts one that was never recorded. There is no ordering that
# closes that, so there is no second channel: the row IS the evidence, and the
# stamp is what makes the rows of this run identifiable (hence the advance).
#
# Reading with a shared lock, since a lane may be appending: without it the last
# line can be read half-written.
run_rows() { # -> machine, backend, outcome, log for each row of this run
  perl -e 'use Fcntl ":flock";
    open(my $h, "<", $ARGV[0]) or exit 1;
    flock($h, LOCK_SH) or exit 1;
    while (my $line = <$h>) {
      chomp $line;
      my @f = split(/\t/, $line, -1);
      next unless @f >= 9 && $f[0] eq $ARGV[1];
      print join("\t", $f[1], $f[2], $f[4], $f[8]), "\n" or exit 1;
    }
    close($h) or exit 1;' "$HISTORY" "$stamp"
}

# Given its path once the lanes have been forked, which is what makes a record
# owed at all: before that nothing has been swept and the absence is the signal.
write_run_record() { # exit-kind -- complete | lane-stopped | cancelled | post-run-failed
  local kind=$1 unit machine backend host outcome log stopped stage rows
  local window window_start window_end bursts
  stage=$RUN_RECORD.stage.$$
  rows=$(run_rows) || return 1
  {
    # Schema 3: that count gained a fourth value, `vm-replaced` (a guest that
    # was destroyed and recreated mid-window). A schema-2 consumer's contract
    # permits only `-`, a number and `unavailable`, so a strict one would reject
    # the record at exactly the moment a replacement happened -- the record it
    # most needs to read. Schema 2: the `unit` row gained the dxg window and burst count
    # (gh-ocannl-979). A consumer picks its parser from this number, so widening a
    # row without it would make a strict schema-1 reader reject a current record
    # and a dxg-aware reader mis-read a historical one. The unit-STATE files below
    # keep their own schema 1: different file, different contract.
    printf 'schema\t3\n'
    printf 'run\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$stamp" "$run_sha" "$REF" "${TARGET:-<all>}" "$SLOW" "$execution" "$kind"
    for unit in "${UNITS[@]}"; do
      IFS=: read -r machine backend host <<<"$unit"
      wanted "$backend" || continue
      # A lane publishes its completion marker as its last act, so its absence
      # covers every way a lane can fail to finish -- its own `die`, a signal
      # relayed to it -- and the top level removes the marker of any lane whose
      # wait status was nonzero, so the flag cannot disagree with the exit kind.
      if [ -e "$LANE_DIR/lane-done.$machine" ]; then stopped=0; else stopped=1; fi
      # The last row this run wrote for the unit, or none. A unit is recorded
      # once, so `tail -1` only matters if a future change records twice: the
      # later row is then the current verdict, which is what the history's own
      # consumers take too.
      outcome=$(printf '%s\n' "$rows" |
        awk -F '\t' -v m="$machine" -v b="$backend" \
          '$1 == m && $2 == b { o = $3; l = $4 } END { if (o != "") print o "\t" l }')
      if [ -z "$outcome" ]; then
        log=-
        outcome=no-row
      else
        log=${outcome#*$'\t'}
        outcome=${outcome%%$'\t'*}
        [ -n "$log" ] || log=-
      fi
      # The unit's dxg window and burst count (gh-ocannl-979), read from the log
      # the row names -- the only artifact that holds them, and one this row
      # already points at, so no third place can disagree. `-` for a unit with
      # no window: a local one, or one that never ran.
      # From the unit's collected-evidence sidecar, never from its log: a log
      # holds whatever the unit's tests printed (see dxg_sidecar).
      window_start=-
      window_end=-
      bursts=-
      if [ "$log" != - ]; then
        window=$(dxg_window_bounds "$log")
        if [ -n "$window" ]; then
          window_start=${window%% *}
          window_end=${window##* }
          # `-` no window, a number a window that was read, `unavailable` a
          # window whose collection failed: three distinguishable states, since
          # "not collected" and "collected and clean" mean opposite things. A
          # collection that failed before the box reported its bounds writes `-`
          # for both of those and `unavailable` here, which is still that state.
          bursts=$(dxg_bursts "$log")
          [ -n "$bursts" ] || bursts=-
        fi
      fi
      printf 'unit\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$machine" "$backend" \
        "$outcome" "$stopped" "$log" "$window_start" "$window_end" "$bursts"
    done
    for unit in "${UNITS[@]}"; do
      IFS=: read -r machine backend host <<<"$unit"
      printf 'backend\t%s\t%s\n' "$backend" "$machine"
    done
  } >"$stage" && mv "$stage" "$RUN_RECORD" || {
    rm -f "$stage"
    return 1
  }
}

# A unit's summary lines. Written to its lane's buffer rather than to stdout, and
# published as one block when the unit finishes (flush_lane_output), so that a
# unit's lines stay contiguous and a line is never split by another lane's.
say() { printf '%s\n' "$*" >>"$LANE_OUT" || die "cannot buffer sweep output in $LANE_OUT"; }

# The error SITES in a log, one per line, in BOTH of dune's spellings: a
# diagnostic anchored to one line says `line N`, one anchored to a span --
# notably a whole stanza whose action exited non-zero, which is how every
# explicit-rule test here fails -- says `lines N-M`. Shared by `fingerprint`,
# which sorts and bounds them, and by `rerun_aliases`, which needs every one.
dune_sites() { # log
  {
    # Matching only the singular left a unit whose ONLY failure
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
      function clear_names( i) {
        for (i in names) delete names[i]
        names_count = 0
      }
      function flush( i) {
        if (loc == "") return
        if (name != "") print prefix ", " name
        else if (names_count > 0) {
          for (i = 1; i <= names_count; i++) print prefix ", names " names[i]
        } else print loc
        loc = ""; name = ""; want = ""; opened = 0; names_done = 0
        clear_names()
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
        if (name != "" || names_done) next
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
          if (tok[i] == "(") {
            opened = 1
            if (want != "names") want = ""
            continue
          }
          if (tok[i] == ")") {
            if (want == "names" && names_count > 0) names_done = 1
            opened = 0; want = ""
            if (names_done) break
            continue
          }
          if (want == "names") { names[++names_count] = tok[i]; continue }
          if (want != "") { name = want " " tok[i]; break }
          if (opened && tok[i] ~ /^(alias|name|names|target|targets)$/) want = tok[i]
          opened = 0
        }
        next
      }
      END { flush() }
    ' "$1"
  } 2>/dev/null
}

# A compact, diffable summary of what went wrong, so a caller can tell a NEW
# failure from a standing one. Metal's operations suite carries known-red tests,
# and a sweep that shouts on every red is a sweep nobody reads.
fingerprint() {
  {
    dune_sites "$1"
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
  # The dxg window's STABLE half (dxg_fingerprint_lines): which signatures the
  # window held, and whether the bridge was losing messages at all. Not the block
  # verbatim -- the window instants, the kernel timestamps and the exact count all
  # differ between two equally broken runs, and a fingerprint is compared bytewise
  # against the previous failure's, so the verbatim block would report `fingerprint
  # moved` on every repeat of a standing environment red, costing the suppression
  # that keeps this output readable. The full block stays in the log, and the
  # window and count are fields of the run record.
  # UNCAPPED, unlike everything above it, and deliberately: this list is already
  # deduplicated, so it is bounded by the number of distinct kernel message shapes
  # the bridge can produce -- a handful, where the raw lines it summarises run to
  # hundreds. A cap here would drop exactly what the list exists for, a signature
  # never seen before, and would do it to the lexicographically last ones, which is
  # no one's idea of the least interesting. The verdict follows them for the reason
  # the serial rerun's line does: it is the one line that must survive.
  dxg_fingerprint_lines "$1"
  # The serial rerun's verdict (serial_rerun), after the sorted block and
  # outside its bound: which of the red stanzas stayed red on their own is the
  # first line a reader of an environment-red unit needs, and the one a
  # 60-entry bound must not be able to drop.
  grep -h '^serial rerun: ' "$1" 2>/dev/null
}

# The rerun targets behind a log's dune-file and inline-expectation sites, one
# per line, each prefixed `alias `, `inline ` or `unmapped `. A `(test (name
# X))` stanza reruns as
# `@<dir>/runtest-X`, the per-test alias dune generates from 3.20 (the
# project's floor); an explicit rule as `@<dir>/<its alias>`. An inline
# expectation is proved by its source-to-`.corrected` diff and tagged with its
# directory's broad `runtest` alias: serial_rerun uses that fallback only when
# one or two such sites were unmapped, so a small environment-red unit still
# gets a serial retry without turning a wider red into a serial directory
# suite. Other source locations, unnamed spans and bare targets stay unmapped.
rerun_aliases() { # log
  dune_sites "$1" | sort -u | awk -v log_file="$1" '
    BEGIN {
      while ((getline raw < log_file) > 0) {
        if (raw ~ /^diff --git a\// && raw ~ /\.ml b\/_build\/default\// &&
            raw ~ /\.ml\.corrected$/) {
          split(raw, parts, /[ \t]+/)
          source = parts[3]
          corrected = parts[4]
          sub(/^a\//, "", source)
          if (corrected == "b/_build/default/" source ".corrected") inline[source] = 1
        }
      }
      close(log_file)
    }
    /^File "[^"]*dune", (alias|name|names) "?[A-Za-z0-9_.-]+"?$/ {
      dir = $2
      sub(/^"/, "", dir)
      sub(/dune",$/, "", dir)
      if (dir ~ /^([A-Za-z0-9_.-]+\/)*$/) {
        value = $4
        sub(/^"/, "", value)
        sub(/"$/, "", value)
        if ($3 == "alias") print "alias @" dir value
        else print "alias @" dir "runtest-" value
        next
      }
    }
    /^File "([A-Za-z0-9_.-]+\/)*[A-Za-z0-9_.-]+\.ml", lines? [0-9]+(-[0-9]+)?$/ {
      path = $2
      sub(/^"/, "", path)
      sub(/",$/, "", path)
      dir = path
      sub("[^/]+$", "", dir)
      if (inline[path]) {
        print "inline @" dir "runtest\t" $0
        next
      }
    }
    { print "unmapped " $0 }
  '
}

# The rerun as shell text for the machine that owns the worktree, one dune call
# per stanza so each has its own status: a single call over all of them would
# report one verdict for the set. The markers are what serial_rerun reads back.
serial_rerun_cmd() { # backend wt alias...
  local backend=$1 wt=$2 a
  shift 2
  printf 'cd "%s" || exit 127; ' "$wt"
  for a in "$@"; do
    printf 'echo "=== serial rerun %s ==="; ' "$a"
    printf 'OCANNL_BACKEND=%s opam exec -- dune build -j 1 %s; ' "$backend" "$a"
    printf 'echo "=== serial rerun %s: exit $? ==="; ' "$a"
  done
  printf 'exit 0'
}

# Rerun an environment-red unit's failing stanzas one at a time, appending to
# its log, then write the `serial rerun:` verdict lines that `fingerprint`
# carries and the summary quotes. The shape is collect_rtc_context's: its own
# phase after the row is recorded, under the worktree lock on whichever side
# owns the tree, with a status that never reaches the outcome. The budget is the
# unit's own CAP rather than CONTEXT_CAP, because this is not a diagnostic of
# fixed size but the suite's red stanzas run again -- 27 of them on the day this
# was measured -- and a stanza the cap cut short is reported `unjudged`, never
# folded into `all clean`.
serial_rerun() { # backend host wt log label [path_prefix]
  local backend=$1 host=$2 wt=$3 log=$4 label=$5 path_prefix=${6:-}
  local line cmd started rc a entry site stanza_count inline_count fallback_suffix=s
  local aliases=() fallback_aliases=() inline_entries=() inline_sites=()
  local unmapped=() red=() unjudged=()
  environment_red "$log" || return 0
  while IFS= read -r line; do
    case $line in
      "alias "*) aliases+=("${line#alias }") ;;
      "inline "*)
        entry=${line#inline }
        inline_entries+=("${entry%%$'\t'*}")
        inline_sites+=("${entry#*$'\t'}")
        ;;
      "unmapped "*) unmapped+=("${line#unmapped }") ;;
    esac
  done < <(rerun_aliases "$log")
  stanza_count=${#aliases[@]}
  inline_count=${#inline_sites[@]}
  if [ "$inline_count" -gt 0 ] && [ "$inline_count" -le 2 ]; then
    for a in "${inline_entries[@]}"; do
      contains "$a" "${fallback_aliases[@]:-}" || fallback_aliases+=("$a")
    done
    aliases+=("${fallback_aliases[@]}")
  elif [ "$inline_count" -gt 2 ]; then
    for site in "${inline_sites[@]}"; do unmapped+=("$site"); done
  fi
  started=$(date +%s)
  if [ ${#aliases[@]} -gt 0 ]; then
    cmd=$(serial_rerun_cmd "$backend" "$wt" "${aliases[@]}")
    echo "=== serial rerun: ${#aliases[@]} stanzas at -j 1 ===" >>"$log"
    if [ -n "$host" ]; then
      run_capped "$(( CAP + 300 ))" ssh -o BatchMode=yes -o ConnectTimeout=8 \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=10 \
        "$host" "$(remote_capped "$CAP" "$path_prefix $(remote_lock_cmd "$wt") $cmd")" \
        >>"$log" 2>&1
    else
      run_capped "$CAP" /bin/sh -c "$cmd" >>"$log" 2>&1
    fi
    rc=$?
    for a in "${aliases[@]}"; do
      line=$(grep -hF -- "=== serial rerun $a: exit " "$log" | tail -1)
      case $line in
        "") unjudged+=("$a") ;;
        *": exit 0 ==="*) ;;
        *) red+=("$a") ;;
      esac
    done
  else
    rc=0
  fi
  {
    if [ ${#red[@]} -gt 0 ]; then
      printf 'serial rerun: still red:'
      printf ' %s' "${red[@]}"
      printf '\n'
    elif [ ${#unjudged[@]} -eq 0 ] && [ ${#aliases[@]} -gt 0 ]; then
      printf 'serial rerun: all clean\n'
    fi
    if [ ${#unjudged[@]} -gt 0 ]; then
      printf 'serial rerun: unjudged (exit %s):' "$rc"
      printf ' %s' "${unjudged[@]}"
      printf '\n'
    fi
    if [ ${#fallback_aliases[@]} -gt 0 ]; then
      printf 'serial rerun: directory fallback (%s inline site%s):' \
        "$inline_count" "$([ "$inline_count" -eq 1 ] || printf s)"
      printf ' %s' "${fallback_aliases[@]}"
      printf '\n'
    fi
    if [ ${#aliases[@]} -eq 0 ]; then
      printf 'serial rerun: nothing to rerun -- no site names a stanza\n'
    fi
    if [ ${#unmapped[@]} -gt 0 ]; then
      printf 'serial rerun: unmapped:'
      printf ' [%s]' "${unmapped[@]}"
      printf '\n'
    fi
  } >>"$log"
  [ ${#fallback_aliases[@]} -eq 1 ] && fallback_suffix=
  say "  $label: environment-red, $stanza_count stanzas and ${#fallback_aliases[@]} directory fallback$fallback_suffix rerun at -j 1 ($(( $(date +%s) - started ))s)"
  grep -h '^serial rerun: ' "$log" | sed "s|^|  $label: |" >>"$LANE_OUT"
  return 0
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
    say "  $label: $EMPTY_FINGERPRINT -- $log"
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
        say "  $label: REGRESSION OR FIX DID NOT TAKE -- previous verdict was $previous_verdict"
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
        say "  $label: fingerprint moved since the previous failure at $short"
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
          say "  $label: REGRESSION OR FIX DID NOT TAKE -- $path last changed at $short (previous failing copy: $old_short)"
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

# One unit, start to finish: preparation, the capped suite, the recorded row, and
# the post-unit phases (RTC context, serial rerun, fingerprint, unit state), all
# on the machine that owns the unit and inside its lane. The post-unit phases
# re-take that machine's worktree lock, so they must not be split off to run
# after the lane has moved on to its next unit.
run_unit() { # machine backend host
  local machine=$1 backend=$2 host=$3
  local log started remote_home remote_probe remote_started remote_boot wt path_prefix= remote_repo
  local remote_prep remote rc elapsed outcome
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
    # wedges its shell would hang this unit's lane here, before the unit records
    # anything, so every ssh in a unit gets an outer bound as well.
    #
    # The one step NOT routed through run_capped: its output is captured, and
    # command substitution runs in a subshell, so a UNIT_PID published there
    # would be invisible to the lane's trap. Its 60s budget bounds how
    # long a cancellation can be delayed here, which is the reason that is
    # tolerable where a 900s preparation leg was not.
    # The probe reads the remote's CLOCK as well as its home, on the same round
    # trip. The dxg window's start has to be an instant in the clock that
    # timestamps that box's kernel log, and this is the moment the unit begins on
    # it; deriving it later by subtracting a locally measured duration assumes the
    # remote clock advanced continuously meanwhile, which is the assumption a WSL
    # VM breaks when it resynchronises after a host resume.
    if ! remote_probe=$(capped 60 ssh -o BatchMode=yes -o ConnectTimeout=8 \
         -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
         "$host" 'printf "%s\n%s\n%s\n" "$HOME" "$(date +%s)" \
           "$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"' 2>/dev/null) ||
       [ -z "$remote_probe" ]; then
      say "  $machine/$backend: skip (unreachable)"
      record "$machine" "$backend" skip 0
      update_unit_state "$machine" "$backend" skip
      return 0
    fi
    remote_home=$(printf '%s\n' "$remote_probe" | sed -n 1p)
    remote_started=$(printf '%s\n' "$remote_probe" | sed -n 2p)
    # WHICH guest, not just which host. A WSL2 VM that is destroyed and recreated
    # mid-unit comes back at the same alias with the same hostname and the same
    # home directory, so nothing else the collection can see tells the two apart
    # -- and the window it then reports spans both boots. Read on the same round
    # trip as the clock, and compared against the same reading at collection time.
    remote_boot=$(printf '%s\n' "$remote_probe" | sed -n 3p)
    [ -n "$remote_home" ] || {
      say "  $machine/$backend: skip (unreachable)"
      record "$machine" "$backend" skip 0
      update_unit_state "$machine" "$backend" skip
      return 0
    }
    # A box whose `date` said nothing leaves no window to bound; the collection
    # below reports that as unavailable rather than guessing one.
    case $remote_started in "" | *[!0-9]*) remote_started= ;; esac
    # The NEXT second, not the probe's own. `date +%s` floors, so the probe's
    # second also contains whatever happened earlier in it -- the previous unit's
    # RTC diagnostic crossing the same bridge, say -- and including it would mark
    # this unit environment-red for a burst that predates it, buying a rerun and
    # filing its real failures under "environment". Rounding the other way can
    # only lose a sub-second sliver in which this unit has done nothing yet: its
    # next act is another ssh round trip, and nothing touches the GPU until dune
    # runs. The end bound rounds outward for the mirror-image reason, so the
    # window is closed on both sides against its neighbours.
    [ -n "$remote_started" ] && remote_started=$(( remote_started + 1 ))
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
      say "  $machine/$backend: error (cannot pin $host to $run_sha)"
      record "$machine" "$backend" error "$(( $(date +%s) - started ))" "$log"
      # The preparation can run for ten minutes, and a guest replaced inside it ends the unit
      # right here -- so this path collects the window exactly like the one below.
      finish_remote_window "$machine" "$backend" "$host" "$log" error \
        "${remote_started:-}" "${remote_boot:-}"
      write_fingerprint "$log" "$machine/$backend"
      update_unit_state "$machine" "$backend" error "$WRITTEN_FINGERPRINT"
      return 0
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
    # serveraliveinterval 0 and connecttimeout none), and because the lane waits
    # on the unit, the lane stalls behind it: no later units on that box, no rows
    # -- and no end of the sweep, which waits for every lane.
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
      say "  $machine/$backend: error (cannot pin $wt to $run_sha)"
      record "$machine" "$backend" error "$(( $(date +%s) - started ))" "$log"
      write_fingerprint "$log" "$machine/$backend"
      update_unit_state "$machine" "$backend" error "$WRITTEN_FINGERPRINT"
      return 0
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
  say "  $machine/$backend: $outcome (${elapsed}s; execution=$execution)"
  record "$machine" "$backend" "$outcome" "$elapsed" "$log" "$execution"
  # The lane is a subshell, so the evidence cannot be appended to the top-level
  # SKIP_RUN_ arrays from here; it is left as a per-unit file that the top level
  # reads back, in table order, once every lane has finished.
  if [ "$outcome" = pass ] && [ -z "$TARGET" ]; then
    printf '%s\n' "$log" >"$LANE_DIR/skip-run.$machine.$backend" ||
      die "cannot stage skip evidence for $machine/$backend"
  fi
  # The kernel's own dxg evidence for THIS unit's window, before the rerun
  # decision that reads it -- and before the RTC diagnostics below, which is not
  # mere ordering: `nvidia-smi` and `rocminfo` cross /dev/dxg themselves, so a
  # window whose end were taken after them could count the DIAGNOSTIC's lost
  # messages as the unit's and mark a plainly test-logic failure environment-red.
  # The end bound is the remote's clock at this point, so those fall outside it.
  # Remote GPU units only: the bridge is what /dev/dxg is, so a local unit has no
  # window and minix's multidev_cc -- CPU, on a WSL box -- would only ever collect
  # another unit's noise. A unit that never ran (`skip`) has no window either.
  # The window's start is the remote's own clock at the unit's beginning, read by the
  # reachability probe; its end is that same clock at collection time. Both ends therefore come
  # from the clock that timestamps the log, and nothing is reconstructed from a duration measured
  # on another machine. Same helper as the preparation-failure path above, so the two cannot drift.
  finish_remote_window "$machine" "$backend" "$host" "$log" "$outcome" \
    "${remote_started:-}" "${remote_boot:-}"
  # Diagnosis, strictly after the row and the elapsed time it reports: this phase
  # has its own budget, and nothing it does can reach $outcome or $elapsed. It
  # runs before the fingerprint so that what it appends to the log is carried in.
  case $outcome:$backend in
    fail:cuda | fail:hip | fail:metal)
      collect_rtc_context "$backend" "$host" "$wt" "$log" "${path_prefix:-}"
      ;;
  esac
  # Only a `fail` can be environment-red: a `timeout` had its process group
  # destroyed and may still hold the box, and an `error` never reached dune.
  # Gated inside on the signature table AND on the collected window, so a red
  # whose failures are the tests' own gets no second run.
  case $outcome in
    fail) serial_rerun "$backend" "$host" "$wt" "$log" "$machine/$backend" "${path_prefix:-}" ;;
  esac
  case $outcome in
    fail | timeout | error) write_fingerprint "$log" "$machine/$backend" ;;
  esac
  update_unit_state "$machine" "$backend" "$outcome" "${WRITTEN_FINGERPRINT:-}" "$log"
  WRITTEN_FINGERPRINT=
}

# Publish the lane's buffered unit lines as one block, under a run-wide lock so
# two lanes finishing together cannot interleave their blocks. The buffer is
# emptied only after a publication that succeeded: a failed one returns non-zero
# with the lines still in it, for the caller to die over and lane_exit to rescue.
flush_lane_output() {
  [ -s "$LANE_OUT" ] || return 0
  perl -e 'use Fcntl ":flock";
    open(my $l, ">>", $ARGV[0]) or exit 1;
    flock($l, LOCK_EX) or exit 1;
    open(my $in, "<", $ARGV[1]) or exit 1;
    print while <$in>;
    close($in) or exit 1;
    close(STDOUT) or exit 1;' "$LANE_DIR/output.lock" "$LANE_OUT" || return 1
  : >"$LANE_OUT"
}

# The lane's EXIT trap. A lane that dies or is cancelled mid-unit still publishes
# what its unit had said -- the serial loop's lines were already on stdout by
# then -- or, if stdout itself is what failed, puts them on stderr rather than
# losing them. bash 3.2 runs a subshell's EXIT trap only on an explicit `exit`,
# which is why run_lane ends with one.
lane_exit() {
  flush_lane_output || cat "$LANE_OUT" >&2
}

# One machine's units, in table order, one at a time: the local units share one
# worktree (and fd 9's lock), a remote box's units share its far-side worktree
# and flock, and on every box they compete for the same CPU or GPU. Run as a
# background subshell per machine, so a lane owns its copies of the per-unit
# globals (UNIT_PID, WRITTEN_FINGERPRINT) outright.
#
# A lane cannot reach the lanes beside it: an unreachable box or an `error` is
# recorded as that unit's outcome and the lane simply moves on, as the serial
# loop did. The one way a lane ends early is `die` -- a row or state file that
# could not be written -- and that fails only its lane; the top level lets the
# others finish recording and then exits 2.
run_lane() { # machine -- only ever as a background job: it ends in `exit`
  local lane=$1 unit machine backend host lane_host= lab_box lab_lock_rc
  LANE_PIDS=
  IN_LANE=1
  UNIT_PID=
  LANE_OUT=$LANE_DIR/output.$lane
  trap 'relay 130' INT
  trap 'relay 143' TERM
  trap lane_exit EXIT
  # Reserve the box before the first unit touches it. A lane is one machine, so one reservation
  # covers all of its units, and it is held for the whole lane rather than per unit: the gap
  # between two units of the same lane is exactly when a restart would land, and a box handed back
  # between minix/hip and minix/multidev_cc is one the second unit can still lose underneath it.
  # Local lanes have no host and reserve nothing -- the lock is about the WSL VM, not the machine.
  for unit in "${UNITS[@]}"; do
    IFS=: read -r machine backend host <<<"$unit"
    [ "$machine" = "$lane" ] || continue
    wanted "$backend" || continue
    [ -n "$host" ] && { lane_host=$host; break; }
  done
  if [ -n "$lane_host" ]; then
    lab_box=$(lab_box_of "$lane_host")
    take_lab_lock "$lab_box"; lab_lock_rc=$?
    # A harness that cannot lock at all fails the lane rather than reporting contention it did not
    # observe: `skip (box ... reserved by ...)` over a read-only state directory is a local fault
    # wearing the costume of a legitimate one.
    [ "$lab_lock_rc" = 2 ] &&
      die "cannot take the lab lock for $lab_box under $LAB_LOCK_DIR (check the directory)"
    if [ "$lab_lock_rc" != 0 ]; then
      # Skipped, not errored: nothing was tested and nothing failed. `error` would fingerprint a
      # box someone else is legitimately working on as a broken one, and the run record's skip
      # coverage is already the channel for a backend that went untested.
      for unit in "${UNITS[@]}"; do
        IFS=: read -r machine backend host <<<"$unit"
        [ "$machine" = "$lane" ] || continue
        wanted "$backend" || continue
        say "  $machine/$backend: skip (box $lab_box reserved by $(lab_lock_holder "$lab_box"))"
        record "$machine" "$backend" skip 0
        update_unit_state "$machine" "$backend" skip
        flush_lane_output || die "cannot publish the $machine/$backend summary to stdout"
      done
      : >"$LANE_DIR/lane-done.$lane" || die "cannot mark the $lane lane finished"
      exit 0
    fi
  fi
  for unit in "${UNITS[@]}"; do
    IFS=: read -r machine backend host <<<"$unit"
    [ "$machine" = "$lane" ] || continue
    wanted "$backend" || continue
    run_unit "$machine" "$backend" "$host"
    flush_lane_output || die "cannot publish the $machine/$backend summary to stdout"
  done
  # The lane's completion marker, published as its last act: the run record
  # reads its absence as a lane that stopped before finishing, so nothing may
  # come between this and the `exit` -- and a lane that dies or is signalled
  # anywhere above leaves it absent without having to know the record exists.
  : >"$LANE_DIR/lane-done.$lane" || die "cannot mark the $lane lane finished"
  exit 0
}

# Lanes, in first-appearance order of the table's machine column: a new box or
# a second backend on an existing box lands in the right lane with no other edit.
LANES=()
for unit in "${UNITS[@]}"; do
  IFS=: read -r machine backend host <<<"$unit"
  wanted "$backend" || continue
  contains "$machine" "${LANES[@]:-}" || LANES+=("$machine")
done
# The --only check above guarantees at least one selected unit.
[ ${#LANES[@]} -gt 0 ] || die "no sweep unit selected"

# Run-scoped coordination files: the lanes' output buffers and lock, and the
# per-unit skip evidence. Removed on every exit of the top level; a lane's own
# exit must not remove it, and a subshell does not inherit this trap.
LANE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ocannl-sweep-lanes.XXXXXX") ||
  die "cannot create the lanes' coordination directory"
trap 'rm -rf "$LANE_DIR"' EXIT

lanes_summary=
for lane in "${LANES[@]}"; do
  lane_units=
  for unit in "${UNITS[@]}"; do
    IFS=: read -r machine backend host <<<"$unit"
    [ "$machine" = "$lane" ] && wanted "$backend" && lane_units=$lane_units${lane_units:+,}$backend
  done
  lanes_summary="$lanes_summary  $lane($lane_units)"
done

echo "sweep $stamp  ref=$REF ($run_sha)  slow=$SLOW  target=${TARGET:-<all>}  execution=$execution"
echo "lanes:$lanes_summary"
echo

# Registration is atomic with respect to the relay: a signal arriving between a
# lane's fork and its entry in LANE_PIDS would otherwise let the relay return
# past a lane it could not see, which would keep running -- and holding fd 9 --
# for the rest of its cap. Signals during the launch are only noted, and relayed
# once every lane is registered.
LANE_PID_LIST=()
pending_signal=
trap 'pending_signal=130' INT
trap 'pending_signal=143' TERM
for lane in "${LANES[@]}"; do
  run_lane "$lane" &
  LANE_PID_LIST+=("$!")
  LANE_PIDS="$LANE_PIDS $!"
done
# Only now is a record owed: a run that ends from here on -- completely, with a
# stopped lane, or cancelled -- swept something and says what. Armed after the
# fork loop rather than with the coordination directory, because a cancellation
# in between would otherwise publish a record of nothing but `no-row` units,
# contradicting the absence that a run which never started a lane must leave.
# The launch window itself only notes a signal (pending_signal above), so the
# relay below is the first thing that can reach this.
RUN_RECORD=$LOGS/$stamp-run.tsv
trap 'relay 130' INT
trap 'relay 143' TERM
[ -z "$pending_signal" ] || relay "$pending_signal"

# Wait for EVERY lane before anything that summarises the run. A lane's non-zero
# exit is its `die` (already reported on stderr), or a lane signalled on its own;
# either way its rows are incomplete, so no skip-coverage verdict is claimed.
failed_lanes=
for ((i = 0; i < ${#LANES[@]}; i++)); do
  wait "${LANE_PID_LIST[$i]}"
  lane_rc=$?
  if [ "$lane_rc" -ne 0 ]; then
    failed_lanes="$failed_lanes ${LANES[$i]} (exit $lane_rc)"
    # The wait status outranks the marker. A lane signalled or killed AFTER
    # publishing its marker but before its shell exited would otherwise be
    # recorded `stopped=0` inside a `lane-stopped` run -- a record contradicting
    # itself and naming no failed lane. Where no status exists to outrank it (the
    # cancellation path, which never reaches this loop), the marker keeps its own
    # meaning: a lane that published it had finished its units.
    rm -f "$LANE_DIR/lane-done.${LANES[$i]}"
  fi
done
LANE_PIDS=

# The record is written before anything that summarises the run, and on BOTH
# paths out of it: a lane-stopped exit 2 is exactly the case where the consumer
# most needs to know which rows are real, and it is also the case a script that
# wrote the record only on the happy path would leave unexplained.
run_exit_kind=complete
[ -z "$failed_lanes" ] || run_exit_kind=lane-stopped
write_run_record "$run_exit_kind" || die "cannot write the run record $RUN_RECORD"
announce_run_record
[ -z "$failed_lanes" ] || die "lane(s) stopped before finishing:$failed_lanes"

# Everything past this point runs with every unit recorded, so it cannot change
# a unit's row -- but it can still abort the run, and a record left saying
# `complete` over an exit 2 would tell the consumer the opposite of what
# happened. Every post-lane harness failure therefore rewrites the exit kind
# first. The record is published BEFORE these steps rather than after them, so
# that a failure here leaves a record that explains itself instead of the
# absence that means a startup refusal.
die_post_run() {
  write_run_record post-run-failed ||
    echo "sweep: cannot rewrite $RUN_RECORD as post-run-failed" >&2
  die "$@"
}

for unit in "${UNITS[@]}"; do
  IFS=: read -r machine backend host <<<"$unit"
  evidence=$LANE_DIR/skip-run.$machine.$backend
  [ -f "$evidence" ] || continue
  SKIP_RUN_BACKENDS+=("$backend")
  SKIP_RUN_BOXES+=("$machine")
  SKIP_RUN_LOGS+=("$(cat "$evidence")")
done

echo
if [ "$FORCE" = 1 ] && [ -z "$TARGET" ]; then
  [ -x "$AGGREGATE_SKIPS" ] || die_post_run "skip aggregator is not executable: $AGGREGATE_SKIPS"
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
    0 | 1) mv "$report_stage" "$report" || die_post_run "cannot publish $report" ;;
    *) rm -f "$report_stage"; die_post_run "skip aggregation failed (exit $aggregate_rc)" ;;
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
