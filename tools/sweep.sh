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
# Wake them before a run that is meant to cover cuda or hip. Each GPU box is
# addressed through the boot its site host table names (`kind_of`, see
# lab_dest): `-linux` for native Ubuntu, `-wsl` for the WSL guest -- and on a WSL
# boot, kick WSL after waking (it starts on demand or at login, never at boot,
# so the `-wsl` host lags the box being up). The caller is expected to notice
# when a backend has been skipped for too long.
#
# tuf, a Wi-Fi-only laptop, is the exception: nothing the caller runs can wake it
# (its own RTC timer does, or a person), so its lane is GATED -- it runs only when
# `wake-lab.sh status tuf` reaches its Linux, records `gate` rather than `skip`
# when it does not, and ends by putting the box back to sleep (see run_lane).
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
#   tools/sweep.sh                     # cc + metal locally; cuda on rog-nv, hip + multidev_cc on minix,
#                                      # and hip again on tuf (discrete memory), if up
#   tools/sweep.sh --slow             # also `dune build @slow`
#   tools/sweep.sh --force             # cold rebuild and re-execute every test alias
#   tools/sweep.sh --only metal        # one backend (repeatable)
#   tools/sweep.sh --target test/einsum  # narrower dune target, for smoke-testing
#   tools/sweep.sh --ref origin/master   # what to test (default: origin/master)
#   OCANNL_TOOL_SWEEP_LOCAL_BOX=m4-max tools/sweep.sh  # required stable local box ID
#   OCANNL_TOOL_SWEEP_DEST_ROG=rog-nv-linux tools/sweep.sh --only cuda
#                                      # name a GPU box's ssh destination (also _MINIX, _TUF),
#                                      # instead of reading its kind from WAKE_LAB_HOSTS
#   OCANNL_TOOL_SWEEP_WAKE_LAB=~/bin/wake-lab.sh  # the lab's power script (the default): its
#                                      # endpoint map names every GPU box's ssh aliases (a run with
#                                      # a remote unit refuses without it), its lock paths are
#                                      # checked at startup, and it is asked whether tuf is up
#                                      # before its lane, and to sleep it after
#   OCANNL_TOOL_SWEEP_TEST_CAP=0 tools/sweep.sh  # lift the per-test cap on GPU units (default:
#                                      # tools/per-test-cap.sh's 1500s; see test_cap_start_cmd)
#   OCANNL_TOOL_SWEEP_DEVICE_PROBE=none tools/sweep.sh  # skip the device probe a GPU unit and
#                                      # its serial rerun start with (or name a stand-in for
#                                      # bin/device_props; _CAP bounds it, default 60s; see
#                                      # device_probe_cmd)
#   OCANNL_TOOL_FLEET_WORKER=none tools/sweep.sh  # do not consult the fleet's execution registry,
#                                      # which otherwise skips a unit whose box is under an
#                                      # exclusive measurement (sweep_fleet_under_measurement), nor
#                                      # a GPU box's own fleet-worker.sh, through which a GPU unit
#                                      # reads the box's GPU-token holders and, where the tokens
#                                      # bind, takes one (tools/gpu-tokens.sh)
#   OCANNL_TOOL_SWEEP_TOKEN_WAIT=600 tools/sweep.sh  # how long a GPU leg waits for a token before
#                                      # its unit records `skip` (default 1800s)

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
# The per-test cap's default and watcher, shared with tools/test-run.sh the same way (test_cmd).
[ -r "$SWEEP_TOOLS/per-test-cap.sh" ] || {
  echo "sweep: cannot read $SWEEP_TOOLS/per-test-cap.sh" >&2
  exit 2
}
# shellcheck source=per-test-cap.sh
. "$SWEEP_TOOLS/per-test-cap.sh"
# And the device probe, shared the same way (device_probe_cmd).
[ -r "$SWEEP_TOOLS/device-probe.sh" ] || {
  echo "sweep: cannot read $SWEEP_TOOLS/device-probe.sh" >&2
  exit 2
}
# shellcheck source=device-probe.sh
. "$SWEEP_TOOLS/device-probe.sh"
# The kernel-window filters (dxg for a WSL boot, native for a native one) and their counts, shared
# with the harness that pins them. After box-jobs.sh, whose transport names pick the kind.
[ -r "$SWEEP_TOOLS/kernel-window.sh" ] || {
  echo "sweep: cannot read $SWEEP_TOOLS/kernel-window.sh" >&2
  exit 2
}
# shellcheck source=kernel-window.sh
. "$SWEEP_TOOLS/kernel-window.sh"
# Where this host's fleet-worker.sh might be, shared with tools/test-run.sh: the reader of the
# fleet's execution registry, which a lane asks before each unit (see sweep_fleet_under_measurement).
[ -r "$SWEEP_TOOLS/fleet-worker-candidates.sh" ] || {
  echo "sweep: cannot read $SWEEP_TOOLS/fleet-worker-candidates.sh" >&2
  exit 2
}
# shellcheck source=fleet-worker-candidates.sh
. "$SWEEP_TOOLS/fleet-worker-candidates.sh"
# The credential deny-list, shared with tools/test-run.sh and tools/machine-verify.sh: every dune
# a leg runs scrubs it first, on the side that runs dune and inside opam's environment
# (switch_dune_cmd).
[ -r "$SWEEP_TOOLS/credential-env.sh" ] || {
  echo "sweep: cannot read $SWEEP_TOOLS/credential-env.sh" >&2
  exit 2
}
# shellcheck source=credential-env.sh
. "$SWEEP_TOOLS/credential-env.sh"
# The process supervisor every unit leg runs under, on both sides of the ssh, and the quoting its
# far-side text is shipped with (capped, capped_bg, remote_capped, sq): sourced, so the fixtures and
# live witnesses that drive it run this very text (gh-ocannl-1380).
[ -r "$SWEEP_TOOLS/sweep-supervisor.sh" ] || {
  echo "sweep: cannot read $SWEEP_TOOLS/sweep-supervisor.sh" >&2
  exit 2
}
# shellcheck source=sweep-supervisor.sh
. "$SWEEP_TOOLS/sweep-supervisor.sh"

# Sourced sweep components; their fixtures use the same functions as a full run.
[ -r "$SWEEP_TOOLS/lab-map.sh" ] || {
  echo "sweep: cannot read $SWEEP_TOOLS/lab-map.sh" >&2
  exit 2
}
# shellcheck source=lab-map.sh
. "$SWEEP_TOOLS/lab-map.sh"
[ -r "$SWEEP_TOOLS/fleet-registry.sh" ] || {
  echo "sweep: cannot read $SWEEP_TOOLS/fleet-registry.sh" >&2
  exit 2
}
# shellcheck source=fleet-registry.sh
. "$SWEEP_TOOLS/fleet-registry.sh"
[ -r "$SWEEP_TOOLS/sweep-fingerprint.sh" ] || {
  echo "sweep: cannot read $SWEEP_TOOLS/sweep-fingerprint.sh" >&2
  exit 2
}
# shellcheck source=sweep-fingerprint.sh
. "$SWEEP_TOOLS/sweep-fingerprint.sh"
[ -r "$SWEEP_TOOLS/sweep-unit-state.sh" ] || {
  echo "sweep: cannot read $SWEEP_TOOLS/sweep-unit-state.sh" >&2
  exit 2
}
# shellcheck source=sweep-unit-state.sh
. "$SWEEP_TOOLS/sweep-unit-state.sh"
[ -r "$SWEEP_TOOLS/gpu-tokens.sh" ] || {
  echo "sweep: cannot read $SWEEP_TOOLS/gpu-tokens.sh" >&2
  exit 2
}
# shellcheck source=gpu-tokens.sh
. "$SWEEP_TOOLS/gpu-tokens.sh"

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
# destroy a VM whose box is reserved. The lock itself is deliberately just a directory, a filename
# and a one-line description -- this sweep takes its own flock without calling wake-lab.sh, so no
# lane's reservation waits on the script answering. `wake-lab.sh lock-path <box>` answers the same
# path for anyone who would rather ask than derive, and the sweep asks it only to check, once per
# run, that the two still agree (lab_contract_check). Which box an ssh alias reserves is the other
# half, and that is not derived here at all: it is a row of wake-lab.sh's endpoint map (lab_map).
#
# There are TWO lock files per box on the wake-lab side, and this is the one a lane takes:
#
#   <box>.lock        the LANE lock -- "no other lane runs on this box". Ours.
#   <box>.hold.lock   the HOLD lock -- "this box's VM must not be destroyed". wake-lab's `--hold`
#                     takes it and its Windows-side holder carries it until `unhold`.
#
# A destroyer (`restart-wsl`, `sleep`, `hibernate`, `down`) takes both and is refused if either is
# held, so a lane holding only the lane lock is still fully protected. Do NOT extend this file to
# take or honour the hold lock as well: a hold says "keep this VM alive", never "nobody else may
# work here", and the day the two claims shared one file the sweep deadlocked against its own
# caller. The cross-machine sweep routine holds both GPU boxes with `--hold` before launching this
# script in the same session, and on 2026-09-18 every remote lane waited out LAB_LOCK_WAIT against
# that routine's own holder and skipped: three of the five backends the routine gates got zero
# coverage (run 20260918T050903Z, ludics-lite#224). A holder line of `wake-lab --hold ...` in a
# `skip (box ... reserved by ...)` means that regression is back.
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
GUEST_IDENTITY_CAP=30
GUEST_IDENTITY_WINDOW=90
GUEST_IDENTITY_PAUSE=10

LAB_MAP= # wake-lab.sh endpoint-map's answer: a box name, then its ssh aliases, per line

# Which ssh destination reaches a GPU box TODAY (gh-ocannl-1030). The boxes are dual-boot: native
# Ubuntu answers at the row's `-linux` alias, the WSL guest under Windows at its `-wsl` one, and only
# the booted one answers. So the destination is not a constant of the unit table but a per-run
# fact, and the owner of that fact is the site's wake-lab host table -- `kind_of <box>` prints
# `linux` or `wsl` -- the same file, and the same function, the scheduled routine reads to decide
# how to WAKE the box before it launches this script. Reading it here too is what keeps the two
# from disagreeing: a sweep that hard-coded the `-wsl` spellings recorded `skip (unreachable)` for
# every GPU unit from the first native boot on, and kept reporting as if nothing had changed.
# Every box is asked, as wake-lab.sh asks every box it acts on (check_targets): a box whose row
# lists one boot today may list two tomorrow, and tuf's already lists a WSL guest.
#
# WAKE_LAB_HOSTS moves the table, exactly as it does for wake-lab.sh. A per-box override,
# OCANNL_TOOL_SWEEP_DEST_ROG / _MINIX / _TUF, names the destination outright for one run (a
# harness, a manual run on a host with no site table); set, it wins and the table is not read for
# that box; empty counts as unset. It must be the `-linux` or `-wsl` alias on THAT box's row, which
# makes it a boot-kind choice spelled as the destination: everything downstream keys on the alias
# -- lab_box_of for the lane's lock, the `-wsl` suffix for the remote PATH -- and an arbitrary alias
# would leave each of those guessing.
#
# No kind means NO destination, and the sweep refuses to start (see the resolution before the lanes
# start): guessing one is the silent skip this exists to remove, since a wrong guess is
# indistinguishable, in the history, from a box that was simply asleep.
LAB_HOSTS=${WAKE_LAB_HOSTS:-$HOME/.config/wake-lab/hosts.sh}

# The box the sweep cannot wake (gh-ocannl-1035). tuf is a Wi-Fi-only laptop: WoL cannot reach it,
# so it is up because its own RTC timer woke it for the sweep (self-improve's Linux bootstrap
# installs that timer) or because a person did. Two things follow, and both are keyed here rather
# than spelled per call site:
#   - its lane is GATED: it runs only when `wake-lab.sh status <box>` reaches the box's Linux, and
#     a box that is asleep or unreachable records `gate` -- a condition outside this run, like a
#     skip, but one the caller was never expected to fix, so it must not read as a failed wake;
#   - its lane ENDS BY SLEEPING the box (`wake-lab.sh sleep <box>`), or a timer-woken laptop stays
#     up all day. wake-lab's sleep is inhibitor-aware, so a box someone else is holding refuses,
#     and that refusal is reported, never a failure of the sweep.
lab_box_gated() { # box
  case $1 in tuf) return 0 ;; *) return 1 ;; esac
}

# The lab's power script, for exactly those two acts -- and, at startup, for the two read-only
# answers the sweep takes from it: the endpoint map (lab_map) and the lock paths the contract check
# compares against (lab_contract_check). The lock itself is still deliberately a directory, not a
# command: no lane calls this to take one.
WAKE_LAB=${OCANNL_TOOL_SWEEP_WAKE_LAB:-$HOME/bin/wake-lab.sh}

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

# ------------------------------------------------------------ the lab lock contract, checked
# The contract above is three facts held in common with a script in another repository: the
# directory, the `<box>.lock` name, and that a lane never takes or honours `<box>.hold.lock`. Which
# box an alias reserves is no longer one of them: the sweep reads it from wake-lab.sh's own map
# (lab_map). ludics-lite's suite checks the lock from its side (scripts/test-wake-lab.sh runs this
# file's take_lab_lock against wake-lab.sh, ludics-lite#170), but it reads a staging checkout it
# never fetches, and it calls the function, not the lane. The gap it leaves closes only here, in
# the revision about to run, against the wake-lab.sh it will actually meet (gh-ocannl-1025): an
# upstream push neither suite has seen yet. So once per run, for every box a selected remote lane
# reserves, this asks `wake-lab.sh lock-path <box>` and requires that it answer the file
# take_lab_lock opens. The negative clause is behaviour, not a name, and the harness pins it (a held
# hold lock, and a lane that still runs).
#
# A broken contract refuses the run at startup, as an unreadable boot kind does (lab_dest): a lane
# reserving a file no destroyer consults is the 2026-09-16 loss waiting to recur, invisible until it
# does, and the repair is a one-line edit on one side. A run with no remote lane asks nothing.
# A startup question's answer, into the variable named first. Through run_capped and a file, never a
# command substitution: bash defers a trap while it waits on a substitution, so a TERM aimed at this
# pid would otherwise wait out the question's whole cap, with its supervisor never published in
# UNIT_PID for the relay to reap.
ask_capped() { # var budget command...
  local ask_var=$1 ask_out ask_rc
  shift
  ask_out=$(mktemp "${TMPDIR:-/tmp}/ocannl-sweep-ask.XXXXXX") || return 1
  run_capped "$@" >"$ask_out" 2>/dev/null </dev/null
  ask_rc=$?
  printf -v "$ask_var" '%s' "$(cat "$ask_out")"
  rm -f "$ask_out"
  return "$ask_rc"
}

LAB_CONTRACT=   # the header's verdict
LAB_LANE_BOXES= # the lab box of each selected remote lane, set with LAB_DESTS
REF=origin/master
TARGET=
SLOW=0
FORCE=0
ONLY=()
# Per-unit wall-clock cap, enforced by the perl supervisor below on both the
# local and the remote side: macOS has no timeout(1) at all, and where timeout(1)
# does exist it is not necessarily one whose -k reaches the process group.
CAP=${OCANNL_TOOL_SWEEP_CAP:-5400}
# Per-test cap for a GPU unit's dune legs (gh-ocannl-1211; test_cap_start_cmd says where it
# applies), the default tools/test-run.sh gives a GPU batch; 0 lifts it.
TEST_CAP=${OCANNL_TOOL_SWEEP_TEST_CAP:-$TEST_CAP_DEFAULT}
case $TEST_CAP in
  '' | *[!0-9]*)
    echo "sweep: OCANNL_TOOL_SWEEP_TEST_CAP must be a whole number of seconds, not '$TEST_CAP'" >&2
    exit 2 ;;
esac
# The device probe a GPU unit and its serial rerun start with (gh-ocannl-1252; device_probe_cmd):
# empty for bin/device_props, `none` for no probe, or an absolute path to a stand-in taking the
# same `--ocannl_backend=<name>` (the harness's); and the bound on each query.
PROBE=${OCANNL_TOOL_SWEEP_DEVICE_PROBE:-}
case $PROBE in
  '' | none | /*) ;;
  *)
    echo "sweep: OCANNL_TOOL_SWEEP_DEVICE_PROBE must be none or an absolute path, not '$PROBE'" >&2
    exit 2 ;;
esac
PROBE_CAP=${OCANNL_TOOL_SWEEP_DEVICE_PROBE_CAP:-$DEVICE_PROBE_CAP_DEFAULT}
# How long a GPU leg waits for one of its box's GPU tokens where the tokens bind (gpu_token_take_cmd).
TOKEN_WAIT=${OCANNL_TOOL_SWEEP_TOKEN_WAIT:-$GPU_TOKEN_WAIT_DEFAULT}
case $TOKEN_WAIT in
  '' | *[!0-9]*)
    echo "sweep: OCANNL_TOOL_SWEEP_TOKEN_WAIT must be a whole number of seconds, not '$TOKEN_WAIT'" >&2
    exit 2 ;;
esac
case $PROBE_CAP in
  '' | 0 | *[!0-9]*)
    echo "sweep: OCANNL_TOOL_SWEEP_DEVICE_PROBE_CAP must be a positive number of seconds, not '$PROBE_CAP'" >&2
    exit 2 ;;
esac
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
# The Linux sides of the GPU boxes, never the native-Windows ones: plain Linux
# toolchain, and Windows portability is covered by the scheduled CI job. Which
# Linux -- the native Ubuntu boot or the WSL guest -- is the box's boot kind
# today, so a GPU unit's ssh-host is written `@<lab box>` here and replaced by
# the destination lab_dest resolves, once the --only selection is known (see
# the resolution before the lanes start: an unselected unit is never resolved,
# so a local-only run needs no site table and no wake-lab.sh).
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
  "rog-nv:cuda:@rog"
  "minix:hip:@minix"
  "minix:multidev_cc:@minix"
  "tuf:hip:@tuf"
)

# hip twice, on purpose (gh-ocannl-1035): minix's gfx1151 is an iGPU whose "device" memory IS host
# memory, so a missing or misplaced host<->device transfer can read the right bytes there anyway,
# and minix's green says nothing about it. tuf's RX 7700S (gfx1102) is the fleet's one discrete AMD
# GPU, where the same mistake reads stale or foreign memory (lukstafi/ludics-lite#320). Its lane is
# GATED (lab_box_gated): a laptop woken by its own timer or by hand, never by the caller. `--only
# hip` selects both units; each is its own row, keyed by machine as every row is.
#
# The memory model each GPU unit exercises, for the run record's `unit` row: which hip row covers
# discrete memory is a fact a consumer otherwise has to know about the fleet's hardware. Declared,
# not probed: it is the box's hardware, and the readings that would probe it cannot tell the two
# apart on these kernels (KFD's topology reports `local_mem_size 0` for both AMD GPUs). The
# architectures are what each box reported on 2026-09-23 (KFD `gfx_target_version` 110501 and
# 110002, nvidia-smi `compute_cap` 12.0). `-` for a CPU unit, which has no device memory.
unit_memory() { # machine backend
  case $1:$2 in
    tuf:hip) printf 'discrete/gfx1102' ;;
    minix:hip) printf 'unified/gfx1151' ;;
    rog-nv:cuda) printf 'discrete/sm_120' ;;
    *:metal) printf 'unified/apple' ;;
    *) printf '%s' - ;;
  esac
}

# The failure names that mean the ENVIRONMENT refused the run rather than a
# test judging it. On minix the WSL2 dxg bridge (tools/box-jobs.sh, and the dxg
# bullet in docs/agent-notes/build-and-test.md#gpu-boxes-job-caps-and-runtime-refusals)
# surfaces its lost messages as three HIP exceptions; rog-nv's CUDA reaches its
# GPU through the same WSL2 bridge, so the cudajit checks at the same three call sites are
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
# <name>:` with the status on the next line. An entry is a name, or a name and
# the one status that qualifies it: the kernel launch is where a concurrent
# suite's device-memory pressure lands (rog-nv's native cuda unit lost
# `schedule_strided_1x1` to `cu_launch_kernel` CUDA_ERROR_OUT_OF_MEMORY on
# 2026-10-02 with an empty kernel window, and passed alone), but it is also
# where a kernel's own resource bug lands deterministically
# (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, INVALID_VALUE for shared memory over the
# limit) and where an earlier kernel's sticky fault surfaces
# (CUDA_ERROR_ILLEGAL_ADDRESS) -- those are a test's own failures, so only the
# memory status buys the rerun. The statuses each name has been
# seen with, and how to read a rerun's verdict, are the signature table in
# docs/agent-notes/build-and-test.md#gpu-boxes-job-caps-and-runtime-refusals
# (the record half of gh-ocannl-927).
ENVIRONMENT_REFUSALS='hip_init
hip_module_load_data_ex
hip_stream_create_with_priority
hip_module_launch_kernel HIP_ERROR_OUT_OF_MEMORY
cu_init
cu_device_primary_ctx_retain
cu_module_load_data_ex
cu_stream_create_with_priority
cu_launch_kernel CUDA_ERROR_OUT_OF_MEMORY'

# The runtime ASSERTIONS that mean the same thing, for a refusal the runtime does not report as an
# error at all but dies of. On a native boot, a HIP process whose SDMA queue KFD refused goes on
# into ROCr's scratch teardown and aborts there -- the 2026-09-23 ladder lost `schedule_conv_gemm`
# at dune's default width to exactly this, beside the kernel's `No more SDMA queue to allocate`
# (gh-ocannl-1029), as one log line:
#   schedule_conv_gemm.exe: ./runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp:2003: virtual void
#   rocr::AMD::GpuAgent::ReleaseQueueMainScratch(rocr::AMD::ScratchInfo&): Assertion
#   `scratch.main_queue_base' failed.
# The kernel window counts the refusal itself; this is the half that lands in the unit's LOG, which
# keeps the rerun when the window could not be read. Keyed as glibc prints a failed assert --
# `<prog>: <file>:<line>: <function>: Assertion `<expr>' failed.` -- with the function as the
# entry, so an unrelated assertion in the same runtime is not caught by it.
ENVIRONMENT_ASSERTIONS='GpuAgent::ReleaseQueueMainScratch'

# Whether the log holds `Fatal error: exception <name>:` with exactly <status> after it: on the
# next line, where the bindings' printer breaks, or on the same line should it ever not break.
refusal_with_status() { # log name status
  awk -v head="Fatal error: exception $2:" -v status="$3" '
    pending { line = $0; sub(/^[ \t]+/, "", line); sub(/[ \t\r]+$/, "", line)
              if (line == status) { found = 1; exit } }
    { pending = 0 }
    index($0, head) == 1 {
      rest = substr($0, length(head) + 1); sub(/^[ \t]+/, "", rest); sub(/[ \t\r]+$/, "", rest)
      if (rest == status) { found = 1; exit }
      pending = rest == ""
    }
    END { exit !found }' "$1"
}

# Either the name lists or the kernel's own evidence (window_red, which reads the unit's
# collected-evidence sidecar and answers only to a positive count). The kernel arm is what lets a
# call site nobody has seen yet -- or a failure with no exception name at all,
# such as the SEGV the 2026-09-15 minix runs produced -- get its serial rerun on
# the FIRST miss instead of after one (gh-ocannl-979), behind a WSL boot's bridge and on a native
# boot's GPU drivers alike (gh-ocannl-1034). The rerun's `still red` /
# `all clean` verdict remains the judge either way: this decides that the unit is
# rerun, never that its failures were the environment's.
#
# The two name lists read the unit's LOG, and only a GPU unit's (gh-ocannl-1039): they name
# device refusals, which a CPU unit cannot meet, while a CPU unit's log holds whatever its tests
# printed -- and this repository's own sweep harness prints refusal fixtures when it fails, so a
# local cc unit whose harness went red read as environment-red and was rerun instead of reported.
# The collected window needs no such gate: only a remote GPU unit has one (window_sidecar).
environment_red() { # log backend
  local name status
  if ! box_jobs_cpu_backend "$2"; then
    while read -r name status; do
      [ -n "$name" ] || continue
      if [ -z "$status" ]; then
        grep -q "^Fatal error: exception $name:" "$1" && return 0
      else
        refusal_with_status "$1" "$name" "$status" && return 0
      fi
    done <<<"$ENVIRONMENT_REFUSALS"
    while IFS= read -r name; do
      [ -n "$name" ] &&
        grep -qE "^[^ ]+: [^ ]+:[0-9]+: .*${name}.*: Assertion .* failed\.$" "$1" && return 0
    done <<<"$ENVIRONMENT_ASSERTIONS"
  fi
  window_red "$1" && return 0
  return 1
}

# Successful forced full-suite units are the only runs from which absence of a
# skip record means execution. Incremental Dune runs serve cached actions
# without running them, and a red or interrupted unit may not have reached
# every test -- unless the serial rerun re-ran every failing stanza clean and
# its completion pass then ran whatever they had held back (rerun_cleared),
# which the report names. What is read is the unit's per-action verdict
# records (stage_skip_evidence), never its log. Keep the qualifying evidence
# from THIS invocation rather than recovering it by timestamp from history (two
# invocations can begin in the same second in the integration harness).
SKIP_RUN_BACKENDS=()
SKIP_RUN_BOXES=()
SKIP_RUN_RECORDS=()
SKIP_RERUN_CLEARED=()

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
# Each backend once, in first-appearance order: hip runs on two boxes (minix and tuf), and the
# vocabulary is of backends -- the skip aggregation refuses a duplicate `--known`.
known_backends=$(for u in "${UNITS[@]}"; do b=${u#*:}; printf '%s\n' "${b%%:*}"; done | awk '!seen[$0]++')
if [ ${#ONLY[@]} -gt 0 ]; then
  for b in "${ONLY[@]}"; do
    known_backend "$b" ||
      die "unknown backend '$b'; known: $(printf '%s' "$known_backends" | tr '\n' ' ')"
  done
fi

wanted() {
  [ ${#ONLY[@]} -eq 0 ] && return 0
  local b
  for b in "${ONLY[@]}"; do [ "$b" = "$1" ] && return 0; done
  return 1
}

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
#
# The WAIT is not gated that way: every registered child is waited for, live or
# not. `wait` on a pid this shell has already reaped returns at once, and waits
# for nothing that is not its own child, so it cannot reach a recycled pid; and
# it is the one way to know a lane's EXIT trap has FINISHED, which is what the
# top level's own exit -- the coordination directory's removal, the record, the
# lanes' last lines (publish_lane_buffers) -- must come after (gh-ocannl-1039).
UNIT_PID=
LANE_PIDS=
# The lanes' coordination directory and the lanes themselves, declared here for
# the relay, which reads both before they are given (see LANE_DIR below).
LANE_DIR=
LANES=()
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
  for pid in $UNIT_PID $LANE_PIDS; do wait "$pid" 2>/dev/null; done
  # A lane stops taking signals once its unit is reaped: the top level relays its own TERM to every
  # lane as well, and a second relay arriving while the lane's EXIT trap runs ends the lane there --
  # before lane_exit has asked for its gated box's sleep and published the line saying so (the
  # harness's cancelled tuf lane lost both that way).
  [ "$IN_LANE" = 1 ] && trap '' INT TERM
  # A cancelled run still ended, and the rows its lanes managed to write before
  # the signal are real; the record says which those were. Only the top level
  # writes it -- a lane runs this same relay for its own supervisor -- and only
  # once there is a coordination directory to read the lanes' evidence from. A
  # failure to write it must not replace the cancellation's exit status, so it
  # is reported rather than fatal.
  # Every lane has exited by now, so what a stopped one buffered is printed here, ahead of the
  # record's locator.
  [ "$IN_LANE" = 1 ] || publish_lane_buffers
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
git -C "$MAIN" -c maintenance.auto=false fetch -q origin master || die "cannot fetch origin master in $MAIN"
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
  # Python's stdout is text mode on Windows: drop the "\r" a line ends with there.
  while IFS= read -r box; do
    box=${box%$'\r'}
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

# Where a unit's per-action verdict records live: beside the worktree, like its
# far-side lock, so neither `git clean` nor `dune clean` reaches them and the
# worktree lock covers them. Verdict (test/support/verdict.ml) writes one file
# per action into the directory OCANNL_TOOL_VERDICT_RECORDS names and rewrites
# it whenever that action runs again, so after the unit's first attempt, serial
# rerun and completion pass -- all three export it -- each action's file holds
# its FINAL attempt by construction: a retry overwrites its own action's
# records and an action never re-run keeps its own (gh-ocannl-1114). Read back
# by collect_verdict_records; the stderr copy of each record stays in the log
# for humans.
verdict_records_dir() { printf '%s.verdict-records' "$1"; }
# Where a GPU unit's per-test cap watcher keeps its records (test_cap_start_cmd), beside the
# worktree, and its device probe (device_probe_cmd).
test_cap_dir() { printf '%s.test-cap' "$1"; }
device_probe_dir() { printf '%s.device-probe' "$1"; }
verdict_records_env() { # wt
  printf 'export OCANNL_TOOL_VERDICT_RECORDS="%s"; ' "$(verdict_records_dir "$1")"
}

# The per-test cap (gh-ocannl-1211) as shell text: tools/test-run.sh's own watcher, started beside
# one dune leg of a GPU unit -- its suite (test_cmd), each stanza of its serial rerun
# (serial_rerun_cmd) and its completion pass (suite_completion_cmd) -- and stopped after it by
# test_cap_stop_cmd. A GPU fault wedges the device, every test dune starts after it hangs in device
# init, and the unit's cap is the only other bound: on 2026-10-14 (sweep 20261014T051751Z) sixteen
# tuf hip tests hung at once and held the unit for its full 5400 s with no test named. The serial
# rerun needs it most, since what triggers it is a kernel-side refusal signature, when the device
# is most likely faulted (gh-ocannl-1252). The first test past the cap sends the leg's supervisor
# (the `sh -c` running this text is its child) the SIGALRM its own cap sends, so the leg ends as
# `timeout` (142) with the hung test named in the log. GPU units only, as test-run.sh's default
# applies, and not for a leg whose targets reach the slow aliases, which keeps only the unit's cap.
# The watcher closes fd 9, the worktree lock, so it can never hold the lock past the leg. Prints
# nothing and fails where the leg gets no watcher.
test_cap_start_cmd() { # backend wt dune-target...
  local backend=$1 wt=$2 watched
  shift 2
  ! box_jobs_cpu_backend "$backend" && [ "$TEST_CAP" -gt 0 ] || return 1
  ! test_cap_reaches_slow "$@" >/dev/null || return 1
  watched=$(test_cap_dir "$wt")
  printf 'rm -rf "%s" && mkdir -p "%s" || exit 126; ' "$watched" "$watched"
  printf 'perl -e %s %s "$$" "$PPID" "%s" gpu sweep %s </dev/null 9>&- & watch=$!; ' \
    "$(sq "$test_cap_watch")" "$TEST_CAP" "$watched" \
    "$(sq "Rerun with OCANNL_TOOL_SWEEP_TEST_CAP=0, or a larger cap")"
}
test_cap_stop_cmd() { printf 'kill $watch 2>/dev/null; wait $watch 2>/dev/null; '; }

# The device probe (gh-ocannl-1252) as shell text, for the side that owns the worktree: the query
# of tools/device-probe.sh, asked of the unit's backend through bin/device_props -- built here,
# through the switch, in the worktree's own `_build` -- or the OCANNL_TOOL_SWEEP_DEVICE_PROBE
# stand-in, bounded at PROBE_CAP. A query that does not answer ends the leg before dune starts,
# with exit 69 and the probe's explanation in the log, which run_unit files as `error` (nothing was
# judged) and names DEVICE UNHEALTHY; a wedged device then costs the unit a minute, not the first
# test's per-test cap. Otherwise the answer goes to the log as a `sweep: device probe:` line and the
# leg goes on, as it does when device_props does not build (recorded as not probed: the suite then
# fails to build the same way). Not for a CPU backend, and not for a local unit on macOS without a
# stand-in: a freshly linked device_props can sit in dlopen for minutes under XProtect, which would
# read as a hung device (tools/test-run.sh's plan_device_probe has the same rule). Prints nothing
# and fails where the leg gets no probe.
device_probe_cmd() { # backend wt host
  local backend=$1 wt=$2 host=$3 rd prog=$PROBE
  ! box_jobs_cpu_backend "$backend" && [ "$PROBE" != none ] || return 1
  [ -n "$PROBE" ] || [ -n "$host" ] || [ "$(uname -s)" != Darwin ] || return 1
  rd=$(device_probe_dir "$wt")
  printf 'rm -rf "%s" && mkdir -p "%s" || exit 126; ' "$rd" "$rd"
  if [ -z "$PROBE" ]; then
    prog=$wt/_build/default/bin/device_props.exe
    printf 'if %s build ./bin/device_props.exe </dev/null >"%s/device-probe-build.log" 2>&1 && [ -x "%s" ]; then ' \
      "$(switch_dune_cmd)" "$rd" "$prog"
  fi
  printf 'sh -c %s device-probe sweep %s %s "%s" "%s" %s; [ $? -ne 69 ] || exit 69; ' \
    "$(sq "$device_probe_sh")" "$(sq "$probe_perl")" "$PROBE_CAP" "$rd" "$prog" "$backend"
  if [ -z "$PROBE" ]; then
    printf 'else echo "not probed: %s did not build (%s/device-probe-build.log)" >>"%s/device-probe"; fi; ' \
      "$prog" "$rd" "$rd"
  fi
  printf 'sed "s/^/sweep: device probe: /" "%s/device-probe"; ' "$rd"
}

# How every leg below starts dune, as shell text: `$(switch_dune_cmd) build ...` runs `dune build
# ...` through `opam exec`. Dune writes the environment of each process it spawns into the
# worktree's `_build/trace.csexp`, and a remote box's login environment carries that box's own
# GitHub token (gh-ocannl-1280). So the deny-list of tools/credential-env.sh is applied where the
# text runs -- the far side of the ssh, or the local `sh -c` -- and INSIDE the environment `opam
# exec` builds, so a variable the switch's own updates set is removed too; never to this script's
# environment, which the remote one does not see. A scrub that cannot complete exits 126, harness
# non-coverage like the forced clean's failure below.
switch_dune_cmd() {
  printf 'opam exec -- sh -c %s sweep-dune dune' \
    "$(sq "{ $(credential_env_scrub_text); } || exit 126; exec \"\$@\"")"
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
test_cmd() { # backend wt [jobs [host]]
  local backend=$1 wt=$2 jobs=${3:-} host=${4:-} force_arg= jobs_arg=
  [ "$FORCE" = 1 ] && force_arg=--force
  [ -n "$jobs" ] && jobs_arg="-j $jobs"
  # 127, not a generic failure: a worktree that is not there means nothing ran,
  # which the outcome mapping treats as non-coverage rather than a red suite.
  printf 'cd "%s" || exit 127; ' "$wt"
  # Every Verdict-linked test action writes its machine records into a file of
  # its own here (verdict_records_env, gh-ocannl-1114), and this unit starts the
  # directory empty: the local backends share one worktree, and what a previous
  # unit's actions recorded is not this unit's evidence. A directory that cannot
  # be reset is harness non-coverage, like the clean below.
  printf 'rm -rf "%s" && mkdir -p "%s" || exit 126; ' "$(verdict_records_dir "$wt")" \
    "$(verdict_records_dir "$wt")"
  verdict_records_env "$wt"
  # Dune's alias --force does not reliably invalidate ppx_expect inline tests.
  # A forced pass therefore starts from an empty build tree, under the worktree
  # lock already held on both local and remote paths. Failure to establish that
  # precondition is harness non-coverage, not a red suite.
  if [ "$FORCE" = 1 ]; then
    printf '%s clean; clean_rc=$?; [ $clean_rc -eq 0 ] || exit 126; ' "$(switch_dune_cmd)"
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
    printf 'OCANNL_BACKEND=%s %s build @check; ' "$backend" "$(switch_dune_cmd)"
  fi
  # The device probe, after the compile-only prebuild has built most of `bin/device_props` and
  # before any test touches the device (device_probe_cmd): a wedged device ends the unit here, as
  # DEVICE UNHEALTHY, rather than at the first test's per-test cap.
  device_probe_cmd "$backend" "$wt" "$host" || :
  # The per-test cap beside the suite's dune (test_cap_start_cmd); the `@slow` leg below, and a
  # --target that reaches the slow aliases, keep only the unit's cap.
  local watched=
  if watched=$(test_cap_start_cmd "$backend" "$wt" "$TARGET"); then
    printf '%s' "$watched"
  else
    watched=
  fi
  if [ -z "$TARGET" ]; then
    # The host-capability record goes first, so the copy below (gh-ocannl-1038) can only find one
    # this invocation produced: a build that fails before reaching its producer leaves none.
    printf 'rm -f _build/default/test/config/ocannl_host_capabilities; '
    printf 'OCANNL_BACKEND=%s %s build %s%s @runtest @train; rc1=$?; ' \
      "$backend" "$(switch_dune_cmd)" "$jobs_arg${jobs_arg:+ }" "$force_arg"
  else
    printf 'OCANNL_BACKEND=%s %s runtest %s%s %s; rc1=$?; ' \
      "$backend" "$(switch_dune_cmd)" "$jobs_arg${jobs_arg:+ }" "$force_arg" "$TARGET"
  fi
  [ -z "$watched" ] || test_cap_stop_cmd
  if [ "$SLOW" = 1 ]; then
    printf 'OCANNL_BACKEND=%s %s build %s%s @slow; rc2=$?; ' \
      "$backend" "$(switch_dune_cmd)" "$jobs_arg${jobs_arg:+ }" "$force_arg"
  else
    printf 'rc2=0; '
  fi
  # The host-capability record the suite just ran under (gh-ocannl-1038): every test's
  # configuration carries it, so the full suite rebuilds it on THIS box, and copying it into the
  # log is what lets sweep_state_update say when a box lost a capability between two sweeps. The
  # suite's call above removed any older copy first, so a record found here is this invocation's.
  # Not for a --target run, whose build need not reach it. Its status is dropped: a missing record
  # is no verdict.
  if [ -z "$TARGET" ]; then
    printf 'sed "s/^/host-capability: /" _build/default/test/config/ocannl_host_capabilities 2>/dev/null; '
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
# under the same lock and PATH, appending to `$log`, which `sweep_fingerprint` then
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
# the CUDA and HIP `compile_program` re-raise their runtime compiler's exception
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
publish_window_sidecar() { # sidecar log writer args... -- stdin is the writer's
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
publish_window_unavailable_fallback() { # kind sidecar log reason label [boot-verdict]
  window_unavailable "$1" - - "$4" "${6:-}" 2>/dev/null >"$3.window-fallback.$$" &&
    mv "$3.window-fallback.$$" "$2" 2>/dev/null && {
      cat "$2" 2>/dev/null >>"$3"
      return 0
    }
  rm -f "$3.window-fallback.$$"
  say "  $5: WARNING -- could not record the $1 window ($4); its record fields say no window"
  return 1
}

# Which guest is on the box now, or nothing if it could not be asked inside the window.
#
# Written to a file rather than captured in a command substitution, for the reason collect_kernel_window
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
  local host=$1 probe=$2 deadline=$(( SECONDS + GUEST_IDENTITY_WINDOW ))
  GUEST_ID=
  while :; do
    run_capped "$GUEST_IDENTITY_CAP" ssh -o BatchMode=yes -o ConnectTimeout=8 \
      "$host" 'cat /proc/sys/kernel/random/boot_id 2>/dev/null' >"$probe" 2>/dev/null
    read -r GUEST_ID < "$probe" 2>/dev/null || GUEST_ID=
    [ -n "$GUEST_ID" ] && return 0
    [ "$SECONDS" -ge "$deadline" ] && return 0
    run_capped "$(( GUEST_IDENTITY_PAUSE + 5 ))" sleep "$GUEST_IDENTITY_PAUSE"
  done
}

# The kind (window_kind_of: `dxg` or `native`) picks the query and the filter; everything else --
# the bounds, the boot comparison, the publication and its fallbacks -- is the same for both.
collect_kernel_window() { # kind host log remote-start-epoch label start-boot-id
  local kind=$1 host=$2 log=$3 remote_start=$4 label=$5 start_boot=${6:-}
  local kernel rc bounds start_utc end_utc sidecar end_boot boot=unknown
  sidecar=$(window_sidecar "$log")
  # No start instant from the box means no window to bound. Reported as a failed
  # collection, which is what it is, rather than guessed.
  if [ -z "$remote_start" ]; then
    publish_window_sidecar "$sidecar" "$log" window_unavailable "$kind" - - \
      "no clock reading from $host" ||
      publish_window_unavailable_fallback "$kind" "$sidecar" "$log" \
        "no clock reading from $host" "$label"
    return 0
  fi
  # Written to a file rather than captured in a command substitution, which runs
  # in a SUBSHELL: the UNIT_PID `run_capped` publishes would be invisible to the
  # lane, so a cancellation could neither relay TERM to this supervisor nor reap
  # it, and it would hold the inherited lock until its own cap expired. The
  # unit's own ssh calls document the same trap.
  kernel=$log.window.$$
  run_capped "$(( CONTEXT_CAP + 60 ))" ssh -o BatchMode=yes -o ConnectTimeout=8 \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
    "$host" "$(remote_capped "$CONTEXT_CAP" "$(window_cmd "$kind" "$remote_start")")" \
    >"$kernel" 2>/dev/null
  rc=$?
  # The bounds the REMOTE used, in its own clock domain -- the only one the log's
  # timestamps are in. Reported rather than recomputed here, so the block and the
  # record row name the window that was actually queried.
  bounds=$(sed -n 's/^window-bounds \([0-9][0-9]*\) \([0-9][0-9]*\)$/\1 \2/p' \
    "$kernel" 2>/dev/null | head -1)
  # Which guest answered THIS collection, against the one the unit started on. The
  # comparison is three-valued on purpose: `replaced` is a finding, `same` is a
  # finding, and a boot id missing from either end is neither -- a box that cannot
  # report one is exactly as trustworthy as it was before this check existed, and
  # turning that into an alarm would retire a working window on every host whose
  # kernel does not publish the file.
  end_boot=$(sed -n 's/^window-boot \(.*\)$/\1/p' "$kernel" 2>/dev/null | head -1)
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
  # the log for whoever reads that. See window_sidecar for why provenance cannot come
  # from the log itself.
  if [ "$rc" -ne 0 ]; then
    rm -f "$kernel"
    publish_window_sidecar "$sidecar" "$log" window_unavailable "$kind" "$start_utc" "$end_utc" \
      "kernel log unreadable on $host (exit $rc)" "$boot" ||
      publish_window_unavailable_fallback "$kind" "$sidecar" "$log" \
        "kernel log unreadable on $host (exit $rc)" "$label" "$boot"
  else
    # Filtered HERE rather than on the far side: the filter is the part with a
    # judgement in it, so it belongs where a fixture can feed it lines directly
    # instead of behind an ssh no test can reach. The publication's status is read
    # BEFORE the cleanup below, which would otherwise replace it with its own.
    publish_window_sidecar "$sidecar" "$log" window_summary "$kind" "$start_utc" "$end_utc" "$boot" \
      <"$kernel"
    rc=$?
    rm -f "$kernel"
    [ "$rc" -eq 0 ] ||
      publish_window_unavailable_fallback "$kind" "$sidecar" "$log" \
        "the collected window could not be published" "$label" "$boot"
  fi
}

# Collect a remote GPU unit's kernel window and say what it shows about the guest.
#
# Factored out because it has to happen on EVERY path that ends a remote unit, not only the one
# that ran dune. A guest replaced during the up-to-600s remote PREPARATION ends the unit through
# the preparation's own `error` return, which collected nothing at all -- so an uncoordinated
# reboot in that window produced neither `vm-replaced` evidence nor the warning, on the path least
# likely to be looked at afterwards. Any future early return from a remote unit belongs here too.
#
# Called BEFORE sweep_fingerprint_write on each path, so the fingerprint carries the window: a replaced
# guest is part of what distinguishes this failure from the same failure on a healthy box.
finish_remote_window() { # machine backend host log outcome remote-start-epoch start-boot-id
  local machine=$1 backend=$2 host=$3 log=$4 outcome=$5 remote_start=$6 start_boot=$7 kind=
  REMOTE_GUEST_REPLACED=0
  # A local unit has no remote guest, and a unit that never ran (`skip`) has nothing to account for.
  [ -n "$host" ] || return 0
  case $outcome in skip) return 0 ;; esac
  # The kernel WINDOW is GPU-only -- its evidence is the device's (the bridge that /dev/dxg is, or
  # a native boot's GPU drivers), so minix's multidev_cc (CPU) would only ever collect another
  # unit's noise -- but the GUEST is not. multidev_cc runs in the same replaceable VM as hip, so a
  # replacement takes it down just the same, and before this it was recorded as a bare `error` with
  # nothing saying the machine had gone. The two questions are separate and only one of them is
  # about the device, so only one of them is gated on the backend.
  #
  # WHICH window is the unit's transport's (window_kind_of, gh-ocannl-1034): the destination the
  # sweep reached the box through names its boot, so a `-wsl` unit reads the bridge and a `-linux`
  # one its GPU drivers. A transport with no kind collects nothing and records `-`: it is not a
  # failed collection, and `unavailable` there would be noise on every GPU unit it ran.
  case $backend in
    cuda | hip) kind=$(window_kind_of "$host") ;;
  esac
  if [ -n "$kind" ]; then
    collect_kernel_window "$kind" "$host" "$log" "$remote_start" "$machine/$backend" "$start_boot"
  elif [ -n "$start_boot" ]; then
    # No sidecar for a unit with no window kind: writing one would make a CPU unit environment-red
    # through window_red and put a device verdict in its record row. The identity answer is
    # reported below instead, which is the part an operator acts on.
    remote_guest_id "$host" "$log.boot.$$"
    [ -n "$GUEST_ID" ] && [ "$GUEST_ID" != "$start_boot" ] && REMOTE_GUEST_REPLACED=1
    rm -f "$log.boot.$$"
  fi
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
      { [ "$REMOTE_GUEST_REPLACED" = 1 ] || window_guest_replaced "$log"; } &&
        say "  $machine/$backend: the guest was REPLACED mid-unit -- this unit tested nothing, and its result is about the box, not the code; rerun it (reruns are incremental)"
      # The same for a box that SLEPT mid-unit (gh-ocannl-1039): its boot id survives, so only the
      # window's power-management lines say it, and a sleep mostly ends a unit as `timeout` -- the
      # per-test cap reads the frozen hours as a hung test.
      window_box_slept "$log" &&
        say "  $machine/$backend: the box SUSPENDED mid-unit -- its result is about the box, not the code; rerun it (reruns are incremental)"
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
    # the compile that just failed. Four lines, not a paragraph: `sweep_fingerprint`
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
    printf '%s build %s --force 2>&1 | sed "s/^/rtc /"; ' "$(switch_dune_cmd)" "$alias_name"
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
      "$host" "$(remote_capped "$CONTEXT_CAP" "$path_prefix $(remote_lock_cmd "$wt") $cmd" \
        "$(sleep_guard_why "$host" rtc-context)")" \
      >>"$log" 2>&1
  else
    run_capped "$CONTEXT_CAP" /bin/sh -c "$cmd" >>"$log" 2>&1
  fi
  return 0
}

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
# flock(1) rather than the perl the local side needs -- the remote boxes are
# Ubuntu, native or WSL, where util-linux provides it. Orphaned remote processes inherit the
# descriptor, so the lock outlives the ssh session that took it and clears when
# the last of them exits: the same property the local lock relies on.
remote_lock_cmd() {
  printf 'mkdir -p "$(dirname "%s")" && exec 9>"%s.lock" && flock -n 9 || exit 126; ' "$1" "$1"
}

# The guard's `why` for one leg of the current unit, as `systemd-inhibit --list` and
# `wake-lab.sh status` show it -- or nothing, where the destination is not a native boot.
# Every far-side leg that takes the worktree lock takes the guard too (preparation, the
# suite, the RTC context, the serial rerun): those are the legs doing work a suspend
# would destroy. The probes and the kernel-window read are seconds of read-only ssh.
sleep_guard_why() { # host leg
  [ "$(box_jobs_dest_transport "$1")" = native ] || return 0
  printf 'ocannl sweep %s %s %s' "$stamp" "$CURRENT_UNIT" "$2"
}
CURRENT_UNIT=

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
#
# A `git worktree prune` running beside the `worktree add` deletes the entry the
# add has just created and not yet locked, and the add dies on `could not open
# '.git/worktrees/<name>/locked'` (gh-ocannl-1532). Under git 2.55 (not 2.53)
# every fetch starts a detached `git maintenance run --auto` whose worktree-prune
# task runs whenever a stale entry exists, as a reused worktree's history leaves
# one: so both of this script's fetches pass `maintenance.auto=false`, and the
# add is tried once more after a prune for the pruners it does not start --
# another session's fetch in the same clone.
prep_cmd() {
  local repo=$1 wt=$2 add
  add=$(printf 'git -C "%s" worktree add -q --detach "%s" %s' "$repo" "$wt" "$full_sha")
  printf 'git -C "%s" worktree prune && ' "$repo"
  printf '{ git -C "%s" rev-parse --git-dir >/dev/null 2>&1 || ' "$wt"
  printf '%s || { git -C "%s" worktree prune && %s; }; } && ' "$add" "$repo" "$add"
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
# documented in docs/agent-notes/build-and-test.md#the-cross-machine-sweep.
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
  local window window_start window_end bursts window_kind_field tokens_field
  stage=$RUN_RECORD.stage.$$
  rows=$(run_rows) || return 1
  {
    # Schema 4: the `unit` row gained a tenth field, the window's KIND (`dxg`, `native`, or `-`),
    # when a native boot got a window of its own (gh-ocannl-1034). The count beside it now means
    # lost bridge messages for one kind and refused GPU queues for the other, so a schema-3 reader
    # -- to which every count is a `vmbus_sendpacket` count -- would mis-read a native one.
    # Schema 3: that count gained a fourth value, `vm-replaced` (a guest that
    # was destroyed and recreated mid-window). A schema-2 consumer's contract
    # permits only `-`, a number and `unavailable`, so a strict one would reject
    # the record at exactly the moment a replacement happened -- the record it
    # most needs to read. Schema 2: the `unit` row gained the dxg window and burst count
    # (gh-ocannl-979). A consumer picks its parser from this number, so widening a
    # row without it would make a strict schema-1 reader reject a current record
    # and a dxg-aware reader mis-read a historical one. The unit-STATE files below
    # keep their own schema 1: different file, different contract.
    # Schema 5: the `unit` row gained an eleventh field, the MEMORY MODEL the unit exercised
    # (unit_memory: `discrete/gfx1102`, `unified/gfx1151`, ... or `-`), and its outcome a value, `gate`,
    # when hip came to run on two boxes of different memory models (gh-ocannl-1035): a consumer
    # keyed on backend alone would merge them, and a schema-4 reader's outcome set has no `gate`.
    # Schema 6: the `unit` row gained a twelfth field, the box's GPU-TOKEN readings
    # (gpu_tokens_summary field: `opened=1/2,closed=0/2`, `?` for an unread one, `-` for none), and
    # the window's count a fifth value, `suspended` (the box slept inside the window), both for
    # gh-ocannl-1039/1177: a schema-5 reader's count set has no `suspended`. A `skip` row can now
    # also come from a unit that waited out its GPU token, and carries a log.
    printf 'schema\t6\n'
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
      # The unit's kernel window, its count (gh-ocannl-979) and the window's kind
      # (gh-ocannl-1034), read from the log the row names -- the only artifact that
      # holds them, and one this row already points at, so no third place can
      # disagree. `-` for a unit with no window: a local one, one that never ran,
      # or one whose transport has no window kind.
      # From the unit's collected-evidence sidecar, never from its log: a log
      # holds whatever the unit's tests printed (see window_sidecar).
      window_start=-
      window_end=-
      bursts=-
      window_kind_field=-
      if [ "$log" != - ]; then
        window=$(window_bounds "$log")
        if [ -n "$window" ]; then
          window_start=${window%% *}
          window_end=${window##* }
          # `-` no window, a number a window that was read, `unavailable` a
          # window whose collection failed: three distinguishable states, since
          # "not collected" and "collected and clean" mean opposite things. A
          # collection that failed before the box reported its bounds writes `-`
          # for both of those and `unavailable` here, which is still that state.
          bursts=$(window_count "$log")
          [ -n "$bursts" ] || bursts=-
          window_kind_field=$(window_kind "$log")
          [ -n "$window_kind_field" ] || window_kind_field=-
        fi
      fi
      # The GPU-token readings, from the unit's own sidecar beside the log the row names
      # (gpu_tokens_sidecar), as the window's fields come from theirs.
      tokens_field=-
      [ "$log" = - ] || tokens_field=$(gpu_tokens_summary field "$(gpu_tokens_sidecar "$log")")
      printf 'unit\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$machine" "$backend" \
        "$outcome" "$stopped" "$log" "$window_start" "$window_end" "$bursts" "$window_kind_field" \
        "$(unit_memory "$machine" "$backend")" "${tokens_field:--}"
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

# The rerun targets behind a log's dune-file, `(test)`-golden and inline-expectation sites, one
# per line, each prefixed `alias `, `inline ` or `unmapped `. A `(test (name X))` stanza reruns as
# `@<dir>/runtest-X`, the per-test alias dune generates from 3.20 (the project's floor); an
# explicit rule as `@<dir>/<its alias>`. A `(test)` stanza's golden diff -- the most common red,
# and a site that names only its `.expected` file -- reruns as the stanza its resolved `diff --git`
# header names: the second operand `<dir>/<name>.exe.output` is the run of `@<dir>/runtest-<name>`
# (gh-ocannl-1464). An explicit rule's `.expected` site stays unmapped even where its dune-file
# site is mapped: nothing in the log pairs the two. Each alias is printed once, however many sites
# name it. An inline expectation is proved by its source-to-`.corrected` diff and tagged with its
# directory's broad `runtest` alias: serial_rerun uses that fallback only when one or two such
# sites were unmapped, so a small environment-red unit still gets a serial retry without turning a
# wider red into a serial directory suite. Other source locations, unnamed spans and bare targets
# stay unmapped.
rerun_aliases() { # log
  sweep_fingerprint_sites "$1" | sort -u | awk -v log_file="$1" '
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
        if (raw ~ /^diff --git a\/_build\/default\/[A-Za-z0-9_.\/-]+\.expected b\/_build\/default\/[A-Za-z0-9_.\/-]+\.exe\.output$/) {
          split(raw, parts, / /)
          golden = parts[3]; output = parts[4]
          sub(/^a\/_build\/default\//, "", golden)
          sub(/^b\/_build\/default\//, "", output)
          sub(/\.exe\.output$/, "", output)
          dir = golden
          sub(/[^\/]+$/, "", dir)
          name = substr(output, length(dir) + 1)
          if (substr(output, 1, length(dir)) == dir && name != "" && name !~ /\//)
            stanza[golden] = "@" dir "runtest-" name
        }
      }
      close(log_file)
    }
    function alias(a) { if (!(a in printed)) { printed[a] = 1; print "alias " a } }
    /^File "[^"]*dune", (alias|name|names) "?[A-Za-z0-9_.-]+"?$/ {
      dir = $2
      sub(/^"/, "", dir)
      sub(/dune",$/, "", dir)
      if (dir ~ /^([A-Za-z0-9_.-]+\/)*$/) {
        value = $4
        sub(/^"/, "", value)
        sub(/"$/, "", value)
        if ($3 == "alias") alias("@" dir value)
        else alias("@" dir "runtest-" value)
        next
      }
    }
    /^File "[^"]+\.expected", lines? [0-9]+(-[0-9]+)?$/ {
      path = $2
      sub(/^"/, "", path)
      sub(/",$/, "", path)
      if (path in stanza) {
        alias(stanza[path])
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

# The targets of a regression's bisect (gh-ocannl-1375), one per line as `target <alias>`, each
# once, or `unmapped <site>` for a failing site no stanza is found for: exactly the stanzas
# rerun_aliases maps, an inline expectation's by its directory `runtest`.
bisect_targets() { # log
  rerun_aliases "$1" | awk '
    {
      line = $0
      if ($0 ~ /^alias /) line = "target " substr($0, 7)
      else if ($0 ~ /^inline /) { line = substr($0, 8); sub(/\t.*/, "", line); line = "target " line }
      if (!(line in seen)) { seen[line] = 1; print line }
    }'
}

# The ready bisect of a unit red on this commit whose previous verdict on the same scope was green
# (sweep_state_update's SWEEP_STATE_PASSED_AT, gh-ocannl-1375): `tools/test-run.sh bisect` with the
# green commit, this one and the failing stanzas (bisect_targets), on the summary beside the
# REGRESSION line, for the box the unit ran on -- a bisect runs where it is invoked. The command is
# printed, never run: a bisect is hours of the box, and it first checks both ends hold there.
say_bisect() { # label where backend good log
  local label=$1 where=$2 backend=$3 good=$4 log=$5 line shown more
  local targets=() unmapped=()
  if [ "$good" = "$full_sha" ]; then
    say "  $label: bisect: none -- the previous pass was at this same commit, so the test or its environment changed, not the code"
    return 0
  fi
  while IFS= read -r line; do
    case $line in
      "target "*) targets+=("${line#target }") ;;
      "unmapped "*) unmapped+=("${line#unmapped }") ;;
    esac
  done < <(bisect_targets "$log")
  if [ ${#targets[@]} -gt 0 ]; then
    say "  $label: bisect on $where: OCANNL_BACKEND=$backend tools/test-run.sh bisect $good $full_sha ${targets[*]}"
  else
    say "  $label: bisect: no command -- no failing site names a stanza"
  fi
  if [ ${#unmapped[@]} -gt 0 ]; then
    shown=$(printf ' [%s]' "${unmapped[@]:0:5}")
    more=
    [ ${#unmapped[@]} -le 5 ] || more=" and $(( ${#unmapped[@]} - 5 )) more"
    say "  $label: bisect: left out, no stanza found for:$shown$more"
  fi
}

# The rerun as shell text for the machine that owns the worktree, one dune call
# per stanza so each has its own status: a single call over all of them would
# report one verdict for the set. The markers are what serial_rerun reads back.
# The actions it re-runs rewrite their own verdict records (verdict_records_dir).
# A GPU unit's rerun starts with the device probe and runs each stanza under the
# per-test cap (device_probe_cmd, test_cap_start_cmd): it is triggered by a
# kernel-side refusal, when the device is most likely faulted, so a wedged device
# ends it as DEVICE UNHEALTHY or at the first hung stanza, never at the unit's CAP
# -- and either way the stanzas it did not reach are reported unjudged.
serial_rerun_cmd() { # backend wt host alias...
  local backend=$1 wt=$2 host=$3 a watched
  shift 3
  printf 'cd "%s" || exit 127; ' "$wt"
  verdict_records_env "$wt"
  device_probe_cmd "$backend" "$wt" "$host" || :
  for a in "$@"; do
    printf 'echo "=== serial rerun %s ==="; ' "$a"
    watched=$(test_cap_start_cmd "$backend" "$wt" "$a") && printf '%s' "$watched"
    printf 'OCANNL_BACKEND=%s %s build -j 1 %s; ' "$backend" "$(switch_dune_cmd)" "$a"
    printf 'rc=$?; '
    [ -z "$watched" ] || test_cap_stop_cmd
    printf 'echo "=== serial rerun %s: exit $rc ==="; ' "$a"
  done
  printf 'exit 0'
}

# The completion pass that follows a clean serial rerun on a forced full-suite
# unit: the unit's own aliases again, at `-j 1` and WITHOUT --force, so Dune
# runs exactly the actions that never completed -- the dependents a red
# prerequisite held back, which rerunning the red stanzas alone does not reach
# -- and replays nothing that already passed. Its exit 0 is the proof that
# every action of the suite completed, its verdict records written, which is
# what lets rerun_cleared stage the unit as skip evidence. A GPU unit's pass runs
# under the per-test cap (test_cap_start_cmd), unless it reaches `@slow`.
suite_completion_cmd() { # backend wt
  local slow_alias= watched
  [ "$SLOW" = 1 ] && slow_alias=' @slow'
  printf 'cd "%s" || exit 127; ' "$2"
  verdict_records_env "$2"
  printf 'echo "=== suite completion ==="; '
  # shellcheck disable=SC2086  # the slow alias is word-split on purpose
  watched=$(test_cap_start_cmd "$1" "$2" @runtest @train $slow_alias) && printf '%s' "$watched"
  printf 'OCANNL_BACKEND=%s %s build -j 1 @runtest @train%s; ' "$1" "$(switch_dune_cmd)" "$slow_alias"
  printf 'rc=$?; '
  [ -z "$watched" ] || test_cap_stop_cmd
  printf 'echo "=== suite completion: exit $rc ==="; '
  printf 'exit 0'
}

# Run a post-unit shell command where the unit's worktree lives, under its lock
# and the unit's CAP, appending to the log (the serial rerun's legs, the box check).
# The serial rerun's legs -- its stanzas and the completion pass -- hold the GPU, so
# where the unit takes a token they each take one too (gh-ocannl-1177); the box check
# runs no GPU test and takes none.
run_on_unit_host() { # host wt cmd log path_prefix [leg]
  local host=$1 wt=$2 cmd=$3 log=$4 path_prefix=$5 leg=${6:-serial-rerun} remote budget=$(( CAP + 300 ))
  if [ -n "$host" ]; then
    remote=$(remote_capped "$CAP" "$path_prefix $(remote_lock_cmd "$wt") $cmd" \
      "$(sleep_guard_why "$host" "$leg")")
    if [ -n "$TOKEN_TAKE" ] && [ "$leg" = serial-rerun ]; then
      remote=$(gpu_token_take_cmd "$TOKEN_FW" "$TOKEN_WAIT" "$wt.gpu-token" "$remote")
      budget=$(( budget + TOKEN_WAIT ))
    fi
    run_capped "$budget" ssh -o BatchMode=yes -o ConnectTimeout=8 \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=10 \
      "$host" "$remote" >>"$log" 2>&1
  else
    run_capped "$CAP" /bin/sh -c "$cmd" >>"$log" 2>&1
  fi
}

# Rerun an environment-red unit's failing stanzas one at a time, appending to
# its log, then write the `serial rerun:` verdict lines that `sweep_fingerprint`
# carries and the summary quotes. The shape is collect_rtc_context's: its own
# phase after the row is recorded, under the worktree lock on whichever side
# owns the tree, with a status that never reaches the outcome. The budget is the
# unit's own CAP rather than CONTEXT_CAP, because this is not a diagnostic of
# fixed size but the suite's red stanzas run again -- 27 of them on the day this
# was measured -- and a stanza the cap cut short is reported `unjudged`, never
# folded into `all clean`. On a forced full-suite unit an `all clean` rerun is
# followed by suite_completion_cmd, reported as `suite completed` or not.
serial_rerun() { # backend host wt log label [path_prefix]
  local backend=$1 host=$2 wt=$3 log=$4 label=$5 path_prefix=${6:-}
  local line cmd started rc a entry site stanza_count inline_count fallback_suffix=s completion
  local log_lines bound
  local aliases=() fallback_aliases=() inline_entries=() inline_sites=()
  local unmapped=() red=() unjudged=()
  environment_red "$log" "$backend" || return 0
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
  # Where this phase's lines start in the log, for the bounds' own lines below.
  log_lines=$(wc -l <"$log" | tr -d ' ')
  if [ ${#aliases[@]} -gt 0 ]; then
    cmd=$(serial_rerun_cmd "$backend" "$wt" "$host" "${aliases[@]}")
    echo "=== serial rerun: ${#aliases[@]} stanzas at -j 1 ===" >>"$log"
    run_on_unit_host "$host" "$wt" "$cmd" "$log" "$path_prefix"
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
  # Only a clean rerun of a forced full-suite unit can become skip evidence, so
  # only that one pays for the completion pass.
  completion=
  if [ ${#red[@]} -eq 0 ] && [ ${#unjudged[@]} -eq 0 ] && [ ${#aliases[@]} -gt 0 ] &&
    [ ${#unmapped[@]} -eq 0 ] && [ "$FORCE" = 1 ] && [ -z "$TARGET" ]; then
    run_on_unit_host "$host" "$wt" "$(suite_completion_cmd "$backend" "$wt")" "$log" "$path_prefix"
    line=$(grep -hF -- '=== suite completion: exit ' "$log" | tail -1)
    case $line in
      "") completion='serial rerun: suite completion unjudged' ;;
      *": exit 0 ==="*) completion='serial rerun: suite completed' ;;
      *) completion="serial rerun: suite completion red (${line#=== suite completion: }"
         completion="${completion% ===})" ;;
    esac
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
    [ -z "$completion" ] || printf '%s\n' "$completion"
  } >>"$log"
  [ ${#fallback_aliases[@]} -eq 1 ] && fallback_suffix=
  say "  $label: environment-red, $stanza_count stanzas and ${#fallback_aliases[@]} directory fallback$fallback_suffix rerun at -j 1 ($(( $(date +%s) - started ))s)"
  grep -h '^serial rerun: ' "$log" | sed "s|^|  $label: |" >>"$LANE_OUT"
  # Why a bounded rerun stopped, from its own lines of the log: for the summary only, never a
  # `serial rerun:` line, which the fingerprint carries and a pid or a timing would move.
  bound=$(tail -n "+$((log_lines + 1))" "$log" | sed -n 's/^sweep: device probe: a fresh device query for \(.*\): the$/\1/p' |
    head -n 1)
  [ -z "$bound" ] || say "  $label: serial rerun: DEVICE UNHEALTHY, no stanza ran: a fresh device query for $bound"
  bound=$(tail -n "+$((log_lines + 1))" "$log" | sed -n 's/^sweep: per-test cap: //p' | head -n 1)
  [ -z "$bound" ] || say "  $label: serial rerun: hung test, past the per-test cap: $bound"
  # A leg of the rerun its GPU token never came to (gh-ocannl-1177): what it would have run is
  # unjudged above, and this says why.
  bound=$(gpu_token_refusal "$log" "$((log_lines + 1))")
  [ -z "$bound" ] || say "  $label: serial rerun: a leg ran nothing, refused its GPU token: $bound"
  say_token_warning "$log" "$label"
  return 0
}

# Whether serial_rerun cleared every failure of the log's unit AND proved the
# rest of the suite complete: `all clean`, `suite completed`, and no verdict
# line that leaves a site unjudged, unmapped or red. A directory fallback is
# judged inside `all clean`, so it does not disqualify.
rerun_cleared() { # log
  local line clean= completed=
  while IFS= read -r line; do
    case $line in
      'serial rerun: all clean') clean=1 ;;
      'serial rerun: suite completed') completed=1 ;;
      'serial rerun: directory fallback '*) ;;
      *) return 1 ;;
    esac
  done < <(grep -h '^serial rerun: ' "$1" 2>/dev/null)
  [ -n "$clean" ] && [ -n "$completed" ]
}

# The unit's per-action verdict records (verdict_records_dir), every action's
# file concatenated, from whichever side owns the worktree and under its lock --
# the next unit on that worktree starts the directory empty. Each file opens with
# its OCANNL_TOOL_VERDICT_ACTION header, so the files stay distinguishable in
# the concatenation; `awk 1` terminates a final line a killed process left
# unterminated. Exit 3 is a missing directory.
verdict_records_cmd() { # wt
  printf 'records="%s"; [ -d "$records" ] || exit 3; ' "$(verdict_records_dir "$1")"
  printf 'for f in "$records"/*.tsv; do [ -f "$f" ] || continue; awk 1 "$f" || exit 4; done; exit 0'
}

collect_verdict_records() { # host wt dest log path_prefix
  local host=$1 wt=$2 dest=$3 log=$4 path_prefix=$5 cmd
  cmd=$(verdict_records_cmd "$wt")
  if [ -n "$host" ]; then
    run_capped "$(( CONTEXT_CAP + 120 ))" ssh -o BatchMode=yes -o ConnectTimeout=8 \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      "$host" "$(remote_capped "$CONTEXT_CAP" "$path_prefix $(remote_lock_cmd "$wt") $cmd" \
        "$(sleep_guard_why "$host" verdict-records)")" \
      >"$dest" 2>>"$log"
  else
    run_capped "$CONTEXT_CAP" /bin/sh -c "$cmd" >"$dest" 2>>"$log"
  fi
}

# Stage a qualifying unit's records for the skip aggregation. The lane is a
# subshell, so the evidence cannot be appended to the top-level SKIP_RUN_ arrays
# from here; it is left as a per-unit file that the top level reads back, in
# table order, once every lane has finished. A unit whose records cannot be
# read, or hold no action at all -- a swept commit whose Verdict predates them
# -- is not evidence, and says so: its absence of skip records would otherwise
# read as every claim executed.
stage_skip_evidence() { # machine backend host wt log path_prefix [cleared]
  local machine=$1 backend=$2 host=$3 wt=$4 log=$5 path_prefix=$6 cleared=${7:-}
  local records=${5%.log}.verdict-records
  if ! collect_verdict_records "$host" "$wt" "$records" "$log" "$path_prefix"; then
    say "  $machine/$backend: skip evidence unavailable -- its per-action verdict records could not be read"
    return 0
  fi
  if ! grep -q "^OCANNL_TOOL_VERDICT_ACTION$(printf '\t')" "$records"; then
    say "  $machine/$backend: skip evidence unavailable -- no per-action verdict records (a swept commit predating gh-ocannl-1114?)"
    return 0
  fi
  printf '%s\n' "$records" >"$LANE_DIR/skip-run.$machine.$backend" ||
    die "cannot stage skip evidence for $machine/$backend"
  if [ -n "$cleared" ]; then
    : >"$LANE_DIR/skip-cleared.$machine.$backend" ||
      die "cannot stage skip evidence for $machine/$backend"
  fi
}

if [ "$FORCE" = 1 ]; then
  execution=forced
else
  execution=incremental
fi

# A native unit that ran without its sleep guard says so on its summary line, where the scheduled
# routine reads the run: the supervisor's warning is in the log, and a missing polkit grant that
# only a log records is one nobody installs (see capped_perl in tools/sweep-supervisor.sh for why
# the unit ran anyway).
# Called after EVERY log-writing phase that ran guarded legs, since a later one may replace the log
# (the suite's `>"$log"` truncates the preparation's); once per unit, so a box refusing every leg
# (a missing polkit grant) says so once rather than per leg.
say_unguarded() { # log label
  local line
  [ "$UNGUARDED_SAID" = 0 ] || return 0
  line=$(grep -m 1 '^sweep-hold: WARNING: ' "$1" 2>/dev/null) || return 0
  UNGUARDED_SAID=1
  say "  $2: ran WITHOUT a sleep guard -- ${line#* -- }"
}
UNGUARDED_SAID=0

# The fleet's GPU tokens around a remote GPU unit (gh-ocannl-1177; tools/gpu-tokens.sh says what each
# reading holds and where the tokens bind). TOKEN_TAKE is set for a unit whose box's tokens bind,
# with the box's own fleet-worker.sh and fleet name beside it; every GPU leg of that unit then runs
# inside a token (gpu_token_take_cmd). Lane-local, reset per unit.
TOKEN_TAKE=
TOKEN_FW=
TOKEN_BOX=
TOKEN_WARNED=0

# One reading of a remote GPU unit's box, appended to the unit's sidecar. Through run_capped and a
# file, for the reason remote_guest_id gives. Asked of the box's own fleet-worker.sh, found under
# the box's home by the same candidate list this host's is found by; `OCANNL_TOOL_FLEET_WORKER=none`
# asks nothing, as it consults no registry.
read_gpu_tokens() { # event host log remote-home
  local event=$1 host=$2 log=$3 answer rc c cands=()
  [ "${OCANNL_TOOL_FLEET_WORKER-}" != none ] || return 0
  while IFS= read -r c; do [ -n "$c" ] && cands+=("$c"); done < <(fleet_worker_candidates "$4")
  answer=$log.tokens.$$
  run_capped 90 ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=2 "$host" \
    "$(remote_capped 60 "$(gpu_tokens_cmd "$event" "${cands[@]:-}")")" >"$answer" 2>/dev/null </dev/null
  rc=$?
  gpu_tokens_append "$(gpu_tokens_sidecar "$log")" "$event" "$answer" "$rc" ||
    say "  $CURRENT_UNIT: WARNING -- could not record its $event GPU-token reading"
  rm -f "$answer"
}

# A leg that ran without the token it should have had says so once per unit, where the routine reads.
say_token_warning() { # log label
  local line
  [ "$TOKEN_WARNED" = 0 ] || return 0
  line=$(grep -m 1 '^sweep: gpu token: WARNING: ' "$1" 2>/dev/null) || return 0
  TOKEN_WARNED=1
  say "  $2: ${line#sweep: }"
}

# One unit, start to finish: preparation, the capped suite, the recorded row, and
# the post-unit phases (RTC context, serial rerun, fingerprint, unit state), all
# on the machine that owns the unit and inside its lane. The post-unit phases
# re-take that machine's worktree lock, so they must not be split off to run
# after the lane has moved on to its next unit.
run_unit() { # machine backend host
  local machine=$1 backend=$2 host=$3
  local log started remote_home remote_probe remote_started remote_boot wt path_prefix= remote_repo
  local remote_prep remote rc elapsed outcome unreachable=skip unguarded hung unhealthy
  local refused take_slots= take_tokens= line
  WRITTEN_FINGERPRINT=
  CURRENT_UNIT=$machine/$backend
  UNGUARDED_SAID=0
  TOKEN_TAKE=
  TOKEN_FW=
  TOKEN_BOX=
  TOKEN_WARNED=0
  # A gated box that answered its status check and then not the unit (its Wi-Fi dropped, it went
  # back to sleep) is still outside this run's control: `gate`, as it would have been a minute
  # earlier, not the `skip` that asks why the caller's wake did not take.
  [ -n "$host" ] && lab_box_gated "$(lab_box_of "$host")" && unreachable=gate

  log=$LOGS/$stamp-$machine-$backend.log
  rm -f "$(gpu_tokens_sidecar "$log")"
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
    # trip. The kernel window's start has to be an instant in the clock that
    # timestamps that box's kernel log, and this is the moment the unit begins on
    # it; deriving it later by subtracting a locally measured duration assumes the
    # remote clock advanced continuously meanwhile, which is the assumption a WSL
    # VM breaks when it resynchronises after a host resume.
    # Armed BEFORE the probe, for a gated box: a cancellation landing while it connects must still
    # leave the EXIT path owing the box its sleep (LANE_DIALLING, sleep_gated_box_after_early_exit).
    LANE_DIALLING=1
    if ! remote_probe=$(capped 60 ssh -o BatchMode=yes -o ConnectTimeout=8 \
         -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
         "$host" 'printf "%s\n%s\n%s\n" "$HOME" "$(date +%s)" \
           "$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"' 2>/dev/null) ||
       [ -z "$remote_probe" ]; then
      LANE_DIALLING=0
      say "  $machine/$backend: $unreachable (unreachable)"
      record "$machine" "$backend" "$unreachable" 0
      sweep_state_update "$machine" "$backend" "$unreachable"
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
    LANE_REACHED=1
    LANE_DIALLING=0
    [ -n "$remote_home" ] || {
      say "  $machine/$backend: $unreachable (unreachable)"
      record "$machine" "$backend" "$unreachable" 0
      sweep_state_update "$machine" "$backend" "$unreachable"
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
    # rog's WSL guest needs the CUDA and WSL lib dirs on PATH; harmless elsewhere. The suffix is a
    # reliable reading of the boot kind because a remote host is always the `-linux` or `-wsl`
    # alias on its box's endpoint-map row -- the override accepts nothing else (lab_dest), and
    # wake-lab's row rules give each OS its suffix. A native-Ubuntu
    # boot has no /usr/lib/wsl/lib, and its non-login ssh PATH already carries the toolchain
    # (~/.bashrc sources the fleet env before its interactive guard), so it keeps only CUDA's bin,
    # which is harmless where the toolchain already put it (gh-ocannl-1030).
    case $host in
      *-wsl) path_prefix="export PATH=/usr/local/cuda/bin:/usr/lib/wsl/lib:\$PATH;" ;;
      *) path_prefix="export PATH=/usr/local/cuda/bin:\$PATH;" ;;
    esac
    # A GPU unit's box: who holds its GPU tokens now (the `opened` reading), what the registry has
    # reserved there, and -- where the tokens bind -- the fleet-worker.sh this unit takes one
    # through (gh-ocannl-1177). Before the preparation, so a contended box is on record however the
    # unit then ends.
    if ! box_jobs_cpu_backend "$backend"; then
      [ -z "${BOX_RESERVATIONS:-}" ] ||
        gpu_tokens_append_reservations "$(gpu_tokens_sidecar "$log")" "$(lab_box_of "$host")" \
          "$BOX_RESERVATIONS" || :
      read_gpu_tokens opened "$host" "$log" "$remote_home"
      # `<fleet-worker.sh> <box> <slots> <tokens>`; the tokens bind where there are fewer of them.
      read -r TOKEN_FW TOKEN_BOX take_slots take_tokens _ \
        <<<"$(gpu_tokens_summary take "$(gpu_tokens_sidecar "$log")")"
      case ${take_slots:-x}${take_tokens:-x} in
        *[!0-9]*) ;;
        *) [ "$take_tokens" -ge "$take_slots" ] || TOKEN_TAKE=1 ;;
      esac
    fi
    # Preparation is its own ssh round trip so that its failure -- a connection
    # dropped after the probe, a full disk, a wedged worktree -- is recorded as
    # `error`, matching the local path. Folded into the test command it would
    # have surfaced as a non-zero status in the generic branch below and been
    # written down as a FAILING SUITE, which is the opposite of the truth: a
    # remote that never got as far as dune tested nothing at all.
    remote_repo="$remote_home/ocannl-staging"
    remote_prep="git -C \"$remote_repo\" -c maintenance.auto=false fetch -q origin master && $(prep_cmd "$remote_repo" "$wt")"
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
         "$host" "$(remote_capped 600 "$path_prefix $(remote_lock_cmd "$wt") $remote_prep" \
           "$(sleep_guard_why "$host" prep)")" \
         >"$log" 2>&1; then
      say "  $machine/$backend: error (cannot pin $host to $run_sha)"
      say_unguarded "$log" "$machine/$backend"
      record "$machine" "$backend" error "$(( $(date +%s) - started ))" "$log"
      # The preparation can run for ten minutes, and a guest replaced inside it ends the unit
      # right here -- so this path collects the window exactly like the one below.
      finish_remote_window "$machine" "$backend" "$host" "$log" error \
        "${remote_started:-}" "${remote_boot:-}"
      sweep_fingerprint_write "$log" "$machine/$backend"
      sweep_state_update "$machine" "$backend" error "$WRITTEN_FINGERPRINT"
      return 0
    fi
    # The suite's log replaces the preparation's, so a guard the preparation alone was refused is
    # read now or never.
    say_unguarded "$log" "$machine/$backend"
    # The cap is applied on the FAR side: killing the local ssh would leave the
    # remote dune running. ONE cap around the whole unit -- the same perl
    # supervisor capped() uses locally, see remote_capped -- because a
    # per-dune-call cap would let a --slow unit run for twice the budget the
    # script advertises.
    remote="$(remote_capped "$CAP" "$path_prefix $(remote_lock_cmd "$wt") $(test_cmd "$backend" "$wt" "$(box_jobs_sweep_jobs "$machine" "$backend" "$host")" "$host")" \
      "$(sleep_guard_why "$host" suite)")"
    # Inside one of the box's GPU tokens where they bind, waited for up to TOKEN_WAIT on the far side.
    [ -z "$TOKEN_TAKE" ] || remote=$(gpu_token_take_cmd "$TOKEN_FW" "$TOKEN_WAIT" "$wt.gpu-token" "$remote")
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
    run_capped "$(( CAP + 300 + ${TOKEN_TAKE:+TOKEN_WAIT+}0 ))" \
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
      sweep_fingerprint_write "$log" "$machine/$backend"
      sweep_state_update "$machine" "$backend" error "$WRITTEN_FINGERPRINT"
      return 0
    fi
    run_capped "$CAP" /bin/sh -c "$(test_cmd "$backend" "$wt" "$(box_jobs_sweep_jobs "$machine" "$backend" "$host")" "$host")" >"$log" 2>&1
    rc=$?
  fi

  elapsed=$(( $(date +%s) - started ))
  # A unit whose box's GPU tokens stayed taken past its wait never started its suite: a `skip`,
  # like a box another lane holds -- nothing was tested and nothing failed -- naming the batches that
  # held the tokens, read the moment the wait ended (gh-ocannl-1177). Never run anyway: that is the
  # third concurrent cuda batch the box's two tokens exist to exclude. 75 with the take's own
  # refusal line only; any other 75 is whatever the suite exited with.
  refused=
  [ "$rc" != 75 ] || [ -z "$TOKEN_TAKE" ] || refused=$(gpu_token_refusal "$log")
  if [ -n "$refused" ]; then
    read_gpu_tokens refused "$host" "$log" "$remote_home"
    say "  $machine/$backend: skip (no GPU token on $TOKEN_BOX: $refused; ${elapsed}s)"
    while IFS= read -r line; do say "  $machine/$backend: $line"; done \
      < <(gpu_tokens_summary report "$(gpu_tokens_sidecar "$log")")
    record "$machine" "$backend" skip "$elapsed" "$log"
    sweep_state_update "$machine" "$backend" skip
    return 0
  fi
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
  #
  # 69 with the device probe's refusal in the log is a wedged device (device_probe_cmd): dune never
  # started, so it is `error` too, and the line below names it DEVICE UNHEALTHY. A 69 without that
  # refusal is whatever the suite exited with, as before.
  unhealthy=
  [ "$rc" != 69 ] ||
    unhealthy=$(sed -n 's/^sweep: device probe: a fresh device query for \(.*\): the$/\1/p' "$log" 2>/dev/null | head -n 1)
  case $rc in
    # `pass` is reserved for the mode that makes Dune re-execute every action
    # attached to the selected aliases. An incremental success is useful, but
    # is an unknown mixture of execution and cache hits, so it must not refresh
    # a consumer that ages actual backend coverage by the latest `pass` row.
    0) [ "$FORCE" = 1 ] && outcome=pass || outcome=incremental-pass ;;
    124 | 137 | 142) outcome=timeout ;;
    126 | 127) outcome=error ;;
    69) [ -n "$unhealthy" ] && outcome=error || outcome=fail ;;
    255) [ -n "$host" ] && outcome=error || outcome=fail ;;
    *) outcome=fail ;;
  esac
  say "  $machine/$backend: $outcome (${elapsed}s; execution=$execution)"
  # A unit the per-test cap ended names its hung test in the log (test_cmd); say which.
  if [ "$outcome" = timeout ]; then
    hung=$(sed -n 's/^sweep: per-test cap: //p' "$log" 2>/dev/null | head -n 1)
    [ -z "$hung" ] || say "  $machine/$backend: hung test, past the per-test cap: $hung"
  fi
  # And a unit the device probe refused, which device it found wedged.
  [ -z "$unhealthy" ] ||
    say "  $machine/$backend: DEVICE UNHEALTHY, dune not started: a fresh device query for $unhealthy"
  record "$machine" "$backend" "$outcome" "$elapsed" "$log" "$execution"
  # Who held the box's GPU tokens as the suite ended (the `closed` reading), and every reading that
  # saw another batch on the GPU, named on the summary: a launch OOM beside them is contention until
  # the serial rerun says otherwise. A remote GPU unit only, as its `opened` reading was.
  if [ -n "$host" ] && ! box_jobs_cpu_backend "$backend"; then
    read_gpu_tokens closed "$host" "$log" "$remote_home"
    while IFS= read -r line; do say "  $machine/$backend: $line"; done \
      < <(gpu_tokens_summary report "$(gpu_tokens_sidecar "$log")")
    say_token_warning "$log" "$machine/$backend"
  fi
  # The kernel's own evidence for THIS unit's window, before the rerun
  # decision that reads it -- and before the RTC diagnostics below, which is not
  # mere ordering: `nvidia-smi` and `rocminfo` cross /dev/dxg themselves, so a
  # window whose end were taken after them could count the DIAGNOSTIC's lost
  # messages as the unit's and mark a plainly test-logic failure environment-red.
  # The end bound is the remote's clock at this point, so those fall outside it.
  # Remote GPU units only, and of the kind their transport names (finish_remote_window):
  # a local unit has no window and minix's multidev_cc -- CPU -- would only ever collect
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
  # A forced full-suite pass is skip evidence, and so is a red the serial rerun
  # wholly cleared whose completion pass then ran every action the red had held
  # back: every action completed in the suite, the rerun or the completion
  # pass, and each one's record file holds its final attempt. Dropping such a
  # red lost minix/hip's evaluations on 2026-09-27 and reported its hip-only
  # claims as skipped on every box.
  if [ -z "$TARGET" ]; then
    if [ "$outcome" = pass ]; then
      stage_skip_evidence "$machine" "$backend" "$host" "$wt" "$log" "${path_prefix:-}"
    elif [ "$outcome" = fail ] && rerun_cleared "$log"; then
      stage_skip_evidence "$machine" "$backend" "$host" "$wt" "$log" "${path_prefix:-}" cleared
    fi
  fi
  case $outcome in
    fail | timeout | error) sweep_fingerprint_write "$log" "$machine/$backend" ;;
  esac
  sweep_state_update "$machine" "$backend" "$outcome" "${WRITTEN_FINGERPRINT:-}" "$log"
  [ -z "${SWEEP_STATE_PASSED_AT:-}" ] ||
    say_bisect "$machine/$backend" "${host:-$machine}" "$backend" "$SWEEP_STATE_PASSED_AT" "$log"
  # After the LAST guarded leg (the RTC context and the serial rerun append to the log too), so a
  # guard refused to any of them is reported, not only one refused to the suite.
  say_unguarded "$log" "$machine/$backend"
  WRITTEN_FINGERPRINT=
  case $outcome in
    pass | incremental-pass | fail)
      box_check "$machine" "$host" "$wt" "${remote_repo:-$MAIN}" "${path_prefix:-}" ;;
  esac
}

# The box's measurement prerequisites (gh-ocannl-1062): benchmarks/check-box.sh, once per lane, in
# the worktree of the lane's first unit whose suite ran -- so pinned to $run_sha -- against the
# box's shared venv, toolchain and the fixture store of its main checkout. Its verdict is the
# BOX's, never the unit's: no sweep unit measures, so a gap is an environment gap for the next
# measurement session, said on a line of its own and kept in <stamp>-<machine>-box-check.log,
# and no history row changes. A swept ref without the script says so; a timed-out unit leaves
# the check to the lane's next unit, since its device may still be wedged.
LANE_BOX_CHECKED=0
box_check() { # machine host wt repo path_prefix
  local machine=$1 host=$2 wt=$3 repo=$4 path_prefix=$5 log rc line
  [ "$LANE_BOX_CHECKED" = 0 ] || return 0
  LANE_BOX_CHECKED=1
  log=$LOGS/$stamp-$machine-box-check.log
  run_on_unit_host "$host" "$wt" "cd \"$wt\" && if [ -f benchmarks/check-box.sh ]; then \
bash benchmarks/check-box.sh --box \"$machine\" --box-only --fixture-dir \"$repo/benchmarks/fixtures\"; \
else echo 'check-box: not at this revision'; exit 3; fi" "$log" "$path_prefix" box-check
  rc=$?
  # Its own guarded leg, logged apart from the unit's: a refusal to it alone is read here or never.
  say_unguarded "$log" "$machine: box check"
  case $rc in
    0) say "  $machine: box check: ready -- $log" ;;
    1)
      say "  $machine: box check: ENVIRONMENT GAP (not a test failure) -- $log"
      while IFS= read -r line; do say "    $line"; done < <(grep '^check-box: .*: GAP -- ' "$log")
      ;;
    3) say "  $machine: box check: not at $run_sha" ;;
    *) say "  $machine: box check: could not run (exit $rc) -- $log" ;;
  esac
}

# Publish the lane's buffered unit lines as one block, under a run-wide lock so
# two lanes finishing together cannot interleave their blocks. The buffer is
# emptied only after a publication that succeeded: a failed one returns non-zero
# with the lines still in it, for the caller to die over and the top level to
# rescue (publish_lane_buffers).
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

# The lane's EXIT trap. bash 3.2 runs a subshell's EXIT trap only on an explicit
# `exit`, which is why run_lane ends with one.
#
# It prints NOTHING. A lane that dies or is cancelled mid-unit leaves what its unit
# had said in its buffer, and the top level prints that once the lane has exited
# (publish_lane_buffers): a trap runs with the stdout of whatever it interrupted,
# and mid-unit that is the unit's LOG -- `run_capped ... >"$log"` holds that
# redirection while it waits, and the relay's `exit` runs the trap inside it. So
# a cancelled lane's last lines, the gated box's early-sleep line among them, went
# into the unit's log and never reached the sweep's output (gh-ocannl-1039; read
# at first as the top level not waiting for its lanes under bash 3.2, which it
# does).
lane_exit() {
  sleep_gated_box_after_early_exit
}

# What the lanes buffered and did not publish themselves, printed by the top level
# in table order once every lane has exited: on the normal path that is a lane that
# stopped on its own `die`, and on a cancellation every lane that was mid-unit. Onto
# stderr where stdout is what fails, rather than lost.
publish_lane_buffers() {
  local lane buffer
  [ -n "$LANE_DIR" ] && [ -d "$LANE_DIR" ] || return 0
  for lane in "${LANES[@]:-}"; do
    buffer=$LANE_DIR/output.$lane
    [ -n "$lane" ] && [ -s "$buffer" ] || continue
    cat "$buffer" 2>/dev/null || cat "$buffer" >&2
    : >"$buffer"
  done
}

# The early-exit half of the gated box's sleep (run_lane has the normal one). A lane that reached
# its gated box and then stopped -- cancelled (relay reaps the unit's supervisor before it exits),
# or dead of its own `die` -- would otherwise leave a timer-woken laptop up all day. Not run in the
# lane's foreground: a cancellation must not wait out wake-lab's confirm-down, so the request is
# DETACHED -- its own session, so the group TERM that cancelled the run does not take it too, and
# without the run's locks, so it holds neither the worktree nor the box -- and BOUNDED, under the
# same group-reaping supervisor and 600 s budget as the normal path's request, since nothing waits
# for it. Owed from the moment the lane dials the box (LANE_DIALLING), not only once the probe has
# answered: a cancellation can land inside the probe. The lane's box reservation
# goes first, or the sleep would be refused by the lane that asked for it. Its output lands beside
# the run's logs, since the lanes' coordination directory is removed when the run ends.
sleep_gated_box_after_early_exit() {
  local box=${LANE_GATED_BOX:-} out
  [ -n "$box" ] && [ "${LANE_SLEPT:-0}" = 0 ] || return 0
  [ "${LANE_REACHED:-0}" = 1 ] || [ "${LANE_DIALLING:-0}" = 1 ] || return 0
  LANE_SLEPT=1
  exec 8>&-
  out=$LOGS/$stamp-$box-sleep.log
  (exec 9>&- </dev/null >"$out" 2>&1
   exec perl -e 'use POSIX (); POSIX::setsid(); exec @ARGV or exit 127' -- \
     perl -e "$capped_perl" -- 600 "$WAKE_LAB" sleep "$box") &
  early_say "  $box: the lane stopped early, so its sleep was requested in the background (wake-lab.sh sleep $box; output in $out)"
}

# `say` for a lane on its way out, which must not `die` over a buffer that is gone. The top level
# waits for its lanes before it removes the coordination directory the buffer lives in, and prints
# the buffer after the lane exits (publish_lane_buffers), so the line normally reaches the sweep's
# output; should the buffer be gone after all, it goes onto stderr, and the durable record of the
# early sleep is its own log beside the run's either way.
early_say() {
  printf '%s\n' "$*" 2>/dev/null >>"$LANE_OUT" || printf '%s\n' "$*" >&2
}

# Whether a gated box is up, by the lab's own reading of it: `wake-lab.sh status <box>` reaching the
# box's native Linux (`linux=UP`). Anything else -- asleep, powered off, booted into another OS, a
# status that could not be read -- is a gate, with LAB_GATE_REASON saying which. A remote lane
# exists only when wake-lab.sh gave its endpoint map (lab_map), so there is always a script to ask. Through run_capped and a file, not a command substitution, so a
# cancellation reaches it (see remote_guest_id).
lab_box_up() { # box
  local box=$1 out line
  LAB_GATE_REASON=
  out=$LANE_DIR/status.$box
  run_capped 120 "$WAKE_LAB" status "$box" >"$out" 2>&1
  line=$(grep -m 1 "^$box[[:space:]]" "$out" 2>/dev/null | tr -s ' \t' '  ')
  case " $line " in *" linux=UP "*) return 0 ;; esac
  line=${line#"$box "}
  LAB_GATE_REASON="$box not up: ${line:-wake-lab.sh status said nothing about it}"
  return 1
}

# Put a gated box back to sleep, and say what happened. Never a failure of the sweep: the units'
# rows are already written, and a box that stays awake costs power, not coverage. A refusal is the
# interlock working -- a block inhibitor (another run on the box) or a lab lock (another session
# here) -- and names its holder; anything else is a WARNING for the operator.
sleep_gated_box() { # box
  local box=$1 out rc line holder
  out=$LANE_DIR/sleep.$box
  run_capped 600 "$WAKE_LAB" sleep "$box" >"$out" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    say "  $box: put back to sleep (wake-lab.sh sleep $box)"
    return 0
  fi
  line=$(grep -m 1 'REFUSED' "$out" 2>/dev/null | sed 's/^ *//')
  if [ -n "$line" ]; then
    holder=$(grep -m 1 'Operation inhibited by' "$out" 2>/dev/null | sed 's/^ *//')
    say "  $box: left awake, not a failure -- $line${holder:+ ($holder)}"
  else
    say "  $box: WARNING -- wake-lab.sh sleep $box exited $rc: $(tail -1 "$out" 2>/dev/null | sed 's/^ *//')"
  fi
  return 0
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
  local lane=$1 unit machine backend host lane_host= lab_box= lab_lock_rc lane_names
  LANE_PIDS=
  LANE_REACHED=0
  LANE_BOX_CHECKED=0
  LANE_DIALLING=0
  # The gated box this lane holds, once it holds one; the EXIT trap puts it to sleep if the lane
  # stops before the normal path does (sleep_gated_box_after_early_exit).
  LANE_GATED_BOX=
  LANE_SLEPT=0
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
    # A gated box is asked whether it is up BEFORE anything reserves it or dials it: asleep is its
    # ordinary state on a day nothing woke it, and every unit of the lane records `gate` saying so.
    if lab_box_gated "$lab_box" && ! lab_box_up "$lab_box"; then
      for unit in "${UNITS[@]}"; do
        IFS=: read -r machine backend host <<<"$unit"
        [ "$machine" = "$lane" ] || continue
        wanted "$backend" || continue
        say "  $machine/$backend: gate ($LAB_GATE_REASON)"
        record "$machine" "$backend" gate 0
        sweep_state_update "$machine" "$backend" gate
        flush_lane_output || die "cannot publish the $machine/$backend summary to stdout"
      done
      : >"$LANE_DIR/lane-done.$lane" || die "cannot mark the $lane lane finished"
      exit 0
    fi
    take_lab_lock "$lab_box"; lab_lock_rc=$?
    [ "$lab_lock_rc" = 0 ] && lab_box_gated "$lab_box" && LANE_GATED_BOX=$lab_box
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
        sweep_state_update "$machine" "$backend" skip
        flush_lane_output || die "cannot publish the $machine/$backend summary to stdout"
      done
      : >"$LANE_DIR/lane-done.$lane" || die "cannot mark the $lane lane finished"
      exit 0
    fi
  fi
  # The fleet's execution registry is asked before every unit, under the lane's reservation: a unit
  # an exclusive measurement holds the box against is skipped, not run (see sweep_fleet_under_measurement).
  lane_names=
  [ -z "$SWEEP_FLEET_FW" ] || lane_names=$(sweep_fleet_lane_names "$lane_host")
  for unit in "${UNITS[@]}"; do
    IFS=: read -r machine backend host <<<"$unit"
    [ "$machine" = "$lane" ] || continue
    wanted "$backend" || continue
    # The registry's reservations on the box, which a GPU unit records beside its token readings;
    # only from THIS unit's read, never a previous unit's.
    BOX_RESERVATIONS=
    if [ -n "$lane_names" ]; then
      # Unquoted on purpose: the names are ssh aliases and a fleet box name, which never hold a space
      # or a glob character, and each must reach the reader as its own argument.
      # shellcheck disable=SC2086
      sweep_fleet_under_measurement $lane_names
      case $? in
        0)
          say "  $machine/$backend: skip (box ${lab_box:-$SWEEP_FLEET_BOX} under an exclusive measurement: $MEASUREMENT_HOLDERS)"
          record "$machine" "$backend" skip 0
          sweep_state_update "$machine" "$backend" skip
          flush_lane_output || die "cannot publish the $machine/$backend summary to stdout"
          continue
          ;;
        2)
          say "  $machine/$backend: WARNING -- the fleet's execution registry could not be read ($REGISTRY_REASON); running without knowing whether a measurement holds the box"
          ;;
      esac
    fi
    run_unit "$machine" "$backend" "$host"
    flush_lane_output || die "cannot publish the $machine/$backend summary to stdout"
  done
  # A gated box goes back to sleep once its lane is done with it -- only if a unit reached it,
  # since one that never answered has nothing to put down. The lane's own reservation goes first:
  # wake-lab's sleep takes the box's lane lock like every destroyer, and would otherwise be refused
  # by the very lane asking for it.
  if [ -n "$lane_host" ] && [ "$LANE_REACHED" = 1 ] && lab_box_gated "$lab_box"; then
    exec 8>&-
    sleep_gated_box "$lab_box"
    # Only once the request has RETURNED: a cancellation that interrupts it (relay reaps its
    # supervisor) must leave the EXIT path to issue the detached request instead.
    LANE_SLEPT=1
    flush_lane_output || die "cannot publish the $lab_box sleep summary to stdout"
  fi
  # The lane's completion marker, published as its last act: the run record
  # reads its absence as a lane that stopped before finishing, so nothing may
  # come between this and the `exit` -- and a lane that dies or is signalled
  # anywhere above leaves it absent without having to know the record exists.
  : >"$LANE_DIR/lane-done.$lane" || die "cannot mark the $lane lane finished"
  exit 0
}

# Resolve each SELECTED remote unit's `@<lab box>` to today's ssh destination (lab_dest above),
# before anything is built or reserved. The endpoint map is read first, and only when there is a
# remote unit to resolve: a local-only run (CI, a developer's machine) needs no wake-lab.sh. No map,
# or a box whose kind cannot be read or whose row has no alias for it, refuses the whole run here,
# with the startup exit 2 and no run record: the alternative is a lane that sshes to a guessed
# alias and files `skip (unreachable)` -- the row a sleeping box also writes -- for coverage that
# was never attempted. Unselected units keep their placeholder; nothing reaches them. The resolved
# destinations are printed in the header, so the routine's report can say which boot each GPU unit
# ran against. Here, after the run lock rather than beside the unit table, because the map is asked
# under `capped`, defined above.
LAB_DESTS=
for unit in "${UNITS[@]}"; do
  IFS=: read -r machine backend host <<<"$unit"
  case $host in @*) wanted "$backend" && { lab_map; break; } ;; esac
done
for ((i = 0; i < ${#UNITS[@]}; i++)); do
  IFS=: read -r machine backend host <<<"${UNITS[$i]}"
  case $host in @*) ;; *) continue ;; esac
  wanted "$backend" || continue
  dest=$(lab_dest "${host#@}") || die "no ssh destination for $machine/$backend; refusing to guess one"
  UNITS[$i]=$machine:$backend:$dest
  case " $LAB_DESTS " in *" $machine=$dest "*) ;; *) LAB_DESTS="$LAB_DESTS $machine=$dest" ;; esac
  case " $LAB_LANE_BOXES " in
    *" ${host#@} "*) ;;
    *) LAB_LANE_BOXES="$LAB_LANE_BOXES${LAB_LANE_BOXES:+ }${host#@}" ;;
  esac
done

# The two other questions this run asks the fleet's own scripts before any lane starts, both
# answered from THIS host and neither touching a box: whether wake-lab.sh still agrees on the lab
# lock contract (a broken one refuses the run here, with no record), and which fleet-worker.sh reads
# the execution registry the lanes consult before each unit.
lab_contract_check
sweep_fleet_probe

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
# Only when a remote unit is selected: which boot of each GPU box this run addresses.
[ -z "$LAB_DESTS" ] || echo "destinations:$LAB_DESTS"
[ -z "$LAB_CONTRACT" ] || echo "lab locks: $LAB_CONTRACT"
echo "reservations: $SWEEP_FLEET_STATUS"
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
# A lane that stopped on its own `die` may have left lines it never published.
publish_lane_buffers

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
  SKIP_RUN_RECORDS+=("$(cat "$evidence")")
  [ -f "$LANE_DIR/skip-cleared.$machine.$backend" ] &&
    SKIP_RERUN_CLEARED+=("$machine/$backend")
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
      "${SKIP_RUN_RECORDS[$i]}")
  done

  report=$LOGS/$stamp-skip-coverage.txt
  report_stage=$report.stage.$$
  scope='@runtest + @train'
  [ "$SLOW" = 1 ] && scope="$scope + @slow"
  {
    echo "skip coverage for $run_sha ($scope; forced execution)"
    [ ${#SKIP_RERUN_CLEARED[@]} -eq 0 ] ||
      echo "red units counted after a clean serial rerun: ${SKIP_RERUN_CLEARED[*]}"
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
