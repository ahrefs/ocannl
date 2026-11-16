#!/usr/bin/env bash
# Rerun the CUDA suite with every compile at the device's own PTX target (cuda_ptx_target=native,
# gh-ocannl-1705) against the same goldens, and say in one line what it found. What ships is the
# FLOOR target, forward-JIT-compiled by the driver; a miscompile or a data race that only one
# target's code exposes then shows as a suite that is green at the floor and red here, with no
# reference of its own (lukstafi/ocannl-staging#1430's lane-0 transfer was wrong at compute_80 to
# compute_90 and right at compute_120; gh-ocannl-1674's ptx_target_parity compares the two targets
# on a few seeds in one process, this compares them on everything). tools/sweep.sh runs this after
# each CUDA unit whose suite was green (its diagnostic legs), in the unit's own worktree but in a
# build directory of its own, `_build-native-ptx` (ignored by git, and by dune as a `_` directory):
# the ambient key is part of every test action's dependencies, so in the unit's `_build` this run
# re-executed every test action and left the next day's floor run nothing to reuse.
#
# Usage: tools/cuda-native-ptx-rerun.sh [dune build arguments...]   (from any directory; default
#                                                                    `@runtest @train`, the
#                                                                    sweep's full-suite unit)
#
# Dune's own output comes first. The last line on stdout is the verdict, and the exit status says
# the same:
#   cuda-native-ptx: verdict: holds (...)            0  the suite passes at the native target
#   cuda-native-ptx: verdict: NATIVE TARGET RED (...) 1  dune failed: a test, a golden or a compile
#                                                       the floor run passed (the sweep runs this only
#                                                       after a green floor unit); the failed rules
#                                                       are counted and the first few named
#   cuda-native-ptx: verdict: unavailable: ...        2  the run could not start (the credential scrub)
#   cuda-native-ptx: verdict: unavailable: runtime refusal (...) 2  a shared allowlisted refusal
# Its boundary: reads dune's status, tools/ci-failed-rules.sh --names, and only the exception/status
# and assertion allowlist in tools/environment-refusals.sh. It reads no kernel journal. A red run
# without a refusal stays red even when no rule is named (dune killed, say). A refusal leaves the
# native target unjudged, including any other failures in that batch: rerun the affected stanzas
# serially in _build-native-ptx before attributing them. Unlike the floor sweep, this diagnostic
# does not automate the serial recovery: that orchestration belongs to sweep.sh; reporting the
# refusal here avoids a second recovery implementation and a false target-miscompile verdict.
set -u
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
cd "$root" || exit 2

verdict() { # status text
  printf 'cuda-native-ptx: verdict: %s\n' "$2"
  exit "$1"
}

[ $# -gt 0 ] || set -- @runtest @train

# Dune records the environment of every process it spawns in _build/trace.csexp (gh-ocannl-1280).
# shellcheck source=credential-env.sh
. tools/credential-env.sh || verdict 2 "unavailable: cannot read tools/credential-env.sh"
eval "$(credential_env_scrub_text)" ||
  verdict 2 "unavailable: cannot remove credential variables:$credential_env_left"

# shellcheck source=environment-refusals.sh
. tools/environment-refusals.sh || verdict 2 "unavailable: cannot read tools/environment-refusals.sh"

work=$(mktemp -d "${TMPDIR:-/tmp}/cuda-native-ptx.XXXXXX") ||
  verdict 2 "unavailable: no temporary directory"
trap 'rm -rf "$work"' EXIT
echo "=== cuda-native-ptx: dune build $* ==="
OCANNL_BACKEND=cuda OCANNL_CUDA_PTX_TARGET=native dune build --build-dir _build-native-ptx "$@" \
  >"$work/out" 2>&1
rc=$?
cat "$work/out"
[ "$rc" != 0 ] || verdict 0 "holds (dune build $*: the suite passes at the native PTX target)"
refusal=$(environment_log_refusal "$work/out") || refusal=
[ -z "$refusal" ] || verdict 2 "unavailable: runtime refusal ($refusal; native target unjudged; rerun affected stanzas at -j 1 in _build-native-ptx)"
rules=$(bash tools/ci-failed-rules.sh --names "$work/out" 2>/dev/null) || rules=
[ -n "$rules" ] || verdict 1 "NATIVE TARGET RED (dune exited $rc; no failed rule in its output)"
count=$(printf '%s\n' "$rules" | grep -c .)
first=$(printf '%s\n' "$rules" | awk 'NR > 3 { exit } { printf "%s%s", (NR > 1 ? ", " : ""), $0 }')
noun=rules more=
[ "$count" != 1 ] || noun=rule
[ "$count" -le 3 ] || more=', ...'
verdict 1 "NATIVE TARGET RED (dune exited $rc; $count failed $noun: $first$more)"
