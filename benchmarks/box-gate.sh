#!/usr/bin/env bash
# The box gate: what every benchmark driver that loads fixtures runs before it measures. It calls
# benchmarks/check-box.sh (gh-ocannl-1062) on the box the run measures on and the fixtures the run
# loads, and refuses to start on a GAP, with the check's own output above the refusal -- so a
# missing venv, a dead device or an absent fixture stops the run before its first cell rather than
# in the middle of it. Sourced, never run (benchmarks/box_gate.py is the Python drivers' twin):
#
#   . "$here/box-gate.sh"                          # before the driver scrubs its environment
#   box_gate_args "$@"; shift "$box_gate_shift"   # first: the gate's options lead the driver's own
#   ...                                           # the driver's own parsing and usage checks
#   box_gate DRIVER FIXTURE-DIR NAME...            # before the driver's first measurement
#
# The options box_gate_args takes, ahead of the driver's own arguments, each at most once:
#   --box BOX         the measurement box this run measures on: an ID declared by
#                     fixtures/DIGESTS.txt's `measurement-boxes` (check-box.sh refuses any other),
#                     or the fleet's name for one, which box_gate_box holds as the box's ID.
#   --skip-box-check  measure without the check, for a deliberate partial run; box_gate says so
#                     on the run's output. With --box too, the box still names the run.
# Scanning stops at the first argument that is neither, so the driver's own options and operands
# are untouched; a repeated option or a --box with no value exits 2.
#
# box_gate: NAME is a workload stem as check-box.sh's --fixture takes it (`gpt2_mini`; a repeat
# is checked once), and FIXTURE-DIR the directory the driver reads NAME.safetensors from. Neither
# --box nor --skip-box-check exits 2: a measuring run states its box. Otherwise check-box.sh's verdict
# decides: READY returns 0, a GAP exits 1 after a `DRIVER: refusing to start` line, and any other
# status (a usage or harness error, nothing judged) exits 2. The check's device item puts one
# tensor on the GPU, so a driver on a fleet box runs under a GPU slot, as its measurement does.
#
# Sourcing it records BENCH_VENV_PY, the box's interpreter override (benchmarks/bench_venv.py),
# for the check alone: a driver that then scrubs BENCH_* from its cells' environment still has
# the check resolve the interpreter the box is set up with, instead of reporting a driver-path GAP.

box_gate_check=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/check-box.sh
box_gate_box= box_gate_skip=0 box_gate_shift=0
box_gate_venv_py=${BENCH_VENV_PY-} box_gate_venv_py_set=${BENCH_VENV_PY+1}

box_gate_args() {
  local seen_box=0
  box_gate_box= box_gate_skip=0 box_gate_shift=0
  while [ $# -gt 0 ]; do
    case $1 in
      --box)
        [ "$seen_box" = 0 ] || { echo "${0##*/}: --box given twice" >&2; exit 2; }
        [ $# -ge 2 ] && [ -n "$2" ] ||
          { echo "${0##*/}: --box needs a value: the measurement box this run measures on" >&2; exit 2; }
        # The measurement box a fleet name stands for (check-box.sh's one table), so a driver
        # records `rog-nv`, not `rog-nv-linux`; a check-box.sh that cannot answer leaves the name
        # as given, for the check itself to judge (gh-ocannl-1433).
        box_gate_box=$(bash "$box_gate_check" --canonical-box "$2" 2>/dev/null) || box_gate_box=
        case $box_gate_box in '' | *[[:space:]]*) box_gate_box=$2 ;; esac
        seen_box=1 box_gate_shift=$((box_gate_shift + 2))
        shift 2
        ;;
      --skip-box-check)
        [ "$box_gate_skip" = 0 ] || { echo "${0##*/}: --skip-box-check given twice" >&2; exit 2; }
        box_gate_skip=1 box_gate_shift=$((box_gate_shift + 1))
        shift
        ;;
      *) break ;;
    esac
  done
}

box_gate() { # DRIVER FIXTURE-DIR NAME...
  local driver=$1 dir=$2 name seen=" " rc=0
  shift 2
  [ $# -gt 0 ] || { echo "$driver: box_gate was given no fixture to check" >&2; exit 2; }
  if [ "$box_gate_skip" = 1 ]; then
    echo "$driver: --skip-box-check: measuring without benchmarks/check-box.sh (a deliberate partial run)"
    return 0
  fi
  [ -n "$box_gate_box" ] || {
    echo "$driver: --box BOX is required to measure: the measurement box benchmarks/check-box.sh" \
      "checks before the first cell (fixtures/DIGESTS.txt's measurement-boxes); --skip-box-check" \
      "for a deliberate partial run" >&2
    exit 2
  }
  for name in "$@"; do # each name once, in first-seen order
    case $seen in *" $name "*) ;; *) set -- "$@" --fixture "$name"; seen="$seen$name " ;; esac
    shift
  done
  if [ -n "$box_gate_venv_py_set" ]; then
    BENCH_VENV_PY=$box_gate_venv_py bash "$box_gate_check" --box "$box_gate_box" --fixture-dir "$dir" "$@" || rc=$?
  else
    bash "$box_gate_check" --box "$box_gate_box" --fixture-dir "$dir" "$@" || rc=$?
  fi
  case $rc in
    0) return 0 ;;
    1)
      echo "$driver: refusing to start: $box_gate_box is not ready for this run -- each GAP above names" \
        "its repair; --skip-box-check is for a deliberate partial run" >&2
      exit 1
      ;;
    *)
      echo "$driver: refusing to start: benchmarks/check-box.sh judged nothing (exit $rc)" >&2
      exit 2
      ;;
  esac
}
