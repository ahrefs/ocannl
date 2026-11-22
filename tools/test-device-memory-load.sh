#!/usr/bin/env bash
# Exercise the public wrapper with a fake benchmark interpreter and no GPU.
# affected-reads: tools/device-memory-load.py benchmarks/bench_venv.py benchmarks/cell_group.py
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
. "$here/../scripts/process-group.sh"
harness_args "$@"
harness_require python3
harness_scratch test-device-memory-load

subject=$here/device-memory-load.sh
export BENCH_VENV_PY="$TMP/stub python" PIDS="$TMP/pids" HOG_ARGS="$TMP/args"
export CMD_PIDS="$TMP/cmd-pids"
unset FLEET_SLOT_HELD
cat >"$BENCH_VENV_PY" <<'PY'
#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def record_group(path):
    # A TERM-ignoring descendant keeps the group populated after its leader exits.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(12)"])
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    Path(path).write_text(f"{os.getpid()} {child.pid}\n")


if sys.argv[1] == "--cmd":
    record_group(os.environ["CMD_PIDS"])
    if sys.argv[2] == "drop-hog":
        target = int(Path(os.environ["PIDS"]).read_text().split()[0])
    else:
        target = os.getppid()
        assert target > 1
    os.kill(target, signal.SIGTERM)
    time.sleep(4)
    print("CMD finished", flush=True)
    sys.exit(0)

mode = os.environ["HOG_MODE"]
assert sys.argv[1] == "-c"
Path(os.environ["HOG_ARGS"]).write_text(" ".join(sys.argv[3:]))
if mode in ("missing-torch", "no-device"):
    if mode == "missing-torch":
        sys.modules["torch"] = None
    else:
        from types import SimpleNamespace

        sys.modules["torch"] = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    sys.argv = sys.argv[2:]
    exec(sys.argv[0])
    raise AssertionError("the worker accepted unavailable hardware")

record_group(os.environ["PIDS"])
if mode == "ready":
    print("ready", flush=True)
elif mode == "wrong":
    print("ready?", flush=True)
elif mode == "cancel":
    supervisor = os.getppid()
    assert supervisor > 1
    os.kill(supervisor, signal.SIGTERM)
else:
    # A partial line must obey the deadline too, even with the pipe readable.
    os.write(1, b"rea")
time.sleep(12)
PY
chmod +x "$BENCH_VENV_PY"

run() {
  local mode=$1
  shift
  : >"$PIDS"
  : >"$HOG_ARGS"
  : >"$CMD_PIDS"
  rc=0
  HOG_MODE=$mode python3 -c '
import subprocess
import sys

try:
    sys.exit(subprocess.run(sys.argv[1:], timeout=8).returncode)
except subprocess.TimeoutExpired:
    sys.exit("fixture deadline expired")
' "$subject" --ready-timeout 3 "$@" >"$TMP/out" 2>"$TMP/err" || rc=$?
}

group_reaped() {
  local leader child
  read -r leader child <"${1:-$PIDS}" || return 1
  if pid_alive "$leader" || pid_alive "$child"; then
    kill -KILL "$leader" "$child" 2>/dev/null || :
    return 1
  fi
}

unexpected_command() { run "$1" -- sh -c 'printf "%s\n" "CMD ran"'; }
details() { printf 'exit %s\n' "$rc"; cat "$TMP/out" "$TMP/err"; }

if harness_leg '1: readiness, command status and group reaping'; then
  run ready --min-mib 7 --max-mib 19 --keep 3 -- sh -c '
    read -r hog child <"$PIDS" || exit 1
    kill -0 "$hog" && kill -0 "$child" || exit 1
    read -r line || exit 1
    printf "%s\n" "$line" "$1"
    exit 43
  ' sh 'arg with spaces' <<<"stdin from caller"
  if [ "$rc" = 43 ] && [ "$(cat "$TMP/out")" = $'stdin from caller\narg with spaces' ] &&
     [ "$(cat "$HOG_ARGS")" = '7 19 3' ]; then
    report 0 'CMD runs after startup and preserves arguments, streams and status'
  else
    report 1 'CMD runs after startup and preserves arguments, streams and status' "$(details)"
  fi
  group_reaped
  report "$?" 'hog and TERM-ignoring descendant are gone after CMD exits' "$(cat "$PIDS")"
fi

if harness_leg '2: incomplete readiness times out and reaps the group'; then
  run hung --ready-timeout 1 -- sh -c 'printf "%s\n" "CMD ran"'
  if [ "$rc" = 1 ] && [ ! -s "$TMP/out" ] && grep -q 'hog did not become ready within 1s' "$TMP/err"; then
    report 0 'the startup deadline refuses before CMD starts'
  else
    report 1 'the startup deadline refuses before CMD starts' "$(details)"
  fi
  group_reaped
  report "$?" 'the timed-out hog and descendant are gone' "$(cat "$PIDS")"
fi

if harness_leg '3: readiness requires the exact line'; then
  unexpected_command wrong
  if [ "$rc" = 1 ] && [ ! -s "$TMP/out" ] && grep -q 'invalid readiness line' "$TMP/err"; then
    report 0 'a prefix of the wrong line cannot start CMD'
  else
    report 1 'a prefix of the wrong line cannot start CMD' "$(details)"
  fi
  group_reaped
  report "$?" 'the malformed hog and descendant are gone' "$(cat "$PIDS")"
fi

if harness_leg '4: unavailable torch or device refuses before CMD'; then
  for mode in missing-torch no-device; do
    unexpected_command "$mode"
    case "$mode" in
      missing-torch) message='torch unavailable' ;;
      no-device) message='torch found no CUDA or HIP device' ;;
    esac
    if [ "$rc" = 1 ] && [ ! -s "$TMP/out" ] && grep -q "$message" "$TMP/err"; then
      report 0 "$mode refuses before CMD"
    else
      report 1 "$mode refuses before CMD" "$(details)"
    fi
  done
fi

if harness_leg '5: an enclosing CPU slot refuses before hog startup'; then
  FLEET_SLOT_HELD='testbox 1 1 cpu' unexpected_command ready
  if [ "$rc" = 1 ] && [ ! -s "$TMP/out" ] && [ ! -s "$HOG_ARGS" ] &&
     grep -q 'enclosing fleet slot must be a GPU slot' "$TMP/err"; then
    report 0 'the helper leaves the CPU slot free of GPU work'
  else
    report 1 'the helper leaves the CPU slot free of GPU work' "$(details)"
  fi
fi

if harness_leg '6: startup cancellation reaps the group'; then
  unexpected_command cancel
  if [ "$rc" = 1 ] && [ ! -s "$TMP/out" ] && grep -q 'terminated by signal 15' "$TMP/err"; then
    report 0 'TERM interrupts the readiness wait before CMD starts'
  else
    report 1 'TERM interrupts the readiness wait before CMD starts' "$(details)"
  fi
  group_reaped
  report "$?" 'the cancelled hog and descendant are gone' "$(cat "$PIDS")"
fi

check_interrupted_command() {
  local mode=$1 message=$2
  run ready -- "$BENCH_VENV_PY" --cmd "$mode"
  if [ "$rc" = 1 ] && [ ! -s "$TMP/out" ] && grep -q "$message" "$TMP/err"; then
    report 0 'the wrapper fails before CMD finishes normally'
  else
    report 1 'the wrapper fails before CMD finishes normally' "$(details)"
  fi
  group_reaped
  report "$?" 'hog group is gone' "$(cat "$PIDS")"
  group_reaped "$CMD_PIDS"
  report "$?" 'CMD group is gone' "$(cat "$CMD_PIDS")"
}

if harness_leg '7: hog loss during CMD invalidates the stress run'; then
  check_interrupted_command drop-hog 'hog exited during CMD'
fi

if harness_leg '8: TERM directed at the wrapper reaps both groups'; then
  check_interrupted_command cancel 'terminated by signal 15'
fi

finish
