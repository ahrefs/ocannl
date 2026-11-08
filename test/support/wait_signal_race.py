"""Exercise Bash's wait-entry race under eight lanes of fork/exit load.

A deterministic unbounded foreground wait must fail the response oracle.
Also diagnose up to 4,000 TERM deliveries against the bare wait, stopping each
lane at its first failure. Require all 20,000 deliveries through the shared wait
to be handled within 2.5 seconds. Child and relay startup are acknowledged
before that deadline starts. Each trial owns a fresh process group.
"""

import concurrent.futures
import os
import json
import shlex
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time


class DiagnosticBudgetExpired(Exception):
    pass


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError("ambiguous control replacement: " + before)
    return text.replace(before, after)


class Lines:
    def __init__(self, stream, cancelled, diagnostic_deadline=None):
        self.fd = stream.fileno()
        self.buffer = b""
        self.cancelled = cancelled
        self.diagnostic_deadline = diagnostic_deadline

    def read(self, seconds):
        deadline = time.monotonic() + seconds
        while b"\n" not in self.buffer:
            if self.cancelled.is_set():
                raise InterruptedError("stress driver cancelled")
            now = time.monotonic()
            if self.diagnostic_deadline is not None and now >= self.diagnostic_deadline:
                raise DiagnosticBudgetExpired()
            remaining = deadline - now
            if remaining <= 0:
                raise TimeoutError("relay did not acknowledge the trial")
            if not select.select([self.fd], [], [], min(remaining, .1))[0]:
                continue
            chunk = os.read(self.fd, 4096)
            if not chunk:
                raise RuntimeError("relay exited before acknowledging the trial")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode()



SCRIPT = r'''
. "$1"
exec 3<>"$2" 4>&1
trap 'trap "" TERM; kill -TERM "$child" 2>/dev/null; echo handled >&4; read -r release; exit 143' TERM
perl -e '$|=1; print "$$\n"; sleep 30' >&3 &
child=$!
read -r ready <&3
echo "$child"
wait_child "$child" 2>/dev/null
exit 99 # Reaching here without the TERM trap is a fixture failure.
'''


def trial(helper, fifo, mode, cancelled, ready_path=None, diagnostic_deadline=None):
    old = mode == "old"
    script = replace_once(SCRIPT, 'wait_child "$child"', 'wait "$child"') if old else SCRIPT
    if mode == "forced":
        # Publishing from the foreground child puts Bash inside an unbounded
        # command before TERM arrives. This deterministically holds its trap;
        # the ordinary bare-wait search below remains a race diagnostic.
        script = replace_once(script, 'echo "$child"\nwait_child "$child" 2>/dev/null',
                                "perl -e '$|=1; print qq{$ARGV[0]\\n}; sleep 30' \"$child\"")
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            ["perl", "-e", '$SIG{INT}=$SIG{HUP}=$SIG{TERM}="DEFAULT"; exec @ARGV',
             "/bin/bash", "-c", script, "wait-race", str(helper), str(fifo)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
            start_new_session=True,
        )
        held = False
        failure = None
        try:
            lines = Lines(process.stdout, cancelled, diagnostic_deadline)
            child = int(lines.read(30))
            if ready_path is not None:
                temporary = ready_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"driver": os.getpid(), "group": process.pid}))
                temporary.replace(ready_path)
            os.kill(process.pid, signal.SIGTERM)
            try:
                handled = lines.read(2.5)
            except TimeoutError:
                held = True
                # This child's natural exit is 30 s away. Releasing its wait
                # must release the held trap, which then acknowledges TERM.
                os.killpg(process.pid, signal.SIGTERM)
                handled = lines.read(10)
            if handled != "handled":
                raise RuntimeError("relay protocol lost a signal")
        except Exception as error:
            failure = error
        finally:
            # The handler parks after acknowledgment, so its live leader
            # still owns the group when we kill it, before reaping anything.
            # It never returns into the old Bash wait (which can also crash).
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                # Darwin can refuse killpg when only zombies remain after
                # the old control crashes. The leader is still unreaped,
                # so this group id cannot have been recycled. Accept only an
                # independently observed empty/zombie-only group, after KILL
                # was attempted; a live member or failed census stays red.
                rows = subprocess.check_output(
                    ["ps", "-A", "-o", "pgid=,stat="], text=True
                ).splitlines()
                if any(int(row.split()[0]) == process.pid
                       and not row.split()[1].upper().startswith("Z") for row in rows):
                    raise
            process.wait()
            process.stdout.close()
            process.stdin.close()
        if isinstance(failure, (InterruptedError, DiagnosticBudgetExpired)):
            raise failure
        # Bash 3.2 can also segfault in the old wait-entry window. Report
        # those trials separately and keep searching for an actual held TERM;
        # a crash must never satisfy the required held-signal control.
        if old and process.returncode == -signal.SIGSEGV and failure is not None:
            return "crashed"
        if failure is not None or process.returncode != -signal.SIGKILL:
            errors.seek(0)
            raise RuntimeError(f"old={old}, exit={process.returncode}: {failure}; "
                               f"stderr={errors.read().decode()}") from failure
        return "held" if held else "prompt"


def lane(helper, mode, deadline, cancelled, limit=2500):
    held = crashes = count = 0
    with tempfile.TemporaryDirectory(prefix="wait-race-") as scratch:
        fifo = Path(scratch) / "ready"
        os.mkfifo(fifo)
        for _ in range(limit):
            if cancelled.is_set():
                raise InterruptedError("stress driver cancelled")
            if time.monotonic() > deadline:
                if mode == "old":
                    break
                raise TimeoutError("shared wait exceeded its 10-minute budget")
            try:
                outcome = trial(helper, fifo, mode, cancelled,
                                diagnostic_deadline=deadline if mode == "old" else None)
            except DiagnosticBudgetExpired:
                break
            count += 1
            crashes += outcome == "crashed"
            held += outcome == "held"
            if held:
                break
    return held, count, crashes


def check_cancellation(harness):
    """Exercise the shipping launcher and its former foreground shape."""
    stopped = threading.Event()
    def stop(signum, frame):
        # Do not unwind while Popen is establishing ownership, or interrupt
        # cleanup if the parent relays a group signal a second time.
        stopped.set()
    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    with tempfile.TemporaryDirectory(prefix="wait-cancel-") as scratch:
        scratch = Path(scratch)
        text = harness.read_text()
        launch = ('  python3 "$HERE/../test/support/wait_signal_race.py" "$@" >"$log" 2>&1 &\n'
                  '  race_driver=$!\n  [ "$driver_cancelled" = 0 ] || exit "$driver_cancelled"\n'
                  '  wait_child "$race_driver"\n')
        here = 'HERE=$(cd "$(dirname "$0")" && pwd)'
        if text.count(launch) != 1 or text.count(here) != 1:
            raise RuntimeError("cannot construct the foreground launcher control")
        mutant = scratch / "foreground.sh"
        mutant.write_text(text.replace(here, "HERE=" + shlex.quote(str(harness.parent)))
                          .replace(launch, launch.splitlines()[0][:-2] + "\n"))
        for foreground, group, subject in [(False, False, harness), (False, True, harness),
                                           (True, False, mutant)]:
            ready = scratch / ("old.ready" if foreground else "group.ready" if group else "new.ready")
            env = dict(os.environ, OCANNL_TOOL_SIGNAL_RACE_READY=str(ready))
            with tempfile.TemporaryFile() as output:
                process = subprocess.Popen(
                    ["perl", "-e", '$SIG{INT}=$SIG{HUP}=$SIG{TERM}="DEFAULT"; exec @ARGV',
                     "/bin/bash", str(subject), "--only", "^2:"], env=env,
                    stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                    start_new_session=True,
                )
                timed_out = False
                try:
                    deadline = time.monotonic() + 30
                    while not ready.exists():
                        if stopped.is_set():
                            raise InterruptedError("cancellation checker stopped")
                        if time.monotonic() > deadline:
                            raise TimeoutError("stress fixture did not publish readiness")
                        time.sleep(.01)
                    # Published only after a trial's child and relay started:
                    # launch time is outside the cancellation response bound.
                    record = json.loads(ready.read_text())
                    if group:
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        os.kill(process.pid, signal.SIGTERM)
                    deadline = time.monotonic() + 5
                    while True:
                        if stopped.is_set():
                            raise InterruptedError("cancellation checker stopped")
                        try:
                            process.wait(timeout=.1)
                            break
                        except subprocess.TimeoutExpired:
                            if time.monotonic() > deadline:
                                timed_out = True
                                break
                finally:
                    if process.returncode is None:
                        # The mutant's shell defers TERM, but its driver is in
                        # this still-owned group and cancels all trial groups.
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                if process.returncode != 143 or timed_out != foreground:
                    output.seek(0)
                    raise RuntimeError(f"foreground={foreground}: timeout={timed_out}, "
                                       f"exit={process.returncode}: {output.read().decode()}")
                states = subprocess.check_output(
                    ["ps", "-A", "-o", "pgid=,stat="], text=True
                ).splitlines()
                if any(int(row.split()[0]) == record["group"]
                       and not row.split()[1].upper().startswith("Z") for row in states):
                    raise RuntimeError("a trial group outlived harness cancellation")
                print("foreground launcher control: rejected by cancellation deadline" if foreground
                      else f"tracked driver: {'group' if group else 'pid'} TERM cancelled and reaped within 5 seconds", flush=True)
    return 0


def main():
    if sys.argv[1] == "--check-cancellation":
        return check_cancellation(Path(sys.argv[2]).resolve())
    helper = Path(sys.argv[1]).resolve()
    ready_path = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
    cancelled = threading.Event()
    def stop(signum, frame):
        cancelled.set()
    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    with tempfile.TemporaryDirectory(prefix="wait-control-") as scratch:
        fifo = Path(scratch) / "ready"
        os.mkfifo(fifo)
        if trial(helper, fifo, "forced", cancelled, ready_path) != "held":
            raise RuntimeError("response oracle accepted an unbounded foreground wait")
        print("deterministic unbounded-wait control: response oracle rejected it", flush=True)
    # The verdict gets its own complete budget before any diagnostic work.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        deadline = time.monotonic() + 600
        shared = list(pool.map(lambda _: lane(helper, "shared", deadline, cancelled), range(8)))
        print(f"shared wait: {sum(h for h, n, c in shared)} held TERM(s) / "
              f"{sum(n for h, n, c in shared)} trials", flush=True)
        passed = all(h == 0 and n == 2500 and c == 0 for h, n, c in shared)
        deadline = time.monotonic() + 60
        old = list(pool.map(lambda _: lane(helper, "old", deadline, cancelled, limit=500), range(8)))
        print(f"old bare wait: {sum(h for h, n, c in old)} held TERM(s) / "
              f"{sum(n for h, n, c in old)} diagnostic trials (limit 4000, budget 60s), "
              f"{sum(c for h, n, c in old)} SIGSEGV(s) reported separately", flush=True)
    if time.monotonic() >= deadline:
        print("note: diagnostic budget exhausted; shared-wait verdict unchanged", flush=True)
    if not any(h for h, n, c in old):
        print("note: bare-wait race not observed in this diagnostic sample", flush=True)
    return 0 if passed else 1



if __name__ == "__main__":
    try:
        sys.exit(main())
    except InterruptedError:
        sys.exit(143)
