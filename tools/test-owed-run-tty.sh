#!/usr/bin/env bash
# A local bare repository backs both PTY legs. The fake Git transport recognizes only
# push and ls-remote in the owed-run Git invocation. The checks read the fixture prompt,
# transport group, exit status and the read-timeout diagnostic. Real credentials, network
# transports, and prompts other than the fixture's are outside this harness.
# affected-reads: tools/owed-run.sh
# Usage: tools/test-owed-run-tty.sh [--subject FILE] [--only REGEX] [--keep]
set -u
. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
subject_file=
harness_own_option --subject subject_file 'a file'
harness_args "$@"
harness_require git perl python3
harness_scratch test-owed-run-tty
HERE=$(cd "$(dirname "$0")" && pwd)
subject=${subject_file:-$HERE/owed-run.sh}

tty_leg() {
  local mode=$1 label=$2 rc=0
  if [ "$(python3 -c 'import os; print(os.name)')" != posix ]; then
    harness_not_applicable 'the interpreter does not provide POSIX controlling terminals'
    return
  fi
  python3 -I -B - "$mode" "$TMP" "$subject" "$BASH" >"$TMP/tty-$mode.log" 2>&1 <<'PYTHON' || rc=$?
import contextlib
import errno
import os
from pathlib import Path
import pty
import select
import shutil
import signal
import subprocess
import sys
import time

mode, scratch, subject, bash = sys.argv[1:]
work = Path(scratch) / mode
work.mkdir()
local, remote = work / "local", work / "remote.git"
git = shutil.which("git")
# Ignore ambient git config and repo selectors, including those for a caller's real checkout.
env = {key: value for key, value in os.environ.items()
       if not key.startswith(("GIT_", "OCANNL_TOOL_OWED_RUN_"))}
env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
           GIT_AUTHOR_NAME="fixture", GIT_AUTHOR_EMAIL="fixture@example.invalid",
           GIT_COMMITTER_NAME="fixture", GIT_COMMITTER_EMAIL="fixture@example.invalid",
           OCANNL_TOOL_OWED_RUN_URL=str(remote),
           OCANNL_TOOL_OWED_RUN_FETCH_CAP="5" if mode == "push" else "2",
           OCANNL_TOOL_OWED_RUN_PUSH_RETRIES="0")
subprocess.run([git, "init", "-q", str(local)], env=env, check=True)
subprocess.run([git, "init", "-q", "--bare", str(remote)], env=env, check=True)
subprocess.run([git, "-C", str(local), "commit", "-q", "--allow-empty", "-m", "fixture"],
               env=env, check=True)
head = subprocess.check_output([git, "-C", str(local), "rev-parse", "HEAD"], env=env).decode().strip()
tools = local / "tools"
tools.mkdir()
script = tools / "owed-run.sh"
shutil.copyfile(subject, script)
position = work / "transport-position"
prompt = b"fixture password> "
fake = work / "git"
fake.write_text(f"""#!{sys.executable}
import os, pathlib, signal, sys
args = sys.argv[1:]
if args[:2] == ['-C', {str(local)!r}] and args[2:3] == [{mode!r}]:
    signal.signal(signal.SIGTTIN, signal.SIG_DFL)
    with open('/dev/tty', 'r+b', buffering=0) as tty:
        position = pathlib.Path({str(position)!r})
        foreground = os.tcgetpgrp(tty.fileno())
        tmp = position.with_suffix('.tmp')
        tmp.write_text(f'{{os.getpgrp()}} {{foreground}}')
        tmp.replace(position)
        tty.write({prompt!r})
        if tty.readline() != b'fixture response\\n':
            sys.exit(84)
os.execv({git!r}, [{git!r}, *args])
""")
fake.chmod(0o700)
env["PATH"] = f"{work}:{env['PATH']}"
args = (["waive", "--head", head, "--reason", "PTY fixture"] if mode == "push"
        else ["check", head, "runtest"])

pid, master = pty.fork()
if pid == 0:
    try:
        os.execve(bash, [bash, str(script), *args], env)
    finally:
        os._exit(127)

output = bytearray()
responded = False
status = None
deadline = time.monotonic() + 12
try:
    while select.select([master], [], [], max(0, deadline - time.monotonic()))[0]:
        try:
            chunk = os.read(master, 8192)
        except OSError as exc:
            if exc.errno != errno.EIO:
                raise
            break
        if not chunk:
            break
        output.extend(chunk)
        if mode == "push" and not responded and prompt in output:
            os.write(master, b"fixture response\n")
            responded = True
        if time.monotonic() >= deadline:
            break
    while time.monotonic() < deadline:
        ended, observed = os.waitpid(pid, os.WNOHANG)
        if ended:
            status = os.waitstatus_to_exitcode(observed)
            break
        time.sleep(0.01)
    transcript = output.decode(errors="replace")
    assert position.exists(), (status, transcript)
    group, foreground = map(int, position.read_text().split())
    if mode == "push":
        assert group == foreground == pid and responded and status == 0, (
            group, foreground, responded, status, transcript)
        subprocess.run([git, "-C", str(remote), "show-ref", "--verify",
                        f"refs/owed-runs/{head}"], env=env, check=True)
    else:
        assert foreground == pid and group != foreground and status == 3, (
            group, foreground, status, transcript)
        assert "could not be read within 2s: unreadable" in transcript, transcript
finally:
    os.close(master)
    if status is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    if position.exists():
        group, _ = map(int, position.read_text().split())
        if group != pid:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group, signal.SIGKILL)
PYTHON
  if [ "$rc" = 0 ]; then report 0 "$label"
  else report 1 "$label" "exit $rc; $(cat "$TMP/tty-$mode.log")"; fi
}

tty_push() { tty_leg push "$harness_leg_running"; }
tty_read() { tty_leg ls-remote "$harness_leg_running"; }
harness_leg_run 'foreground owed-run push reads a response from the controlling terminal' tty_push
harness_leg_run 'isolated owed-run read hits its bound under terminal job control' tty_read
finish
