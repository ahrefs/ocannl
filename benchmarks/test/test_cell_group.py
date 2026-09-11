import ast
import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

import cell_group
import gh675_cells


HERE = Path(__file__).resolve().parent.parent


class CellGroupTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def python(self, source, *args):
        return [sys.executable, "-c", source, *map(str, args)]

    def wait_file(self, path, timeout=10):
        deadline = time.monotonic() + timeout
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(path.exists(), f"child did not publish {path}")

    def alive(self, pid):
        # Through `cell_group`, not `os.kill(pid, 0)`: on Windows that call TERMINATES the process
        # it is asked about and raises `WinError 87` once it has exited.  The zombie refinement
        # below stays here, since only POSIX has zombies.
        if not cell_group.process_is_alive(pid):
            return False
        if Path(f"/proc/{pid}/stat").exists():
            with contextlib.suppress(OSError, IndexError):
                return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        return True

    def wait_gone(self, pid, timeout=10):
        deadline = time.monotonic() + timeout
        while self.alive(pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        return not self.alive(pid)

    def test_a_sleep_chain_is_killed_and_reaped_as_one_group(self):
        pidfile = self.dir / "grandchild.pid"
        child = cell_group.spawn(
            self.python(
                "import os, signal, subprocess, sys, time\n"
                "code = ('import signal, time; '\n"
                "        'signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)')\n"
                "kid = subprocess.Popen([sys.executable, '-c', code],\n"
                "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                "open(sys.argv[1], 'w').write(str(kid.pid))\n"
                "time.sleep(300)\n",
                pidfile,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.wait_file(pidfile)
        grandchild = int(pidfile.read_text())

        result = cell_group.terminate(child, grace=0.2)

        self.assertIs(result.observation, cell_group.GONE)
        self.assertTrue(result.reaped)
        self.assertTrue(self.wait_gone(grandchild), f"pid {grandchild} survived group cleanup")

    def test_a_child_killed_mid_stream_preserves_its_partial_stdout(self):
        ready = self.dir / "stdout-ready"
        child = cell_group.spawn(
            self.python(
                "import signal, sys, time\n"
                "grace_signal = getattr(signal, 'SIGBREAK', signal.SIGTERM)\n"
                "signal.signal(grace_signal, signal.SIG_IGN)\n"
                "sys.stdout.write('partial child output')\n"
                "sys.stdout.flush()\n"
                "open(sys.argv[1], 'w').write('ready')\n"
                "time.sleep(300)\n",
                ready,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # The child ignores the signal ManagedProcess uses for its graceful phase: SIGTERM on
        # POSIX, SIGBREAK (sent as CTRL_BREAK_EVENT) on Windows. The readiness marker is published
        # only after that handler is installed and stdout was flushed, so the parent cannot signal
        # the child before either prerequisite of the intended grace-then-kill path exists.
        self.wait_file(ready)
        real_communicate = child.communicate

        def shorter_final_snapshot(*args, **kwargs):
            # Model the macOS failure mode deterministically: the grace-period communicate raises
            # TimeoutExpired carrying the partial bytes, then the post-kill reap returns a shorter
            # snapshot. The termination primitive must retain the longest cumulative observation.
            out, err = real_communicate(*args, **kwargs)
            return out[:0] if out is not None else None, err

        child.communicate = shorter_final_snapshot

        result = cell_group.terminate(child, grace=0.2)

        self.assertIs(result.observation, cell_group.GONE)
        self.assertTrue(result.reaped)
        self.assertEqual(result.stdout, b"partial child output")

    def test_text_output_snapshots_are_compared_as_encoded_bytes(self):
        group = unittest.mock.Mock()
        group.encoding = "utf-8"
        group.errors = "strict"
        group.communicate.side_effect = [
            subprocess.TimeoutExpired(
                "child",
                0.1,
                output="éé".encode(),
            ),
            ("ééX", None),
        ]
        group.observe.return_value = cell_group.GONE

        result = cell_group.terminate(group, grace=0.1, poll_interval=0)

        # Raw lengths choose the four-byte partial snapshot over this three-code-point complete
        # one. Comparing both as UTF-8 makes the complete five-byte snapshot authoritative.
        self.assertEqual(result.stdout, "ééX")
        self.assertTrue(result.reaped)

    def test_an_orphan_spawner_is_observed_and_collected_after_its_leader_exits(self):
        pidfile = self.dir / "orphan.pid"
        child = cell_group.spawn(
            self.python(
                "import subprocess, sys\n"
                "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
                "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                "open(sys.argv[1], 'w').write(str(kid.pid))\n",
                pidfile,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        child.communicate(timeout=10)
        self.wait_file(pidfile)
        orphan = int(pidfile.read_text())
        # `signal.SIGKILL` does not exist on Windows, and this lambda only reaches the kill when
        # the orphan survived -- so the platform where the cleanup matters most is the one where it
        # would have raised `AttributeError` instead of killing anything.  `os.kill` with any other
        # signal is `TerminateProcess` there, which is what is wanted.
        hard_kill = getattr(signal, "SIGKILL", signal.SIGTERM)
        self.addCleanup(lambda: self.alive(orphan) and os.kill(orphan, hard_kill))

        self.assertIsNot(child.observe(), cell_group.GONE)
        result = cell_group.terminate(child, grace=0.2)

        self.assertIs(result.observation, cell_group.GONE)
        self.assertTrue(self.wait_gone(orphan), f"pid {orphan} survived orphan cleanup")

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "zombie census needs procfs")
    def test_a_zombie_only_group_is_observed_gone(self):
        child = cell_group.spawn(self.python("pass"), stdout=subprocess.DEVNULL)
        deadline = time.monotonic() + 10
        observed = child.observe(allow_zombie_gone=True)
        while observed is not cell_group.GONE and time.monotonic() < deadline:
            time.sleep(0.02)
            observed = child.observe(allow_zombie_gone=True)
        os.kill(child.pid, 0)  # the unreaped process-table entry still exists

        self.assertIs(child.observe(), cell_group.UNKNOWN)
        self.assertIs(observed, cell_group.GONE)
        child.wait()

    def test_sweep_drivers_have_no_unmanaged_spawn_site(self):
        offenders = []
        for path in (HERE / "orchestrate.py", HERE / "gh675_cells.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                owner = node.func.value
                if (
                    isinstance(owner, ast.Name)
                    and owner.id == "subprocess"
                    and node.func.attr in ("Popen", "run", "call", "check_call", "check_output")
                ):
                    offenders.append(f"{path.name}:{node.lineno} subprocess.{node.func.attr}")
        self.assertEqual(offenders, [], "unmanaged benchmark child sites: " + ", ".join(offenders))

    def test_a_failed_windows_job_assignment_kills_the_unassigned_child_too(self):
        job = unittest.mock.Mock()
        child = unittest.mock.Mock()

        cell_group._cleanup_failed_windows_spawn(job, child)

        job.terminate.assert_called_once_with()
        job.close.assert_called_once_with()
        child.kill.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=1)

    def test_a_held_signal_does_not_replace_a_cleanup_failure(self):
        cancellation = cell_group.CancellationDeferral("test driver")

        with self.assertRaises(cell_group.CleanupFailed) as raised:
            with cancellation.deferring():
                cancellation.held_signal = signal.SIGTERM
                raise cell_group.CleanupFailed("SURVIVORS still hold the device")

        self.assertIn("SURVIVORS", str(raised.exception))
        self.assertNotIn("cleaned first", str(raised.exception))
        self.assertIsNone(cancellation.held_signal)

    @unittest.skipUnless(os.name == "posix", "spawn-window signal fixture uses POSIX delivery")
    def test_the_gh675_spawn_window_defers_cancellation_until_cleanup_is_owned(self):
        spawned = []
        real_popen = cell_group.subprocess.Popen
        cancellation = gh675_cells._cancellation
        cancellation.depth = 0
        cancellation.held_signal = None
        previous_term = signal.getsignal(signal.SIGTERM)
        previous_int = signal.getsignal(signal.SIGINT)
        self.addCleanup(signal.signal, signal.SIGTERM, previous_term)
        self.addCleanup(signal.signal, signal.SIGINT, previous_int)
        self.addCleanup(setattr, cancellation, "depth", 0)
        self.addCleanup(setattr, cancellation, "held_signal", None)
        cancellation.install()

        def popen_then_cancel(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            spawned.append(proc.pid)
            os.kill(os.getpid(), signal.SIGTERM)
            return proc

        with unittest.mock.patch.object(
            cell_group.subprocess, "Popen", side_effect=popen_then_cancel
        ):
            with self.assertRaises(SystemExit):
                gh675_cells.run_managed(
                    self.python("import time; time.sleep(300)"),
                    timeout=60,
                    context="cancelled probe",
                )

        self.assertEqual(len(spawned), 1)
        self.assertTrue(self.wait_gone(spawned[0]), "the spawn-window child outlived cancellation")


if __name__ == "__main__":
    unittest.main()
