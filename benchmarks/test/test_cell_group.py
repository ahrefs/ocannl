import ast
import contextlib
import os
import re
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


def publish_pid(path, pid):
    """One line of Python source, for a child a test spawns, that publishes `pid` to the file named
    by `path` (both are Python expressions in the child) so that the file never exists without it.

    `open(path, 'w').write(...)` creates the file EMPTY and fills it a moment later, so a parent
    polling `exists()` can read `''` and fail on `int('')` -- which reddened the per-PR matrix
    (gh-ocannl-1041). Writing a sibling and `os.replace`-ing it into place makes the final name
    appear only with its content, which fixes every reader at once rather than each poll; the
    rename is atomic on POSIX and on Windows alike, and `write_text` has closed the sibling before
    it is renamed. `test_every_published_pid_goes_through_publish_pid` keeps new fixtures on it.
    """
    return (
        "import os, pathlib; "
        f"pathlib.Path({path} + '.pending').write_text(str({pid})); "
        f"os.replace({path} + '.pending', {path})\n"
    )


def kill_the_group(pid):
    """SIGKILL what is left of `pid`'s process group, if it is still shaped like a cell's.

    The pid may name a process long gone and the number reused, so the group is killed only while
    it is the whole of a session -- which `cell_group.spawn` gives every child it starts -- and
    never when it is this process's own. POSIX only: on Windows there is no group to kill and no
    need, since every spawn there sits in a kill-on-close Job.
    """
    if os.name != "posix":
        return
    # ProcessLookupError: everything is gone (on macOS a zombie already answers so);
    # PermissionError: the number was reused by a process that is not ours to signal.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        group = os.getpgid(pid)
        if group == os.getsid(pid) and group != os.getpgrp():
            os.killpg(group, signal.SIGKILL)


def kill_the_group_on_cleanup(case, pidfile):
    """Register on the TestCase `case` a cleanup that SIGKILLs whatever is left of the group of
    the pid a child published (through `publish_pid`) to `pidfile`.

    Every fixture that parks a process in `time.sleep(300)` publishes its pid, and it is the test's
    own assertions that establish the code under test killed it. When one of them fails, nothing
    else will: a cell sits in a session of its own by design, so killing the test's direct child
    -- or a sweep driver, which the cancellation tests SIGKILL, running no handler -- leaves it
    orphaned for the full 300 s (gh-ocannl-1054: a forced failure left three such processes behind,
    and on a shared box or a CI runner they hold whatever a later timing test measures against). So
    the kill is owed on every path; registered after `setUp`, it runs before the directory holding
    the pidfile is removed.

    It kills the GROUP, not the pid: where the published pid is a grandchild, its sleeping parent
    -- the cell -- is the same leak, and `cell_group.spawn` keeps both in the cell's group. The
    pid is read at cleanup time, which is why `kill_the_group` guards against a reused number.
    `test_every_sleeping_fixture_is_killed_on_cleanup` keeps new fixtures registering it.
    """

    def kill_what_is_left():
        if pidfile.exists():
            kill_the_group(int(pidfile.read_text()))

    case.addCleanup(kill_what_is_left)


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
        kill_the_group_on_cleanup(self, pidfile)
        child = cell_group.spawn(
            self.python(
                "import signal, subprocess, sys, time\n"
                "code = ('import signal, time; '\n"
                "        'signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)')\n"
                "kid = subprocess.Popen([sys.executable, '-c', code],\n"
                "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                + publish_pid("sys.argv[1]", "kid.pid")
                + "time.sleep(300)\n",
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

    # Windows implements communicate timeouts by joining pipe-reader threads and raises before
    # copying their buffers into TimeoutExpired. The later successful communicate is therefore
    # the first observable snapshot, and this fixture deliberately truncates that return to model
    # the regression -- leaving no earlier partial snapshot whose preservation it can exercise.
    @unittest.skipIf(
        os.name == "nt",
        "communicate timeouts expose no partial pipe snapshot on Windows",
    )
    def test_a_child_killed_mid_stream_preserves_its_partial_stdout(self):
        # The readiness marker is the child's published pid, so that a failure before `terminate`
        # does not leave a SIGTERM-ignoring sleeper behind.
        pidfile = self.dir / "stdout-ready.pid"
        kill_the_group_on_cleanup(self, pidfile)
        child = cell_group.spawn(
            self.python(
                "import signal, sys, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "sys.stdout.write('partial child output')\n"
                "sys.stdout.flush()\n"
                + publish_pid("sys.argv[1]", "os.getpid()")
                + "time.sleep(300)\n",
                pidfile,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # The readiness marker is published only after stdout was flushed. This ordering makes
        # the test deterministic: the parent cannot kill the child before the asserted bytes
        # exist, which was the macOS CI flake in the old combined sleep-chain test.
        self.wait_file(pidfile)
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
        kill_the_group_on_cleanup(self, pidfile)
        child = cell_group.spawn(
            self.python(
                "import subprocess, sys\n"
                "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
                "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                + publish_pid("sys.argv[1]", "kid.pid"),
                pidfile,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        child.communicate(timeout=10)
        self.wait_file(pidfile)
        orphan = int(pidfile.read_text())
        # On POSIX the group kill registered above collects the orphan on every path; Windows has
        # no group to kill, so the orphan is killed by pid there.  `os.kill` with any signal is
        # `TerminateProcess` on Windows, which is what is wanted (`signal.SIGKILL` does not exist).
        if os.name != "posix":
            self.addCleanup(lambda: self.alive(orphan) and os.kill(orphan, signal.SIGTERM))

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
        for path in (HERE / "orchestrate.py", HERE / "gh675_cells.py", HERE / "gh1181_cells.py"):
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

    def test_a_published_pid_file_never_exists_without_its_pid(self):
        # The property the pollers rely on, observed at the one moment it could fail: when the
        # final name appears (the rename), it must not have existed before, and what lands there
        # must already be the whole pid.
        pidfile = self.dir / "published.pid"
        seen = []
        real_replace = os.replace

        def observing_replace(src, dst):
            seen.append((Path(dst).exists(), Path(src).read_text()))
            return real_replace(src, dst)

        with unittest.mock.patch.object(os, "replace", observing_replace):
            exec(publish_pid("path", "4242"), {"path": str(pidfile)})

        self.assertEqual(seen, [(False, "4242")])
        self.assertEqual(pidfile.read_text(), "4242")
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["published.pid"])

    def test_every_published_pid_goes_through_publish_pid(self):
        # Fixtures here are written by copying a neighbour, so one truncating pid write left in
        # any test source is how the empty-read flake comes back (gh-ocannl-1041).
        truncating = re.compile(r"open\([^()\n]*,\s*\\*'w\\*'\)\.write\(str\(")
        # Two-sided, so the scan cannot pass by matching nothing: it catches both spellings it
        # replaced -- plain, and escaped inside a driver's nested source -- and not the helper.
        # (Each is split across two literals so that this file does not match itself.)
        for old in (
            "open(sys.argv[1], 'w')" ".write(str(kid.pid))",
            "open(sys.argv[1], \\'w\\')" ".write(str(os.getpid()))",
        ):
            self.assertRegex(old, truncating)
        self.assertNotRegex(publish_pid("sys.argv[1]", "kid.pid"), truncating)

        sources = sorted([*HERE.glob("test*.py"), *HERE.glob("test/test*.py")])
        self.assertIn(HERE / "test_orchestrate.py", sources)
        self.assertIn(HERE / "test" / "test_cell_group.py", sources)
        offenders = [
            f"{path.relative_to(HERE)}:{lineno}"
            for path in sources
            for lineno, line in enumerate(path.read_text().splitlines(), 1)
            if truncating.search(line)
        ]
        self.assertEqual(offenders, [], "pid files written in place, not via publish_pid")

    def test_every_sleeping_fixture_is_killed_on_cleanup(self):
        # Fixtures here are written by copying a neighbour, and one that parks a sleeper with no
        # kill on cleanup passes every green run: the leak shows only when an assertion fails
        # (gh-ocannl-1054, gh-ocannl-1089). The census is keyed on the SLEEP, not on a published
        # pid: a fixture that never publishes is exactly the one a pid-keyed census cannot see,
        # and nine of them did not (gh-ocannl-1105). A sleeper is a string literal -- the source
        # of a child -- sleeping a second or more; the test process's own polling sleeps are
        # calls, not literals, and a docstring describes rather than spawns.
        sleep = re.compile(r"sleep\(\s*(\d+(?:\.\d*)?)\s*\)")

        def parks(literal):
            return any(float(seconds) >= 1 for seconds in sleep.findall(literal))

        # Two-sided on the predicate, with the needles built at run time so that this test's own
        # literals are not sleepers: the fixtures' spelling matches, a short child sleep does not.
        self.assertTrue(parks("import time; time.sleep(" + "300)"))
        self.assertTrue(parks("'import time; time.sleep(" + "300)'])\n"))
        self.assertFalse(parks("time.sleep(" + "0.2)\n"))
        # (Split so that this test's own source matches neither.)
        publishes, registers = "publish" "_pid(", "kill_the_group" "_on_cleanup(self, pidfile)"
        sources = sorted([*HERE.glob("test*.py"), *HERE.glob("test/test*.py")])
        sleepers, offenders = set(), set()
        for path in sources:
            text = path.read_text()
            tree = ast.parse(text, filename=str(path))
            docstrings = {
                id(node.body[0].value)
                for node in ast.walk(tree)
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
            }
            fixture_of = {
                id(node): method
                for case in ast.walk(tree)
                if isinstance(case, ast.ClassDef)
                for method in case.body
                if isinstance(method, ast.FunctionDef) and method.name.startswith("test")
                for node in ast.walk(method)
            }
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                    and parks(node.value)
                ):
                    continue
                method = fixture_of.get(id(node))
                where = f"{path.relative_to(HERE)}:{node.lineno}"
                if method is None:
                    # A sleeper built in a helper or at module level: no fixture's source can be
                    # read for its cleanup, so it is reported rather than trusted.
                    offenders.add(f"{where} (outside a test method)")
                    continue
                sleepers.add(method.name)
                source = ast.get_source_segment(text, method)
                if publishes not in source or registers not in source:
                    offenders.add(f"{path.relative_to(HERE)}:{method.lineno} {method.name}")
        # Two-sided: the census reaches fixtures in both files -- ones that always published, the
        # ones gh-ocannl-1105 found publishing nothing, and the one whose pid is taken at `Popen`
        # -- so it cannot pass by finding no sleeper at all.
        for found in (
            "test_a_sigterm_to_the_sweep_takes_the_running_cell_with_it",
            "test_a_sleep_chain_is_killed_and_reaped_as_one_group",
            "test_a_cell_over_the_cap_is_a_runner_failure_naming_the_cap",
            "test_a_killed_cell_leaves_a_log_of_what_it_printed",
            "test_a_cancellation_inside_the_spawn_window_still_kills_the_cell",
            "test_the_gh675_spawn_window_defers_cancellation_until_cleanup_is_owned",
        ):
            self.assertIn(found, sleepers)
        self.assertEqual(
            sorted(offenders), [], "sleeping fixtures that publish no pid or kill none on cleanup"
        )

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
        # The child sleeps 300 s in a session of its own; when the cancellation this test pins
        # fails to collect it, nothing else will (gh-ocannl-1089). The pid taken at `Popen` covers
        # a child killed before it ran a line; the published one, one the census can see.
        self.addCleanup(lambda: [kill_the_group(pid) for pid in spawned])
        pidfile = self.dir / "spawn-window.pid"
        kill_the_group_on_cleanup(self, pidfile)
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
                    self.python(
                        "import sys, time\n"
                        + publish_pid("sys.argv[1]", "os.getpid()")
                        + "time.sleep(300)\n",
                        pidfile,
                    ),
                    timeout=60,
                    context="cancelled probe",
                )

        self.assertEqual(len(spawned), 1)
        self.assertTrue(self.wait_gone(spawned[0]), "the spawn-window child outlived cancellation")


if __name__ == "__main__":
    unittest.main()
